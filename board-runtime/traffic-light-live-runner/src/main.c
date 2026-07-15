#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cvi_comm.h>
#include <cvi_sys.h>
#include <sample_comm.h>
#include <core/utils/vpss_helper.h>
#include "cvi_tdl.h"

#include "middleware_utils.h"
#include "live_inference.h"

#define MAX_SHARED_DETECTIONS 3

static volatile sig_atomic_t g_exit_requested = 0;

typedef struct {
  CVI_U32 rtsp_width;
  CVI_U32 rtsp_height;
  CVI_U32 analysis_width;
  CVI_U32 analysis_height;
  int fps;
  int rtsp_port;
  const char *model_path;
  const char *profile;
  float conf;
  float iou;
  int max_det;
  int infer_every;
} AppConfig;

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *mw_context;
  VPSS_GRP group;
  VPSS_CHN channel;
  const char *thread_name;
  unsigned int log_every;
} ChannelThreadArgs;

typedef struct {
  TLLR_InferenceContext *infer_ctx;
  const AppConfig *app_cfg;
  VPSS_GRP group;
  VPSS_CHN channel;
  const char *thread_name;
  unsigned int log_every;
} AnalysisThreadArgs;

/* ── Shared detection buffer between analysis and RTSP threads ── */
typedef struct {
  pthread_mutex_t lock;
  int count;
  unsigned long long frame_id;
  TLLR_Detection detections[MAX_SHARED_DETECTIONS];
} SharedDetections;

static void shared_dets_init(SharedDetections *sd) {
  pthread_mutex_init(&sd->lock, NULL);
  sd->count = 0;
  sd->frame_id = 0;
  memset(sd->detections, 0, sizeof(sd->detections));
}

static void shared_dets_update(SharedDetections *sd, int count,
                               const TLLR_Detection *dets,
                               unsigned long long frame_id) {
  pthread_mutex_lock(&sd->lock);
  sd->count = (count > MAX_SHARED_DETECTIONS) ? MAX_SHARED_DETECTIONS : count;
  sd->frame_id = frame_id;
  if (sd->count > 0 && dets != NULL) {
    memcpy(sd->detections, dets, (size_t)sd->count * sizeof(TLLR_Detection));
  }
  pthread_mutex_unlock(&sd->lock);
}

static int shared_dets_read(SharedDetections *sd, TLLR_Detection *dets,
                            int max_dets) {
  pthread_mutex_lock(&sd->lock);
  int n = (sd->count < max_dets) ? sd->count : max_dets;
  if (n > 0) {
    memcpy(dets, sd->detections, (size_t)n * sizeof(TLLR_Detection));
  }
  pthread_mutex_unlock(&sd->lock);
  return n;
}

static void shared_dets_destroy(SharedDetections *sd) {
  pthread_mutex_destroy(&sd->lock);
}

static float clamp_float(float value, float lo, float hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

static void compute_letterbox_rect(int src_w, int src_h, int canvas_w, int canvas_h,
                                   float *out_x, float *out_y,
                                   float *out_w, float *out_h) {
  const float scale_x = (float)canvas_w / (float)src_w;
  const float scale_y = (float)canvas_h / (float)src_h;
  const float scale = (scale_x < scale_y) ? scale_x : scale_y;
  const float valid_w = (float)src_w * scale;
  const float valid_h = (float)src_h * scale;
  *out_x = ((float)canvas_w - valid_w) * 0.5f;
  *out_y = ((float)canvas_h - valid_h) * 0.5f;
  *out_w = valid_w;
  *out_h = valid_h;
}

static void detections_to_object_meta(const TLLR_Detection *dets, int det_count,
                                      int src_w, int src_h, int dst_w, int dst_h,
                                      cvtdl_object_info_t *infos,
                                      cvtdl_object_t *meta) {
  memset(meta, 0, sizeof(*meta));
  meta->width = (uint32_t)dst_w;
  meta->height = (uint32_t)dst_h;
  meta->info = infos;

  if (dets == NULL || infos == NULL || det_count <= 0 || src_w <= 0 || src_h <= 0) {
    return;
  }

  float valid_x = 0.0f;
  float valid_y = 0.0f;
  float valid_w = (float)src_w;
  float valid_h = (float)src_h;
  compute_letterbox_rect(dst_w, dst_h, src_w, src_h,
                         &valid_x, &valid_y, &valid_w, &valid_h);

  const float sx = (float)dst_w / valid_w;
  const float sy = (float)dst_h / valid_h;
  int out_count = 0;
  for (int i = 0; i < det_count && out_count < MAX_SHARED_DETECTIONS; i++) {
    const float x1 = clamp_float((dets[i].x1 - valid_x) * sx, 0.0f, (float)(dst_w - 1));
    const float y1 = clamp_float((dets[i].y1 - valid_y) * sy, 0.0f, (float)(dst_h - 1));
    const float x2 = clamp_float((dets[i].x2 - valid_x) * sx, 0.0f, (float)(dst_w - 1));
    const float y2 = clamp_float((dets[i].y2 - valid_y) * sy, 0.0f, (float)(dst_h - 1));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    cvtdl_object_info_t *info = &infos[out_count++];
    memset(info, 0, sizeof(*info));
    info->bbox.x1 = x1;
    info->bbox.y1 = y1;
    info->bbox.x2 = x2;
    info->bbox.y2 = y2;
    info->bbox.score = dets[i].confidence;
    info->classes = dets[i].class_id;
    snprintf(info->name, sizeof(info->name), "%s: %.2f",
             tllr_class_name(dets[i].class_id), dets[i].confidence);
  }
  meta->size = (uint32_t)out_count;
}

static void blend_mask_pixel_nv21(uint8_t *y_plane, uint8_t *uv_plane,
                                  int y_stride, int uv_stride,
                                  int x, int y) {
  const int alpha_num = 5;
  const int alpha_den = 10;
  const uint8_t overlay_y = 100;
  const uint8_t overlay_u = 210;
  const uint8_t overlay_v = 90;

  uint8_t *yp = y_plane + (size_t)y * y_stride + x;
  *yp = (uint8_t)(((*yp) * (alpha_den - alpha_num) + overlay_y * alpha_num) / alpha_den);

  if ((x & 1) == 0 && (y & 1) == 0) {
    uint8_t *uv = uv_plane + (size_t)(y / 2) * uv_stride + x;
    uv[0] = (uint8_t)((uv[0] * (alpha_den - alpha_num) + overlay_v * alpha_num) / alpha_den);
    uv[1] = (uint8_t)((uv[1] * (alpha_den - alpha_num) + overlay_u * alpha_num) / alpha_den);
  }
}

static void overlay_segmentation_masks_nv21(const TLLR_Detection *dets, int det_count,
                                            VIDEO_FRAME_INFO_S *frame,
                                            int analysis_w, int analysis_h,
                                            int rtsp_w, int rtsp_h) {
  if (dets == NULL || det_count <= 0 || frame == NULL) {
    return;
  }

  VIDEO_FRAME_S *vf = &frame->stVFrame;
  if (vf->enPixelFormat != VI_PIXEL_FORMAT) {
    return;
  }

  uint8_t *y_plane = vf->pu8VirAddr[0];
  uint8_t *uv_plane = vf->pu8VirAddr[1];
  bool mapped_y = false;
  bool mapped_uv = false;
  if (y_plane == NULL) {
    y_plane = (uint8_t *)CVI_SYS_Mmap(vf->u64PhyAddr[0], vf->u32Length[0]);
    mapped_y = true;
  }
  if (uv_plane == NULL) {
    uv_plane = (uint8_t *)CVI_SYS_Mmap(vf->u64PhyAddr[1], vf->u32Length[1]);
    mapped_uv = true;
  }
  if (y_plane == NULL || uv_plane == NULL) {
    if (mapped_y && y_plane != NULL) CVI_SYS_Munmap(y_plane, vf->u32Length[0]);
    if (mapped_uv && uv_plane != NULL) CVI_SYS_Munmap(uv_plane, vf->u32Length[1]);
    return;
  }

  float valid_x = 0.0f;
  float valid_y = 0.0f;
  float valid_w = 0.0f;
  float valid_h = 0.0f;
  compute_letterbox_rect(rtsp_w, rtsp_h, analysis_w, analysis_h,
                         &valid_x, &valid_y, &valid_w, &valid_h);

  const float sx = (float)rtsp_w / valid_w;
  const float sy = (float)rtsp_h / valid_h;
  const int y_stride = (int)vf->u32Stride[0];
  const int uv_stride = (int)vf->u32Stride[1];

  for (int i = 0; i < det_count; ++i) {
    if (!dets[i].has_mask) {
      continue;
    }
    for (int my = 0; my < 160; my += 4) {
      for (int mx = 0; mx < 160; mx += 4) {
        bool active = false;
        for (int ty = 0; ty < 4 && !active; ++ty) {
          for (int tx = 0; tx < 4; ++tx) {
            if (dets[i].mask160[(my + ty) * 160 + (mx + tx)] != 0) {
              active = true;
              break;
            }
          }
        }
        if (!active) {
          continue;
        }
        const float ax = ((float)mx + 2.0f) * 4.0f;
        const float ay = ((float)my + 2.0f) * 4.0f;
        if (ax < valid_x || ay < valid_y || ax >= valid_x + valid_w || ay >= valid_y + valid_h) {
          continue;
        }
        const int cx = (int)clamp_float((ax - valid_x) * sx, 0.0f, (float)(rtsp_w - 1));
        const int cy = (int)clamp_float((ay - valid_y) * sy, 0.0f, (float)(rtsp_h - 1));
        int rx1 = (int)clamp_float((float)(cx - 5), 0.0f, (float)(rtsp_w - 1));
        int ry1 = (int)clamp_float((float)(cy - 3), 0.0f, (float)(rtsp_h - 1));
        int rx2 = (int)clamp_float((float)(cx + 6), 0.0f, (float)rtsp_w);
        int ry2 = (int)clamp_float((float)(cy + 4), 0.0f, (float)rtsp_h);
        for (int y = ry1; y < ry2; ++y) {
          for (int x = rx1; x < rx2; ++x) {
            blend_mask_pixel_nv21(y_plane, uv_plane, y_stride, uv_stride, x, y);
          }
        }
      }
    }
  }

  if (mapped_y) CVI_SYS_Munmap(y_plane, vf->u32Length[0]);
  if (mapped_uv) CVI_SYS_Munmap(uv_plane, vf->u32Length[1]);
}

/* ── CLI / usage ── */

static void print_usage(const char *prog) {
  printf(
      "Usage: %s [options]\n\n"
      "Options:\n"
      "  --model <path>         Cvimodel path (required)\n"
      "  --profile <name>       Model profile: traffic | crosswalk | mangdao (default: traffic)\n"
      "  --rtsp-width <n>       RTSP output width (default: 1280)\n"
      "  --rtsp-height <n>      RTSP output height (default: 720)\n"
      "  --analysis-width <n>   Analysis channel width (default: 640)\n"
      "  --analysis-height <n>  Analysis channel height (default: 640)\n"
      "  --fps <n>              Sensor / encoder FPS hint (default: 20)\n"
      "  --rtsp-port <n>        RTSP port (default: 554)\n"
      "  --conf <float>         Confidence threshold (default: 0.25)\n"
      "  --iou <float>          IOU threshold (default: 0.50)\n"
      "  --max-det <n>          Max detections per frame (default: 100)\n"
      "  --infer-every <n>      Run inference every N frames (default: 1)\n"
      "  --no-overlay           Disable detection box overlay on RTSP\n"
      "  --help                 Show this message\n\n"
      "Live traffic-light detection: camera -> VPSS -> RTSP + analysis inference.\n",
      prog);
}

static int parse_positive_int(const char *value, int *out) {
  char *end = NULL;
  errno = 0;
  long parsed = strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > 65535) {
    return -1;
  }
  *out = (int)parsed;
  return 0;
}

static int parse_float_arg(const char *value, float *out) {
  char *end = NULL;
  errno = 0;
  float parsed = strtof(value, &end);
  if (errno != 0 || end == value || *end != '\0') {
    return -1;
  }
  *out = parsed;
  return 0;
}

static int parse_args(int argc, char **argv, AppConfig *config, int *no_overlay) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--help") == 0) {
      print_usage(argv[0]);
      return 1;
    }
    if (strcmp(arg, "--no-overlay") == 0) {
      *no_overlay = 1;
      continue;
    }
    if (strcmp(arg, "--model") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --model\n");
        return -1;
      }
      config->model_path = argv[++i];
      continue;
    }
    if (strcmp(arg, "--profile") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --profile\n");
        return -1;
      }
      config->profile = argv[++i];
      continue;
    }
    if (strcmp(arg, "--conf") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --conf\n");
        return -1;
      }
      if (parse_float_arg(argv[++i], &config->conf) != 0) {
        fprintf(stderr, "Invalid float for --conf: %s\n", argv[i]);
        return -1;
      }
      continue;
    }
    if (strcmp(arg, "--iou") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --iou\n");
        return -1;
      }
      if (parse_float_arg(argv[++i], &config->iou) != 0) {
        fprintf(stderr, "Invalid float for --iou: %s\n", argv[i]);
        return -1;
      }
      continue;
    }
    if (strcmp(arg, "--max-det") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --max-det\n");
        return -1;
      }
      int value = 0;
      if (parse_positive_int(argv[++i], &value) != 0) {
        fprintf(stderr, "Invalid value for --max-det: %s\n", argv[i]);
        return -1;
      }
      config->max_det = value;
      continue;
    }
    if (strcmp(arg, "--infer-every") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value after --infer-every\n");
        return -1;
      }
      int value = 0;
      if (parse_positive_int(argv[++i], &value) != 0) {
        fprintf(stderr, "Invalid value for --infer-every: %s\n", argv[i]);
        return -1;
      }
      config->infer_every = value;
      continue;
    }

    if (i + 1 >= argc) {
      fprintf(stderr, "Missing value after %s\n", arg);
      return -1;
    }

    int value = 0;
    if (parse_positive_int(argv[i + 1], &value) != 0) {
      fprintf(stderr, "Invalid numeric value for %s: %s\n", arg, argv[i + 1]);
      return -1;
    }

    if (strcmp(arg, "--rtsp-width") == 0) {
      config->rtsp_width = (CVI_U32)value;
    } else if (strcmp(arg, "--rtsp-height") == 0) {
      config->rtsp_height = (CVI_U32)value;
    } else if (strcmp(arg, "--analysis-width") == 0) {
      config->analysis_width = (CVI_U32)value;
    } else if (strcmp(arg, "--analysis-height") == 0) {
      config->analysis_height = (CVI_U32)value;
    } else if (strcmp(arg, "--fps") == 0) {
      config->fps = value;
    } else if (strcmp(arg, "--rtsp-port") == 0) {
      config->rtsp_port = value;
    } else {
      fprintf(stderr, "Unknown option: %s\n", arg);
      return -1;
    }
    ++i;
  }

  return 0;
}

static double monotonic_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void handle_signal(int signo) {
  if (g_exit_requested) {
    return;
  }

  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);
  signal(SIGHUP, SIG_IGN);
  printf("[signal] received %d, exiting...\n", signo);
  fflush(stdout);
  g_exit_requested = 1;
}

/* ── Middleware config ── */

static CVI_S32 prepare_middleware_config(const AppConfig *app_cfg, SAMPLE_TDL_MW_CONFIG_S *mw_cfg,
                                         SIZE_S *sensor_size) {
  memset(mw_cfg, 0, sizeof(*mw_cfg));

  CVI_S32 ret = SAMPLE_TDL_Get_VI_Config(&mw_cfg->stViConfig);
  if (ret != CVI_SUCCESS || mw_cfg->stViConfig.s32WorkingViNum <= 0) {
    printf("Failed to get VI config from /mnt/data/sensor_cfg.ini, ret=%#x\n", ret);
    return CVI_FAILURE;
  }

  PIC_SIZE_E pic_size;
  ret = SAMPLE_COMM_VI_GetSizeBySensor(
      mw_cfg->stViConfig.astViInfo[0].stSnsInfo.enSnsType,
      &pic_size);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_SYS_GetPicSize(pic_size, sensor_size);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", ret);
    return ret;
  }

  mw_cfg->stVBPoolConfig.u32VBPoolCount = 2;

  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 5;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].u32Height = sensor_size->u32Height;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].u32Width = sensor_size->u32Width;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].bBind = CVI_TRUE;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;

  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].enFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 5;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].u32Height = app_cfg->analysis_height;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].u32Width = app_cfg->analysis_width;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].bBind = CVI_TRUE;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  mw_cfg->stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  mw_cfg->stVPSSPoolConfig.u32VpssGrpCount = 1;
#ifndef __CV186X__
  mw_cfg->stVPSSPoolConfig.stVpssMode.aenInput[0] = VPSS_INPUT_MEM;
  mw_cfg->stVPSSPoolConfig.stVpssMode.enMode = VPSS_MODE_DUAL;
  mw_cfg->stVPSSPoolConfig.stVpssMode.ViPipe[0] = 0;
  mw_cfg->stVPSSPoolConfig.stVpssMode.aenInput[1] = VPSS_INPUT_ISP;
  mw_cfg->stVPSSPoolConfig.stVpssMode.ViPipe[1] = 0;
#endif

  SAMPLE_TDL_VPSS_CONFIG_S *vpss_cfg = &mw_cfg->stVPSSPoolConfig.astVpssConfig[0];
  vpss_cfg->bBindVI = CVI_TRUE;
  vpss_cfg->u32ChnBindVI = 0;
  vpss_cfg->u32ChnCount = 2;

  VPSS_GRP_DEFAULT_HELPER2(&vpss_cfg->stVpssGrpAttr, sensor_size->u32Width, sensor_size->u32Height,
                           VI_PIXEL_FORMAT, 1);
  VPSS_CHN_DEFAULT_HELPER(&vpss_cfg->astVpssChnAttr[0], app_cfg->rtsp_width, app_cfg->rtsp_height,
                          VI_PIXEL_FORMAT, CVI_TRUE);
  VPSS_CHN_DEFAULT_HELPER(&vpss_cfg->astVpssChnAttr[1], app_cfg->analysis_width,
                          app_cfg->analysis_height, PIXEL_FORMAT_RGB_888_PLANAR, CVI_TRUE);
  vpss_cfg->astVpssChnAttr[1].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  vpss_cfg->astVpssChnAttr[1].stAspectRatio.bEnableBgColor = CVI_TRUE;
  vpss_cfg->astVpssChnAttr[1].stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  vpss_cfg->astVpssChnAttr[0].u32Depth = 1;
  vpss_cfg->astVpssChnAttr[1].u32Depth = 1;

  SAMPLE_TDL_Get_Input_Config(&mw_cfg->stVencConfig.stChnInputCfg);
  mw_cfg->stVencConfig.stChnInputCfg.srcFramerate = app_cfg->fps;
  mw_cfg->stVencConfig.stChnInputCfg.framerate = app_cfg->fps;
  mw_cfg->stVencConfig.u32FrameWidth = app_cfg->rtsp_width;
  mw_cfg->stVencConfig.u32FrameHeight = app_cfg->rtsp_height;

  SAMPLE_TDL_Get_RTSP_Config(&mw_cfg->stRTSPConfig.stRTSPConfig);
  mw_cfg->stRTSPConfig.stRTSPConfig.port = app_cfg->rtsp_port;

  return CVI_SUCCESS;
}

/* ── RTSP output thread (with overlay) ── */

typedef struct {
  ChannelThreadArgs base;
  SharedDetections *shared_dets;
  cvitdl_service_handle_t service_handle;
  int rtsp_w;
  int rtsp_h;
  int analysis_w;
  int analysis_h;
  int no_overlay;
} RtspThreadArgs;

static void *run_rtsp_output_thread(void *opaque) {
  RtspThreadArgs *args = (RtspThreadArgs *)opaque;
  VIDEO_FRAME_INFO_S frame;
  uint64_t frame_count = 0;
  double start = monotonic_seconds();
  TLLR_Detection dets[MAX_SHARED_DETECTIONS];
  cvtdl_object_info_t obj_infos[MAX_SHARED_DETECTIONS];
  cvtdl_object_t obj_meta;

  printf("[%s] started on VPSS grp=%d chn=%d overlay=%s\n",
         args->base.thread_name, args->base.group, args->base.channel,
         args->no_overlay ? "off" : "on");
  while (!g_exit_requested) {
    memset(&frame, 0, sizeof(frame));
    CVI_S32 ret = CVI_VPSS_GetChnFrame(args->base.group, args->base.channel, &frame, 2000);
    if (ret != CVI_SUCCESS) {
      printf("[%s] CVI_VPSS_GetChnFrame failed with %#x\n", args->base.thread_name, ret);
      if (!g_exit_requested) {
        g_exit_requested = 1;
      }
      break;
    }

    /* Draw overlay before sending to RTSP */
    if (!args->no_overlay) {
      int n = shared_dets_read(args->shared_dets, dets, MAX_SHARED_DETECTIONS);
      overlay_segmentation_masks_nv21(dets, n, &frame,
                                      args->analysis_w, args->analysis_h,
                                      args->rtsp_w, args->rtsp_h);
      detections_to_object_meta(dets, n,
                                args->analysis_w, args->analysis_h,
                                args->rtsp_w, args->rtsp_h,
                                obj_infos, &obj_meta);
      ret = CVI_TDL_Service_ObjectDrawRect(args->service_handle, &obj_meta, &frame, true,
                                           CVI_TDL_Service_GetDefaultBrush());
      if (ret != CVI_TDL_SUCCESS) {
        printf("[%s] CVI_TDL_Service_ObjectDrawRect failed with %#x\n",
               args->base.thread_name, ret);
      }
    }

    ret = SAMPLE_TDL_Send_Frame_RTSP(&frame, args->base.mw_context);
    CVI_S32 release_ret = CVI_VPSS_ReleaseChnFrame(args->base.group, args->base.channel, &frame);
    if (release_ret != CVI_SUCCESS) {
      printf("[%s] CVI_VPSS_ReleaseChnFrame failed with %#x\n", args->base.thread_name, release_ret);
      g_exit_requested = 1;
      break;
    }
    if (ret != CVI_SUCCESS) {
      printf("[%s] SAMPLE_TDL_Send_Frame_RTSP failed with %#x\n", args->base.thread_name, ret);
      g_exit_requested = 1;
      break;
    }

    frame_count++;
    if (args->base.log_every > 0 && frame_count % args->base.log_every == 0) {
      double elapsed = monotonic_seconds() - start;
      double fps = elapsed > 0.0 ? (double)frame_count / elapsed : 0.0;
      printf("[%s] forwarded %llu frames, avg_fps=%.2f\n", args->base.thread_name,
             (unsigned long long)frame_count, fps);
    }
  }

  printf("[%s] stopped\n", args->base.thread_name);
  return NULL;
}

/* ── Analysis + inference thread ── */

typedef struct {
  AnalysisThreadArgs base;
  SharedDetections *shared_dets;
} InferThreadArgs;

static void *run_analysis_inference_thread(void *opaque) {
  InferThreadArgs *args = (InferThreadArgs *)opaque;
  VIDEO_FRAME_INFO_S frame;
  uint64_t frame_count = 0;
  double start = monotonic_seconds();
  double total_infer_ms = 0.0;

  printf("[%s] started on VPSS grp=%d chn=%d (model=%s, conf=%.2f, iou=%.2f, infer_every=%d)\n",
         args->base.thread_name, args->base.group, args->base.channel,
         args->base.app_cfg->model_path, args->base.app_cfg->conf, args->base.app_cfg->iou,
         args->base.app_cfg->infer_every);

  while (!g_exit_requested) {
    memset(&frame, 0, sizeof(frame));
    CVI_S32 ret = CVI_VPSS_GetChnFrame(args->base.group, args->base.channel, &frame, 2000);
    if (ret != CVI_SUCCESS) {
      printf("[%s] CVI_VPSS_GetChnFrame failed with %#x\n", args->base.thread_name, ret);
      if (!g_exit_requested) {
        g_exit_requested = 1;
      }
      break;
    }

    frame_count++;

    if (frame_count % args->base.app_cfg->infer_every == 0) {
      TLLR_Detection detections[MAX_SHARED_DETECTIONS];
      int det_count = 0;
      double forward_ms = 0.0;
      char summary[512];
      summary[0] = '\0';

      int infer_ret = tllr_inference_run_frame(
          args->base.infer_ctx, &frame,
          detections, MAX_SHARED_DETECTIONS, &det_count, &forward_ms,
          summary, sizeof(summary));

      if (infer_ret != CVI_SUCCESS) {
        printf("[%s] inference failed on frame=%llu\n", args->base.thread_name,
               (unsigned long long)frame_count);
      } else {
        total_infer_ms += forward_ms;
        printf("[%s] frame=%llu %s\n", args->base.thread_name,
               (unsigned long long)frame_count, summary);

        /* Update shared buffer for RTSP overlay */
        shared_dets_update(args->shared_dets, det_count, detections, frame_count);

        if (det_count > 0) {
          const char *first_class = tllr_class_name(detections[0].class_id);
          printf("[%s] *** DETECTED: %s conf=%.3f x1y1=%.0f,%.0f x2y2=%.0f,%.0f ***\n",
                 args->base.thread_name, first_class, detections[0].confidence,
                 detections[0].x1, detections[0].y1,
                 detections[0].x2, detections[0].y2);
        }
      }
    }

    ret = CVI_VPSS_ReleaseChnFrame(args->base.group, args->base.channel, &frame);
    if (ret != CVI_SUCCESS) {
      printf("[%s] CVI_VPSS_ReleaseChnFrame failed with %#x\n", args->base.thread_name, ret);
      g_exit_requested = 1;
      break;
    }
  }

  double elapsed = monotonic_seconds() - start;
  uint64_t infer_count = frame_count / args->base.app_cfg->infer_every;
  double avg_infer_ms = infer_count > 0 ? total_infer_ms / (double)infer_count : 0.0;
  printf("[%s] stopped (frames=%llu infers=%llu avg_forward_ms=%.1f)\n",
         args->base.thread_name, (unsigned long long)frame_count,
         (unsigned long long)infer_count, avg_infer_ms);
  return NULL;
}

/* ── main ── */

int main(int argc, char **argv) {
  AppConfig app_cfg = {
      .rtsp_width = 1280,
      .rtsp_height = 720,
      .analysis_width = 640,
      .analysis_height = 640,
      .fps = 20,
      .rtsp_port = 554,
      .model_path = NULL,
      .profile = "traffic",
      .conf = 0.25f,
      .iou = 0.50f,
      .max_det = 100,
      .infer_every = 1,
  };
  int no_overlay = 0;

  int parse_ret = parse_args(argc, argv, &app_cfg, &no_overlay);
  if (parse_ret == 1) {
    return 0;
  }
  if (parse_ret != 0) {
    print_usage(argv[0]);
    return 1;
  }

  if (app_cfg.model_path == NULL) {
    fprintf(stderr, "Error: --model is required\n");
    print_usage(argv[0]);
    return 1;
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);
  signal(SIGHUP, handle_signal);

  printf("[live] starting RTSP=%ux%u analysis=%ux%u fps=%d port=%d overlay=%s\n",
         app_cfg.rtsp_width, app_cfg.rtsp_height, app_cfg.analysis_width, app_cfg.analysis_height,
         app_cfg.fps, app_cfg.rtsp_port, no_overlay ? "off" : "on");
  printf("[live] model=%s conf=%.2f iou=%.2f max_det=%d infer_every=%d\n",
         app_cfg.model_path, app_cfg.conf, app_cfg.iou, app_cfg.max_det, app_cfg.infer_every);

  SharedDetections shared_dets;
  shared_dets_init(&shared_dets);

  TLLR_InferenceConfig infer_cfg = {
      .model_path = app_cfg.model_path,
      .profile = app_cfg.profile,
      .program_id = 0,
      .conf_threshold = app_cfg.conf,
      .iou_threshold = app_cfg.iou,
      .max_det = app_cfg.max_det,
      .print_tensor_info = 1,
  };
  TLLR_InferenceContext *infer_ctx = NULL;
  int infer_init_ret = tllr_inference_init(&infer_cfg, &infer_ctx);
  if (infer_init_ret != CVI_SUCCESS) {
    printf("[live] tllr_inference_init failed\n");
    shared_dets_destroy(&shared_dets);
    return 1;
  }
  printf("[live] model loaded successfully\n");

  SAMPLE_TDL_MW_CONFIG_S mw_cfg;
  SIZE_S sensor_size;
  CVI_S32 ret = prepare_middleware_config(&app_cfg, &mw_cfg, &sensor_size);
  if (ret != CVI_SUCCESS) {
    printf("[live] prepare_middleware_config failed with %#x\n", ret);
    tllr_inference_destroy(infer_ctx);
    shared_dets_destroy(&shared_dets);
    return 1;
  }

  printf("[live] sensor size=%ux%u\n", sensor_size.u32Width, sensor_size.u32Height);

  SAMPLE_TDL_MW_CONTEXT mw_context;
  memset(&mw_context, 0, sizeof(mw_context));
  ret = SAMPLE_TDL_Init_WM(&mw_cfg, &mw_context);
  if (ret != CVI_SUCCESS) {
    printf("[live] SAMPLE_TDL_Init_WM failed with %#x\n", ret);
    tllr_inference_destroy(infer_ctx);
    shared_dets_destroy(&shared_dets);
    return 1;
  }

  printf("[live] middleware init complete\n");
  printf("[live] RTSP path: rtsp://<board-ip>:%d/h264\n", app_cfg.rtsp_port);

  cvitdl_handle_t tdl_handle = NULL;
  cvitdl_service_handle_t service_handle = NULL;
  ret = CVI_TDL_CreateHandle2(&tdl_handle, 1, 0);
  if (ret != CVI_TDL_SUCCESS) {
    printf("[live] CVI_TDL_CreateHandle2 failed with %#x\n", ret);
    SAMPLE_TDL_Destroy_MW(&mw_context);
    tllr_inference_destroy(infer_ctx);
    shared_dets_destroy(&shared_dets);
    return 1;
  }
  ret = CVI_TDL_Service_CreateHandle(&service_handle, tdl_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("[live] CVI_TDL_Service_CreateHandle failed with %#x\n", ret);
    CVI_TDL_DestroyHandle(tdl_handle);
    SAMPLE_TDL_Destroy_MW(&mw_context);
    tllr_inference_destroy(infer_ctx);
    shared_dets_destroy(&shared_dets);
    return 1;
  }

  RtspThreadArgs rtsp_args = {
      .base = {
          .mw_context = &mw_context,
          .group = 0,
          .channel = VPSS_CHN0,
          .thread_name = "rtsp-output",
          .log_every = 120,
      },
      .shared_dets = &shared_dets,
      .service_handle = service_handle,
      .rtsp_w = (int)app_cfg.rtsp_width,
      .rtsp_h = (int)app_cfg.rtsp_height,
      .analysis_w = (int)app_cfg.analysis_width,
      .analysis_h = (int)app_cfg.analysis_height,
      .no_overlay = no_overlay,
  };

  InferThreadArgs infer_args = {
      .base = {
          .infer_ctx = infer_ctx,
          .app_cfg = &app_cfg,
          .group = 0,
          .channel = VPSS_CHN1,
          .thread_name = "analysis-infer",
          .log_every = 60,
      },
      .shared_dets = &shared_dets,
  };

  pthread_t rtsp_thread;
  pthread_t analysis_thread;
  int thread_created_rtsp = 0;
  int thread_created_analysis = 0;

  if (pthread_create(&rtsp_thread, NULL, run_rtsp_output_thread, &rtsp_args) == 0) {
    thread_created_rtsp = 1;
  } else {
    printf("[live] failed to create rtsp-output thread\n");
    g_exit_requested = 1;
  }

  if (!g_exit_requested && pthread_create(&analysis_thread, NULL, run_analysis_inference_thread,
                                          &infer_args) == 0) {
    thread_created_analysis = 1;
  } else if (!g_exit_requested) {
    printf("[live] failed to create analysis-infer thread\n");
    g_exit_requested = 1;
  }

  while (!g_exit_requested) {
    sleep(1);
  }

  if (thread_created_rtsp) {
    pthread_join(rtsp_thread, NULL);
  }
  if (thread_created_analysis) {
    pthread_join(analysis_thread, NULL);
  }

  CVI_TDL_Service_DestroyHandle(service_handle);
  CVI_TDL_DestroyHandle(tdl_handle);
  SAMPLE_TDL_Destroy_MW(&mw_context);
  tllr_inference_destroy(infer_ctx);
  shared_dets_destroy(&shared_dets);
  printf("[live] shutdown complete\n");
  return 0;
}
