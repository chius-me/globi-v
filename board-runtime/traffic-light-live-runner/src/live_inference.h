#ifndef TLLR_LIVE_INFERENCE_H_
#define TLLR_LIVE_INFERENCE_H_

#include <stddef.h>

#include <cvi_comm.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *model_path;
  const char *profile;
  int program_id;
  float conf_threshold;
  float iou_threshold;
  int max_det;
  int print_tensor_info;
} TLLR_InferenceConfig;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float confidence;
  int class_id;
  int has_mask;
  unsigned char mask160[160 * 160];
} TLLR_Detection;

typedef struct TLLR_InferenceContext TLLR_InferenceContext;

int tllr_inference_init(const TLLR_InferenceConfig *config, TLLR_InferenceContext **out_ctx);
void tllr_inference_destroy(TLLR_InferenceContext *ctx);
int tllr_inference_run_frame(TLLR_InferenceContext *ctx, const VIDEO_FRAME_INFO_S *frame,
                             TLLR_Detection *detections, int max_detections,
                             int *out_count, double *out_forward_ms,
                             char *summary, size_t summary_size);
const char *tllr_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif  // TLLR_LIVE_INFERENCE_H_
