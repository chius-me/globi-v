/**
 * camera_capture.c v4 - Uses exact same VI/VPSS init as working sample_vi_fd
 * (720p NV21 → VPSS → VENC JPEG → file)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <fcntl.h>
#include <endian.h>

typedef struct _ST_AUDIO_UNIT_TEST_CFG { int dummy; } ST_AudioUnitTestCfg;

#include "cvi_comm.h"
#include "cvi_sys.h"
#include "cvi_vi.h"
#include "cvi_vpss.h"
#include "cvi_venc.h"
#include "cvi_vb.h"
#include "cvi_buffer.h"
#include "cvi_comm_vb.h"
#include "core/utils/vpss_helper.h"
#include "sample_comm.h"

#define LOG_TAG "cam_cap"

#define VIDEO_WIDTH  1280
#define VIDEO_HEIGHT 720
#define VI_PIPE_ID   0
#define VPSS_GRP_ID  0
#define VPSS_CHN0    0  /* for TDL (unused) */
#define VPSS_CHN1    1  /* for VENC JPEG */
#define VENC_CHN     0

static int low_mem_profile = 1;
static SAMPLE_VI_CONFIG_S g_viConfig;

#define ION_TOTALMEM "/sys/firmware/devicetree/base/reserved-memory/ion/size"
static void load_ion_totalmem(void) {
    FILE *fp = fopen(ION_TOTALMEM, "r");
    if (!fp) return;
    char buf[16] = ""; int64_t m = 0;
    if (fread(buf, 1, sizeof(buf), fp) > 0) {
        memcpy(&m, buf, sizeof(int64_t));
        if (be64toh(m) > 0 && be64toh(m) < 65*1024*1024) low_mem_profile = 1;
    }
    fclose(fp);
}

/* ─── InitVI: same as vi_vo_utils.c InitVI ─── */
static int init_vi(SAMPLE_VI_CONFIG_S *cfg, SIZE_S *viSize) {
    int ret;
    SAMPLE_INI_CFG_S ini = {};
    VB_CONFIG_S vb = {};
    PIC_SIZE_E picSz;
    CVI_U32 blk;
    VI_PIPE_ATTR_S pa;
    int snsId = 0;

    DYNAMIC_RANGE_E dr = DYNAMIC_RANGE_SDR8;
    PIXEL_FORMAT_E   pf = VI_PIXEL_FORMAT;
    VIDEO_FORMAT_E   vf = VIDEO_FORMAT_LINEAR;
    COMPRESS_MODE_E  cm = low_mem_profile ? COMPRESS_MODE_TILE : COMPRESS_MODE_NONE;

    ini.enSource = VI_PIPE_FRAME_SOURCE_DEV;
    ini.devNum = 2;
    ini.enSnsType[0] = SONY_IMX327_MIPI_2M_30FPS_12BIT;
    ini.enWDRMode[0] = WDR_MODE_NONE;
    ini.s32BusId[0] = 3; ini.MipiDev[0] = 0xFF;
    ini.s32BusId[1] = 0; ini.MipiDev[1] = 0xFF;

    SAMPLE_COMM_VI_ParseIni(&ini);
    SAMPLE_PRT("sensor: %d devNum=%d\n", ini.enSnsType[0], ini.devNum);

    SAMPLE_COMM_VI_GetSensorInfo(cfg);

    /* Manual VI config setup for GC2083 */
    extern ISP_SNS_OBJ_S stSnsGc2083_Obj;
    (void)stSnsGc2083_Obj; /* referenced later */

    for (snsId = 0; snsId < ini.devNum; snsId++) {
        cfg->s32WorkingViNum = 1 + snsId;
        cfg->as32WorkingViId[snsId] = snsId;
        cfg->astViInfo[snsId].stSnsInfo.enSnsType = ini.enSnsType[snsId];
        cfg->astViInfo[snsId].stSnsInfo.MipiDev  = 0;
        cfg->astViInfo[snsId].stSnsInfo.s32BusId = 3;
        cfg->astViInfo[snsId].stSnsInfo.s32SnsI2cAddr = 0x37;
        cfg->astViInfo[snsId].stSnsInfo.as16LaneId[0] = 2;
        cfg->astViInfo[snsId].stSnsInfo.as16LaneId[1] = 0;
        cfg->astViInfo[snsId].stSnsInfo.as16LaneId[2] = 1;
        cfg->astViInfo[snsId].stSnsInfo.as16LaneId[3] = -1;
        cfg->astViInfo[snsId].stSnsInfo.as16LaneId[4] = -1;
        cfg->astViInfo[snsId].stDevInfo.ViDev = snsId;
        cfg->astViInfo[snsId].stPipeInfo.aPipe[0] = snsId;
        cfg->astViInfo[snsId].stDevInfo.enWDRMode = WDR_MODE_NONE;
        cfg->astViInfo[snsId].stPipeInfo.enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;
        cfg->astViInfo[snsId].stPipeInfo.aPipe[1] = -1;
        cfg->astViInfo[snsId].stPipeInfo.aPipe[2] = -1;
        cfg->astViInfo[snsId].stPipeInfo.aPipe[3] = -1;
        cfg->astViInfo[snsId].stChnInfo.ViChn = snsId;
        cfg->astViInfo[snsId].stChnInfo.enPixFormat = pf;
        cfg->astViInfo[snsId].stChnInfo.enDynamicRange = dr;
        cfg->astViInfo[snsId].stChnInfo.enVideoFormat = vf;
        cfg->astViInfo[snsId].stChnInfo.enCompressMode = cm;
    }
    SAMPLE_PRT("s32WorkingViNum=%d\n", cfg->s32WorkingViNum);

    ret = SAMPLE_COMM_VI_GetSizeBySensor(cfg->astViInfo[0].stSnsInfo.enSnsType, &picSz);
    if (ret) return ret;
    ret = SAMPLE_COMM_SYS_GetPicSize(picSz, viSize);
    SAMPLE_PRT("viSize=%dx%d\n", viSize->u32Width, viSize->u32Height);

    /* VB pools: 720p NV21 main + 720p NV21 for VPSS output */
    SIZE_S voSz = {VIDEO_WIDTH, VIDEO_HEIGHT};
    memset(&vb, 0, sizeof(vb)); vb.u32MaxPoolCnt = 2;
    vb.astCommPool[0].u32BlkSize = COMMON_GetPicBufferSize(viSize->u32Width, viSize->u32Height,
        SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    vb.astCommPool[0].u32BlkCnt = low_mem_profile ? 3 : 7;
    vb.astCommPool[1].u32BlkSize = COMMON_GetPicBufferSize(VIDEO_WIDTH, VIDEO_HEIGHT,
        VI_PIXEL_FORMAT, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    vb.astCommPool[1].u32BlkCnt = 3;
    ret = SAMPLE_COMM_SYS_Init(&vb);
    if (ret) { SAMPLE_PRT("SYS_Init %#x\n", ret); return ret; }
    ret = CVI_SYS_SetVPSSMode(VPSS_MODE_DUAL);
    if (ret) { SAMPLE_PRT("SetVPSSMode %#x\n", ret); return ret; }

    /* StartSensor for GC2083 */
    for (int i = 0; i < cfg->s32WorkingViNum; i++) {
        CVI_U32 sid = cfg->astViInfo[i].stSnsInfo.s32SnsId;
        ret = SAMPLE_COMM_ISP_SetSnsObj(sid, cfg->astViInfo[i].stSnsInfo.enSnsType);
        if (ret) return ret;
        ret = SAMPLE_COMM_ISP_SetSnsInit(sid, 0);
        if (ret) return ret;

        RX_INIT_ATTR_S rx = {};
        rx.MipiDev = cfg->astViInfo[i].stSnsInfo.MipiDev;
        for (int j = 0; j < 5; j++) rx.as16LaneId[j] = cfg->astViInfo[i].stSnsInfo.as16LaneId[j];
        ret = stSnsGc2083_Obj.pfnPatchRxAttr ? stSnsGc2083_Obj.pfnPatchRxAttr(&rx) : CVI_SUCCESS;
        if (ret) return ret;

        ISP_INIT_ATTR_S ia = {}; ia.enGainMode = SNS_GAIN_MODE_SHARE;
        if (stSnsGc2083_Obj.pfnSetInit) {
            ret = stSnsGc2083_Obj.pfnSetInit(VI_PIPE_ID, &ia);
            if (ret) return ret;
        }
        ISP_SNS_COMMBUS_U bi = {.s8I2cDev = (CVI_S8)cfg->astViInfo[i].stSnsInfo.s32BusId};
        ret = stSnsGc2083_Obj.pfnSetBusInfo(VI_PIPE_ID, bi);
        if (ret) return ret;
        if (stSnsGc2083_Obj.pfnPatchI2cAddr)
            stSnsGc2083_Obj.pfnPatchI2cAddr(cfg->astViInfo[i].stSnsInfo.s32SnsI2cAddr);
    }

    ret = SAMPLE_COMM_VI_StartDev(&cfg->astViInfo[0]);
    if (ret) return ret;
    ret = SAMPLE_COMM_VI_StartMIPI(cfg);
    if (ret) return ret;

    /* VI Pipe */
    memset(&pa, 0, sizeof(pa));
    pa.u32MaxW = viSize->u32Width; pa.u32MaxH = viSize->u32Height;
    pa.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP; pa.enBitWidth = DATA_BITWIDTH_12;
    pa.stFrameRate.s32SrcFrameRate = -1; pa.stFrameRate.s32DstFrameRate = -1;
    pa.bNrEn = CVI_TRUE; pa.enCompressMode = cm;
    ret = CVI_VI_CreatePipe(VI_PIPE_ID, &pa);
    if (ret) return ret;
    ret = CVI_VI_StartPipe(VI_PIPE_ID);
    if (ret) return ret;
    ret = SAMPLE_COMM_VI_CreateIsp(cfg);
    if (ret) return ret;
    ret = SAMPLE_COMM_VI_StartViChn(cfg);
    if (ret) return ret;
    SAMPLE_PRT("VI init OK\n");
    return 0;
}

/* ─── VPSS: exact same as working vi_vo_utils.c InitVPSS ─── */
static int init_vpss(SIZE_S *viSize) {
    VPSS_GRP_ATTR_S ga;
    VPSS_CHN_ATTR_S ca[VPSS_MAX_PHY_CHN_NUM] = {};
    CVI_BOOL en[VPSS_MAX_PHY_CHN_NUM] = {};

    /* Channel 0: 720p NV21 for TDL (unused but needed for VI bind) */
    en[VPSS_CHN0] = CVI_TRUE;
    memset(&ca[VPSS_CHN0], 0, sizeof(VPSS_CHN_ATTR_S));
    ca[VPSS_CHN0].u32Width = VIDEO_WIDTH;
    ca[VPSS_CHN0].u32Height = VIDEO_HEIGHT;
    ca[VPSS_CHN0].enVideoFormat = VIDEO_FORMAT_LINEAR;
    ca[VPSS_CHN0].enPixelFormat = VI_PIXEL_FORMAT;
    ca[VPSS_CHN0].u32Depth = 1;
    ca[VPSS_CHN0].stFrameRate.s32SrcFrameRate = -1;
    ca[VPSS_CHN0].stFrameRate.s32DstFrameRate = -1;
    ca[VPSS_CHN0].stAspectRatio.enMode = ASPECT_RATIO_NONE;

    /* Channel 1: 720p NV21 for VENC JPEG */
    en[VPSS_CHN1] = CVI_TRUE;
    memset(&ca[VPSS_CHN1], 0, sizeof(VPSS_CHN_ATTR_S));
    ca[VPSS_CHN1].u32Width = VIDEO_WIDTH;
    ca[VPSS_CHN1].u32Height = VIDEO_HEIGHT;
    ca[VPSS_CHN1].enVideoFormat = VIDEO_FORMAT_LINEAR;
    ca[VPSS_CHN1].enPixelFormat = VI_PIXEL_FORMAT;
    ca[VPSS_CHN1].u32Depth = 1;
    ca[VPSS_CHN1].stFrameRate.s32SrcFrameRate = -1;
    ca[VPSS_CHN1].stFrameRate.s32DstFrameRate = -1;
    ca[VPSS_CHN1].stAspectRatio.enMode = ASPECT_RATIO_NONE;

    VPSS_GRP_DEFAULT_HELPER2(&ga, viSize->u32Width, viSize->u32Height, VI_PIXEL_FORMAT, 0);

    int ret = SAMPLE_COMM_VPSS_Init(VPSS_GRP_ID, en, &ga, ca);
    if (ret) { SAMPLE_PRT("VPSS_Init %#x\n", ret); return ret; }
    ret = SAMPLE_COMM_VPSS_Start(VPSS_GRP_ID, en, &ga, ca);
    if (ret) { SAMPLE_PRT("VPSS_Start %#x\n", ret); return ret; }
    ret = SAMPLE_COMM_VI_Bind_VPSS(VI_PIPE_ID, VPSS_CHN0, VPSS_GRP_ID);
    if (ret) { SAMPLE_PRT("Bind_VPSS %#x\n", ret); return ret; }
    SAMPLE_PRT("VPSS OK\n");
    return 0;
}

/* ─── VENC JPEG ─── */
static int init_venc(void) {
    chnInputCfg ic = {};
    strcpy(ic.codec, "jpg");
    ic.initialDelay = CVI_INITIAL_DELAY_DEFAULT;
    ic.width = VIDEO_WIDTH; ic.height = VIDEO_HEIGHT;
    ic.vpssGrp = VPSS_GRP_ID; ic.vpssChn = VPSS_CHN1;
    ic.num_frames = -1; ic.rcMode = SAMPLE_RC_FIXQP; ic.quality = 90;
    ic.srcFramerate = 25; ic.framerate = 25;

    VENC_GOP_ATTR_S gop = {};
    int ret = SAMPLE_COMM_VENC_GetGopAttr(VENC_GOPMODE_NORMALP, &gop);
    if (ret) { SAMPLE_PRT("GetGopAttr %#x\n", ret); return ret; }
    ret = SAMPLE_COMM_VENC_Start(&ic, VENC_CHN, PT_JPEG, PIC_720P,
                                  (SAMPLE_RC_E)ic.rcMode, 0, CVI_FALSE, &gop);
    if (ret) { SAMPLE_PRT("VENC_Start %#x\n", ret); return ret; }
    SAMPLE_PRT("VENC JPEG OK\n");
    return 0;
}

/* ─── Capture Frame to JPEG ─── */
static int capture(const char *path) {
    VIDEO_FRAME_INFO_S f;
    int ret = CVI_VPSS_GetChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f, 3000);
    if (ret) { SAMPLE_PRT("GetChnFrame %#x\n", ret); return ret; }

    ret = CVI_VENC_SendFrame(VENC_CHN, &f, 20000);
    if (ret) { SAMPLE_PRT("SendFrame %#x\n", ret); CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f); return ret; }

    VENC_CHN_STATUS_S st;
    ret = CVI_VENC_QueryStatus(VENC_CHN, &st);
    if (ret || !st.u32CurPacks) {
        CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f);
        SAMPLE_PRT("QueryStatus %#x packs=%d\n", ret, st.u32CurPacks);
        return CVI_FAILURE;
    }

    VENC_STREAM_S s;
    s.pstPack = malloc(sizeof(VENC_PACK_S) * st.u32CurPacks);
    if (!s.pstPack) { CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f); return -1; }
    ret = CVI_VENC_GetStream(VENC_CHN, &s, -1);
    if (ret) { free(s.pstPack); CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f); return ret; }

    FILE *fp = fopen(path, "wb");
    size_t total = 0;
    if (fp) {
        for (unsigned i = 0; i < s.u32PackCount; i++)
            total += fwrite(s.pstPack[i].pu8Addr + s.pstPack[i].u32Offset, 1,
                            s.pstPack[i].u32Len - s.pstPack[i].u32Offset, fp);
        fclose(fp);
    }
    SAMPLE_PRT("JPEG: %s (%zu bytes)\n", path, total);

    CVI_VENC_ReleaseStream(VENC_CHN, &s);
    CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f);
    free(s.pstPack);
    return fp ? 0 : -1;
}

static void cleanup(void) {
    SAMPLE_COMM_VENC_Stop(VENC_CHN);
    CVI_BOOL en[VPSS_MAX_PHY_CHN_NUM] = {};
    en[VPSS_CHN0] = CVI_TRUE; en[VPSS_CHN1] = CVI_TRUE;
    SAMPLE_COMM_VI_UnBind_VPSS(VI_PIPE_ID, VPSS_CHN0, VPSS_GRP_ID);
    SAMPLE_COMM_VPSS_Stop(VPSS_GRP_ID, en);
    SAMPLE_COMM_VI_DestroyVi(&g_viConfig);
    SAMPLE_COMM_SYS_Exit();
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <output.jpg>\n", argv[0]); return 1; }
    SIZE_S vs = {};
    int ret;
    memset(&g_viConfig, 0, sizeof(g_viConfig));
    SAMPLE_PRT("Camera Capture v4\n");
    load_ion_totalmem();

    ret = init_vi(&g_viConfig, &vs);
    if (ret) { SAMPLE_PRT("VI fail %#x\n", ret); cleanup(); return 1; }

    ret = init_vpss(&vs);
    if (ret) { SAMPLE_PRT("VPSS fail\n"); cleanup(); return 1; }

    ret = init_venc();
    if (ret) { SAMPLE_PRT("VENC fail\n"); cleanup(); return 1; }

    SAMPLE_PRT("Warmup...\n");
    for (int i = 0; i < 5; i++) {
        VIDEO_FRAME_INFO_S f;
        ret = CVI_VPSS_GetChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f, 2000);
        if (ret == CVI_SUCCESS) CVI_VPSS_ReleaseChnFrame(VPSS_GRP_ID, VPSS_CHN1, &f);
        usleep(100000);
    }

    SAMPLE_PRT("Capture...\n");
    ret = capture(argv[1]);
    cleanup();
    SAMPLE_PRT("Done ret=%d\n", ret);
    return ret;
}
