#include "live_inference.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cviruntime.h>
#include <cvi_sys.h>

namespace {

constexpr int kRegMax = 16;
constexpr int kMaxClasses = 8;
constexpr std::array<int, 3> kExpectedStrides{{8, 16, 32}};

struct ModelProfile {
  const char *name;
  int class_count;
  std::array<const char *, kMaxClasses> class_names;
  std::array<std::array<std::array<float, 2>, 3>, 3> anchors;
};

const ModelProfile kTrafficProfile{
    "traffic",
    4,
    {{"red_light", "yellow_light", "green_light", "off", nullptr, nullptr, nullptr, nullptr}},
    {{
    {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}},
    {{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}},
    {{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}},
    }}};

const ModelProfile kCrosswalkProfile{
    "crosswalk",
    2,
    {{"crosswalk", "guide_arrows", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}},
    {{
    {{{105.375f, 15.609375f}, {224.5f, 18.40625f}, {362.0f, 30.9375f}}},
    {{{426.0f, 53.5f}, {321.75f, 71.75f}, {539.0f, 67.8125f}}},
    {{{468.75f, 109.0625f}, {553.0f, 92.5f}, {639.5f, 121.5625f}}},
    }}};

const ModelProfile kMangdaoProfile{
    "mangdao",
    1,
    {{"tactile_paving", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}},
    {{
    {{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}},
    {{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}},
    {{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}},
    }}};

const ModelProfile *g_active_profile = &kTrafficProfile;

const ModelProfile *FindProfile(const char *name) {
  if (name == nullptr || std::strcmp(name, "traffic") == 0) {
    return &kTrafficProfile;
  }
  if (std::strcmp(name, "crosswalk") == 0) {
    return &kCrosswalkProfile;
  }
  if (std::strcmp(name, "mangdao") == 0) {
    return &kMangdaoProfile;
  }
  return nullptr;
}

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float confidence = 0.0f;
  int class_id = -1;
  int anchor_index = -1;
  bool has_mask = false;
  std::array<uint8_t, 160 * 160> mask160{};
};

struct HeadPair {
  CVI_TENSOR *reg = nullptr;
  CVI_TENSOR *cls = nullptr;
  int stride = 0;
};

struct YoloV5SplitHead {
  CVI_TENSOR *bbox = nullptr;
  CVI_TENSOR *obj = nullptr;
  CVI_TENSOR *cls = nullptr;
  int stride = 0;
  std::array<std::array<float, 2>, 3> anchors{};
};

const char *FormatToString(CVI_FMT fmt) {
  switch (fmt) {
    case CVI_FMT_FP32:
      return "fp32";
    case CVI_FMT_INT32:
      return "i32";
    case CVI_FMT_UINT32:
      return "u32";
    case CVI_FMT_BF16:
      return "bf16(raw-u16)";
    case CVI_FMT_INT16:
      return "i16";
    case CVI_FMT_UINT16:
      return "u16";
    case CVI_FMT_INT8:
      return "i8";
    case CVI_FMT_UINT8:
      return "u8";
    default:
      return "unknown";
  }
}

std::string TensorName(const CVI_TENSOR &tensor) {
  return tensor.name ? std::string(tensor.name) : std::string("<unnamed>");
}

void PrintTensorList(const char *title, CVI_TENSOR *tensors, int tensor_num) {
  std::printf("%s (%d)\n", title, tensor_num);
  for (int i = 0; i < tensor_num; ++i) {
    CVI_TENSOR &tensor = tensors[i];
    std::printf("  [%d] %s <%d,%d,%d,%d> fmt=%s count=%zu mem=%zu qscale=%g zp=%d\n",
                i, TensorName(tensor).c_str(),
                tensor.shape.dim[0], tensor.shape.dim[1], tensor.shape.dim[2], tensor.shape.dim[3],
                FormatToString(tensor.fmt),
                CVI_NN_TensorCount(&tensor),
                static_cast<size_t>(tensor.mem_size),
                CVI_NN_TensorQuantScale(&tensor),
                CVI_NN_TensorQuantZeroPoint(&tensor));
  }
}

template <typename T>
T ClampValue(T value, T lo, T hi) {
  return std::max(lo, std::min(hi, value));
}

float Sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float TensorValueAsFloat(const CVI_TENSOR &tensor, size_t index) {
  const float scale = CVI_NN_TensorQuantScale(const_cast<CVI_TENSOR *>(&tensor));
  const int zero_point = CVI_NN_TensorQuantZeroPoint(const_cast<CVI_TENSOR *>(&tensor));
  switch (tensor.fmt) {
    case CVI_FMT_FP32:
      return static_cast<const float *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index];
    case CVI_FMT_INT8:
      return (static_cast<const int8_t *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index] - zero_point) * scale;
    case CVI_FMT_UINT8:
      return (static_cast<const uint8_t *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index] - zero_point) * scale;
    case CVI_FMT_INT16:
      return static_cast<const int16_t *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index] * scale;
    case CVI_FMT_UINT16:
      return (static_cast<const uint16_t *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index] - zero_point) * scale;
    case CVI_FMT_BF16:
      return Bf16ToFloat(static_cast<const uint16_t *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor)))[index]);
    default:
      return 0.0f;
  }
}

int SetFp32TensorFromRgbPlanarFrame(CVI_TENSOR *input, const VIDEO_FRAME_S *vf) {
  if (input == nullptr || vf == nullptr || input->fmt != CVI_FMT_FP32) {
    return CVI_FAILURE;
  }
  float *dst = static_cast<float *>(CVI_NN_TensorPtr(input));
  if (dst == nullptr) {
    return CVI_FAILURE;
  }

  const int width = static_cast<int>(vf->u32Width);
  const int height = static_cast<int>(vf->u32Height);
  const size_t plane_size = static_cast<size_t>(width) * height;
  std::array<uint8_t *, 3> planes{{nullptr, nullptr, nullptr}};
  std::array<bool, 3> mapped{{false, false, false}};

  for (int c = 0; c < 3; ++c) {
    if (vf->pu8VirAddr[c] != nullptr) {
      planes[c] = vf->pu8VirAddr[c];
    } else {
      planes[c] = static_cast<uint8_t *>(CVI_SYS_Mmap(vf->u64PhyAddr[c], vf->u32Length[c]));
      mapped[c] = true;
    }
    if (planes[c] == nullptr) {
      for (int j = 0; j < c; ++j) {
        if (mapped[j] && planes[j] != nullptr) {
          CVI_SYS_Munmap(planes[j], vf->u32Length[j]);
        }
      }
      return CVI_FAILURE;
    }
  }

  for (int c = 0; c < 3; ++c) {
    float *dst_plane = dst + static_cast<size_t>(c) * plane_size;
    const uint8_t *src_plane = planes[c];
    const int stride = static_cast<int>(vf->u32Stride[c]);
    for (int y = 0; y < height; ++y) {
      const uint8_t *src_row = src_plane + static_cast<size_t>(y) * stride;
      float *dst_row = dst_plane + static_cast<size_t>(y) * width;
      for (int x = 0; x < width; ++x) {
        dst_row[x] = static_cast<float>(src_row[x]) * (1.0f / 255.0f);
      }
    }
  }

  for (int c = 0; c < 3; ++c) {
    if (mapped[c]) {
      CVI_SYS_Munmap(planes[c], vf->u32Length[c]);
    }
  }
  return CVI_SUCCESS;
}

float DecodeDflDistance(const CVI_TENSOR &reg_tensor, int side, size_t anchor_index) {
  const size_t anchors = static_cast<size_t>(reg_tensor.shape.dim[2]) * reg_tensor.shape.dim[3];
  const int channel_base = side * kRegMax;

  float max_logit = -std::numeric_limits<float>::infinity();
  std::array<float, kRegMax> logits{};
  for (int i = 0; i < kRegMax; ++i) {
    const size_t index = static_cast<size_t>(channel_base + i) * anchors + anchor_index;
    logits[i] = TensorValueAsFloat(reg_tensor, index);
    max_logit = std::max(max_logit, logits[i]);
  }

  float sum_exp = 0.0f;
  float weighted_sum = 0.0f;
  for (int i = 0; i < kRegMax; ++i) {
    const float e = std::exp(logits[i] - max_logit);
    sum_exp += e;
    weighted_sum += e * static_cast<float>(i);
  }
  if (sum_exp <= 0.0f) {
    return 0.0f;
  }
  return weighted_sum / sum_exp;
}

std::vector<HeadPair> CollectHeadPairs(CVI_TENSOR *outputs, int output_num,
                                       const ModelProfile &profile) {
  std::vector<CVI_TENSOR *> reg_heads;
  std::vector<CVI_TENSOR *> cls_heads;
  for (int i = 0; i < output_num; ++i) {
    CVI_TENSOR *tensor = &outputs[i];
    if (tensor->shape.dim[0] != 1) {
      continue;
    }
    if (tensor->shape.dim[1] == 4 * kRegMax) {
      reg_heads.push_back(tensor);
    } else if (tensor->shape.dim[1] == profile.class_count) {
      cls_heads.push_back(tensor);
    }
  }

  auto sort_by_hw_desc = [](CVI_TENSOR *a, CVI_TENSOR *b) {
    const int area_a = a->shape.dim[2] * a->shape.dim[3];
    const int area_b = b->shape.dim[2] * b->shape.dim[3];
    return area_a > area_b;
  };
  std::sort(reg_heads.begin(), reg_heads.end(), sort_by_hw_desc);
  std::sort(cls_heads.begin(), cls_heads.end(), sort_by_hw_desc);

  if (reg_heads.size() != 3 || cls_heads.size() != 3) {
    return {};
  }

  std::vector<HeadPair> pairs;
  for (size_t i = 0; i < 3; ++i) {
    if (reg_heads[i]->shape.dim[2] != cls_heads[i]->shape.dim[2] ||
        reg_heads[i]->shape.dim[3] != cls_heads[i]->shape.dim[3]) {
      return {};
    }
    const int feature_h = reg_heads[i]->shape.dim[2];
    if (feature_h <= 0) {
      return {};
    }
    HeadPair pair;
    pair.reg = reg_heads[i];
    pair.cls = cls_heads[i];
    pair.stride = 640 / feature_h;
    pairs.push_back(pair);
  }

  if (pairs.size() != 3 ||
      pairs[0].stride != kExpectedStrides[0] ||
      pairs[1].stride != kExpectedStrides[1] ||
      pairs[2].stride != kExpectedStrides[2]) {
    return {};
  }
  return pairs;
}

std::vector<Detection> DecodeYolov5sSplitDetections(const CVI_TENSOR *bbox_out,
                                                   const CVI_TENSOR *cls_out,
                                                   float conf_threshold) {
  std::vector<Detection> detections;
  if (bbox_out == nullptr || cls_out == nullptr) return detections;

  const int nc = cls_out->shape.dim[1];
  const int anchors = bbox_out->shape.dim[2];

  for (int a = 0; a < anchors; ++a) {
    float best_conf = -1.0f;
    int best_class = -1;
    for (int c = 0; c < nc && c < g_active_profile->class_count; ++c) {
      const size_t idx = static_cast<size_t>(c) * anchors + a;
      float conf = TensorValueAsFloat(*cls_out, idx);
      if (conf > best_conf) { best_conf = conf; best_class = c; }
    }
    if (best_conf < conf_threshold) continue;

    /* Bbox format is [cx, cy, w, h] — convert to xyxy */
    float cx = TensorValueAsFloat(*bbox_out, a);
    float cy = TensorValueAsFloat(*bbox_out, static_cast<size_t>(anchors) + a);
    float w  = TensorValueAsFloat(*bbox_out, static_cast<size_t>(2) * anchors + a);
    float h  = TensorValueAsFloat(*bbox_out, static_cast<size_t>(3) * anchors + a);

    Detection det;
    det.x1 = cx - w * 0.5f;
    det.y1 = cy - h * 0.5f;
    det.x2 = cx + w * 0.5f;
    det.y2 = cy + h * 0.5f;
    det.confidence = best_conf;
    det.class_id = best_class;
    detections.push_back(det);
  }
  return detections;
}

std::vector<Detection> DecodeYolov5sDetections(const CVI_TENSOR *output, float conf_threshold) {
  std::vector<Detection> detections;
  if (output == nullptr || output->shape.dim[1] < 5) {
    return detections;
  }

  const int nc = output->shape.dim[1] - 4;   /* class count */
  const int anchors = output->shape.dim[2];     /* total anchors (e.g. 8400) */

  for (int a = 0; a < anchors; ++a) {
    /* Find best class and its confidence */
    float best_conf = -1.0f;
    int best_class = -1;
    for (int c = 0; c < nc && c < g_active_profile->class_count; ++c) {
      /* layout: [1, 4+nc, anchors, 1] → index = (4+c)*anchors + a */
      const size_t idx = static_cast<size_t>(4 + c) * anchors + a;
      float conf = TensorValueAsFloat(*output, idx);
      if (conf > best_conf) {
        best_conf = conf;
        best_class = c;
      }
    }
    if (best_conf < conf_threshold) continue;

    /* Read bbox [cx, cy, w, h] — convert to xyxy */
    const size_t base = static_cast<size_t>(a);
    const float cx = TensorValueAsFloat(*output, base);
    const float cy = TensorValueAsFloat(*output, static_cast<size_t>(anchors) + base);
    const float w  = TensorValueAsFloat(*output, static_cast<size_t>(2) * anchors + base);
    const float h  = TensorValueAsFloat(*output, static_cast<size_t>(3) * anchors + base);

    Detection det;
    det.x1 = cx - w * 0.5f;
    det.y1 = cy - h * 0.5f;
    det.x2 = cx + w * 0.5f;
    det.y2 = cy + h * 0.5f;
    det.confidence = best_conf;
    det.class_id = best_class;
    detections.push_back(det);
  }

  return detections;
}

std::vector<Detection> DecodeMangdaoYolov8SegBoxes(const CVI_TENSOR *det_out,
                                                   float conf_threshold) {
  std::vector<Detection> detections;
  if (det_out == nullptr || det_out->shape.dim[0] != 1 ||
      det_out->shape.dim[1] < 5 || det_out->shape.dim[2] <= 0) {
    return detections;
  }

  const int anchors = det_out->shape.dim[2];
  for (int a = 0; a < anchors; ++a) {
    const float score = TensorValueAsFloat(*det_out, static_cast<size_t>(4) * anchors + a);
    if (score < conf_threshold) {
      continue;
    }

    const float cx = TensorValueAsFloat(*det_out, a);
    const float cy = TensorValueAsFloat(*det_out, static_cast<size_t>(anchors) + a);
    const float w = TensorValueAsFloat(*det_out, static_cast<size_t>(2) * anchors + a);
    const float h = TensorValueAsFloat(*det_out, static_cast<size_t>(3) * anchors + a);

    Detection det;
    det.x1 = cx - w * 0.5f;
    det.y1 = cy - h * 0.5f;
    det.x2 = cx + w * 0.5f;
    det.y2 = cy + h * 0.5f;
    det.confidence = score;
    det.class_id = 0;
    det.anchor_index = a;
    detections.push_back(det);
  }
  return detections;
}

void FillMangdaoMasks(std::vector<Detection> *detections,
                      const CVI_TENSOR *det_out,
                      const CVI_TENSOR *proto_out) {
  if (detections == nullptr || det_out == nullptr || proto_out == nullptr ||
      proto_out->shape.dim[1] != 32 || proto_out->shape.dim[2] != 160 ||
      proto_out->shape.dim[3] != 160) {
    return;
  }
  constexpr int kMaskSize = 160;
  constexpr int kInputSize = 640;
  constexpr int kMaskPixels = kMaskSize * kMaskSize;
  const int anchors = det_out->shape.dim[2];

  for (Detection &det : *detections) {
    if (det.anchor_index < 0 || det.anchor_index >= anchors) {
      continue;
    }

    std::array<float, 32> coeff{};
    for (int c = 0; c < 32; ++c) {
      coeff[c] = TensorValueAsFloat(*det_out, static_cast<size_t>(5 + c) * anchors + det.anchor_index);
    }

    const int x1 = ClampValue(static_cast<int>(std::floor(det.x1 * kMaskSize / kInputSize)), 0, kMaskSize);
    const int y1 = ClampValue(static_cast<int>(std::floor(det.y1 * kMaskSize / kInputSize)), 0, kMaskSize);
    const int x2 = ClampValue(static_cast<int>(std::ceil(det.x2 * kMaskSize / kInputSize)), 0, kMaskSize);
    const int y2 = ClampValue(static_cast<int>(std::ceil(det.y2 * kMaskSize / kInputSize)), 0, kMaskSize);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    det.mask160.fill(0);
    for (int y = y1; y < y2; ++y) {
      for (int x = x1; x < x2; ++x) {
        const int p = y * kMaskSize + x;
        float sum = 0.0f;
        for (int c = 0; c < 32; ++c) {
          sum += coeff[c] * TensorValueAsFloat(*proto_out, static_cast<size_t>(c) * kMaskPixels + p);
        }
        if (Sigmoid(sum) > 0.5f) {
          det.mask160[p] = 255;
          det.has_mask = true;
        }
      }
    }
  }
}

bool LooksLikeYoloV5Split(CVI_TENSOR *outputs, int output_num,
                          const ModelProfile &profile) {
  if (output_num != 9) {
    return false;
  }
  const std::array<int, 3> sizes{{80, 40, 20}};
  for (int i = 0; i < 3; ++i) {
    CVI_TENSOR &bbox = outputs[i * 3 + 0];
    CVI_TENSOR &obj = outputs[i * 3 + 1];
    CVI_TENSOR &cls = outputs[i * 3 + 2];
    const int size = sizes[i];
    if (bbox.shape.dim[0] != 3 || bbox.shape.dim[1] != size ||
        bbox.shape.dim[2] != size || bbox.shape.dim[3] != 4) {
      return false;
    }
    if (obj.shape.dim[0] != 3 || obj.shape.dim[1] != size ||
        obj.shape.dim[2] != size || obj.shape.dim[3] != 1) {
      return false;
    }
    if (cls.shape.dim[0] != 3 || cls.shape.dim[1] != size ||
        cls.shape.dim[2] != size || cls.shape.dim[3] != profile.class_count) {
      return false;
    }
  }
  return true;
}

std::vector<YoloV5SplitHead> CollectYoloV5SplitHeads(CVI_TENSOR *outputs, int output_num,
                                                     const ModelProfile &profile) {
  std::vector<YoloV5SplitHead> heads;
  if (!LooksLikeYoloV5Split(outputs, output_num, profile)) {
    return heads;
  }
  for (int i = 0; i < 3; ++i) {
    YoloV5SplitHead head;
    head.bbox = &outputs[i * 3 + 0];
    head.obj = &outputs[i * 3 + 1];
    head.cls = &outputs[i * 3 + 2];
    head.stride = kExpectedStrides[i];
    head.anchors = profile.anchors[i];
    heads.push_back(head);
  }
  return heads;
}

size_t YoloV5SplitIndex(int anchor, int y, int x, int channel, int h, int w, int channels) {
  return (((static_cast<size_t>(anchor) * h + y) * w + x) * channels) + channel;
}

std::vector<Detection> DecodeYoloV5SplitDetections(CVI_TENSOR *outputs, int output_num,
                                                   float conf_threshold,
                                                   const ModelProfile &profile) {
  std::vector<Detection> detections;
  const std::vector<YoloV5SplitHead> heads = CollectYoloV5SplitHeads(outputs, output_num, profile);

  for (const YoloV5SplitHead &head : heads) {
    const CVI_TENSOR &bbox = *head.bbox;
    const CVI_TENSOR &obj = *head.obj;
    const CVI_TENSOR &cls = *head.cls;
    const int h = bbox.shape.dim[1];
    const int w = bbox.shape.dim[2];

    for (int a = 0; a < 3; ++a) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const size_t obj_index = YoloV5SplitIndex(a, y, x, 0, h, w, 1);
          const float objectness = Sigmoid(TensorValueAsFloat(obj, obj_index));

          float best_conf = -1.0f;
          int best_class = -1;
          for (int c = 0; c < profile.class_count; ++c) {
            const size_t cls_index = YoloV5SplitIndex(a, y, x, c, h, w, profile.class_count);
            const float conf = objectness * Sigmoid(TensorValueAsFloat(cls, cls_index));
            if (conf > best_conf) {
              best_conf = conf;
              best_class = c;
            }
          }
          if (best_conf < conf_threshold) {
            continue;
          }

          const float raw_x = TensorValueAsFloat(bbox, YoloV5SplitIndex(a, y, x, 0, h, w, 4));
          const float raw_y = TensorValueAsFloat(bbox, YoloV5SplitIndex(a, y, x, 1, h, w, 4));
          const float raw_w = TensorValueAsFloat(bbox, YoloV5SplitIndex(a, y, x, 2, h, w, 4));
          const float raw_h = TensorValueAsFloat(bbox, YoloV5SplitIndex(a, y, x, 3, h, w, 4));

          const float cx = (Sigmoid(raw_x) * 2.0f - 0.5f + static_cast<float>(x)) * head.stride;
          const float cy = (Sigmoid(raw_y) * 2.0f - 0.5f + static_cast<float>(y)) * head.stride;
          const float bw = std::pow(Sigmoid(raw_w) * 2.0f, 2.0f) * head.anchors[a][0];
          const float bh = std::pow(Sigmoid(raw_h) * 2.0f, 2.0f) * head.anchors[a][1];

          Detection det;
          det.x1 = cx - bw * 0.5f;
          det.y1 = cy - bh * 0.5f;
          det.x2 = cx + bw * 0.5f;
          det.y2 = cy + bh * 0.5f;
          det.confidence = best_conf;
          det.class_id = best_class;
          detections.push_back(det);
        }
      }
    }
  }

  return detections;
}

std::vector<Detection> DecodeYolov8Detections(CVI_TENSOR *outputs, int output_num,
                                              float conf_threshold,
                                              const ModelProfile &profile) {
  std::vector<Detection> detections;
  const std::vector<HeadPair> heads = CollectHeadPairs(outputs, output_num, profile);
  if (heads.empty()) {
    return detections;
  }

  for (const HeadPair &head : heads) {
    const CVI_TENSOR &reg = *head.reg;
    const CVI_TENSOR &cls = *head.cls;
    const int h = reg.shape.dim[2];
    const int w = reg.shape.dim[3];
    const size_t anchors = static_cast<size_t>(h) * w;

    for (size_t anchor = 0; anchor < anchors; ++anchor) {
      float best_conf = -1.0f;
      int best_class = -1;
      for (int c = 0; c < profile.class_count; ++c) {
        const size_t cls_index = static_cast<size_t>(c) * anchors + anchor;
        const float conf = Sigmoid(TensorValueAsFloat(cls, cls_index));
        if (conf > best_conf) {
          best_conf = conf;
          best_class = c;
        }
      }
      if (best_conf < conf_threshold) {
        continue;
      }

      const float left = DecodeDflDistance(reg, 0, anchor);
      const float top = DecodeDflDistance(reg, 1, anchor);
      const float right = DecodeDflDistance(reg, 2, anchor);
      const float bottom = DecodeDflDistance(reg, 3, anchor);

      const float grid_x = static_cast<float>(anchor % w) + 0.5f;
      const float grid_y = static_cast<float>(anchor / w) + 0.5f;
      Detection det;
      det.x1 = (grid_x - left) * head.stride;
      det.y1 = (grid_y - top) * head.stride;
      det.x2 = (grid_x + right) * head.stride;
      det.y2 = (grid_y + bottom) * head.stride;
      det.confidence = best_conf;
      det.class_id = best_class;
      detections.push_back(det);
    }
  }

  return detections;
}

float BoxIou(const Detection &a, const Detection &b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);
  const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter_area;
  if (denom <= 0.0f) {
    return 0.0f;
  }
  return inter_area / denom;
}

std::vector<Detection> ApplyNms(const std::vector<Detection> &detections,
                                float iou_threshold, int max_det,
                                const ModelProfile &profile) {
  std::vector<Detection> final_dets;

  for (int class_id = 0; class_id < profile.class_count; ++class_id) {
    std::vector<Detection> cls_dets;
    for (const auto &det : detections) {
      if (det.class_id == class_id) {
        cls_dets.push_back(det);
      }
    }

    std::sort(cls_dets.begin(), cls_dets.end(), [](const Detection &a, const Detection &b) {
      return a.confidence > b.confidence;
    });

    std::vector<Detection> kept;
    for (const auto &candidate : cls_dets) {
      bool suppressed = false;
      for (const auto &accepted : kept) {
        if (BoxIou(candidate, accepted) > iou_threshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept.push_back(candidate);
      }
    }

    final_dets.insert(final_dets.end(), kept.begin(), kept.end());
  }

  std::sort(final_dets.begin(), final_dets.end(), [](const Detection &a, const Detection &b) {
    return a.confidence > b.confidence;
  });
  if (static_cast<int>(final_dets.size()) > max_det) {
    final_dets.resize(static_cast<size_t>(max_det));
  }
  return final_dets;
}

void ClampDetections(std::vector<Detection> *detections, int width, int height) {
  const float max_x = static_cast<float>(width - 1);
  const float max_y = static_cast<float>(height - 1);
  for (auto &det : *detections) {
    det.x1 = std::max(0.0f, std::min(max_x, det.x1));
    det.x2 = std::max(0.0f, std::min(max_x, det.x2));
    det.y1 = std::max(0.0f, std::min(max_y, det.y1));
    det.y2 = std::max(0.0f, std::min(max_y, det.y2));
  }
}

}  // namespace

struct TLLR_InferenceContext {
  TLLR_InferenceConfig config;
  const ModelProfile *profile = &kTrafficProfile;
  CVI_MODEL_HANDLE model = nullptr;
  CVI_TENSOR *inputs = nullptr;
  CVI_TENSOR *outputs = nullptr;
  int input_num = 0;
  int output_num = 0;
  bool ready = false;
};

extern "C" {

int tllr_inference_init(const TLLR_InferenceConfig *config, TLLR_InferenceContext **out_ctx) {
  if (config == nullptr || out_ctx == nullptr || config->model_path == nullptr) {
    return CVI_FAILURE;
  }

  TLLR_InferenceContext *ctx = new TLLR_InferenceContext();
  ctx->config = *config;
  ctx->profile = FindProfile(config->profile);
  if (ctx->profile == nullptr) {
    std::printf("[live-infer] unsupported profile=%s (expected traffic|crosswalk|mangdao)\n",
                config->profile ? config->profile : "<null>");
    delete ctx;
    return CVI_FAILURE;
  }
  g_active_profile = ctx->profile;
  std::printf("[live-infer] profile=%s classes=%d\n",
              ctx->profile->name, ctx->profile->class_count);

  CVI_RC rc = CVI_NN_RegisterModel(config->model_path, &ctx->model);
  if (rc != CVI_RC_SUCCESS) {
    std::printf("[live-infer] CVI_NN_RegisterModel failed rc=%d path=%s\n", (int)rc, config->model_path);
    delete ctx;
    return CVI_FAILURE;
  }

  rc = CVI_NN_SetConfig(ctx->model, OPTION_PROGRAM_INDEX, config->program_id);
  if (rc != CVI_RC_SUCCESS) {
    std::printf("[live-infer] OPTION_PROGRAM_INDEX failed rc=%d\n", (int)rc);
    CVI_NN_CleanupModel(ctx->model);
    delete ctx;
    return CVI_FAILURE;
  }

  rc = CVI_NN_GetInputOutputTensors(ctx->model, &ctx->inputs, &ctx->input_num, &ctx->outputs, &ctx->output_num);
  if (rc != CVI_RC_SUCCESS) {
    std::printf("[live-infer] CVI_NN_GetInputOutputTensors failed rc=%d\n", (int)rc);
    CVI_NN_CleanupModel(ctx->model);
    delete ctx;
    return CVI_FAILURE;
  }

  if (ctx->input_num != 1) {
    std::printf("[live-infer] expected 1 input tensor, got %d\n", ctx->input_num);
    CVI_NN_CleanupModel(ctx->model);
    delete ctx;
    return CVI_FAILURE;
  }

  CVI_TENSOR *input = &ctx->inputs[0];
  if (input->shape.dim[0] != 1 || input->shape.dim[1] != 3 || input->shape.dim[2] != 640 || input->shape.dim[3] != 640) {
    std::printf("[live-infer] unexpected input tensor shape <%d,%d,%d,%d>\n",
                input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]);
    CVI_NN_CleanupModel(ctx->model);
    delete ctx;
    return CVI_FAILURE;
  }

  if (config->print_tensor_info) {
    PrintTensorList("[live-infer] Inputs", ctx->inputs, ctx->input_num);
    PrintTensorList("[live-infer] Outputs", ctx->outputs, ctx->output_num);
  }

  ctx->ready = true;
  *out_ctx = ctx;
  return CVI_SUCCESS;
}

void tllr_inference_destroy(TLLR_InferenceContext *ctx) {
  if (ctx == nullptr) {
    return;
  }
  if (ctx->model != nullptr) {
    CVI_NN_CleanupModel(ctx->model);
    ctx->model = nullptr;
  }
  delete ctx;
}

int tllr_inference_run_frame(TLLR_InferenceContext *ctx, const VIDEO_FRAME_INFO_S *frame,
                             TLLR_Detection *detections, int max_detections,
                             int *out_count, double *out_forward_ms,
                             char *summary, size_t summary_size) {
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (out_forward_ms != nullptr) {
    *out_forward_ms = 0.0;
  }
  if (summary != nullptr && summary_size > 0) {
    summary[0] = '\0';
  }

  if (ctx == nullptr || !ctx->ready || frame == nullptr || detections == nullptr || max_detections <= 0) {
    return CVI_FAILURE;
  }

  const VIDEO_FRAME_S *vf = &frame->stVFrame;
  if (vf->enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR) {
    std::printf("[live-infer] unsupported pixel format=%d\n", vf->enPixelFormat);
    return CVI_FAILURE;
  }
  if (vf->u32Width != 640 || vf->u32Height != 640) {
    std::printf("[live-infer] unexpected frame size=%ux%u\n", vf->u32Width, vf->u32Height);
    return CVI_FAILURE;
  }

  CVI_RC rc = CVI_RC_SUCCESS;
  if (ctx->inputs[0].fmt == CVI_FMT_FP32) {
    if (SetFp32TensorFromRgbPlanarFrame(&ctx->inputs[0], vf) != CVI_SUCCESS) {
      std::printf("[live-infer] SetFp32TensorFromRgbPlanarFrame failed\n");
      return CVI_FAILURE;
    }
  } else {
    uint64_t frame_paddrs[3] = {
        vf->u64PhyAddr[0],
        vf->u64PhyAddr[1],
        vf->u64PhyAddr[2],
    };
    rc = CVI_NN_SetTensorWithAlignedFrames(&ctx->inputs[0], frame_paddrs, 1, CVI_NN_PIXEL_RGB_PLANAR);
    if (rc != CVI_RC_SUCCESS) {
      std::printf("[live-infer] CVI_NN_SetTensorWithAlignedFrames failed rc=%d\n", (int)rc);
      return CVI_FAILURE;
    }
  }

  auto start = std::chrono::steady_clock::now();
  rc = CVI_NN_Forward(ctx->model, ctx->inputs, ctx->input_num, ctx->outputs, ctx->output_num);
  auto end = std::chrono::steady_clock::now();
  if (rc != CVI_RC_SUCCESS) {
    std::printf("[live-infer] CVI_NN_Forward failed rc=%d\n", (int)rc);
    return CVI_FAILURE;
  }

  const double forward_ms = std::chrono::duration<double, std::milli>(end - start).count();
  if (out_forward_ms != nullptr) {
    *out_forward_ms = forward_ms;
  }

  std::vector<Detection> decoded;
  if (std::strcmp(ctx->profile->name, "mangdao") == 0 &&
      ctx->output_num == 2 && ctx->outputs[0].shape.dim[1] == 37 &&
      ctx->outputs[0].shape.dim[2] == 8400) {
    decoded = DecodeMangdaoYolov8SegBoxes(&ctx->outputs[0], ctx->config.conf_threshold);
  } else if (LooksLikeYoloV5Split(ctx->outputs, ctx->output_num, *ctx->profile)) {
    decoded = DecodeYoloV5SplitDetections(ctx->outputs, ctx->output_num,
                                          ctx->config.conf_threshold, *ctx->profile);
  } else if (ctx->output_num == 2 && ctx->outputs[0].shape.dim[3] == 1 &&
      ctx->outputs[0].shape.dim[1] == 4 && ctx->outputs[1].shape.dim[1] >= 1) {
    /* YOLOv5s split format: 2 outputs — bbox [1,4,N,1] + cls [1,nc,N,1] */
    decoded = DecodeYolov5sSplitDetections(&ctx->outputs[0], &ctx->outputs[1],
                                          ctx->config.conf_threshold);
  } else if (ctx->output_num == 1 && ctx->outputs[0].shape.dim[1] >= 5 &&
             ctx->outputs[0].shape.dim[3] == 1) {
    /* YOLOv5s single-output format: [1, 4+nc, anchors, 1] */
    decoded = DecodeYolov5sDetections(&ctx->outputs[0], ctx->config.conf_threshold);
  } else {
    decoded = DecodeYolov8Detections(ctx->outputs, ctx->output_num,
                                     ctx->config.conf_threshold, *ctx->profile);
  }
  decoded = ApplyNms(decoded, ctx->config.iou_threshold, ctx->config.max_det, *ctx->profile);
  if (std::strcmp(ctx->profile->name, "mangdao") == 0 &&
      ctx->output_num == 2 && ctx->outputs[0].shape.dim[1] == 37 &&
      ctx->outputs[0].shape.dim[2] == 8400) {
    FillMangdaoMasks(&decoded, &ctx->outputs[0], &ctx->outputs[1]);
  }
  ClampDetections(&decoded, static_cast<int>(vf->u32Width), static_cast<int>(vf->u32Height));

  const int result_count = std::min(static_cast<int>(decoded.size()), max_detections);
  for (int i = 0; i < result_count; ++i) {
    detections[i].x1 = decoded[i].x1;
    detections[i].y1 = decoded[i].y1;
    detections[i].x2 = decoded[i].x2;
    detections[i].y2 = decoded[i].y2;
    detections[i].confidence = decoded[i].confidence;
    detections[i].class_id = decoded[i].class_id;
    detections[i].has_mask = decoded[i].has_mask ? 1 : 0;
    if (decoded[i].has_mask) {
      std::memcpy(detections[i].mask160, decoded[i].mask160.data(), decoded[i].mask160.size());
    } else {
      std::memset(detections[i].mask160, 0, sizeof(detections[i].mask160));
    }
  }
  if (out_count != nullptr) {
    *out_count = result_count;
  }

  if (summary != nullptr && summary_size > 0) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "det=" << result_count << " forward_ms=" << forward_ms;
    for (int i = 0; i < result_count; ++i) {
      oss << " | " << ctx->profile->class_names[detections[i].class_id] << ':';
      oss.setf(std::ios::fixed);
      oss.precision(3);
      oss << detections[i].confidence;
      oss.precision(0);
      oss << '@' << std::lround(detections[i].x1) << ',' << std::lround(detections[i].y1)
          << ',' << std::lround(detections[i].x2) << ',' << std::lround(detections[i].y2);
      oss.precision(2);
    }
    std::snprintf(summary, summary_size, "%s", oss.str().c_str());
  }

  return CVI_SUCCESS;
}

const char *tllr_class_name(int class_id) {
  if (class_id < 0 || class_id >= g_active_profile->class_count) {
    return "unknown";
  }
  return g_active_profile->class_names[class_id];
}

}  // extern "C"
