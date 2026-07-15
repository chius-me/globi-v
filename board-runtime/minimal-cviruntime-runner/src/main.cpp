#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cnpy.h"
#include "cviruntime.h"

namespace {

struct Options {
  std::string model_path;
  std::string input_npz;
  std::string image_path;
  std::string image_dir;
  std::string output_npz;
  std::string dump_input_f32_npz;
  std::string save_vis_path;
  std::string save_json_path;
  std::string json_dir;
  std::string vis_dir;
  int program_id = 0;
  int count = 1;
  int max_det = 100;
  bool dump_all_tensors = false;
  bool keep_aspect_ratio = true;
  bool forward_only = false;
  std::string pixel_format = "rgb";
  int pad_value = 0;
  float conf_threshold = 0.25f;
  float iou_threshold = 0.50f;
  std::array<float, 3> mean{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> scale{{0.0039216f, 0.0039216f, 0.0039216f}};
};

struct PreprocessInfo {
  std::string image_path;
  cv::Mat original_bgr;
  int orig_w = 0;
  int orig_h = 0;
  int target_w = 0;
  int target_h = 0;
  int resized_w = 0;
  int resized_h = 0;
  int paste_x = 0;
  int paste_y = 0;
  bool keep_aspect_ratio = true;
};

struct ImageInputResult {
  std::vector<float> nchw;
  PreprocessInfo prep;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float confidence = 0.0f;
  int class_id = -1;
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

constexpr int kRegMax = 16;
constexpr int kNumClasses = 2;
const std::array<const char *, kNumClasses> kClassNames{{"crosswalk", "guide_arrows"}};
const std::array<cv::Scalar, kNumClasses> kClassColors{{
    cv::Scalar(70, 220, 70),
    cv::Scalar(255, 80, 220),
}};
const std::array<int, 3> kExpectedStrides{{8, 16, 32}};
const std::array<std::array<std::array<float, 2>, 3>, 3> kYoloV5Anchors{{
    {{{105.375f, 15.609375f}, {224.5f, 18.40625f}, {362.0f, 30.9375f}}},
    {{{426.0f, 53.5f}, {321.75f, 71.75f}, {539.0f, 67.8125f}}},
    {{{468.75f, 109.0625f}, {553.0f, 92.5f}, {639.5f, 121.5625f}}},
}};

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
}

void Check(bool condition, const std::string &message) {
  if (!condition) {
    Fail(message);
  }
}

void CheckRc(CVI_RC rc, const std::string &message) {
  if (rc != CVI_RC_SUCCESS) {
    Fail(message + " (rc=" + std::to_string(static_cast<int>(rc)) + ")");
  }
}

std::string TensorName(const CVI_TENSOR &tensor) {
  return tensor.name ? std::string(tensor.name) : std::string("<unnamed>");
}

void PrintUsage(const char *argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " --model <model.cvimodel> (--input <input.npz> | --image <input.jpg>) --output <output.npz> [options]\n\n"
      << "  " << argv0
      << " --model <model.cvimodel> --image-dir <frames_dir> --json-dir <json_dir> [--vis-dir <vis_dir>] [options]\n\n"
      << "Options:\n"
      << "  --program-id <id>        Select program index (default: 0)\n"
      << "  --count <n>              Forward count for timing/debug (default: 1)\n"
      << "  --dump-all-tensors       Ask runtime to expose all tensors\n"
      << "  --forward-only           Save output tensors and skip detection decode\n"
      << "  --dump-input-f32 <file>  Save image-preprocess float tensor as npz before quantization\n"
      << "  --save-vis <file>        Save board-side annotated result image\n"
      << "  --save-json <file>       Save board-side decoded detections as JSON\n"
      << "  --image-dir <dir>        Batch image directory; model is loaded once\n"
      << "  --json-dir <dir>         Batch JSON output directory\n"
      << "  --vis-dir <dir>          Optional batch annotated image output directory\n"
      << "  --conf <float>           Confidence threshold after sigmoid (default: 0.25)\n"
      << "  --iou <float>            IoU threshold for NMS (default: 0.50)\n"
      << "  --max-det <n>            Maximum detections to keep after NMS (default: 100)\n"
      << "  --pixel-format <rgb|bgr> Image input color order before mean/scale (default: rgb)\n"
      << "  --mean <m1,m2,m3>        Per-channel mean for --image route (default: 0,0,0)\n"
      << "  --scale <s1,s2,s3>       Per-channel scale for --image route (default: 0.0039216,...)\n"
      << "  --pad-value <0-255>      Letterbox pad value for --image route (default: 0)\n"
      << "  --stretch                Disable keep-aspect-ratio and resize directly\n"
      << "  --help                   Show this help\n";
}

std::array<float, 3> ParseFloatTriplet(const std::string &text, const std::string &flag_name) {
  std::array<float, 3> values{};
  size_t start = 0;
  for (int i = 0; i < 3; ++i) {
    size_t end = text.find(',', start);
    std::string token = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    Check(!token.empty(), "invalid value for " + flag_name + ": " + text);
    values[i] = std::stof(token);
    if (i < 2) {
      Check(end != std::string::npos, flag_name + " requires exactly 3 comma-separated values");
      start = end + 1;
    } else {
      Check(end == std::string::npos, flag_name + " requires exactly 3 comma-separated values");
    }
  }
  return values;
}

Options ParseArgs(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        Fail("missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--model") {
      opt.model_path = require_value(arg);
    } else if (arg == "--input") {
      opt.input_npz = require_value(arg);
    } else if (arg == "--image") {
      opt.image_path = require_value(arg);
    } else if (arg == "--image-dir") {
      opt.image_dir = require_value(arg);
    } else if (arg == "--output") {
      opt.output_npz = require_value(arg);
    } else if (arg == "--program-id") {
      opt.program_id = std::stoi(require_value(arg));
    } else if (arg == "--count") {
      opt.count = std::stoi(require_value(arg));
    } else if (arg == "--max-det") {
      opt.max_det = std::stoi(require_value(arg));
    } else if (arg == "--dump-all-tensors") {
      opt.dump_all_tensors = true;
    } else if (arg == "--forward-only") {
      opt.forward_only = true;
    } else if (arg == "--dump-input-f32") {
      opt.dump_input_f32_npz = require_value(arg);
    } else if (arg == "--save-vis") {
      opt.save_vis_path = require_value(arg);
    } else if (arg == "--save-json") {
      opt.save_json_path = require_value(arg);
    } else if (arg == "--json-dir") {
      opt.json_dir = require_value(arg);
    } else if (arg == "--vis-dir") {
      opt.vis_dir = require_value(arg);
    } else if (arg == "--conf") {
      opt.conf_threshold = std::stof(require_value(arg));
    } else if (arg == "--iou") {
      opt.iou_threshold = std::stof(require_value(arg));
    } else if (arg == "--pixel-format") {
      opt.pixel_format = require_value(arg);
    } else if (arg == "--mean") {
      opt.mean = ParseFloatTriplet(require_value(arg), arg);
    } else if (arg == "--scale") {
      opt.scale = ParseFloatTriplet(require_value(arg), arg);
    } else if (arg == "--pad-value") {
      opt.pad_value = std::stoi(require_value(arg));
    } else if (arg == "--stretch") {
      opt.keep_aspect_ratio = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      Fail("unknown argument: " + arg);
    }
  }

  Check(!opt.model_path.empty(), "--model is required");
  const int input_modes = (!opt.input_npz.empty() ? 1 : 0) +
                          (!opt.image_path.empty() ? 1 : 0) +
                          (!opt.image_dir.empty() ? 1 : 0);
  Check(input_modes == 1, "exactly one of --input, --image, or --image-dir is required");
  if (opt.image_dir.empty()) {
    Check(!opt.output_npz.empty(), "--output is required");
  } else {
    Check(!opt.json_dir.empty(), "--json-dir is required with --image-dir");
    Check(opt.output_npz.empty(), "--output is not used with --image-dir");
    Check(opt.dump_input_f32_npz.empty(), "--dump-input-f32 is not used with --image-dir");
    Check(opt.save_json_path.empty(), "--save-json is not used with --image-dir");
    Check(opt.save_vis_path.empty(), "--save-vis is not used with --image-dir");
  }
  Check(opt.count >= 1, "--count must be >= 1");
  Check(opt.max_det >= 1, "--max-det must be >= 1");
  Check(opt.pad_value >= 0 && opt.pad_value <= 255, "--pad-value must be between 0 and 255");
  Check(opt.conf_threshold >= 0.0f && opt.conf_threshold <= 1.0f, "--conf must be between 0 and 1");
  Check(opt.iou_threshold >= 0.0f && opt.iou_threshold <= 1.0f, "--iou must be between 0 and 1");
  Check(opt.pixel_format == "rgb" || opt.pixel_format == "bgr",
        "--pixel-format only supports rgb or bgr in this minimal runner");
  if (!opt.save_vis_path.empty() || !opt.save_json_path.empty()) {
    Check(!opt.image_path.empty(), "--save-vis/--save-json currently require --image route");
  }
  return opt;
}

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

std::vector<size_t> TensorShape(const CVI_TENSOR &tensor) {
  return {
      static_cast<size_t>(tensor.shape.dim[0]),
      static_cast<size_t>(tensor.shape.dim[1]),
      static_cast<size_t>(tensor.shape.dim[2]),
      static_cast<size_t>(tensor.shape.dim[3]),
  };
}

void PrintTensorList(const char *title, CVI_TENSOR *tensors, int tensor_num) {
  std::cout << title << " (" << tensor_num << ")" << std::endl;
  for (int i = 0; i < tensor_num; ++i) {
    auto &tensor = tensors[i];
    std::cout << "  [" << i << "] " << TensorName(tensor) << " <"
              << tensor.shape.dim[0] << "," << tensor.shape.dim[1] << ","
              << tensor.shape.dim[2] << "," << tensor.shape.dim[3] << ">"
              << " fmt=" << FormatToString(tensor.fmt)
              << " count=" << CVI_NN_TensorCount(&tensor)
              << " mem=" << tensor.mem_size
              << " qscale=" << CVI_NN_TensorQuantScale(&tensor)
              << " zp=" << CVI_NN_TensorQuantZeroPoint(&tensor)
              << std::endl;
  }
}

void ConvertFp32ToInt8(const float *src, int8_t *dst, size_t count,
                      float qscale, int zero_point) {
  for (size_t i = 0; i < count; ++i) {
    int value = static_cast<int>(std::lround(src[i] * qscale)) + zero_point;
    value = std::max(-128, std::min(127, value));
    dst[i] = static_cast<int8_t>(value);
  }
}

void ConvertFp32ToUint8(const float *src, uint8_t *dst, size_t count,
                       float qscale, int zero_point) {
  for (size_t i = 0; i < count; ++i) {
    int value = static_cast<int>(std::lround(src[i] * qscale)) + zero_point;
    value = std::max(0, std::min(255, value));
    dst[i] = static_cast<uint8_t>(value);
  }
}

void ConvertFp32ToBf16(const float *src, uint16_t *dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    float value = src[i];
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    dst[i] = reinterpret_cast<uint16_t *>(&value)[0];
#else
    dst[i] = reinterpret_cast<uint16_t *>(&value)[1];
#endif
  }
}

void CopyFp32BufferIntoTensor(const float *src, size_t count, CVI_TENSOR *tensor) {
  const size_t tensor_count = CVI_NN_TensorCount(tensor);
  Check(count == tensor_count, "input element count mismatch for tensor " + TensorName(*tensor));

  const int zero_point = CVI_NN_TensorQuantZeroPoint(tensor);
  const float quant_scale = CVI_NN_TensorQuantScale(tensor);
  void *dst = CVI_NN_TensorPtr(tensor);

  switch (tensor->fmt) {
    case CVI_FMT_FP32:
      std::memcpy(dst, src, tensor->mem_size);
      return;
    case CVI_FMT_INT8:
      ConvertFp32ToInt8(src, static_cast<int8_t *>(dst), tensor_count, quant_scale, zero_point);
      return;
    case CVI_FMT_UINT8:
      ConvertFp32ToUint8(src, static_cast<uint8_t *>(dst), tensor_count, quant_scale, zero_point);
      return;
    case CVI_FMT_BF16:
      ConvertFp32ToBf16(src, static_cast<uint16_t *>(dst), tensor_count);
      return;
    default:
      Fail("image route does not support tensor format " + std::string(FormatToString(tensor->fmt)) +
           " for input tensor " + TensorName(*tensor));
  }
}

const cnpy::NpyArray *FindInputArray(const cnpy::npz_t &npz, const std::string &tensor_name) {
  auto it = npz.find(tensor_name);
  if (it != npz.end()) {
    return &it->second;
  }
  if (npz.size() == 1) {
    return &npz.begin()->second;
  }
  return nullptr;
}

void CopyArrayIntoTensor(const cnpy::NpyArray &arr, CVI_TENSOR *tensor) {
  const size_t tensor_count = CVI_NN_TensorCount(tensor);
  const int zero_point = CVI_NN_TensorQuantZeroPoint(tensor);
  const float quant_scale = CVI_NN_TensorQuantScale(tensor);
  void *dst = CVI_NN_TensorPtr(tensor);

  if (arr.type == 'f' && tensor->fmt == CVI_FMT_FP32) {
    Check(arr.num_vals == tensor_count, "FP32 input count mismatch for tensor " + TensorName(*tensor));
    std::memcpy(dst, arr.data<float>(), tensor->mem_size);
    return;
  }
  if (arr.type == 'f' && tensor->fmt == CVI_FMT_INT8) {
    Check(arr.num_vals == tensor_count, "INT8 input count mismatch for tensor " + TensorName(*tensor));
    ConvertFp32ToInt8(arr.data<float>(), static_cast<int8_t *>(dst), tensor_count, quant_scale, zero_point);
    return;
  }
  if (arr.type == 'f' && tensor->fmt == CVI_FMT_UINT8) {
    Check(arr.num_vals == tensor_count, "UINT8 input count mismatch for tensor " + TensorName(*tensor));
    ConvertFp32ToUint8(arr.data<float>(), static_cast<uint8_t *>(dst), tensor_count, quant_scale, zero_point);
    return;
  }
  if (arr.type == 'f' && tensor->fmt == CVI_FMT_BF16) {
    Check(arr.num_vals == tensor_count, "BF16 input count mismatch for tensor " + TensorName(*tensor));
    ConvertFp32ToBf16(arr.data<float>(), static_cast<uint16_t *>(dst), tensor_count);
    return;
  }

  Check(arr.num_bytes() == tensor->mem_size,
        "raw input byte size mismatch for tensor " + TensorName(*tensor));
  std::memcpy(dst, arr.data<uint8_t>(), tensor->mem_size);
}

void LoadInputNpz(const std::string &input_path, CVI_TENSOR *inputs, int input_num) {
  cnpy::npz_t npz = cnpy::npz_load(input_path);
  Check(!npz.empty(), "failed to load input npz: " + input_path);

  for (int i = 0; i < input_num; ++i) {
    CVI_TENSOR *tensor = &inputs[i];
    const cnpy::NpyArray *arr = FindInputArray(npz, TensorName(*tensor));
    Check(arr != nullptr, "cannot find matching input array for tensor " + TensorName(*tensor));
    CopyArrayIntoTensor(*arr, tensor);
  }
}

ImageInputResult BuildImageInput(const Options &opt, const CVI_TENSOR &tensor) {
  Check(tensor.shape.dim[0] == 1, "image route currently only supports batch size 1");
  Check(tensor.shape.dim[1] == 3, "image route currently only supports 3-channel input tensors");

  const int target_h = tensor.shape.dim[2];
  const int target_w = tensor.shape.dim[3];
  cv::Mat image = cv::imread(opt.image_path, cv::IMREAD_COLOR);
  Check(!image.empty(), "failed to read image: " + opt.image_path);

  cv::Mat prepared;
  int paste_x = 0;
  int paste_y = 0;
  int resized_w = target_w;
  int resized_h = target_h;

  if (opt.keep_aspect_ratio) {
    const double ratio = std::min(static_cast<double>(target_w) / image.cols,
                                  static_cast<double>(target_h) / image.rows);
    resized_w = std::max(1, static_cast<int>(image.cols * ratio));
    resized_h = std::max(1, static_cast<int>(image.rows * ratio));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    prepared = cv::Mat(target_h, target_w, CV_8UC3,
                       cv::Scalar(opt.pad_value, opt.pad_value, opt.pad_value));
    paste_x = (target_w - resized_w) / 2;
    paste_y = (target_h - resized_h) / 2;
    resized.copyTo(prepared(cv::Rect(paste_x, paste_y, resized_w, resized_h)));
  } else {
    cv::resize(image, prepared, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);
  }

  if (opt.pixel_format == "rgb") {
    cv::cvtColor(prepared, prepared, cv::COLOR_BGR2RGB);
  }

  std::vector<cv::Mat> channels;
  cv::split(prepared, channels);
  Check(static_cast<int>(channels.size()) == tensor.shape.dim[1],
        "unexpected channel count after preprocessing");

  std::vector<float> output(static_cast<size_t>(target_h) * target_w * tensor.shape.dim[1]);
  size_t offset = 0;
  for (int c = 0; c < tensor.shape.dim[1]; ++c) {
    for (int y = 0; y < target_h; ++y) {
      const uint8_t *row = channels[c].ptr<uint8_t>(y);
      for (int x = 0; x < target_w; ++x) {
        output[offset++] = (static_cast<float>(row[x]) - opt.mean[c]) * opt.scale[c];
      }
    }
  }

  std::cout << "Image preprocess: path=" << opt.image_path
            << " original=" << image.cols << "x" << image.rows
            << " resized=" << resized_w << "x" << resized_h
            << " paste_xy=" << paste_x << "," << paste_y
            << " pixel_format=" << opt.pixel_format
            << " keep_aspect_ratio=" << (opt.keep_aspect_ratio ? "true" : "false")
            << std::endl;

  ImageInputResult result;
  result.nchw = std::move(output);
  result.prep.image_path = opt.image_path;
  result.prep.original_bgr = image;
  result.prep.orig_w = image.cols;
  result.prep.orig_h = image.rows;
  result.prep.target_w = target_w;
  result.prep.target_h = target_h;
  result.prep.resized_w = resized_w;
  result.prep.resized_h = resized_h;
  result.prep.paste_x = paste_x;
  result.prep.paste_y = paste_y;
  result.prep.keep_aspect_ratio = opt.keep_aspect_ratio;
  return result;
}

ImageInputResult LoadImageIntoTensor(const Options &opt, CVI_TENSOR *inputs, int input_num) {
  Check(input_num == 1, "image route currently expects exactly one input tensor");
  ImageInputResult image_input = BuildImageInput(opt, inputs[0]);
  CopyFp32BufferIntoTensor(image_input.nchw.data(), image_input.nchw.size(), &inputs[0]);
  return image_input;
}

template <typename T>
void AddTensorToNpz(cnpy::npz_t &npz, const CVI_TENSOR &tensor) {
  cnpy::npz_add_array<T>(
      npz, TensorName(tensor),
      static_cast<const T *>(CVI_NN_TensorPtr(const_cast<CVI_TENSOR *>(&tensor))),
      TensorShape(tensor));
}

void SaveOutputNpz(const std::string &output_path, CVI_TENSOR *outputs, int output_num) {
  cnpy::npz_t npz;
  for (int i = 0; i < output_num; ++i) {
    const CVI_TENSOR &tensor = outputs[i];
    switch (tensor.fmt) {
      case CVI_FMT_FP32:
        AddTensorToNpz<float>(npz, tensor);
        break;
      case CVI_FMT_INT32:
        Fail("CVI_FMT_INT32 output is not wired in this minimal runner yet: " + TensorName(tensor));
        break;
      case CVI_FMT_UINT32:
        AddTensorToNpz<uint32_t>(npz, tensor);
        break;
      case CVI_FMT_BF16:
        AddTensorToNpz<uint16_t>(npz, tensor);
        break;
      case CVI_FMT_INT16:
        AddTensorToNpz<int16_t>(npz, tensor);
        break;
      case CVI_FMT_UINT16:
        AddTensorToNpz<uint16_t>(npz, tensor);
        break;
      case CVI_FMT_INT8:
        AddTensorToNpz<int8_t>(npz, tensor);
        break;
      case CVI_FMT_UINT8:
        AddTensorToNpz<uint8_t>(npz, tensor);
        break;
      default:
        Fail("unsupported output tensor format for " + TensorName(tensor));
    }
  }
  cnpy::npz_save_all(output_path, npz);
}

void SaveF32NchwToNpz(const std::string &output_path, const std::string &tensor_name,
                     const std::vector<float> &data, int n, int c, int h, int w) {
  cnpy::npz_t npz;
  cnpy::npz_add_array<float>(npz, tensor_name, data.data(),
                             {static_cast<size_t>(n), static_cast<size_t>(c),
                              static_cast<size_t>(h), static_cast<size_t>(w)});
  cnpy::npz_save_all(output_path, npz);
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
      Fail("unsupported tensor format for decode on tensor " + TensorName(tensor));
  }
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
  Check(sum_exp > 0.0f, "invalid DFL sum for tensor " + TensorName(reg_tensor));
  return weighted_sum / sum_exp;
}

std::vector<HeadPair> CollectHeadPairs(CVI_TENSOR *outputs, int output_num) {
  std::vector<CVI_TENSOR *> reg_heads;
  std::vector<CVI_TENSOR *> cls_heads;
  for (int i = 0; i < output_num; ++i) {
    CVI_TENSOR *tensor = &outputs[i];
    if (tensor->shape.dim[0] != 1) {
      continue;
    }
    if (tensor->shape.dim[1] == 4 * kRegMax) {
      reg_heads.push_back(tensor);
    } else if (tensor->shape.dim[1] == kNumClasses) {
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

  Check(reg_heads.size() == 3, "expected 3 regression heads for YOLOv8-style decode");
  Check(cls_heads.size() == 3, "expected 3 classification heads for YOLOv8-style decode");

  std::vector<HeadPair> pairs;
  for (size_t i = 0; i < 3; ++i) {
    Check(reg_heads[i]->shape.dim[2] == cls_heads[i]->shape.dim[2] &&
              reg_heads[i]->shape.dim[3] == cls_heads[i]->shape.dim[3],
          "reg/cls head spatial size mismatch");
    const int feature_h = reg_heads[i]->shape.dim[2];
    Check(feature_h > 0, "invalid feature height");
    const int stride = 640 / feature_h;
    HeadPair pair;
    pair.reg = reg_heads[i];
    pair.cls = cls_heads[i];
    pair.stride = stride;
    pairs.push_back(pair);
  }

  Check(pairs[0].stride == kExpectedStrides[0] &&
            pairs[1].stride == kExpectedStrides[1] &&
            pairs[2].stride == kExpectedStrides[2],
        "unexpected strides while collecting heads");
  return pairs;
}

std::vector<Detection> DecodeYolov8Detections(CVI_TENSOR *outputs, int output_num,
                                              float conf_threshold) {
  std::vector<Detection> detections;
  const std::vector<HeadPair> heads = CollectHeadPairs(outputs, output_num);

  for (const HeadPair &head : heads) {
    const CVI_TENSOR &reg = *head.reg;
    const CVI_TENSOR &cls = *head.cls;
    const int h = reg.shape.dim[2];
    const int w = reg.shape.dim[3];
    const size_t anchors = static_cast<size_t>(h) * w;

    for (size_t anchor = 0; anchor < anchors; ++anchor) {
      float best_conf = -1.0f;
      int best_class = -1;
      for (int c = 0; c < kNumClasses; ++c) {
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

bool LooksLikeYoloV5Split(CVI_TENSOR *outputs, int output_num) {
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
        cls.shape.dim[2] != size || cls.shape.dim[3] != kNumClasses) {
      return false;
    }
  }
  return true;
}

std::vector<YoloV5SplitHead> CollectYoloV5SplitHeads(CVI_TENSOR *outputs, int output_num) {
  Check(LooksLikeYoloV5Split(outputs, output_num),
        "expected 9 YOLOv5 split outputs ordered as bbox/objectness/classes for 80,40,20 grids");
  std::vector<YoloV5SplitHead> heads;
  for (int i = 0; i < 3; ++i) {
    YoloV5SplitHead head;
    head.bbox = &outputs[i * 3 + 0];
    head.obj = &outputs[i * 3 + 1];
    head.cls = &outputs[i * 3 + 2];
    head.stride = kExpectedStrides[i];
    head.anchors = kYoloV5Anchors[i];
    heads.push_back(head);
  }
  return heads;
}

size_t YoloV5SplitIndex(int anchor, int y, int x, int channel, int h, int w, int channels) {
  return (((static_cast<size_t>(anchor) * h + y) * w + x) * channels) + channel;
}

std::vector<Detection> DecodeYoloV5SplitDetections(CVI_TENSOR *outputs, int output_num,
                                                   float conf_threshold) {
  std::vector<Detection> detections;
  const std::vector<YoloV5SplitHead> heads = CollectYoloV5SplitHeads(outputs, output_num);

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
          if (objectness <= 0.0f) {
            continue;
          }

          float best_conf = -1.0f;
          int best_class = -1;
          for (int c = 0; c < kNumClasses; ++c) {
            const size_t cls_index = YoloV5SplitIndex(a, y, x, c, h, w, kNumClasses);
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

bool LooksLikeYoloV5Concat(CVI_TENSOR *outputs, int output_num) {
  if (output_num != 1) {
    return false;
  }
  const CVI_TENSOR &out = outputs[0];
  const int no = 5 + kNumClasses;
  return out.shape.dim[0] == 1 &&
         ((out.shape.dim[1] == 25200 && out.shape.dim[2] == no) ||
          (out.shape.dim[1] == no && out.shape.dim[2] == 25200));
}

float MaybeSigmoidProbability(float value) {
  if (value >= 0.0f && value <= 1.0f) {
    return value;
  }
  return Sigmoid(value);
}

float YoloV5ConcatValue(const CVI_TENSOR &out, int anchor_index, int channel) {
  const int no = 5 + kNumClasses;
  if (out.shape.dim[1] == 25200 && out.shape.dim[2] == no) {
    return TensorValueAsFloat(out, (static_cast<size_t>(anchor_index) * no) + channel);
  }
  return TensorValueAsFloat(out, (static_cast<size_t>(channel) * 25200) + anchor_index);
}

std::vector<Detection> DecodeYoloV5ConcatDetections(CVI_TENSOR *outputs, int output_num,
                                                    float conf_threshold) {
  std::vector<Detection> detections;
  if (!LooksLikeYoloV5Concat(outputs, output_num)) {
    return detections;
  }

  const CVI_TENSOR &out = outputs[0];
  int offset = 0;
  const std::array<int, 3> sizes{{80, 40, 20}};
  for (int scale = 0; scale < 3; ++scale) {
    const int size = sizes[scale];
    const int stride = kExpectedStrides[scale];
    for (int a = 0; a < 3; ++a) {
      for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
          const int anchor_index = offset + ((a * size + y) * size + x);
          const float objectness = MaybeSigmoidProbability(YoloV5ConcatValue(out, anchor_index, 4));

          float best_conf = -1.0f;
          int best_class = -1;
          for (int c = 0; c < kNumClasses; ++c) {
            const float class_score = MaybeSigmoidProbability(YoloV5ConcatValue(out, anchor_index, 5 + c));
            const float conf = objectness * class_score;
            if (conf > best_conf) {
              best_conf = conf;
              best_class = c;
            }
          }
          if (best_conf < conf_threshold) {
            continue;
          }

          const float raw_x = YoloV5ConcatValue(out, anchor_index, 0);
          const float raw_y = YoloV5ConcatValue(out, anchor_index, 1);
          const float raw_w = YoloV5ConcatValue(out, anchor_index, 2);
          const float raw_h = YoloV5ConcatValue(out, anchor_index, 3);

          const float cx = (Sigmoid(raw_x) * 2.0f - 0.5f + static_cast<float>(x)) * stride;
          const float cy = (Sigmoid(raw_y) * 2.0f - 0.5f + static_cast<float>(y)) * stride;
          const float bw = std::pow(Sigmoid(raw_w) * 2.0f, 2.0f) * kYoloV5Anchors[scale][a][0];
          const float bh = std::pow(Sigmoid(raw_h) * 2.0f, 2.0f) * kYoloV5Anchors[scale][a][1];

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
    offset += 3 * size * size;
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
                                float iou_threshold, int max_det) {
  std::vector<Detection> final_dets;

  for (int class_id = 0; class_id < kNumClasses; ++class_id) {
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

void UndoPreprocess(std::vector<Detection> *detections, const PreprocessInfo &prep) {
  if (detections->empty()) {
    return;
  }

  if (!prep.keep_aspect_ratio) {
    const float scale_x = static_cast<float>(prep.target_w) / prep.orig_w;
    const float scale_y = static_cast<float>(prep.target_h) / prep.orig_h;
    for (auto &det : *detections) {
      det.x1 /= scale_x;
      det.x2 /= scale_x;
      det.y1 /= scale_y;
      det.y2 /= scale_y;
    }
  } else {
    const float ratio = std::min(static_cast<float>(prep.target_w) / prep.orig_w,
                                 static_cast<float>(prep.target_h) / prep.orig_h);
    for (auto &det : *detections) {
      det.x1 = (det.x1 - prep.paste_x) / ratio;
      det.x2 = (det.x2 - prep.paste_x) / ratio;
      det.y1 = (det.y1 - prep.paste_y) / ratio;
      det.y2 = (det.y2 - prep.paste_y) / ratio;
    }
  }

  const float max_x = static_cast<float>(prep.orig_w - 1);
  const float max_y = static_cast<float>(prep.orig_h - 1);
  for (auto &det : *detections) {
    det.x1 = std::max(0.0f, std::min(max_x, det.x1));
    det.x2 = std::max(0.0f, std::min(max_x, det.x2));
    det.y1 = std::max(0.0f, std::min(max_y, det.y1));
    det.y2 = std::max(0.0f, std::min(max_y, det.y2));
  }
}

cv::Mat DrawDetections(const PreprocessInfo &prep, const std::vector<Detection> &detections) {
  cv::Mat canvas = prep.original_bgr.clone();
  for (const auto &det : detections) {
    const cv::Scalar color = kClassColors[det.class_id];
    const cv::Point p1(static_cast<int>(std::round(det.x1)), static_cast<int>(std::round(det.y1)));
    const cv::Point p2(static_cast<int>(std::round(det.x2)), static_cast<int>(std::round(det.y2)));
    cv::rectangle(canvas, p1, p2, color, 2);

    std::ostringstream label_stream;
    label_stream << kClassNames[det.class_id] << " " << std::fixed << std::setprecision(3) << det.confidence;
    const std::string label = label_stream.str();

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
    const int label_y1 = std::max(0, p1.y - text_size.height - 8);
    const int label_y2 = label_y1 + text_size.height + 8;
    cv::rectangle(canvas,
                  cv::Point(p1.x, label_y1),
                  cv::Point(p1.x + text_size.width + 8, label_y2),
                  color, cv::FILLED);
    cv::putText(canvas, label,
                cv::Point(p1.x + 4, label_y2 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
  return canvas;
}

std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch))
              << std::dec << std::setfill(' ');
        } else {
          out << ch;
        }
    }
  }
  return out.str();
}

std::string FloatToString(float value, int precision = 4) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

void SaveDetectionsJson(const std::string &path,
                        const Options &opt,
                        const PreprocessInfo &prep,
                        const std::vector<Detection> &detections) {
  std::ofstream ofs(path.c_str());
  Check(ofs.good(), "failed to open JSON output path: " + path);

  ofs << "{\n";
  ofs << "  \"image\": \"" << JsonEscape(prep.image_path) << "\",\n";
  ofs << "  \"output_npz\": \"" << JsonEscape(opt.output_npz) << "\",\n";
  if (!opt.save_vis_path.empty()) {
    ofs << "  \"save_vis\": \"" << JsonEscape(opt.save_vis_path) << "\",\n";
  }
  ofs << "  \"count\": " << detections.size() << ",\n";
  ofs << "  \"conf\": " << FloatToString(opt.conf_threshold, 4) << ",\n";
  ofs << "  \"iou\": " << FloatToString(opt.iou_threshold, 4) << ",\n";
  ofs << "  \"detections\": [\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto &det = detections[i];
    ofs << "    {\n";
    ofs << "      \"class_id\": " << det.class_id << ",\n";
    ofs << "      \"class_name\": \"" << kClassNames[det.class_id] << "\",\n";
    ofs << "      \"confidence\": " << FloatToString(det.confidence, 4) << ",\n";
    ofs << "      \"xyxy\": ["
        << FloatToString(det.x1, 4) << ", "
        << FloatToString(det.y1, 4) << ", "
        << FloatToString(det.x2, 4) << ", "
        << FloatToString(det.y2, 4) << "]\n";
    ofs << "    }" << (i + 1 == detections.size() ? "\n" : ",\n");
  }
  ofs << "  ]\n";
  ofs << "}\n";
}

void PrintDetectionsSummary(const std::vector<Detection> &detections) {
  std::cout << "Decoded detections: " << detections.size() << std::endl;
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto &det = detections[i];
    std::cout << "  [" << i << "] class=" << kClassNames[det.class_id]
              << " conf=" << FloatToString(det.confidence, 4)
              << " xyxy=[" << FloatToString(det.x1, 2) << ", "
              << FloatToString(det.y1, 2) << ", "
              << FloatToString(det.x2, 2) << ", "
              << FloatToString(det.y2, 2) << "]"
              << std::endl;
  }
}

bool HasImageExtension(const std::string &name) {
  const size_t dot = name.find_last_of('.');
  if (dot == std::string::npos) {
    return false;
  }
  std::string ext = name.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext == "jpg" || ext == "jpeg" || ext == "png";
}

std::vector<std::string> ListImages(const std::string &dir) {
  DIR *dp = opendir(dir.c_str());
  Check(dp != nullptr, "failed to open image directory: " + dir);

  std::vector<std::string> paths;
  while (dirent *entry = readdir(dp)) {
    const std::string name = entry->d_name;
    if (name == "." || name == ".." || !HasImageExtension(name)) {
      continue;
    }
    paths.push_back(dir + "/" + name);
  }
  closedir(dp);
  std::sort(paths.begin(), paths.end());
  Check(!paths.empty(), "no images found in directory: " + dir);
  return paths;
}

std::string PathStem(const std::string &path) {
  size_t slash = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  size_t dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

std::vector<Detection> DecodeOutputs(CVI_TENSOR *outputs,
                                     int output_num,
                                     const Options &opt,
                                     const PreprocessInfo &prep) {
  std::vector<Detection> decoded;
  if (LooksLikeYoloV5Split(outputs, output_num)) {
    decoded = DecodeYoloV5SplitDetections(outputs, output_num, opt.conf_threshold);
  } else if (LooksLikeYoloV5Concat(outputs, output_num)) {
    decoded = DecodeYoloV5ConcatDetections(outputs, output_num, opt.conf_threshold);
  } else {
    decoded = DecodeYolov8Detections(outputs, output_num, opt.conf_threshold);
  }
  decoded = ApplyNms(decoded, opt.iou_threshold, opt.max_det);
  UndoPreprocess(&decoded, prep);
  return decoded;
}

ImageInputResult RunImageOnce(const Options &opt,
                              const std::string &image_path,
                              CVI_MODEL_HANDLE model,
                              CVI_TENSOR *inputs,
                              int input_num,
                              CVI_TENSOR *outputs,
                              int output_num,
                              double *forward_ms) {
  Options frame_opt = opt;
  frame_opt.image_path = image_path;
  ImageInputResult image_input = LoadImageIntoTensor(frame_opt, inputs, input_num);

  auto start = std::chrono::steady_clock::now();
  CheckRc(CVI_NN_Forward(model, inputs, input_num, outputs, output_num), "forward failed");
  auto end = std::chrono::steady_clock::now();
  if (forward_ms != nullptr) {
    *forward_ms = std::chrono::duration<double, std::milli>(end - start).count();
  }
  return image_input;
}

void RunImageDir(const Options &opt,
                 CVI_MODEL_HANDLE model,
                 CVI_TENSOR *inputs,
                 int input_num,
                 CVI_TENSOR *outputs,
                 int output_num) {
  const std::vector<std::string> images = ListImages(opt.image_dir);
  std::cout << "Batch images: " << images.size() << std::endl;
  double total_forward_ms = 0.0;

  for (size_t i = 0; i < images.size(); ++i) {
    double forward_ms = 0.0;
    ImageInputResult image_input = RunImageOnce(opt, images[i], model, inputs, input_num,
                                                outputs, output_num, &forward_ms);
    total_forward_ms += forward_ms;

    std::vector<Detection> decoded = DecodeOutputs(outputs, output_num, opt, image_input.prep);
    const std::string stem = PathStem(images[i]);
    Options frame_opt = opt;
    frame_opt.image_path = images[i];
    frame_opt.output_npz = "";
    frame_opt.save_json_path = opt.json_dir + "/" + stem + ".json";
    SaveDetectionsJson(frame_opt.save_json_path, frame_opt, image_input.prep, decoded);

    if (!opt.vis_dir.empty()) {
      frame_opt.save_vis_path = opt.vis_dir + "/" + stem + ".jpg";
      cv::Mat vis = DrawDetections(image_input.prep, decoded);
      Check(cv::imwrite(frame_opt.save_vis_path, vis),
            "failed to save annotated image: " + frame_opt.save_vis_path);
    }

    if ((i + 1) % 25 == 0 || i + 1 == images.size()) {
      std::cout << "Batch progress: " << (i + 1) << "/" << images.size()
                << " avg_forward_ms=" << (total_forward_ms / static_cast<double>(i + 1))
                << " last_det=" << decoded.size()
                << std::endl;
    }
  }
}

class ModelGuard {
 public:
  ~ModelGuard() {
    if (model_ != nullptr) {
      CVI_NN_CleanupModel(model_);
    }
  }

  CVI_MODEL_HANDLE *ptr() { return &model_; }
  CVI_MODEL_HANDLE get() const { return model_; }

 private:
  CVI_MODEL_HANDLE model_ = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
  try {
    Options opt = ParseArgs(argc, argv);

    ModelGuard model;
    CheckRc(CVI_NN_RegisterModel(opt.model_path.c_str(), model.ptr()), "failed to register model");

    CheckRc(CVI_NN_SetConfig(model.get(), OPTION_PROGRAM_INDEX, opt.program_id),
            "failed to set OPTION_PROGRAM_INDEX");
    CheckRc(CVI_NN_SetConfig(model.get(), OPTION_OUTPUT_ALL_TENSORS, opt.dump_all_tensors),
            "failed to set OPTION_OUTPUT_ALL_TENSORS");

    CVI_TENSOR *inputs = nullptr;
    CVI_TENSOR *outputs = nullptr;
    int input_num = 0;
    int output_num = 0;
    CheckRc(CVI_NN_GetInputOutputTensors(model.get(), &inputs, &input_num, &outputs, &output_num),
            "failed to query input/output tensors");

    PrintTensorList("Inputs", inputs, input_num);
    PrintTensorList("Outputs", outputs, output_num);

    if (!opt.image_dir.empty()) {
      RunImageDir(opt, model.get(), inputs, input_num, outputs, output_num);
      return 0;
    }

    bool used_image_route = false;
    ImageInputResult image_input;
    if (!opt.input_npz.empty()) {
      LoadInputNpz(opt.input_npz, inputs, input_num);
    } else {
      image_input = LoadImageIntoTensor(opt, inputs, input_num);
      used_image_route = true;
      if (!opt.dump_input_f32_npz.empty()) {
        SaveF32NchwToNpz(opt.dump_input_f32_npz, TensorName(inputs[0]), image_input.nchw,
                         inputs[0].shape.dim[0], inputs[0].shape.dim[1],
                         inputs[0].shape.dim[2], inputs[0].shape.dim[3]);
        std::cout << "Saved input f32 npz: " << opt.dump_input_f32_npz << std::endl;
      }
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.count; ++i) {
      CheckRc(CVI_NN_Forward(model.get(), inputs, input_num, outputs, output_num),
              "forward failed");
    }
    auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Forward done: count=" << opt.count
              << ", total_ms=" << elapsed_ms
              << ", avg_ms=" << (elapsed_ms / opt.count)
              << std::endl;

    SaveOutputNpz(opt.output_npz, outputs, output_num);
    std::cout << "Saved output npz: " << opt.output_npz << std::endl;

    if (used_image_route && !opt.forward_only) {
      std::vector<Detection> decoded;
      if (LooksLikeYoloV5Split(outputs, output_num)) {
        std::cout << "Decode mode: YOLOv5 9-output split" << std::endl;
        decoded = DecodeYoloV5SplitDetections(outputs, output_num, opt.conf_threshold);
      } else if (LooksLikeYoloV5Concat(outputs, output_num)) {
        std::cout << "Decode mode: YOLOv5 concat" << std::endl;
        decoded = DecodeYoloV5ConcatDetections(outputs, output_num, opt.conf_threshold);
      } else {
        std::cout << "Decode mode: YOLOv8 DFL heads" << std::endl;
        decoded = DecodeYolov8Detections(outputs, output_num, opt.conf_threshold);
      }
      decoded = ApplyNms(decoded, opt.iou_threshold, opt.max_det);
      UndoPreprocess(&decoded, image_input.prep);
      PrintDetectionsSummary(decoded);

      if (!opt.save_vis_path.empty()) {
        cv::Mat vis = DrawDetections(image_input.prep, decoded);
        Check(cv::imwrite(opt.save_vis_path, vis), "failed to save annotated image: " + opt.save_vis_path);
        std::cout << "Saved annotated image: " << opt.save_vis_path << std::endl;
      }
      if (!opt.save_json_path.empty()) {
        SaveDetectionsJson(opt.save_json_path, opt, image_input.prep, decoded);
        std::cout << "Saved detections json: " << opt.save_json_path << std::endl;
      }
    }

    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
