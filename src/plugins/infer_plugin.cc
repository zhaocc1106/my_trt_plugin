#include "infer_plugin.h"
#include "mat_mul_plugin/mat_mul_plugin.h"
#include <NvInferRuntime.h>

namespace zcc {
namespace plugin {
REGISTER_TENSORRT_PLUGIN(MatMulPluginDynamicCreator);
} // namespace plugin
} // namespace zcc

extern "C" {

bool initLibZccInferPlugins() { return true; }
} // extern "C"
