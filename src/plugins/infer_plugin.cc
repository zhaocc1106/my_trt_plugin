#include "infer_plugin.h"

#include <NvInferRuntime.h>

#include "mat_mul_plugin/mat_mul_plugin.h"

// 支持设置命名空间的插件注册器
template <typename T>
class PluginRegistrarExt : public nvinfer1::PluginRegistrar<T> {
public:
  PluginRegistrarExt(const char* pluginNamespace) { getPluginRegistry()->registerCreator(instance, pluginNamespace); }

private:
  //! Plugin instance.
  T instance{};
};
#define REGISTER_TENSORRT_PLUGIN_EXT(plugin, name_space) \
  static PluginRegistrarExt<plugin> plugin##_plugin_registrar(name_space)

namespace zcc {
namespace plugin {

// REGISTER_TENSORRT_PLUGIN(MatMulPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN_EXT(MatMulPluginDynamicCreator, "zcc");

} // namespace plugin
} // namespace zcc

extern "C" {

bool initLibZccInferPlugins() {
  return true;
}
} // extern "C"
