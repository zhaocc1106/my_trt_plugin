#pragma once
#include <string>
#include <vector>

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

namespace zcc {
namespace plugin {
#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif

class PluginDynamicBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  PluginDynamicBase(const std::string &name) : layer_name_(name) {}

  const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; }
  int initialize() PLUGIN_NOEXCEPT override { return 0; }
  void terminate() PLUGIN_NOEXCEPT override {}
  void destroy() PLUGIN_NOEXCEPT override { delete this; }
  void setPluginNamespace(const char *pluginNamespace)
      PLUGIN_NOEXCEPT override {
    name_space_ = pluginNamespace;
  }
  const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
    return name_space_.c_str();
  }

 protected:
  const std::string layer_name_;
  std::string name_space_;
};

class PluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection *getFieldNames()
      PLUGIN_NOEXCEPT override {
    return &fc_;
  }

  void setPluginNamespace(const char *pluginNamespace)
      PLUGIN_NOEXCEPT override {
    name_space_ = pluginNamespace;
  }

  const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
    return name_space_.c_str();
  }

 protected:
  nvinfer1::PluginFieldCollection fc_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
  std::string name_space_;
};
} // namespace plugin
} // namespace zcc