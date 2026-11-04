#ifndef FLUTTER_PLUGIN_LLAMA_FLUTTER_ANDROID_PLUGIN_H_
#define FLUTTER_PLUGIN_LLAMA_FLUTTER_ANDROID_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>

namespace llama_flutter_android {

class LlamaFlutterAndroidPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  LlamaFlutterAndroidPlugin(
      flutter::PluginRegistrarWindows* registrar,
      std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> host_channel,
      std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> cluster_channel);

  virtual ~LlamaFlutterAndroidPlugin();

  // Disallow copy and assign.
  LlamaFlutterAndroidPlugin(const LlamaFlutterAndroidPlugin&) = delete;
  LlamaFlutterAndroidPlugin& operator=(const LlamaFlutterAndroidPlugin&) = delete;

 private:
  void HandleHostMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleClusterMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  flutter::PluginRegistrarWindows* registrar_;
  std::shared_ptr<flutter::MethodChannel<flutter::EncodableValue>> host_channel_;
  std::shared_ptr<flutter::MethodChannel<flutter::EncodableValue>> cluster_channel_;
};

}  // namespace llama_flutter_android

#endif  // FLUTTER_PLUGIN_LLAMA_FLUTTER_ANDROID_PLUGIN_H_
