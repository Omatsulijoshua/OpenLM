#include "llama_flutter_android_plugin.h"

#include "../src/llama_desktop_core.h"

#include <windows.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace llama_flutter_android {

namespace {

constexpr UINT WM_FLUTTER_TASK = WM_APP + 108;
static HWND g_plugin_hwnd = nullptr;

static void PostToMainThread(std::function<void()> task) {
  if (!g_plugin_hwnd) {
    task();
    return;
  }
  auto* fn = new std::function<void()>(std::move(task));
  if (!PostMessage(g_plugin_hwnd, WM_FLUTTER_TASK, 0, reinterpret_cast<LPARAM>(fn))) {
    delete fn;
  }
}

static std::string GetStringVal(const flutter::EncodableValue& val, const std::string& def = "") {
  if (std::holds_alternative<std::string>(val)) {
    return std::get<std::string>(val);
  }
  return def;
}

static int64_t GetInt64Val(const flutter::EncodableValue& val, int64_t def = 0) {
  if (std::holds_alternative<int32_t>(val)) {
    return std::get<int32_t>(val);
  }
  if (std::holds_alternative<int64_t>(val)) {
    return std::get<int64_t>(val);
  }
  return def;
}

static double GetDoubleVal(const flutter::EncodableValue& val, double def = 0.0) {
  if (std::holds_alternative<double>(val)) {
    return std::get<double>(val);
  }
  if (std::holds_alternative<int32_t>(val)) {
    return (double)std::get<int32_t>(val);
  }
  if (std::holds_alternative<int64_t>(val)) {
    return (double)std::get<int64_t>(val);
  }
  return def;
}

static bool GetBoolVal(const flutter::EncodableValue& val, bool def = false) {
  if (std::holds_alternative<bool>(val)) {
    return std::get<bool>(val);
  }
  return def;
}

}  // namespace

// static
void LlamaFlutterAndroidPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto host_channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "llama_flutter_android/host",
          &flutter::StandardMethodCodec::GetInstance());

  auto cluster_channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "llama_flutter_android/cluster",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<LlamaFlutterAndroidPlugin>(
      registrar, std::move(host_channel), std::move(cluster_channel));

  plugin->host_channel_->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleHostMethodCall(call, std::move(result));
      });

  plugin->cluster_channel_->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleClusterMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

LlamaFlutterAndroidPlugin::LlamaFlutterAndroidPlugin(
    flutter::PluginRegistrarWindows* registrar,
    std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> host_channel,
    std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> cluster_channel)
    : registrar_(registrar),
      host_channel_(std::move(host_channel)),
      cluster_channel_(std::move(cluster_channel)) {
  if (registrar_ && registrar_->GetView()) {
    g_plugin_hwnd = registrar_->GetView()->GetNativeWindow();
  }

  registrar_->RegisterTopLevelWindowProcDelegate(
      [](HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) -> std::optional<LRESULT> {
        if (message == WM_FLUTTER_TASK) {
          auto* fn = reinterpret_cast<std::function<void()>*>(lparam);
          if (fn) {
            (*fn)();
            delete fn;
          }
          return 0;
        }
        return std::nullopt;
      });
}

LlamaFlutterAndroidPlugin::~LlamaFlutterAndroidPlugin() = default;

void LlamaFlutterAndroidPlugin::HandleHostMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  const std::string& method = method_call.method_name();
  auto& core = llama_desktop::LlamaDesktopCore::instance();

  if (method == "loadModel") {
    const auto* list = std::get_if<flutter::EncodableList>(method_call.arguments());
    if (!list || list->size() < 4) {
      result->Error("INVALID_ARGUMENTS", "Expected [modelPath, nThreads, contextSize, nGpuLayers, mmprojPath]");
      return;
    }
    std::string path = GetStringVal((*list)[0]);
    int threads = (int)GetInt64Val((*list)[1], 4);
    int ctx = (int)GetInt64Val((*list)[2], 2048);
    int gpu_layers = (int)GetInt64Val((*list)[3], 0);
    std::string mmproj = (list->size() > 4) ? GetStringVal((*list)[4]) : "";

    auto host_chan = host_channel_;
    std::shared_ptr<flutter::MethodResult<flutter::EncodableValue>> shared_result = std::move(result);

    std::thread([&core, path, threads, ctx, gpu_layers, mmproj, host_chan, shared_result]() {
      std::string vision_warning;
      std::string err = core.loadModel(
          path, threads, ctx, gpu_layers, mmproj,
          [host_chan](double progress) {
            PostToMainThread([host_chan, progress]() {
              host_chan->InvokeMethod("onLoadProgress", std::make_unique<flutter::EncodableValue>(progress));
            });
          },
          &vision_warning);

      PostToMainThread([shared_result, err, vision_warning]() {
        if (!err.empty()) {
          shared_result->Error("LOAD_FAILED", err);
        } else if (!vision_warning.empty()) {
          shared_result->Error("VISION_WARNING", vision_warning);
        } else {
          shared_result->Success();
        }
      });
    }).detach();

  } else if (method == "generate") {
    const auto* list = std::get_if<flutter::EncodableList>(method_call.arguments());
    if (!list || list->size() < 16) {
      result->Error("INVALID_ARGUMENTS", "Expected generate request list");
      return;
    }
    llama_desktop::DesktopGenerateRequest req;
    req.prompt = GetStringVal((*list)[0]);
    req.max_tokens = (int)GetInt64Val((*list)[1], 512);
    req.temperature = GetDoubleVal((*list)[2], 0.7);
    req.top_p = GetDoubleVal((*list)[3], 0.9);
    req.top_k = (int)GetInt64Val((*list)[4], 40);
    req.min_p = GetDoubleVal((*list)[5], 0.05);
    req.typical_p = GetDoubleVal((*list)[6], 1.0);
    req.repeat_penalty = GetDoubleVal((*list)[7], 1.1);
    req.frequency_penalty = GetDoubleVal((*list)[8], 0.0);
    req.presence_penalty = GetDoubleVal((*list)[9], 0.0);
    req.repeat_last_n = (int)GetInt64Val((*list)[10], 64);
    req.mirostat = (int)GetInt64Val((*list)[11], 0);
    req.mirostat_tau = GetDoubleVal((*list)[12], 5.0);
    req.mirostat_eta = GetDoubleVal((*list)[13], 0.1);
    req.seed = GetInt64Val((*list)[14], -1);
    req.penalize_newline = GetBoolVal((*list)[15], true);

    auto host_chan = host_channel_;
    std::shared_ptr<flutter::MethodResult<flutter::EncodableValue>> shared_result = std::move(result);

    std::thread([&core, req, host_chan, shared_result]() {
      std::string err = core.generate(req, [host_chan](const std::string& token) {
        PostToMainThread([host_chan, token]() {
          host_chan->InvokeMethod("onToken", std::make_unique<flutter::EncodableValue>(token));
        });
      });

      PostToMainThread([host_chan, shared_result, err]() {
        if (!err.empty()) {
          host_chan->InvokeMethod("onError", std::make_unique<flutter::EncodableValue>(err));
          shared_result->Error("GENERATE_FAILED", err);
        } else {
          host_chan->InvokeMethod("onDone", nullptr);
          shared_result->Success();
        }
      });
    }).detach();

  } else if (method == "generateChat") {
    const auto* list = std::get_if<flutter::EncodableList>(method_call.arguments());
    if (!list || list->size() < 17) {
      result->Error("INVALID_ARGUMENTS", "Expected generateChat request list");
      return;
    }
    llama_desktop::DesktopChatRequest chat_req;
    const auto* raw_msgs = std::get_if<flutter::EncodableList>(&(*list)[0]);
    if (raw_msgs) {
      for (const auto& item : *raw_msgs) {
        const auto* mlist = std::get_if<flutter::EncodableList>(&item);
        if (mlist && mlist->size() >= 2) {
          llama_desktop::DesktopChatMessage msg;
          msg.role = GetStringVal((*mlist)[0]);
          msg.content = GetStringVal((*mlist)[1]);
          if (mlist->size() > 2) msg.image_path = GetStringVal((*mlist)[2]);
          chat_req.messages.push_back(msg);
        }
      }
    }
    chat_req.template_name = GetStringVal((*list)[1]);
    chat_req.gen.max_tokens = (int)GetInt64Val((*list)[2], 512);
    chat_req.gen.temperature = GetDoubleVal((*list)[3], 0.7);
    chat_req.gen.top_p = GetDoubleVal((*list)[4], 0.9);
    chat_req.gen.top_k = (int)GetInt64Val((*list)[5], 40);
    chat_req.gen.min_p = GetDoubleVal((*list)[6], 0.05);
    chat_req.gen.typical_p = GetDoubleVal((*list)[7], 1.0);
    chat_req.gen.repeat_penalty = GetDoubleVal((*list)[8], 1.1);
    chat_req.gen.frequency_penalty = GetDoubleVal((*list)[9], 0.0);
    chat_req.gen.presence_penalty = GetDoubleVal((*list)[10], 0.0);
    chat_req.gen.repeat_last_n = (int)GetInt64Val((*list)[11], 64);
    chat_req.gen.mirostat = (int)GetInt64Val((*list)[12], 0);
    chat_req.gen.mirostat_tau = GetDoubleVal((*list)[13], 5.0);
    chat_req.gen.mirostat_eta = GetDoubleVal((*list)[14], 0.1);
    chat_req.gen.seed = GetInt64Val((*list)[15], -1);
    chat_req.gen.penalize_newline = GetBoolVal((*list)[16], true);

    auto host_chan = host_channel_;
    std::shared_ptr<flutter::MethodResult<flutter::EncodableValue>> shared_result = std::move(result);

    std::thread([&core, chat_req, host_chan, shared_result]() {
      std::string err = core.generateChat(chat_req, [host_chan](const std::string& token) {
        PostToMainThread([host_chan, token]() {
          host_chan->InvokeMethod("onToken", std::make_unique<flutter::EncodableValue>(token));
        });
      });

      PostToMainThread([host_chan, shared_result, err]() {
        if (!err.empty()) {
          host_chan->InvokeMethod("onError", std::make_unique<flutter::EncodableValue>(err));
          shared_result->Error("GENERATE_FAILED", err);
        } else {
          host_chan->InvokeMethod("onDone", nullptr);
          shared_result->Success();
        }
      });
    }).detach();

  } else if (method == "getSupportedTemplates") {
    auto templates = core.getSupportedTemplates();
    flutter::EncodableList res;
    for (const auto& t : templates) res.push_back(flutter::EncodableValue(t));
    result->Success(flutter::EncodableValue(res));

  } else if (method == "stop") {
    core.stop();
    result->Success();

  } else if (method == "dispose") {
    core.dispose();
    result->Success();

  } else if (method == "isModelLoaded") {
    result->Success(flutter::EncodableValue(core.isModelLoaded()));

  } else if (method == "getContextInfo") {
    auto info = core.getContextInfo();
    flutter::EncodableList res = {
        flutter::EncodableValue(info.tokens_used),
        flutter::EncodableValue(info.context_size),
        flutter::EncodableValue(info.usage_percentage)};
    result->Success(flutter::EncodableValue(res));

  } else if (method == "clearContext") {
    core.clearContext();
    result->Success();

  } else if (method == "setSystemPromptLength") {
    int len = (int)GetInt64Val(*method_call.arguments(), 0);
    core.setSystemPromptLength(len);
    result->Success();

  } else if (method == "registerCustomTemplate") {
    const auto* list = std::get_if<flutter::EncodableList>(method_call.arguments());
    if (list && list->size() >= 2) {
      core.registerCustomTemplate(GetStringVal((*list)[0]), GetStringVal((*list)[1]));
    }
    result->Success();

  } else if (method == "unregisterCustomTemplate") {
    std::string name = GetStringVal(*method_call.arguments());
    core.unregisterCustomTemplate(name);
    result->Success();

  } else if (method == "detectGpu") {
    auto gpu = core.detectGpu();
    flutter::EncodableList res = {
        flutter::EncodableValue(gpu.vulkan_supported),
        flutter::EncodableValue(gpu.gpu_name),
        flutter::EncodableValue(gpu.vulkan_api_version),
        flutter::EncodableValue(gpu.device_local_memory_bytes),
        flutter::EncodableValue(gpu.free_ram_bytes),
        flutter::EncodableValue(gpu.recommended_gpu_layers)};
    result->Success(flutter::EncodableValue(res));

  } else {
    result->NotImplemented();
  }
}

void LlamaFlutterAndroidPlugin::HandleClusterMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  const std::string& method = method_call.method_name();
  auto& core = llama_desktop::LlamaDesktopCore::instance();

  if (method == "startWorker") {
    const auto* map = std::get_if<flutter::EncodableMap>(method_call.arguments());
    int port = 50052;
    int threads = (int)std::max<unsigned int>(1, std::thread::hardware_concurrency());
    if (map) {
      auto it_port = map->find(flutter::EncodableValue("port"));
      if (it_port != map->end()) port = (int)GetInt64Val(it_port->second, 50052);
      auto it_th = map->find(flutter::EncodableValue("threads"));
      if (it_th != map->end() && !it_th->second.IsNull()) {
        threads = (int)GetInt64Val(it_th->second, threads);
      }
    }
    std::string err = core.startWorker(port, threads, "");
    if (!err.empty()) {
      result->Error("START_FAILED", err);
    } else {
      result->Success();
    }

  } else if (method == "stopWorker") {
    core.stopWorker();
    result->Success();

  } else if (method == "workerStatus") {
    auto st = core.workerStatus();
    flutter::EncodableMap map;
    map[flutter::EncodableValue("state")] = flutter::EncodableValue(st.state);
    if (!st.error.empty()) map[flutter::EncodableValue("error")] = flutter::EncodableValue(st.error);
    if (!st.peer.empty()) map[flutter::EncodableValue("peer")] = flutter::EncodableValue(st.peer);
    map[flutter::EncodableValue("heldMb")] = flutter::EncodableValue(st.held_mb);
    map[flutter::EncodableValue("ramMb")] = flutter::EncodableValue(st.ram_mb);
    map[flutter::EncodableValue("bandwidth")] = flutter::EncodableValue(st.bandwidth);
    map[flutter::EncodableValue("throttled")] = flutter::EncodableValue(st.throttled);
    map[flutter::EncodableValue("id")] = flutter::EncodableValue(st.id);
    map[flutter::EncodableValue("batteryUnrestricted")] = flutter::EncodableValue(st.battery_unrestricted);
    map[flutter::EncodableValue("batteryPct")] = flutter::EncodableValue(st.battery_pct);
    map[flutter::EncodableValue("charging")] = flutter::EncodableValue(st.charging);
    result->Success(flutter::EncodableValue(map));

  } else if (method == "localFreeRamMb") {
    result->Success(flutter::EncodableValue(core.localFreeRamMb()));

  } else if (method == "benchmark") {
    int threads = (int)std::max<unsigned int>(1, std::thread::hardware_concurrency());
    double gbps = core.benchmark(threads);
    result->Success(flutter::EncodableValue(gbps));

  } else if (method == "lastLoadBuffers") {
    auto bufs = core.lastLoadBuffers();
    flutter::EncodableList list;
    for (const auto& item : bufs) {
      flutter::EncodableMap map;
      map[flutter::EncodableValue("name")] = flutter::EncodableValue(item.first);
      map[flutter::EncodableValue("mb")] = flutter::EncodableValue(item.second);
      list.push_back(flutter::EncodableValue(map));
    }
    result->Success(flutter::EncodableValue(list));

  } else if (method == "probe") {
    const auto* map = std::get_if<flutter::EncodableMap>(method_call.arguments());
    std::string endpoint;
    if (map) {
      auto it = map->find(flutter::EncodableValue("endpoint"));
      if (it != map->end()) endpoint = GetStringVal(it->second);
    }
    bool ok = core.probe(endpoint, 2000);
    result->Success(flutter::EncodableValue(ok));

  } else if (method == "setCluster") {
    const auto* map = std::get_if<flutter::EncodableMap>(method_call.arguments());
    std::vector<std::string> endpoints;
    std::vector<double> weights;
    double local_weight = 1.0;
    if (map) {
      auto it_ep = map->find(flutter::EncodableValue("endpoints"));
      if (it_ep != map->end()) {
        const auto* elist = std::get_if<flutter::EncodableList>(&it_ep->second);
        if (elist) {
          for (const auto& v : *elist) endpoints.push_back(GetStringVal(v));
        }
      }
      auto it_w = map->find(flutter::EncodableValue("weights"));
      if (it_w != map->end()) {
        const auto* wlist = std::get_if<flutter::EncodableList>(&it_w->second);
        if (wlist) {
          for (const auto& v : *wlist) weights.push_back(GetDoubleVal(v, 1.0));
        }
      }
      auto it_lw = map->find(flutter::EncodableValue("localWeight"));
      if (it_lw != map->end()) local_weight = GetDoubleVal(it_lw->second, 1.0);
    }
    std::string err = core.setCluster(endpoints, weights, local_weight);
    if (!err.empty()) {
      result->Error("UNREACHABLE", err);
    } else {
      result->Success();
    }

  } else if (method == "setKeepScreenOn") {
    const auto* map = std::get_if<flutter::EncodableMap>(method_call.arguments());
    bool on = false;
    if (map) {
      auto it = map->find(flutter::EncodableValue("on"));
      if (it != map->end()) on = GetBoolVal(it->second, false);
    }
    core.setKeepScreenOn(on);
    result->Success();

  } else if (method == "requestBatteryUnrestricted") {
    result->Success();

  } else if (method == "discover") {
    // Pure Dart UDP discovery handles this across all platforms without mDNS daemon issues
    result->Success(flutter::EncodableValue(flutter::EncodableList()));

  } else {
    result->NotImplemented();
  }
}

}  // namespace llama_flutter_android
