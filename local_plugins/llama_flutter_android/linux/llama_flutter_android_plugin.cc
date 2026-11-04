#include "include/llama_flutter_android/llama_flutter_android_plugin.h"

#include "../src/llama_desktop_core.h"

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define LLAMA_FLUTTER_ANDROID_PLUGIN(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), llama_flutter_android_plugin_get_type(), \
                              LlamaFlutterAndroidPlugin))

struct _LlamaFlutterAndroidPlugin {
  GObject parent_instance;
  FlMethodChannel* host_channel;
  FlMethodChannel* cluster_channel;
};

G_DEFINE_TYPE(LlamaFlutterAndroidPlugin, llama_flutter_android_plugin, g_object_get_type())

namespace {

struct DispatchData {
  std::function<void()> fn;
};

static gboolean dispatch_callback(gpointer user_data) {
  auto* data = static_cast<DispatchData*>(user_data);
  if (data && data->fn) {
    data->fn();
  }
  delete data;
  return G_SOURCE_REMOVE;
}

static void PostToMainThread(std::function<void()> task) {
  auto* data = new DispatchData{std::move(task)};
  g_idle_add(dispatch_callback, data);
}

static std::string GetStringVal(FlValue* val, const std::string& def = "") {
  if (val && fl_value_get_type(val) == FL_VALUE_TYPE_STRING) {
    return fl_value_get_string(val);
  }
  return def;
}

static int64_t GetInt64Val(FlValue* val, int64_t def = 0) {
  if (!val) return def;
  if (fl_value_get_type(val) == FL_VALUE_TYPE_INT) {
    return fl_value_get_int(val);
  }
  if (fl_value_get_type(val) == FL_VALUE_TYPE_FLOAT) {
    return (int64_t)fl_value_get_float(val);
  }
  return def;
}

static double GetDoubleVal(FlValue* val, double def = 0.0) {
  if (!val) return def;
  if (fl_value_get_type(val) == FL_VALUE_TYPE_FLOAT) {
    return fl_value_get_float(val);
  }
  if (fl_value_get_type(val) == FL_VALUE_TYPE_INT) {
    return (double)fl_value_get_int(val);
  }
  return def;
}

static bool GetBoolVal(FlValue* val, bool def = false) {
  if (val && fl_value_get_type(val) == FL_VALUE_TYPE_BOOL) {
    return fl_value_get_bool(val);
  }
  return def;
}

}  // namespace

static void handle_host_method_call(
    LlamaFlutterAndroidPlugin* self,
    FlMethodCall* method_call) {
  const gchar* method = fl_method_call_get_name(method_call);
  FlValue* args = fl_method_call_get_args(method_call);
  auto& core = llama_desktop::LlamaDesktopCore::instance();

  if (strcmp(method, "loadModel") == 0) {
    if (!args || fl_value_get_type(args) != FL_VALUE_TYPE_LIST || fl_value_get_length(args) < 4) {
      fl_method_call_respond_error(method_call, "INVALID_ARGUMENTS", "Expected list", nullptr, nullptr);
      return;
    }
    std::string path = GetStringVal(fl_value_get_list_value(args, 0));
    int threads = (int)GetInt64Val(fl_value_get_list_value(args, 1), 4);
    int ctx = (int)GetInt64Val(fl_value_get_list_value(args, 2), 2048);
    int gpu_layers = (int)GetInt64Val(fl_value_get_list_value(args, 3), 0);
    std::string mmproj = (fl_value_get_length(args) > 4) ? GetStringVal(fl_value_get_list_value(args, 4)) : "";

    g_object_ref(method_call);
    FlMethodChannel* host_channel = self->host_channel;
    g_object_ref(host_channel);

    std::thread([&core, path, threads, ctx, gpu_layers, mmproj, method_call, host_channel]() {
      std::string vision_warning;
      std::string err = core.loadModel(
          path, threads, ctx, gpu_layers, mmproj,
          [host_channel](double progress) {
            PostToMainThread([host_channel, progress]() {
              g_autoptr(FlValue) val = fl_value_new_float(progress);
              fl_method_channel_invoke_method(host_channel, "onLoadProgress", val, nullptr, nullptr, nullptr);
            });
          },
          &vision_warning);

      PostToMainThread([method_call, host_channel, err, vision_warning]() {
        if (!err.empty()) {
          fl_method_call_respond_error(method_call, "LOAD_FAILED", err.c_str(), nullptr, nullptr);
        } else if (!vision_warning.empty()) {
          fl_method_call_respond_error(method_call, "VISION_WARNING", vision_warning.c_str(), nullptr, nullptr);
        } else {
          g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
          fl_method_call_respond(method_call, response, nullptr);
        }
        g_object_unref(method_call);
        g_object_unref(host_channel);
      });
    }).detach();

  } else if (strcmp(method, "generate") == 0) {
    if (!args || fl_value_get_type(args) != FL_VALUE_TYPE_LIST || fl_value_get_length(args) < 16) {
      fl_method_call_respond_error(method_call, "INVALID_ARGUMENTS", "Expected generate request list", nullptr, nullptr);
      return;
    }
    llama_desktop::DesktopGenerateRequest req;
    req.prompt = GetStringVal(fl_value_get_list_value(args, 0));
    req.max_tokens = (int)GetInt64Val(fl_value_get_list_value(args, 1), 512);
    req.temperature = GetDoubleVal(fl_value_get_list_value(args, 2), 0.7);
    req.top_p = GetDoubleVal(fl_value_get_list_value(args, 3), 0.9);
    req.top_k = (int)GetInt64Val(fl_value_get_list_value(args, 4), 40);
    req.min_p = GetDoubleVal(fl_value_get_list_value(args, 5), 0.05);
    req.typical_p = GetDoubleVal(fl_value_get_list_value(args, 6), 1.0);
    req.repeat_penalty = GetDoubleVal(fl_value_get_list_value(args, 7), 1.1);
    req.frequency_penalty = GetDoubleVal(fl_value_get_list_value(args, 8), 0.0);
    req.presence_penalty = GetDoubleVal(fl_value_get_list_value(args, 9), 0.0);
    req.repeat_last_n = (int)GetInt64Val(fl_value_get_list_value(args, 10), 64);
    req.mirostat = (int)GetInt64Val(fl_value_get_list_value(args, 11), 0);
    req.mirostat_tau = GetDoubleVal(fl_value_get_list_value(args, 12), 5.0);
    req.mirostat_eta = GetDoubleVal(fl_value_get_list_value(args, 13), 0.1);
    req.seed = GetInt64Val(fl_value_get_list_value(args, 14), -1);
    req.penalize_newline = GetBoolVal(fl_value_get_list_value(args, 15), true);

    g_object_ref(method_call);
    FlMethodChannel* host_channel = self->host_channel;
    g_object_ref(host_channel);

    std::thread([&core, req, method_call, host_channel]() {
      std::string err = core.generate(req, [host_channel](const std::string& token) {
        PostToMainThread([host_channel, token]() {
          g_autoptr(FlValue) val = fl_value_new_string(token.c_str());
          fl_method_channel_invoke_method(host_channel, "onToken", val, nullptr, nullptr, nullptr);
        });
      });

      PostToMainThread([method_call, host_channel, err]() {
        if (!err.empty()) {
          g_autoptr(FlValue) val = fl_value_new_string(err.c_str());
          fl_method_channel_invoke_method(host_channel, "onError", val, nullptr, nullptr, nullptr);
          fl_method_call_respond_error(method_call, "GENERATE_FAILED", err.c_str(), nullptr, nullptr);
        } else {
          fl_method_channel_invoke_method(host_channel, "onDone", nullptr, nullptr, nullptr, nullptr);
          g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
          fl_method_call_respond(method_call, response, nullptr);
        }
        g_object_unref(method_call);
        g_object_unref(host_channel);
      });
    }).detach();

  } else if (strcmp(method, "generateChat") == 0) {
    if (!args || fl_value_get_type(args) != FL_VALUE_TYPE_LIST || fl_value_get_length(args) < 17) {
      fl_method_call_respond_error(method_call, "INVALID_ARGUMENTS", "Expected generateChat request list", nullptr, nullptr);
      return;
    }
    llama_desktop::DesktopChatRequest chat_req;
    FlValue* raw_msgs = fl_value_get_list_value(args, 0);
    if (raw_msgs && fl_value_get_type(raw_msgs) == FL_VALUE_TYPE_LIST) {
      size_t count = fl_value_get_length(raw_msgs);
      for (size_t i = 0; i < count; i++) {
        FlValue* mlist = fl_value_get_list_value(raw_msgs, i);
        if (mlist && fl_value_get_type(mlist) == FL_VALUE_TYPE_LIST && fl_value_get_length(mlist) >= 2) {
          llama_desktop::DesktopChatMessage msg;
          msg.role = GetStringVal(fl_value_get_list_value(mlist, 0));
          msg.content = GetStringVal(fl_value_get_list_value(mlist, 1));
          if (fl_value_get_length(mlist) > 2) msg.image_path = GetStringVal(fl_value_get_list_value(mlist, 2));
          chat_req.messages.push_back(msg);
        }
      }
    }
    chat_req.template_name = GetStringVal(fl_value_get_list_value(args, 1));
    chat_req.gen.max_tokens = (int)GetInt64Val(fl_value_get_list_value(args, 2), 512);
    chat_req.gen.temperature = GetDoubleVal(fl_value_get_list_value(args, 3), 0.7);
    chat_req.gen.top_p = GetDoubleVal(fl_value_get_list_value(args, 4), 0.9);
    chat_req.gen.top_k = (int)GetInt64Val(fl_value_get_list_value(args, 5), 40);
    chat_req.gen.min_p = GetDoubleVal(fl_value_get_list_value(args, 6), 0.05);
    chat_req.gen.typical_p = GetDoubleVal(fl_value_get_list_value(args, 7), 1.0);
    chat_req.gen.repeat_penalty = GetDoubleVal(fl_value_get_list_value(args, 8), 1.1);
    chat_req.gen.frequency_penalty = GetDoubleVal(fl_value_get_list_value(args, 9), 0.0);
    chat_req.gen.presence_penalty = GetDoubleVal(fl_value_get_list_value(args, 10), 0.0);
    chat_req.gen.repeat_last_n = (int)GetInt64Val(fl_value_get_list_value(args, 11), 64);
    chat_req.gen.mirostat = (int)GetInt64Val(fl_value_get_list_value(args, 12), 0);
    chat_req.gen.mirostat_tau = GetDoubleVal(fl_value_get_list_value(args, 13), 5.0);
    chat_req.gen.mirostat_eta = GetDoubleVal(fl_value_get_list_value(args, 14), 0.1);
    chat_req.gen.seed = GetInt64Val(fl_value_get_list_value(args, 15), -1);
    chat_req.gen.penalize_newline = GetBoolVal(fl_value_get_list_value(args, 16), true);

    g_object_ref(method_call);
    FlMethodChannel* host_channel = self->host_channel;
    g_object_ref(host_channel);

    std::thread([&core, chat_req, method_call, host_channel]() {
      std::string err = core.generateChat(chat_req, [host_channel](const std::string& token) {
        PostToMainThread([host_channel, token]() {
          g_autoptr(FlValue) val = fl_value_new_string(token.c_str());
          fl_method_channel_invoke_method(host_channel, "onToken", val, nullptr, nullptr, nullptr);
        });
      });

      PostToMainThread([method_call, host_channel, err]() {
        if (!err.empty()) {
          g_autoptr(FlValue) val = fl_value_new_string(err.c_str());
          fl_method_channel_invoke_method(host_channel, "onError", val, nullptr, nullptr, nullptr);
          fl_method_call_respond_error(method_call, "GENERATE_FAILED", err.c_str(), nullptr, nullptr);
        } else {
          fl_method_channel_invoke_method(host_channel, "onDone", nullptr, nullptr, nullptr, nullptr);
          g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
          fl_method_call_respond(method_call, response, nullptr);
        }
        g_object_unref(method_call);
        g_object_unref(host_channel);
      });
    }).detach();

  } else if (strcmp(method, "getSupportedTemplates") == 0) {
    auto list = core.getSupportedTemplates();
    g_autoptr(FlValue) res = fl_value_new_list();
    for (const auto& item : list) {
      fl_value_append_take(res, fl_value_new_string(item.c_str()));
    }
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "stop") == 0) {
    core.stop();
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "dispose") == 0) {
    core.dispose();
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "isModelLoaded") == 0) {
    g_autoptr(FlValue) res = fl_value_new_bool(core.isModelLoaded());
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "getContextInfo") == 0) {
    auto info = core.getContextInfo();
    g_autoptr(FlValue) res = fl_value_new_list();
    fl_value_append_take(res, fl_value_new_int(info.tokens_used));
    fl_value_append_take(res, fl_value_new_int(info.context_size));
    fl_value_append_take(res, fl_value_new_float(info.usage_percentage));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "clearContext") == 0) {
    core.clearContext();
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "setSystemPromptLength") == 0) {
    core.setSystemPromptLength((int)GetInt64Val(args, 0));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "registerCustomTemplate") == 0) {
    if (args && fl_value_get_type(args) == FL_VALUE_TYPE_LIST && fl_value_get_length(args) >= 2) {
      core.registerCustomTemplate(
          GetStringVal(fl_value_get_list_value(args, 0)),
          GetStringVal(fl_value_get_list_value(args, 1)));
    }
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "unregisterCustomTemplate") == 0) {
    core.unregisterCustomTemplate(GetStringVal(args));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "detectGpu") == 0) {
    auto gpu = core.detectGpu();
    g_autoptr(FlValue) res = fl_value_new_list();
    fl_value_append_take(res, fl_value_new_bool(gpu.vulkan_supported));
    fl_value_append_take(res, fl_value_new_string(gpu.gpu_name.c_str()));
    fl_value_append_take(res, fl_value_new_int(gpu.vulkan_api_version));
    fl_value_append_take(res, fl_value_new_int(gpu.device_local_memory_bytes));
    fl_value_append_take(res, fl_value_new_int(gpu.free_ram_bytes));
    fl_value_append_take(res, fl_value_new_int(gpu.recommended_gpu_layers));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
    fl_method_call_respond(method_call, response, nullptr);
  }
}

static void handle_cluster_method_call(
    LlamaFlutterAndroidPlugin* self,
    FlMethodCall* method_call) {
  const gchar* method = fl_method_call_get_name(method_call);
  FlValue* args = fl_method_call_get_args(method_call);
  auto& core = llama_desktop::LlamaDesktopCore::instance();

  if (strcmp(method, "startWorker") == 0) {
    int port = 50052;
    int threads = (int)std::max<unsigned int>(1, std::thread::hardware_concurrency());
    if (args && fl_value_get_type(args) == FL_VALUE_TYPE_MAP) {
      FlValue* p = fl_value_lookup_string(args, "port");
      if (p) port = (int)GetInt64Val(p, 50052);
      FlValue* t = fl_value_lookup_string(args, "threads");
      if (t && fl_value_get_type(t) != FL_VALUE_TYPE_NULL) {
        threads = (int)GetInt64Val(t, threads);
      }
    }
    std::string err = core.startWorker(port, threads, "");
    if (!err.empty()) {
      fl_method_call_respond_error(method_call, "START_FAILED", err.c_str(), nullptr, nullptr);
    } else {
      g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
      fl_method_call_respond(method_call, response, nullptr);
    }

  } else if (strcmp(method, "stopWorker") == 0) {
    core.stopWorker();
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "workerStatus") == 0) {
    auto st = core.workerStatus();
    g_autoptr(FlValue) map = fl_value_new_map();
    fl_value_set_string_take(map, "state", fl_value_new_string(st.state.c_str()));
    if (!st.error.empty()) fl_value_set_string_take(map, "error", fl_value_new_string(st.error.c_str()));
    if (!st.peer.empty()) fl_value_set_string_take(map, "peer", fl_value_new_string(st.peer.c_str()));
    fl_value_set_string_take(map, "heldMb", fl_value_new_int(st.held_mb));
    fl_value_set_string_take(map, "ramMb", fl_value_new_int(st.ram_mb));
    fl_value_set_string_take(map, "bandwidth", fl_value_new_float(st.bandwidth));
    fl_value_set_string_take(map, "throttled", fl_value_new_bool(st.throttled));
    fl_value_set_string_take(map, "id", fl_value_new_string(st.id.c_str()));
    fl_value_set_string_take(map, "batteryUnrestricted", fl_value_new_bool(st.battery_unrestricted));
    fl_value_set_string_take(map, "batteryPct", fl_value_new_int(st.battery_pct));
    fl_value_set_string_take(map, "charging", fl_value_new_bool(st.charging));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(map));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "localFreeRamMb") == 0) {
    g_autoptr(FlValue) res = fl_value_new_int(core.localFreeRamMb());
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "benchmark") == 0) {
    int threads = (int)std::max<unsigned int>(1, std::thread::hardware_concurrency());
    g_autoptr(FlValue) res = fl_value_new_float(core.benchmark(threads));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "lastLoadBuffers") == 0) {
    auto bufs = core.lastLoadBuffers();
    g_autoptr(FlValue) list = fl_value_new_list();
    for (const auto& item : bufs) {
      g_autoptr(FlValue) map = fl_value_new_map();
      fl_value_set_string_take(map, "name", fl_value_new_string(item.first.c_str()));
      fl_value_set_string_take(map, "mb", fl_value_new_float(item.second));
      fl_value_append(list, map);
    }
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(list));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "probe") == 0) {
    std::string endpoint;
    if (args && fl_value_get_type(args) == FL_VALUE_TYPE_MAP) {
      FlValue* ep = fl_value_lookup_string(args, "endpoint");
      if (ep) endpoint = GetStringVal(ep);
    }
    g_autoptr(FlValue) res = fl_value_new_bool(core.probe(endpoint, 2000));
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "setCluster") == 0) {
    std::vector<std::string> endpoints;
    std::vector<double> weights;
    double local_weight = 1.0;
    if (args && fl_value_get_type(args) == FL_VALUE_TYPE_MAP) {
      FlValue* ep_val = fl_value_lookup_string(args, "endpoints");
      if (ep_val && fl_value_get_type(ep_val) == FL_VALUE_TYPE_LIST) {
        size_t count = fl_value_get_length(ep_val);
        for (size_t i = 0; i < count; i++) {
          endpoints.push_back(GetStringVal(fl_value_get_list_value(ep_val, i)));
        }
      }
      FlValue* w_val = fl_value_lookup_string(args, "weights");
      if (w_val && fl_value_get_type(w_val) == FL_VALUE_TYPE_LIST) {
        size_t count = fl_value_get_length(w_val);
        for (size_t i = 0; i < count; i++) {
          weights.push_back(GetDoubleVal(fl_value_get_list_value(w_val, i), 1.0));
        }
      }
      FlValue* lw_val = fl_value_lookup_string(args, "localWeight");
      if (lw_val) local_weight = GetDoubleVal(lw_val, 1.0);
    }
    std::string err = core.setCluster(endpoints, weights, local_weight);
    if (!err.empty()) {
      fl_method_call_respond_error(method_call, "UNREACHABLE", err.c_str(), nullptr, nullptr);
    } else {
      g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
      fl_method_call_respond(method_call, response, nullptr);
    }

  } else if (strcmp(method, "setKeepScreenOn") == 0 || strcmp(method, "requestBatteryUnrestricted") == 0) {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
    fl_method_call_respond(method_call, response, nullptr);

  } else if (strcmp(method, "discover") == 0) {
    g_autoptr(FlValue) res = fl_value_new_list();
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_success_response_new(res));
    fl_method_call_respond(method_call, response, nullptr);

  } else {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
    fl_method_call_respond(method_call, response, nullptr);
  }
}

static void llama_flutter_android_plugin_dispose(GObject* object) {
  LlamaFlutterAndroidPlugin* self = LLAMA_FLUTTER_ANDROID_PLUGIN(object);
  g_clear_object(&self->host_channel);
  g_clear_object(&self->cluster_channel);
  G_OBJECT_CLASS(llama_flutter_android_plugin_parent_class)->dispose(object);
}

static void llama_flutter_android_plugin_class_init(LlamaFlutterAndroidPluginClass* klass) {
  G_OBJECT_CLASS(klass)->dispose = llama_flutter_android_plugin_dispose;
}

static void llama_flutter_android_plugin_init(LlamaFlutterAndroidPlugin* self) {}

void llama_flutter_android_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  LlamaFlutterAndroidPlugin* plugin = LLAMA_FLUTTER_ANDROID_PLUGIN(
      g_object_new(llama_flutter_android_plugin_get_type(), nullptr));

  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();

  plugin->host_channel = fl_method_channel_new(
      fl_plugin_registrar_get_messenger(registrar),
      "llama_flutter_android/host",
      FL_METHOD_CODEC(codec));

  plugin->cluster_channel = fl_method_channel_new(
      fl_plugin_registrar_get_messenger(registrar),
      "llama_flutter_android/cluster",
      FL_METHOD_CODEC(codec));

  fl_method_channel_set_method_call_handler(
      plugin->host_channel,
      [](FlMethodChannel* channel, FlMethodCall* method_call, gpointer user_data) {
        handle_host_method_call(LLAMA_FLUTTER_ANDROID_PLUGIN(user_data), method_call);
      },
      g_object_ref(plugin),
      g_object_unref);

  fl_method_channel_set_method_call_handler(
      plugin->cluster_channel,
      [](FlMethodChannel* channel, FlMethodCall* method_call, gpointer user_data) {
        handle_cluster_method_call(LLAMA_FLUTTER_ANDROID_PLUGIN(user_data), method_call);
      },
      g_object_ref(plugin),
      g_object_unref);

  g_object_unref(plugin);
}
