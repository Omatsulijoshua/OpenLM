#include "llama_desktop_core.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace llama_desktop {

namespace {

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static mtmd_context* g_mtmd_ctx = nullptr;
static std::string g_current_model_path;
static std::atomic<bool> g_stop_flag{false};
static int g_n_past = 0;
static int g_system_prompt_length = 0;
static std::vector<llama_token> g_cached_tokens;
static std::mutex g_infer_mutex;

// Custom chat templates
static std::mutex g_template_mutex;
static std::map<std::string, std::string> g_custom_templates;

// Cluster state
static std::mutex g_cluster_mutex;
static std::vector<std::string> g_cluster_endpoints;
static std::vector<float> g_cluster_weights;
static float g_cluster_local_weight = 0.0f;
static std::vector<std::string> g_registered_endpoints;

// Log capture
static std::mutex g_log_mutex;
static std::string g_load_buffers;
static std::string g_load_error;
static bool g_capture_load_error = false;
static std::string g_worker_peer;
static ggml_log_level g_last_log_level = GGML_LOG_LEVEL_INFO;

// Worker state
static std::atomic<bool> g_worker_running{false};
static std::string g_worker_error;
static int64_t g_baseline_rss_mb = 0;
static double g_cached_bandwidth = 0.0;
static std::string g_worker_id;

static void desktopLlamaLog(ggml_log_level level, const char* text, void*) {
    if (!text) return;
    const ggml_log_level effective =
        level == GGML_LOG_LEVEL_CONT ? g_last_log_level : level;
    if (level != GGML_LOG_LEVEL_CONT) g_last_log_level = level;
    if (effective == GGML_LOG_LEVEL_DEBUG) return;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (const char* at = std::strstr(text, " model buffer size = ")) {
        const char* colon = std::strchr(text, ':');
        const char* start = colon && colon < at ? colon + 1 : text;
        while (start < at && *start == ' ') start++;
        const double mib = std::strtod(at + std::strlen(" model buffer size = "), nullptr);
        g_load_buffers.append(start, at - start).append("\t")
            .append(std::to_string(mib)).append("\n");
    }
    static const char kConnected[] = "RPC client connected: ";
    if (std::strncmp(text, kConnected, sizeof(kConnected) - 1) == 0) {
        g_worker_peer.assign(text + sizeof(kConnected) - 1);
        while (!g_worker_peer.empty() && std::isspace((unsigned char)g_worker_peer.back())) {
            g_worker_peer.pop_back();
        }
    } else if (std::strncmp(text, "RPC client disconnected", 23) == 0) {
        g_worker_peer.clear();
    }
    if (g_capture_load_error) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            g_load_error.assign(text);
        } else if (level == GGML_LOG_LEVEL_CONT && !g_load_error.empty()) {
            g_load_error.append(text);
        }
        if (g_load_error.size() > 4096) {
            g_load_error.erase(0, g_load_error.size() - 4096);
        }
    }
}

static bool isRpcDevice(ggml_backend_dev_t dev) {
    const char* name = ggml_backend_dev_name(dev);
    return name && std::strncmp(name, "RPC", 3) == 0;
}

static ggml_backend_dev_t firstLocalGpuDevice() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU ||
             type == GGML_BACKEND_DEVICE_TYPE_IGPU) && !isRpcDevice(dev)) {
            return dev;
        }
    }
    return nullptr;
}

static int readBlockCount(const char* model_path) {
    gguf_init_params params = {/* .no_alloc = */ true, /* .ctx = */ nullptr};
    gguf_context* gguf = gguf_init_from_file(model_path, params);
    if (!gguf) return -1;
    int count = -1;
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id >= 0) {
        const std::string key = std::string(gguf_get_val_str(gguf, arch_id)) + ".block_count";
        const int64_t id = gguf_find_key(gguf, key.c_str());
        if (id >= 0) {
            switch (gguf_get_kv_type(gguf, id)) {
                case GGUF_TYPE_UINT32: count = (int)gguf_get_val_u32(gguf, id); break;
                case GGUF_TYPE_INT32:  count = gguf_get_val_i32(gguf, id); break;
                default: break;
            }
        }
    }
    gguf_free(gguf);
    return count;
}

struct ClusterPlan {
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> split;
    int n_gpu_layers = 0;
    llama_model_tensor_buft_override overrides[2] = {};
};

static std::string planCluster(const char* model_path, bool use_local_gpu, ClusterPlan& plan) {
    std::vector<std::string> endpoints;
    std::vector<float> weights;
    float local_weight;
    {
        std::lock_guard<std::mutex> lock(g_cluster_mutex);
        endpoints = g_cluster_endpoints;
        weights = g_cluster_weights;
        local_weight = g_cluster_local_weight;
    }
    const int n_layer = readBlockCount(model_path);
    if (n_layer <= 0) return "Could not read the model's layer count";

    ggml_backend_dev_t local_gpu = use_local_gpu ? firstLocalGpuDevice() : nullptr;
    float remote_weight = 0.0f;
    std::vector<float> device_weights;
    if (local_gpu && local_weight > 0.0f) {
        plan.devices.push_back(local_gpu);
        device_weights.push_back(std::max(local_weight, 0.01f));
    }
    for (size_t i = 0; i < endpoints.size(); i++) {
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoints[i].c_str());
        if (!reg) return "Worker " + endpoints[i] + " reported no devices";
        {
            std::lock_guard<std::mutex> lock(g_cluster_mutex);
            if (std::find(g_registered_endpoints.begin(), g_registered_endpoints.end(),
                          endpoints[i]) == g_registered_endpoints.end()) {
                ggml_backend_register(reg);
                g_registered_endpoints.push_back(endpoints[i]);
            }
        }
        const size_t n_dev = ggml_backend_reg_dev_count(reg);
        for (size_t d = 0; d < n_dev; d++) {
            plan.devices.push_back(ggml_backend_reg_dev_get(reg, d));
            device_weights.push_back(std::max(weights[i], 0.01f) / n_dev);
        }
        remote_weight += std::max(weights[i], 0.01f);
    }
    if (plan.devices.empty()) return "No cluster workers";
    plan.devices.push_back(nullptr);

    plan.split.assign(llama_max_devices(), 0.0f);
    for (size_t d = 0; d < device_weights.size() && d < plan.split.size(); d++) {
        plan.split[d] = device_weights[d];
    }

    const float total = std::max(local_weight, 0.0f) + remote_weight;
    int local = (int)std::lround(n_layer * std::max(local_weight, 0.0f) / total);
    local = std::clamp(local, 0, n_layer);
    plan.n_gpu_layers = (local_gpu && local_weight > 0.0f) ? (n_layer + 1) : (n_layer + 1 - local);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        plan.overrides[0] = {"^(output|token_embd)\\.weight$", ggml_backend_dev_buffer_type(cpu)};
        plan.overrides[1] = {nullptr, nullptr};
    }
    return {};
}

static bool isValidUTF8(const char* str, size_t len) {
    if (!str) return false;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    while (i < len) {
        unsigned char c = bytes[i];
        if ((c & 0x80) == 0) { i++; continue; }
        int nb = 0;
        if ((c & 0xE0) == 0xC0) nb = 2;
        else if ((c & 0xF0) == 0xE0) nb = 3;
        else if ((c & 0xF8) == 0xF0) nb = 4;
        else return false;
        if (i + nb > len) return false;
        for (int j = 1; j < nb; j++) {
            if ((bytes[i + j] & 0xC0) != 0x80) return false;
        }
        if (nb == 2 && (c & 0x1E) == 0) return false;
        if (nb == 3) {
            if (c == 0xED && (bytes[i + 1] & 0x20) == 0x20) return false;
            if (c == 0xE0 && (bytes[i + 1] & 0x20) == 0) return false;
        }
        if (nb == 4) {
            if (c > 0xF4) return false;
            if (c == 0xF0 && (bytes[i + 1] & 0x30) == 0) return false;
            if (c == 0xF4 && bytes[i + 1] > 0x8F) return false;
        }
        i += nb;
    }
    return true;
}

static std::string sanitizeUTF8(const char* str, size_t len) {
    if (!str || len == 0) return "";
    if (isValidUTF8(str, len)) return std::string(str, len);
    std::string result;
    result.reserve(len);
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    while (i < len) {
        unsigned char c = bytes[i];
        if ((c & 0x80) == 0) { result += (char)c; i++; continue; }
        int nb = 0;
        if ((c & 0xE0) == 0xC0) nb = 2;
        else if ((c & 0xF0) == 0xE0) nb = 3;
        else if ((c & 0xF8) == 0xF0) nb = 4;
        else { result += "\xEF\xBF\xBD"; i++; continue; }
        if (i + nb > len) { result += "\xEF\xBF\xBD"; break; }
        std::string seq(reinterpret_cast<const char*>(bytes + i), nb);
        if (isValidUTF8(seq.c_str(), nb)) result += seq;
        else result += "\xEF\xBF\xBD";
        i += nb;
    }
    return result;
}

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string detectTemplateFromPath(const std::string& path) {
    const std::string lower = toLower(path);
    if (lower.find("llama-3") != std::string::npos || lower.find("llama3") != std::string::npos) return "llama3";
    if (lower.find("deepseek") != std::string::npos) return "deepseek";
    if (lower.find("gemma") != std::string::npos) return "gemma";
    if (lower.find("phi-3") != std::string::npos || lower.find("phi3") != std::string::npos || lower.find("phi-4") != std::string::npos) return "phi";
    if (lower.find("mistral") != std::string::npos || lower.find("zephyr") != std::string::npos) return "zephyr";
    if (lower.find("vicuna") != std::string::npos) return "vicuna";
    if (lower.find("alpaca") != std::string::npos) return "alpaca";
    return "chatml";
}

static std::string formatChatMessages(
    const std::vector<DesktopChatMessage>& messages,
    const std::string& requested_template,
    const std::string& model_path) {
    std::string tmpl = requested_template.empty()
        ? detectTemplateFromPath(model_path)
        : toLower(requested_template);

    {
        std::lock_guard<std::mutex> lock(g_template_mutex);
        auto it = g_custom_templates.find(tmpl);
        if (it != g_custom_templates.end()) {
            std::string custom = it->second;
            std::string sys_text, user_text, asst_text;
            for (const auto& m : messages) {
                if (m.role == "system") sys_text += m.content + "\n";
                else if (m.role == "user") user_text = m.content;
                else if (m.role == "assistant") asst_text = m.content;
            }
            auto replaceAll = [](std::string& str, const std::string& from, const std::string& to) {
                size_t pos = 0;
                while ((pos = str.find(from, pos)) != std::string::npos) {
                    str.replace(pos, from.length(), to);
                    pos += to.length();
                }
            };
            replaceAll(custom, "{system}", sys_text);
            replaceAll(custom, "{user}", user_text);
            replaceAll(custom, "{assistant}", asst_text);
            return custom;
        }
    }

    std::ostringstream out;
    if (tmpl == "llama3") {
        out << "<|begin_of_text|>";
        for (const auto& m : messages) {
            out << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n"
                << m.content << "<|eot_id|>";
        }
        out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (tmpl == "gemma") {
        for (const auto& m : messages) {
            std::string role = (m.role == "assistant") ? "model" : "user";
            out << "<start_of_turn>" << role << "\n" << m.content << "<end_of_turn>\n";
        }
        out << "<start_of_turn>model\n";
    } else if (tmpl == "phi") {
        for (const auto& m : messages) {
            out << "<|" << m.role << "|>\n" << m.content << "<|end|>\n";
        }
        out << "<|assistant|>\n";
    } else if (tmpl == "alpaca") {
        for (const auto& m : messages) {
            if (m.role == "system") out << m.content << "\n\n";
            else if (m.role == "user") out << "### Instruction:\n" << m.content << "\n\n";
            else if (m.role == "assistant") out << "### Response:\n" << m.content << "\n\n";
        }
        out << "### Response:\n";
    } else if (tmpl == "vicuna") {
        for (const auto& m : messages) {
            if (m.role == "system") out << m.content << "\n\n";
            else if (m.role == "user") out << "USER: " << m.content << "\n";
            else if (m.role == "assistant") out << "ASSISTANT: " << m.content << "</s>\n";
        }
        out << "ASSISTANT: ";
    } else if (tmpl == "deepseek") {
        for (const auto& m : messages) {
            if (m.role == "system") out << m.content << "\n\n";
            else if (m.role == "user") out << "<｜User｜>" << m.content;
            else if (m.role == "assistant") out << "<｜Assistant｜>" << m.content << "<｜end▁of▁sentence｜>";
        }
        out << "<｜Assistant｜>";
    } else {
        // Default: ChatML (Qwen, SmolLM, OpenHermes, Dolphin, etc.)
        for (const auto& m : messages) {
            out << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
        }
        out << "<|im_start|>assistant\n";
    }
    return out.str();
}

static int64_t currentProcessRssMb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (int64_t)(pmc.WorkingSetSize / (1024 * 1024));
    }
    return 0;
#else
    std::ifstream statm("/proc/self/statm");
    long total_pages = 0, rss_pages = 0;
    if (statm >> total_pages >> rss_pages) {
        long page_size = sysconf(_SC_PAGESIZE);
        return (int64_t)((uint64_t)rss_pages * (uint64_t)page_size / (1024 * 1024));
    }
    return 0;
#endif
}

static std::string ensureWorkerId() {
    if (!g_worker_id.empty()) return g_worker_id;
    static const char kHex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    g_worker_id.reserve(12);
    for (int i = 0; i < 12; i++) {
        g_worker_id.push_back(kHex[dist(gen)]);
    }
    return g_worker_id;
}

}  // namespace

LlamaDesktopCore& LlamaDesktopCore::instance() {
    static LlamaDesktopCore inst;
    return inst;
}

LlamaDesktopCore::LlamaDesktopCore() {
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
    llama_backend_init();
    llama_log_set(desktopLlamaLog, nullptr);
}

std::string LlamaDesktopCore::loadModel(
    const std::string& model_path,
    int n_threads,
    int context_size,
    int n_gpu_layers,
    const std::string& mmproj_path,
    std::function<void(double)> on_progress,
    std::string* out_vision_warning) {
    std::lock_guard<std::mutex> lock(g_infer_mutex);

    if (g_model || g_ctx) {
        dispose();
    }

    std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
    if (!model_file) {
        return "GGUF model file is missing or unreadable: " + model_path;
    }
    if (model_file.tellg() < 4) {
        return "GGUF model file is empty or incomplete";
    }
    model_file.seekg(0, std::ios::beg);
    char magic[4] = {};
    model_file.read(magic, sizeof(magic));
    if (!model_file || std::memcmp(magic, "GGUF", sizeof(magic)) != 0) {
        return "Invalid GGUF model header";
    }
    model_file.close();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    static ggml_backend_dev_t no_gpu_devices[] = {nullptr};
    ggml_backend_dev_t local_gpu_devices[] = {firstLocalGpuDevice(), nullptr};
    if (n_gpu_layers <= 0) {
        model_params.devices = no_gpu_devices;
    } else if (local_gpu_devices[0] != nullptr) {
        model_params.devices = local_gpu_devices;
    }

    ClusterPlan cluster;
    bool has_cluster = false;
    {
        std::lock_guard<std::mutex> clock(g_cluster_mutex);
        has_cluster = !g_cluster_endpoints.empty();
    }
    if (has_cluster) {
        const std::string err = planCluster(model_path.c_str(), n_gpu_layers > 0, cluster);
        if (!err.empty()) {
            return "Cluster: " + err;
        }
        model_params.devices = cluster.devices.data();
        model_params.tensor_split = cluster.split.data();
        model_params.n_gpu_layers = cluster.n_gpu_layers;
        if (cluster.overrides[0].pattern) {
            model_params.tensor_buft_overrides = cluster.overrides;
        }
    }

    llama_log_set(desktopLlamaLog, nullptr);
    {
        std::lock_guard<std::mutex> llock(g_log_mutex);
        g_load_error.clear();
        g_load_buffers.clear();
        g_capture_load_error = true;
    }

    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    {
        std::lock_guard<std::mutex> llock(g_log_mutex);
        g_capture_load_error = false;
    }
    if (!g_model) {
        std::lock_guard<std::mutex> llock(g_log_mutex);
        return g_load_error.empty()
            ? "Failed to load GGUF model; check model compatibility and available RAM"
            : ("Failed to load GGUF model: " + g_load_error);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::max(512, context_size);
    ctx_params.n_threads = std::max(1, n_threads);
    ctx_params.n_threads_batch = std::max(1, n_threads);
    ctx_params.n_batch = 512;
    ctx_params.op_offload = (n_gpu_layers > 0) && !has_cluster;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return "Failed to create llama_context";
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        llama_free(g_ctx);
        llama_model_free(g_model);
        g_ctx = nullptr;
        g_model = nullptr;
        return "Failed to retrieve model vocabulary";
    }

    g_current_model_path = model_path;
    g_n_past = 0;
    g_cached_tokens.clear();

    if (!mmproj_path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = false;
        mparams.n_threads = std::max(1, n_threads);
        mparams.warmup = false;
        g_mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), g_model, mparams);
        if (!g_mtmd_ctx && out_vision_warning) {
            *out_vision_warning = "VISION_LOAD_FAILED: Could not initialize vision projector at " + mmproj_path;
        }
    }

    if (on_progress) {
        on_progress(1.0);
    }
    return {};
}

std::string LlamaDesktopCore::generate(
    const DesktopGenerateRequest& req,
    std::function<void(const std::string&)> on_token) {
    std::lock_guard<std::mutex> lock(g_infer_mutex);

    if (!g_model || !g_ctx || !g_vocab) {
        return "Model not loaded";
    }
    g_stop_flag = false;

    const std::string sanitized = sanitizeUTF8(req.prompt.c_str(), req.prompt.size());
    const int n_prompt_tokens = -llama_tokenize(
        g_vocab, sanitized.c_str(), (int)sanitized.size(), nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        return "Failed to tokenize prompt";
    }

    std::vector<llama_token> tokens(n_prompt_tokens);
    const int actual = llama_tokenize(
        g_vocab, sanitized.c_str(), (int)sanitized.size(),
        tokens.data(), (int)tokens.size(), true, true);
    if (actual < 0) {
        return "Tokenize failed";
    }
    tokens.resize(actual);

    // Prefix cache reuse
    size_t common = 0;
    const size_t limit = std::min(g_cached_tokens.size(), tokens.size());
    while (common < limit && g_cached_tokens[common] == tokens[common]) {
        common++;
    }
    if (common > 0 && common == tokens.size()) {
        common--;
    }
    if ((int)common < g_n_past) {
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, (int)common, -1);
        g_n_past = (int)common;
    }

    const int n_ctx = (int)llama_n_ctx(g_ctx);
    const int new_tokens_count = (int)tokens.size() - g_n_past;
    if (g_n_past + new_tokens_count > n_ctx) {
        const int n_discard = n_ctx / 4;
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
        llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, -1, -n_discard);
        g_n_past = std::max(0, g_n_past - n_discard);
        g_cached_tokens.clear();
    }

    // Evaluate uncached prompt tokens in batches of 512
    {
        llama_batch batch = llama_batch_init(512, 0, 1);
        int idx = g_n_past;
        while (idx < (int)tokens.size()) {
            batch.n_tokens = 0;
            while (batch.n_tokens < 512 && idx < (int)tokens.size()) {
                batch.token[batch.n_tokens] = tokens[idx];
                batch.pos[batch.n_tokens] = idx;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = (idx == (int)tokens.size() - 1) ? 1 : 0;
                batch.n_tokens++;
                idx++;
            }
            if (llama_decode(g_ctx, batch) != 0) {
                llama_batch_free(batch);
                return "llama_decode failed during prompt evaluation";
            }
        }
        g_n_past = (int)tokens.size();
        g_cached_tokens = tokens;
        llama_batch_free(batch);
    }

    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(chain_params);

    const uint32_t seed_val = (req.seed < 0) ? (uint32_t)std::time(nullptr) : (uint32_t)req.seed;

    if (req.repeat_penalty != 1.0 || req.frequency_penalty != 0.0 || req.presence_penalty != 0.0) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(g_vocab),
            req.repeat_last_n,
            (float)req.repeat_penalty,
            (float)req.frequency_penalty,
            (float)req.presence_penalty));
    }

    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp((float)req.temperature));

    if (req.mirostat == 1) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat(
            llama_vocab_n_tokens(g_vocab),
            seed_val,
            (float)req.mirostat_tau,
            (float)req.mirostat_eta,
            100));
    } else if (req.mirostat == 2) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat_v2(
            seed_val,
            (float)req.mirostat_tau,
            (float)req.mirostat_eta));
    } else {
        if (req.min_p > 0.0 && req.min_p < 1.0) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p((float)req.min_p, 1));
        }
        if (req.typical_p < 1.0) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_typical((float)req.typical_p, 1));
        }
        if (req.top_k > 0) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(req.top_k));
        }
        if (req.top_p < 1.0) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p((float)req.top_p, 1));
        }
    }
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(seed_val));

    const llama_token eos_token = llama_vocab_eos(g_vocab);
    std::string partial_token;

    for (int i = 0; i < req.max_tokens && !g_stop_flag; i++) {
        const llama_token token_id = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (token_id == eos_token || llama_vocab_is_eog(g_vocab, token_id)) {
            break;
        }

        char buf[256] = {0};
        int n = llama_token_to_piece(g_vocab, token_id, buf, sizeof(buf) - 1, 0, true);
        if (n < 0) n = 0;
        buf[n] = '\0';

        partial_token.append(buf, n);
        if (isValidUTF8(partial_token.c_str(), partial_token.size())) {
            if (on_token && !g_stop_flag) {
                on_token(partial_token);
            }
            partial_token.clear();
        }

        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0] = token_id;
        next_batch.pos[0] = g_n_past;
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0] = 1;
        next_batch.n_tokens = 1;
        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        g_n_past++;
        g_cached_tokens.push_back(token_id);
    }

    if (!partial_token.empty() && !g_stop_flag && on_token) {
        on_token(sanitizeUTF8(partial_token.c_str(), partial_token.size()));
    }
    return {};
}

std::string LlamaDesktopCore::generateChat(
    const DesktopChatRequest& req,
    std::function<void(const std::string&)> on_token) {
    DesktopGenerateRequest gen_req = req.gen;
    gen_req.prompt = formatChatMessages(req.messages, req.template_name, g_current_model_path);
    return generate(gen_req, on_token);
}

std::vector<std::string> LlamaDesktopCore::getSupportedTemplates() {
    std::vector<std::string> list = {
        "chatml", "llama3", "gemma", "phi", "alpaca", "vicuna", "zephyr", "deepseek"
    };
    std::lock_guard<std::mutex> lock(g_template_mutex);
    for (const auto& kv : g_custom_templates) {
        list.push_back(kv.first);
    }
    return list;
}

void LlamaDesktopCore::registerCustomTemplate(const std::string& name, const std::string& content) {
    std::lock_guard<std::mutex> lock(g_template_mutex);
    g_custom_templates[toLower(name)] = content;
}

void LlamaDesktopCore::unregisterCustomTemplate(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_template_mutex);
    g_custom_templates.erase(toLower(name));
}

void LlamaDesktopCore::stop() {
    g_stop_flag = true;
}

void LlamaDesktopCore::dispose() {
    g_stop_flag = true;
    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    g_n_past = 0;
    g_cached_tokens.clear();
}

bool LlamaDesktopCore::isModelLoaded() const {
    return g_model != nullptr && g_ctx != nullptr;
}

DesktopContextInfo LlamaDesktopCore::getContextInfo() const {
    DesktopContextInfo info;
    if (!g_ctx) return info;
    info.tokens_used = g_n_past;
    info.context_size = (int64_t)llama_n_ctx(g_ctx);
    info.usage_percentage = info.context_size > 0
        ? (double)info.tokens_used * 100.0 / (double)info.context_size
        : 0.0;
    return info;
}

void LlamaDesktopCore::clearContext() {
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    if (g_ctx) {
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, g_system_prompt_length, -1);
        g_n_past = g_system_prompt_length;
        if ((int)g_cached_tokens.size() > g_system_prompt_length) {
            g_cached_tokens.resize(g_system_prompt_length);
        }
    }
}

void LlamaDesktopCore::setSystemPromptLength(int length) {
    g_system_prompt_length = std::max(0, length);
}

DesktopGpuInfo LlamaDesktopCore::detectGpu() const {
    DesktopGpuInfo info;
    info.free_ram_bytes = localFreeRamMb() * 1024LL * 1024LL;
    ggml_backend_dev_t dev = firstLocalGpuDevice();
    if (!dev) {
        info.vulkan_supported = false;
        info.gpu_name = "CPU";
        info.vulkan_api_version = -1;
        info.device_local_memory_bytes = -1;
        info.recommended_gpu_layers = 0;
        return info;
    }
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    const char* desc = ggml_backend_dev_description(dev);
    const char* name = ggml_backend_dev_name(dev);
    info.vulkan_supported = true;
    info.gpu_name = (desc && desc[0]) ? desc : ((name && name[0]) ? name : "GPU");
    info.vulkan_api_version = 1;
    info.device_local_memory_bytes = total_mem > 0 ? (int64_t)total_mem : info.free_ram_bytes;
    info.recommended_gpu_layers = 99;
    return info;
}

std::string LlamaDesktopCore::startWorker(int port, int threads, const std::string& cache_dir) {
    if (g_worker_running) return {};
    ggml_log_set(desktopLlamaLog, nullptr);

    ggml_backend_dev_t dev = firstLocalGpuDevice();
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return "No compute device found for worker";
    }

    g_baseline_rss_mb = currentProcessRssMb();
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_worker_error.clear();
        g_worker_peer.clear();
    }

    const int n_threads = std::max(1, threads);
    if (g_cached_bandwidth <= 0.0) {
        g_cached_bandwidth = benchmark(n_threads);
    }

    const std::string endpoint = "0.0.0.0:" + std::to_string(port);
    g_worker_running = true;
    setKeepScreenOn(true);

    std::thread([endpoint, cache_dir, n_threads, dev]() mutable {
        ggml_backend_dev_t devices[] = {dev};
        ggml_backend_rpc_start_server(
            endpoint.c_str(),
            cache_dir.empty() ? nullptr : cache_dir.c_str(),
            (size_t)n_threads,
            1,
            devices);
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_worker_error = "Worker stopped: couldn't listen on " + endpoint + " (port in use?)";
        }
        g_worker_running = false;
    }).detach();

    return {};
}

void LlamaDesktopCore::stopWorker() {
    setKeepScreenOn(false);
}

DesktopWorkerStatus LlamaDesktopCore::workerStatus() {
    DesktopWorkerStatus st;
    const bool running = g_worker_running.load();
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        st.error = g_worker_error;
        st.peer = running ? g_worker_peer : "";
    }
    st.state = running ? "running" : (!st.error.empty() ? "error" : "stopped");
    st.held_mb = running ? std::max<int64_t>(0, currentProcessRssMb() - g_baseline_rss_mb) : 0;
    st.ram_mb = localFreeRamMb();
    st.bandwidth = g_cached_bandwidth;
    st.throttled = false;
    st.id = ensureWorkerId();
    st.battery_unrestricted = true;
#ifdef _WIN32
    SYSTEM_POWER_STATUS sps{};
    if (GetSystemPowerStatus(&sps)) {
        st.battery_pct = (sps.BatteryLifePercent <= 100) ? (int)sps.BatteryLifePercent : -1;
        st.charging = (sps.ACLineStatus == 1) || ((sps.BatteryFlag & 8) != 0);
    }
#else
    std::ifstream cap("/sys/class/power_supply/BAT0/capacity");
    if (cap) {
        cap >> st.battery_pct;
    }
    std::ifstream status("/sys/class/power_supply/BAT0/status");
    if (status) {
        std::string s;
        status >> s;
        st.charging = (s != "Discharging");
    }
#endif
    return st;
}

int64_t LlamaDesktopCore::localFreeRamMb() const {
#ifdef _WIN32
    MEMORYSTATUSEX mem_info{};
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        return (int64_t)(mem_info.ullAvailPhys / (1024ULL * 1024ULL));
    }
    return 4096;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t val_kb = 0;
    std::string unit;
    while (meminfo >> key >> val_kb >> unit) {
        if (key == "MemAvailable:") {
            return (int64_t)(val_kb / 1024ULL);
        }
    }
    struct sysinfo si{};
    if (sysinfo(&si) == 0) {
        return (int64_t)((uint64_t)si.freeram * (uint64_t)si.mem_unit / (1024ULL * 1024ULL));
    }
    return 4096;
#endif
}

double LlamaDesktopCore::benchmark(int threads) {
    const int n = std::max(1, threads);
    const size_t per_thread = (64u << 20) / sizeof(uint64_t) / n;
    std::vector<uint64_t> data(per_thread * n, 1);
    std::vector<uint64_t> sums(n, 0);
    const int passes = 4;
    auto run = [&](int t) {
        const uint64_t* p = data.data() + per_thread * t;
        uint64_t a = 0, b = 0, c = 0, d = 0;
        for (int pass = 0; pass < passes; pass++) {
            for (size_t i = 0; i + 4 <= per_thread; i += 4) {
                a += p[i]; b += p[i + 1]; c += p[i + 2]; d += p[i + 3];
            }
        }
        sums[t] = a + b + c + d;
    };
    run(0);
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < n; t++) pool.emplace_back(run, t);
    for (auto& th : pool) th.join();
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double bytes = (double)data.size() * sizeof(uint64_t) * passes;
    const double gbps = secs > 0 ? bytes / secs / 1e9 : 0.0;
    g_cached_bandwidth = gbps;
    return gbps;
}

std::vector<std::pair<std::string, double>> LlamaDesktopCore::lastLoadBuffers() const {
    std::string raw;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        raw = g_load_buffers;
    }
    std::vector<std::pair<std::string, double>> out;
    std::istringstream iss(raw);
    std::string line;
    while (std::getline(iss, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string name = line.substr(0, tab);
        double mb = std::strtod(line.substr(tab + 1).c_str(), nullptr);
        if (!name.empty() && mb > 0.0) {
            out.emplace_back(name, mb);
        }
    }
    return out;
}

bool LlamaDesktopCore::probe(const std::string& endpoint, int timeout_ms) const {
    const size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) return false;
    const std::string host = endpoint.substr(0, colon);
    const int port = std::atoi(endpoint.substr(colon + 1).c_str());
    if (port <= 0 || port > 65535) return false;

#ifdef _WIN32
    SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) return false;
    u_long non_blocking = 1;
    ioctlsocket(fd, FIONBIO, &non_blocking);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        closesocket(fd);
        return false;
    }
    int res = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (res == 0) {
        closesocket(fd);
        return true;
    }
    if (WSAGetLastError() != WSAEWOULDBLOCK) {
        closesocket(fd);
        return false;
    }
    fd_set write_fds;
    FD_ZERO(&write_fds);
    FD_SET(fd, &write_fds);
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    res = select(0, nullptr, &write_fds, nullptr, &tv);
    if (res <= 0) {
        closesocket(fd);
        return false;
    }
    int so_error = 0;
    int len = sizeof(so_error);
    getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_error), &len);
    closesocket(fd);
    return so_error == 0;
#else
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        return false;
    }
    int res = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (res == 0) {
        close(fd);
        return true;
    }
    if (errno != EINPROGRESS) {
        close(fd);
        return false;
    }
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLOUT;
    if (poll(&pfd, 1, timeout_ms) <= 0) {
        close(fd);
        return false;
    }
    int so_error = 0;
    socklen_t len = sizeof(so_error);
    getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len);
    close(fd);
    return so_error == 0;
#endif
}

std::string LlamaDesktopCore::setCluster(
    const std::vector<std::string>& endpoints,
    const std::vector<double>& weights,
    double local_weight) {
    std::vector<std::string> unreachable;
    for (const auto& ep : endpoints) {
        if (!probe(ep, 2000)) {
            unreachable.push_back(ep);
        }
    }
    if (!unreachable.empty()) {
        std::string msg = "Can't reach ";
        for (size_t i = 0; i < unreachable.size(); i++) {
            if (i > 0) msg += ", ";
            msg += unreachable[i];
        }
        return msg;
    }

    std::lock_guard<std::mutex> lock(g_cluster_mutex);
    g_cluster_endpoints = endpoints;
    g_cluster_weights.clear();
    for (size_t i = 0; i < endpoints.size(); i++) {
        g_cluster_weights.push_back(i < weights.size() ? (float)weights[i] : 1.0f);
    }
    g_cluster_local_weight = (float)local_weight;
    setKeepScreenOn(!endpoints.empty());
    return {};
}

void LlamaDesktopCore::setKeepScreenOn(bool on) {
#ifdef _WIN32
    if (on) {
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    } else {
        SetThreadExecutionState(ES_CONTINUOUS);
    }
#else
    (void)on;
#endif
}

}  // namespace llama_desktop
