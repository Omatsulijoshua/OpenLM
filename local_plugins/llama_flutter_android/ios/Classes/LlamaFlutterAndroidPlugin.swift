#if os(iOS)
import Flutter
import UIKit
#elseif os(macOS)
import FlutterMacOS
import AppKit
#endif
import Foundation
import Metal

public class LlamaFlutterAndroidPlugin: NSObject, FlutterPlugin, LlamaHostApi {

    private var flutterApi: LlamaFlutterApi?
    private let wrapper = LlamaIosWrapper()

    // Dedicated serial thread for blocking C++ inference
    private let inferenceQueue = DispatchQueue(
        label: "com.write4me.llama_flutter_ios.inference",
        qos: .userInitiated
    )

    public static func register(with registrar: FlutterPluginRegistrar) {
        let instance = LlamaFlutterAndroidPlugin()
        #if os(iOS)
        let messenger = registrar.messenger()
        #elseif os(macOS)
        let messenger = registrar.messenger
        #endif
        instance.flutterApi = LlamaFlutterApi(binaryMessenger: messenger)
        LlamaHostApiSetup.setUp(binaryMessenger: messenger, api: instance)
        ClusterChannel.register(messenger: messenger)
    }

    // MARK: - LlamaHostApi

    func loadModel(config: ModelConfig, completion: @escaping (Result<Void, Error>) -> Void) {
        let gpuLayers = Int32(config.nGpuLayers ?? 99) // Default: full Metal offload on Apple Silicon

        inferenceQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try ObjcExceptionBridge.catch {
                    self.wrapper.loadModel(
                        atPath: config.modelPath,
                        nThreads: Int32(config.nThreads),
                        contextSize: Int32(config.contextSize),
                        nGpuLayers: gpuLayers,
                        progressCallback: { [weak self] progress in
                            DispatchQueue.main.async {
                                self?.flutterApi?.onLoadProgress(progress: progress) { _ in }
                            }
                        }
                    )
                }
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async {
                    self.flutterApi?.onError(error: error.localizedDescription) { _ in }
                    completion(.failure(error))
                }
            }
        }
    }

    func generate(request: GenerateRequest, completion: @escaping (Result<Void, Error>) -> Void) {
        inferenceQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try ObjcExceptionBridge.catch {
                    self.wrapper.generate(
                        withPrompt: request.prompt,
                        maxTokens: Int32(request.maxTokens),
                        temperature: request.temperature,
                        topP: request.topP,
                        topK: Int32(request.topK),
                        minP: request.minP,
                        typicalP: request.typicalP,
                        repeatPenalty: request.repeatPenalty,
                        frequencyPenalty: request.frequencyPenalty,
                        presencePenalty: request.presencePenalty,
                        repeatLastN: Int32(request.repeatLastN),
                        mirostat: Int32(request.mirostat),
                        mirostatTau: request.mirostatTau,
                        mirostatEta: request.mirostatEta,
                        seed: request.seed ?? -1,
                        penalizeNewline: request.penalizeNewline,
                        tokenCallback: { [weak self] token in
                            DispatchQueue.main.async {
                                self?.flutterApi?.onToken(token: token) { _ in }
                            }
                        }
                    )
                }
                DispatchQueue.main.async {
                    self.flutterApi?.onDone { _ in }
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    self.flutterApi?.onError(error: error.localizedDescription) { _ in }
                    completion(.failure(error))
                }
            }
        }
    }

    func generateChat(request: ChatRequest, completion: @escaping (Result<Void, Error>) -> Void) {
        inferenceQueue.async { [weak self] in
            guard let self = self else { return }
            let formattedPrompt = ChatTemplateManager.shared.formatMessages(
                request.messages,
                templateName: request.template,
                modelPath: nil
            )
            let genReq = GenerateRequest(
                prompt: formattedPrompt,
                maxTokens: request.maxTokens,
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                typicalP: request.typicalP,
                repeatPenalty: request.repeatPenalty,
                frequencyPenalty: request.frequencyPenalty,
                presencePenalty: request.presencePenalty,
                repeatLastN: request.repeatLastN,
                mirostat: request.mirostat,
                mirostatTau: request.mirostatTau,
                mirostatEta: request.mirostatEta,
                seed: request.seed,
                penalizeNewline: request.penalizeNewline
            )
            self.generate(request: genReq, completion: completion)
        }
    }

    func getSupportedTemplates() throws -> [String] {
        return ChatTemplateManager.shared.getSupportedTemplates()
    }

    func stop(completion: @escaping (Result<Void, Error>) -> Void) {
        wrapper.stop()
        completion(.success(()))
    }

    func dispose(completion: @escaping (Result<Void, Error>) -> Void) {
        wrapper.stop()
        inferenceQueue.async { [weak self] in
            self?.wrapper.freeModel()
            DispatchQueue.main.async { completion(.success(())) }
        }
    }

    func isModelLoaded() throws -> Bool {
        return wrapper.isModelLoaded()
    }

    func getContextInfo() throws -> ContextInfo {
        let used = Int64(wrapper.tokensUsed())
        let size = Int64(wrapper.contextSize())
        let pct  = size > 0 ? (Double(used) / Double(size)) * 100.0 : 0.0
        return ContextInfo(tokensUsed: used, contextSize: size, usagePercentage: pct)
    }

    func clearContext(completion: @escaping (Result<Void, Error>) -> Void) {
        inferenceQueue.async { [weak self] in
            self?.wrapper.clearContext()
            DispatchQueue.main.async { completion(.success(())) }
        }
    }

    func setSystemPromptLength(length: Int64) throws {
        wrapper.setSystemPromptLength(Int32(length))
    }

    func registerCustomTemplate(name: String, content: String) throws {
        ChatTemplateManager.shared.registerCustomTemplate(name: name, content: content)
    }

    func unregisterCustomTemplate(name: String) throws {
        ChatTemplateManager.shared.unregisterCustomTemplate(name: name)
    }

    func detectGpu(completion: @escaping (Result<GpuInfo, Error>) -> Void) {
        let freeRamBytes = Int64(LlamaClusterWorker.availableMemoryMb()) * 1024 * 1024
        if let device = MTLCreateSystemDefaultDevice() {
            var vramBytes: Int64 = -1
            if #available(iOS 16.0, macOS 10.13, *) {
                vramBytes = Int64(device.recommendedMaxWorkingSetSize)
            }
            let info = GpuInfo(
                vulkanSupported: true,
                gpuName: "\(device.name) (Metal)",
                vulkanApiVersion: 3,
                deviceLocalMemoryBytes: vramBytes > 0 ? vramBytes : freeRamBytes,
                freeRamBytes: freeRamBytes,
                recommendedGpuLayers: 99
            )
            completion(.success(info))
        } else {
            let info = GpuInfo(
                vulkanSupported: false,
                gpuName: "Apple Metal GPU",
                vulkanApiVersion: -1,
                deviceLocalMemoryBytes: -1,
                freeRamBytes: freeRamBytes,
                recommendedGpuLayers: 99
            )
            completion(.success(info))
        }
    }
}
