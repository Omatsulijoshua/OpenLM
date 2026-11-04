#if os(iOS)
import Flutter
import UIKit
#elseif os(macOS)
import FlutterMacOS
import AppKit
import IOKit.ps
#endif
import Foundation

/// MethodChannel for the device cluster feature on iOS and macOS.
///
/// Supports both Master role (`setCluster`, `lastLoadBuffers`, `probe`, `discover`)
/// and Worker role (`startWorker`, `stopWorker`, `workerStatus`, `benchmark`).
final class ClusterChannel: NSObject, NetServiceDelegate, NetServiceBrowserDelegate {
    private static let channelName = "llama_flutter_android/cluster"
    private static let serviceType = "_openlm._tcp."
    private static let rpcVersion = "rpc7"
    private static let idKey = "cluster_worker_id"

    private var netService: NetService?
    private var baselineFootprintMb: Int = 0
    private var cachedBandwidth: Double = 0
    private var activityToken: NSObjectProtocol?

    // Active mDNS discovery state
    private var browser: NetServiceBrowser?
    private var resolvingServices: [NetService] = []
    private var discoveredWorkers: [String: [String: Any]] = [:]

    static func register(messenger: FlutterBinaryMessenger) {
        let channel = FlutterMethodChannel(name: channelName, binaryMessenger: messenger)
        let instance = ClusterChannel()
        channel.setMethodCallHandler(instance.handle)
    }

    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as? [String: Any]
        switch call.method {
        case "startWorker":
            let port = args?["port"] as? Int ?? 50052
            let threads = args?["threads"] as? Int ?? max(2, ProcessInfo.processInfo.activeProcessorCount - 2)
            startWorker(port: port, threads: threads, result: result)
        case "stopWorker":
            // ggml's RPC server loop has no stop flag; stop advertising so
            // masters ignore it until the app restarts.
            netService?.stop()
            netService = nil
            setIdleTimer(disabled: false)
            result(nil)
        case "workerStatus":
            result(workerStatus())
        case "localFreeRamMb":
            result(LlamaClusterWorker.availableMemoryMb())
        case "setKeepScreenOn":
            let on = args?["on"] as? Bool ?? false
            setIdleTimer(disabled: on)
            result(nil)
        case "requestBatteryUnrestricted":
            result(nil)
        case "benchmark":
            let threads = max(2, ProcessInfo.processInfo.activeProcessorCount - 2)
            DispatchQueue.global(qos: .userInitiated).async {
                let gbps = LlamaClusterWorker.memoryBandwidth(withThreads: Int32(threads))
                DispatchQueue.main.async {
                    self.cachedBandwidth = gbps
                    result(gbps)
                }
            }
        case "lastLoadBuffers":
            let raw = LlamaIosWrapper.lastLoadBuffers()
            let buffers: [[String: Any]] = raw.split(separator: "\n").compactMap { line in
                let parts = line.split(separator: "\t")
                guard parts.count >= 2,
                      let mb = Double(parts[1].trimmingCharacters(in: .whitespaces)) else {
                    return nil
                }
                let name = String(parts[0]).trimmingCharacters(in: .whitespaces)
                return name.isEmpty ? nil : ["name": name, "mb": mb]
            }
            result(buffers)
        case "probe":
            let endpoint = args?["endpoint"] as? String ?? ""
            DispatchQueue.global(qos: .userInitiated).async {
                let ok = Self.probeEndpoint(endpoint, timeoutMs: 2000)
                DispatchQueue.main.async { result(ok) }
            }
        case "setCluster":
            let endpoints = args?["endpoints"] as? [String] ?? []
            let weights = (args?["weights"] as? [Double] ?? []).map { NSNumber(value: $0) }
            let localWeight = args?["localWeight"] as? Double ?? 1.0
            DispatchQueue.global(qos: .userInitiated).async {
                let unreachable = endpoints.filter { !Self.probeEndpoint($0, timeoutMs: 2000) }
                DispatchQueue.main.async {
                    if !unreachable.isEmpty {
                        result(FlutterError(
                            code: "UNREACHABLE",
                            message: "Can't reach \(unreachable.joined(separator: ", "))",
                            details: nil
                        ))
                        return
                    }
                    LlamaIosWrapper.setClusterWithEndpoints(
                        endpoints,
                        weights: weights,
                        localWeight: localWeight
                    )
                    self.setIdleTimer(disabled: !endpoints.isEmpty)
                    result(nil)
                }
            }
        case "discover":
            let timeoutMs = args?["timeoutMs"] as? Int ?? 4000
            discoverWorkers(timeoutMs: timeoutMs, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func startWorker(port: Int, threads: Int, result: @escaping FlutterResult) {
        let cache = (NSTemporaryDirectory() as NSString).appendingPathComponent("rpc_cache")
        baselineFootprintMb = LlamaClusterWorker.footprintMb()
        if let err = LlamaClusterWorker.start(onPort: Int32(port), threads: Int32(threads), cacheDir: cache) {
            result(FlutterError(code: "START_FAILED", message: err, details: nil))
            return
        }
        setIdleTimer(disabled: true)
        DispatchQueue.global(qos: .userInitiated).async {
            if self.cachedBandwidth == 0 {
                self.cachedBandwidth = LlamaClusterWorker.memoryBandwidth(withThreads: Int32(threads))
            }
            DispatchQueue.main.async {
                self.advertise(port: port, bandwidth: self.cachedBandwidth)
                result(nil)
            }
        }
    }

    private func setIdleTimer(disabled: Bool) {
        DispatchQueue.main.async {
            #if os(iOS)
            UIApplication.shared.isIdleTimerDisabled = disabled
            #elseif os(macOS)
            if disabled {
                if self.activityToken == nil {
                    self.activityToken = ProcessInfo.processInfo.beginActivity(
                        options: [.userInitiated, .idleSystemSleepDisabled],
                        reason: "OpenLM cluster worker / master active"
                    )
                }
            } else if let token = self.activityToken {
                ProcessInfo.processInfo.endActivity(token)
                self.activityToken = nil
            }
            #endif
        }
    }

    private func workerStatus() -> [String: Any?] {
        let running = LlamaClusterWorker.isRunning()
        let error = LlamaClusterWorker.lastError()
        let state = running ? "running" : (error != nil ? "error" : "stopped")
        let held = running ? max(0, LlamaClusterWorker.footprintMb() - baselineFootprintMb) : 0
        let (batPct, charging) = battery()
        return [
            "state": state,
            "error": error,
            "peer": running ? LlamaClusterWorker.connectedPeer() : nil,
            "heldMb": held,
            "ramMb": LlamaClusterWorker.availableMemoryMb(),
            "bandwidth": cachedBandwidth,
            "throttled": false,
            "id": workerId(),
            "batteryUnrestricted": true,
            "batteryPct": batPct,
            "charging": charging,
        ]
    }

    private func advertise(port: Int, bandwidth: Double) {
        netService?.stop()
        let name = deviceName()
        let svc = NetService(domain: "local.", type: Self.serviceType, name: name, port: Int32(port))
        let txt: [String: Data] = [
            "ram": "\(LlamaClusterWorker.availableMemoryMb())".data(using: .utf8)!,
            "bw": String(format: "%.1f", bandwidth).data(using: .utf8)!,
            "model": name.data(using: .utf8)!,
            "id": workerId().data(using: .utf8)!,
            "v": Self.rpcVersion.data(using: .utf8)!,
        ]
        svc.setTXTRecord(NetService.data(fromTXTRecord: txt))
        svc.delegate = self
        svc.publish()
        netService = svc
    }

    // MARK: - mDNS Discovery (NetServiceBrowser)

    private func discoverWorkers(timeoutMs: Int, result: @escaping FlutterResult) {
        browser?.stop()
        resolvingServices.removeAll()
        discoveredWorkers.removeAll()

        let b = NetServiceBrowser()
        b.delegate = self
        browser = b
        b.searchForServices(ofType: Self.serviceType, inDomain: "local.")

        let delay = DispatchTimeInterval.milliseconds(max(500, timeoutMs))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self = self else { return }
            self.browser?.stop()
            self.browser = nil
            for svc in self.resolvingServices {
                svc.stop()
            }
            self.resolvingServices.removeAll()
            let myId = self.workerId()
            let list = self.discoveredWorkers.values.filter { ($0["id"] as? String) != myId }
            result(Array(list))
        }
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        service.delegate = self
        resolvingServices.append(service)
        service.resolve(withTimeout: 3.0)
    }

    func netServiceDidResolveAddress(_ sender: NetService) {
        guard let addresses = sender.addresses else { return }
        var ipv4Host: String?
        for addrData in addresses {
            addrData.withUnsafeBytes { (rawBuf: UnsafeRawBufferPointer) in
                guard let base = rawBuf.baseAddress else { return }
                let family = base.assumingMemoryBound(to: sockaddr.self).pointee.sa_family
                if family == sa_family_t(AF_INET) {
                    var addr = base.assumingMemoryBound(to: sockaddr_in.self).pointee.sin_addr
                    var buf = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
                    if inet_ntop(AF_INET, &addr, &buf, socklen_t(INET_ADDRSTRLEN)) != nil {
                        let ip = String(cString: buf)
                        if !ip.hasPrefix("127.") {
                            ipv4Host = ip
                        }
                    }
                }
            }
            if ipv4Host != nil { break }
        }
        guard let host = ipv4Host, sender.port > 0 else { return }

        var txtDict: [String: String] = [:]
        if let txtData = sender.txtRecordData() {
            let raw = NetService.dictionary(fromTXTRecord: txtData)
            for (k, v) in raw {
                if let s = String(data: v, encoding: .utf8) {
                    txtDict[k] = s
                }
            }
        }
        let ver = txtDict["v"] ?? ""
        let compatible = ver.isEmpty || ver == Self.rpcVersion
        let name = txtDict["model"] ?? sender.name
        let ramMb = Int(txtDict["ram"] ?? "") ?? 0
        let bw = Double(txtDict["bw"] ?? "") ?? 0.0
        let id = txtDict["id"] ?? "\(host):\(sender.port)"

        discoveredWorkers["\(host):\(sender.port)"] = [
            "name": name,
            "host": host,
            "port": sender.port,
            "ramMb": ramMb,
            "bandwidth": bw,
            "id": id,
            "compatible": compatible,
        ]
    }

    // MARK: - TCP Probe

    private static func probeEndpoint(_ endpoint: String, timeoutMs: Int) -> Bool {
        let parts = endpoint.split(separator: ":")
        guard parts.count == 2,
              let port = UInt16(parts[1]) else { return false }
        let host = String(parts[0])

        let fd = socket(AF_INET, SOCK_STREAM, 0)
        if fd < 0 { return false }
        defer { close(fd) }

        let flags = fcntl(fd, F_GETFL, 0)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else { return false }

        let res = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if res == 0 { return true }
        if errno != EINPROGRESS { return false }

        var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
        let polled = poll(&pfd, 1, Int32(timeoutMs))
        if polled <= 0 { return false }

        var soError: Int32 = 0
        var len = socklen_t(MemoryLayout<Int32>.size)
        if getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &len) < 0 { return false }
        return soError == 0
    }

    private func workerId() -> String {
        let defaults = UserDefaults.standard
        if let existing = defaults.string(forKey: Self.idKey), !existing.isEmpty {
            return existing
        }
        let fresh = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))
        defaults.set(fresh, forKey: Self.idKey)
        return fresh
    }

    private func battery() -> (Int, Bool) {
        #if os(iOS)
        let dev = UIDevice.current
        dev.isBatteryMonitoringEnabled = true
        let pct = dev.batteryLevel >= 0 ? Int((dev.batteryLevel * 100).rounded()) : -1
        let charging = dev.batteryState == .charging || dev.batteryState == .full
        return (pct, charging)
        #elseif os(macOS)
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef],
              let first = sources.first,
              let desc = IOPSGetPowerSourceDescription(snapshot, first)?.takeUnretainedValue() as? [String: Any] else {
            return (-1, true)
        }
        let cur = desc[kIOPSCurrentCapacityKey as String] as? Int ?? -1
        let maxCap = desc[kIOPSMaxCapacityKey as String] as? Int ?? 100
        let pct = (cur >= 0 && maxCap > 0) ? (cur * 100 / maxCap) : -1
        let powerState = desc[kIOPSPowerSourceStateKey as String] as? String
        let charging = (powerState == kIOPSACPowerValue as String) ||
                       ((desc[kIOPSIsChargingKey as String] as? Bool) == true)
        return (pct, charging)
        #endif
    }

    private func deviceName() -> String {
        #if os(macOS)
        if let localized = Host.current().localizedName, !localized.isEmpty {
            return localized
        }
        var size: Int = 0
        if sysctlbyname("hw.model", nil, &size, nil, 0) == 0 && size > 0 {
            var buf = [CChar](repeating: 0, count: size)
            if sysctlbyname("hw.model", &buf, &size, nil, 0) == 0 {
                return String(cString: buf)
            }
        }
        return "Mac"
        #else
        var sys = utsname()
        uname(&sys)
        let code = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        return Self.appleModels[code] ?? UIDevice.current.name
        #endif
    }

    private static let appleModels: [String: String] = [
        // iPhone 12 – 17
        "iPhone13,1": "iPhone 12 mini", "iPhone13,2": "iPhone 12",
        "iPhone13,3": "iPhone 12 Pro", "iPhone13,4": "iPhone 12 Pro Max",
        "iPhone14,4": "iPhone 13 mini", "iPhone14,5": "iPhone 13",
        "iPhone14,2": "iPhone 13 Pro", "iPhone14,3": "iPhone 13 Pro Max",
        "iPhone14,7": "iPhone 14", "iPhone14,8": "iPhone 14 Plus",
        "iPhone15,2": "iPhone 14 Pro", "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone15,4": "iPhone 15", "iPhone15,5": "iPhone 15 Plus",
        "iPhone16,1": "iPhone 15 Pro", "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone17,1": "iPhone 16 Pro", "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16", "iPhone17,4": "iPhone 16 Plus",
        "iPhone17,5": "iPhone 16e",
        // iPad Air / Pro / mini (M-series & recent A-series)
        "iPad13,1": "iPad Air 4", "iPad13,2": "iPad Air 4",
        "iPad13,16": "iPad Air (M1)", "iPad13,17": "iPad Air (M1)",
        "iPad14,8": "iPad Air 11\" (M2)", "iPad14,9": "iPad Air 11\" (M2)",
        "iPad14,10": "iPad Air 13\" (M2)", "iPad14,11": "iPad Air 13\" (M2)",
        "iPad13,4": "iPad Pro 11\" (M1)", "iPad13,8": "iPad Pro 12.9\" (M1)",
        "iPad14,3": "iPad Pro 11\" (M2)", "iPad14,5": "iPad Pro 12.9\" (M2)",
        "iPad16,3": "iPad Pro 11\" (M4)", "iPad16,5": "iPad Pro 13\" (M4)",
        "iPad14,1": "iPad mini 6", "iPad16,1": "iPad mini (A17 Pro)",
    ]
}
