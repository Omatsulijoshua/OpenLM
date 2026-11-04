import 'dart:async';
import 'dart:convert';
import 'dart:io'
    show
        File,
        InternetAddress,
        InternetAddressType,
        NetworkInterface,
        Platform,
        RawDatagramSocket,
        RawSocketEvent;
import 'dart:math' as math;

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:llama_flutter_android/llama_flutter_android.dart';

import '../core/constants.dart';
import '../core/ui.dart';
import 'app_log_service.dart';
import 'hive_service.dart';

/// What this device does in a device cluster.
enum ClusterRole {
  off,

  /// Runs the model, with part of it on the workers.
  master,

  /// Lends its processor and memory to a master.
  worker,
}

/// One device's part of the loaded model.
class ClusterShare {
  final String name;
  final double mb;
  final bool isThisPhone;
  const ClusterShare(this.name, this.mb, {this.isThisPhone = false});
}

/// A device the master can put layers on, for planning a load.
class _Phone {
  final ClusterWorker? worker; // null: the master itself
  final double capacityMb;
  final double speed; // GB/s
  double assignedMb = 0;
  _Phone(this.worker, this.capacityMb, this.speed);
}

/// Device cluster: spread a GGUF model across devices on the same LAN/Wi-Fi.
/// The master runs the model; workers hold part of its layers.
class ClusterService extends GetxService {
  final HiveService _hive = Get.find<HiveService>();

  static bool get isSupported =>
      !kIsWeb &&
      (Platform.isAndroid ||
          Platform.isIOS ||
          Platform.isMacOS ||
          Platform.isWindows ||
          Platform.isLinux);

  /// Any platform can be a master or worker in a cross-platform cluster.
  static bool get canBeMaster => isSupported;

  /// Workers stop when their process stops.
  static bool get workerStopsWithApp =>
      !kIsWeb &&
      (Platform.isIOS ||
          Platform.isMacOS ||
          Platform.isWindows ||
          Platform.isLinux);

  /// QR codes a worker shows: openlm://worker?host=…&port=…
  static const _qrScheme = 'openlm';

  /// Speed assumed for workers paired before they reported one (GB/s).
  static const _unknownSpeed = 4.0;

  /// Workers the master can use on its next GGUF load.
  final workers = <ClusterWorker>[].obs;

  /// Master mode: use [workers] when loading GGUF models.
  final enabled = false.obs;

  final found = <ClusterWorker>[].obs;
  final isScanning = false.obs;

  /// This device's worker service.
  final worker = const ClusterWorkerStatus(state: 'stopped').obs;
  final _workerRequested = false.obs;

  /// The API server is running (other devices, e.g. a PC, use the model).
  final serving = false.obs;

  /// This device's LAN/Wi-Fi address, or null when it isn't connected.
  final wifiAddress = RxnString();

  /// How the loaded model is spread across devices; empty when it isn't.
  final split = <ClusterShare>[].obs;

  String deviceName = '';
  double? _localSpeed;
  Timer? _statusTimer;
  Timer? _beaconTimer;
  RawDatagramSocket? _beaconSocket;

  ClusterRole get role => _workerRequested.value || worker.value.running
      ? ClusterRole.worker
      : enabled.value
          ? ClusterRole.master
          : ClusterRole.off;

  /// "Pixel 8 72% · POCO F1 28%", or null when the model isn't split.
  String? get splitSummary {
    final total = split.fold<double>(0, (a, s) => a + s.mb);
    if (split.length < 2 || total <= 0) return null;
    return split
        .map((s) => '${s.name} ${(s.mb / total * 100).round()}%')
        .join(' · ');
  }

  @override
  void onInit() {
    super.onInit();
    enabled.value =
        _hive.getSetting<bool>(AppConstants.keyClusterEnabled) ?? false;
    final saved =
        _hive.getSetting<List>(AppConstants.keyClusterWorkers) ?? const [];
    workers.assignAll(saved.whereType<Map>().map(ClusterWorker.fromMap));
    if (isSupported) {
      if (Platform.isIOS) {
        DeviceInfoPlugin().iosInfo.then((info) => deviceName = info.modelName);
      } else if (Platform.isMacOS) {
        DeviceInfoPlugin()
            .macOsInfo
            .then((info) => deviceName = info.computerName);
      } else if (Platform.isWindows) {
        DeviceInfoPlugin()
            .windowsInfo
            .then((info) => deviceName = info.computerName);
      } else if (Platform.isLinux) {
        DeviceInfoPlugin().linuxInfo.then((info) => deviceName = info.name);
      } else if (Platform.isAndroid) {
        DeviceInfoPlugin().androidInfo.then((info) => deviceName = info.model);
      }
      refreshWorkerStatus().then((_) {
        if (worker.value.running) _watchWorker();
      });
      ever<ClusterWorkerStatus>(worker, (_) => _updateKeepScreenOn());
      ever<bool>(serving, (_) => _updateKeepScreenOn());
      refreshWifiAddress();
    }
  }

  @override
  void onClose() {
    _statusTimer?.cancel();
    _stopBeacon();
    super.onClose();
  }

  /// Android throttles a phone whose screen is off (slow cores, low CPU
  /// share), so keep it on while this device works for others: as a worker,
  /// or serving the model over the network.
  bool? _screenKeptOn;
  void _updateKeepScreenOn() {
    final s = worker.value;
    final on = s.running || s.starting || serving.value;
    if (on == _screenKeptOn) return;
    _screenKeptOn = on;
    LlamaCluster.setKeepScreenOn(on).catchError((_) {});
  }

  Future<void> setRole(ClusterRole next) async {
    if (next == role) return;
    if (next != ClusterRole.worker && role == ClusterRole.worker) {
      if (workerStopsWithApp) {
        showToast('Still sharing', 'Close OpenLM to stop sharing this device.');
        return;
      }
      await stopWorker();
    }
    await _setEnabled(next == ClusterRole.master);
    if (next == ClusterRole.worker) await startWorker();
    await refreshWifiAddress();
  }

  // ── Worker ─────────────────────────────────────────────────────────────

  /// What the master scans to add this phone, once the worker is running.
  String? get workerQrData {
    final host = wifiAddress.value;
    final status = worker.value;
    if (host == null || !status.running) return null;
    return Uri(scheme: _qrScheme, host: 'worker', queryParameters: {
      'host': host,
      'port': '${LlamaCluster.defaultPort}',
      if (deviceName.isNotEmpty) 'name': deviceName,
      if (status.ramMb > 0) 'ram': '${status.ramMb}',
      if (status.bandwidth > 0) 'bw': status.bandwidth.toStringAsFixed(1),
      if (status.id != null) 'id': status.id!,
    }).toString();
  }

  Future<void> startWorker() async {
    if (Platform.isAndroid) {
      // Unrestricted battery keeps Android (Samsung especially) from moving
      // the worker to slow cores once it gets busy.
      final status = await LlamaCluster.workerStatus();
      if (!status.batteryUnrestricted) await LlamaCluster.requestBatteryUnrestricted();
    }
    _workerRequested.value = true;
    worker.value = const ClusterWorkerStatus(state: 'starting');
    await LlamaCluster.startWorker();
    _watchWorker();
  }

  Future<void> stopWorker() async {
    _workerRequested.value = false;
    _statusTimer?.cancel();
    _stopBeacon();
    await LlamaCluster.stopWorker();
    await Future.delayed(const Duration(milliseconds: 600));
    await refreshWorkerStatus();
  }

  /// Follows the worker service (its own process) while it runs: whether a
  /// master is connected and how much it holds.
  void _watchWorker() {
    _statusTimer?.cancel();
    var checks = 0;
    _statusTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      await refreshWorkerStatus();
      final s = worker.value;
      if (s.running) {
        _startBeacon();
      } else {
        _stopBeacon();
      }
      // Give the service a few seconds to come up before calling it stopped.
      if (!s.running && !s.starting && ++checks > 3) {
        _workerRequested.value = false;
        _stopBeacon();
        _statusTimer?.cancel();
      }
    });
  }

  void _startBeacon() {
    if (_beaconTimer != null) return;
    _beaconTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      final host = wifiAddress.value;
      final s = worker.value;
      if (host == null || !s.running) return;
      try {
        _beaconSocket ??=
            await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
        _beaconSocket!.broadcastEnabled = true;
        final payload = jsonEncode({
          'name': deviceName.isNotEmpty ? deviceName : 'Worker ($host)',
          'host': host,
          'port': LlamaCluster.defaultPort,
          'ramMb': s.ramMb,
          'bandwidth': s.bandwidth,
          'model': deviceName,
          'id': s.id ?? host,
          'v': 'rpc7',
        });
        _beaconSocket!.send(
            utf8.encode(payload), InternetAddress('255.255.255.255'), 50053);
      } catch (_) {}
    });
  }

  void _stopBeacon() {
    _beaconTimer?.cancel();
    _beaconTimer = null;
    _beaconSocket?.close();
    _beaconSocket = null;
  }

  Future<void> refreshWorkerStatus() async {
    try {
      worker.value = await LlamaCluster.workerStatus();
    } catch (e) {
      Get.find<AppLogService>().error('Cluster worker status failed', details: e);
    }
  }

  /// Finds this device's IPv4 address on LAN / Wi-Fi / Ethernet across all platforms.
  Future<void> refreshWifiAddress() async {
    try {
      final interfaces =
          await NetworkInterface.list(type: InternetAddressType.IPv4);
      InternetAddress? best;
      for (final i in interfaces) {
        final name = i.name.toLowerCase();
        if (name.contains('wsl') ||
            name.contains('vethernet') ||
            name.contains('docker') ||
            name.contains('vmnet') ||
            name.contains('virtual') ||
            name.contains('pseudo')) {
          continue;
        }
        for (final a in i.addresses) {
          if (a.isLoopback || a.isLinkLocal) continue;
          final ip = a.address;
          if (ip.startsWith('127.') || ip.startsWith('169.254.')) continue;
          if (name.startsWith('wlan') ||
              name == 'en0' ||
              name.contains('wi-fi') ||
              name.contains('wireless')) {
            best = a;
            break;
          }
          best ??= a;
        }
        if (best != null &&
            (name.startsWith('wlan') ||
                name == 'en0' ||
                name.contains('wi-fi') ||
                name.contains('wireless'))) {
          break;
        }
      }
      wifiAddress.value = best?.address;
    } catch (_) {
      wifiAddress.value = null;
    }
  }

  // ── Master: workers ────────────────────────────────────────────────────

  /// Adds the worker from a scanned QR code. Returns an error message, or
  /// null on success.
  Future<String?> addFromQr(String data) async {
    final uri = Uri.tryParse(data.trim());
    final params = uri?.queryParameters ?? const {};
    final host = params['host'];
    final port = int.tryParse(params['port'] ?? '');
    if (uri == null ||
        (uri.scheme != _qrScheme && uri.scheme != 'privatelm') ||
        host == null ||
        port == null) {
      return 'That isn\'t an OpenLM worker code';
    }
    return _add(ClusterWorker(
      name: params['name'] ?? host,
      host: host,
      port: port,
      ramMb: int.tryParse(params['ram'] ?? ''),
      bandwidth: double.tryParse(params['bw'] ?? ''),
      model: params['name'],
      id: params['id'],
    ));
  }

  /// Adds a worker by IP ("192.168.1.20" or "192.168.1.20:50052"). Returns
  /// an error message, or null on success.
  Future<String?> addManual(String input) async {
    final text = input.trim();
    if (text.isEmpty) return 'Enter the worker phone\'s IP address';
    final host = text.contains(':') ? text.substring(0, text.lastIndexOf(':')) : text;
    final port = text.contains(':')
        ? int.tryParse(text.substring(text.lastIndexOf(':') + 1))
        : LlamaCluster.defaultPort;
    if (host.isEmpty || port == null) return 'Use the form 192.168.1.20:50052';
    return _add(ClusterWorker(name: host, host: host, port: port));
  }

  Future<String?> addFound(ClusterWorker w) => _add(w);

  /// Finds saved workers that no longer answer at their address (they moved
  /// to another Wi-Fi, a hotspot, or got a new IP) by their device ID on
  /// the network, and updates them. Returns the ones still unreachable.
  Future<List<ClusterWorker>> relocateWorkers() async {
    final missing = <ClusterWorker>[];
    for (final w in workers) {
      if (!await LlamaCluster.probe(w.endpoint)) missing.add(w);
    }
    if (missing.isEmpty) return const [];
    final seen = await LlamaCluster.discover();
    var moved = 0;
    final unreachable = <ClusterWorker>[];
    for (final w in missing) {
      final match = seen.firstWhereOrNull((s) => w.id != null && s.id == w.id);
      final i = workers.indexWhere((x) => x.endpoint == w.endpoint);
      if (match == null || match.endpoint == w.endpoint || i < 0) {
        unreachable.add(w);
        continue;
      }
      workers[i] = w.movedTo(match.host, match.port);
      moved++;
    }
    if (moved > 0) {
      await _save();
      Get.find<AppLogService>().info('Phone cluster: found $moved worker(s) at a new address');
    }
    return unreachable;
  }

  Future<String?> _add(ClusterWorker worker) async {
    if (!await LlamaCluster.probe(worker.endpoint)) {
      return 'Can\'t reach ${worker.endpoint}. Check that the phone is a worker '
          'and on the same Wi-Fi.';
    }
    workers.removeWhere((w) => w.endpoint == worker.endpoint);
    workers.add(worker);
    found.removeWhere((w) => w.endpoint == worker.endpoint);
    await _save();
    return null;
  }

  Future<void> remove(ClusterWorker w) async {
    workers.removeWhere((s) => s.endpoint == w.endpoint);
    await _save();
  }

  bool isAdded(ClusterWorker w) => workers.any((s) => s.endpoint == w.endpoint);

  /// Looks for workers on the LAN (mDNS + UDP beacon hybrid).
  Future<void> scanNearby() async {
    if (isScanning.value) return;
    isScanning.value = true;
    try {
      final seen = <String, ClusterWorker>{};

      // 1. Native discovery (mDNS / NetService / NSD)
      try {
        final list = await LlamaCluster.discover(
            timeout: const Duration(milliseconds: 2500));
        for (final w in list) {
          if (!isAdded(w)) seen[w.endpoint] = w;
        }
      } catch (_) {}

      // 2. Cross-platform UDP beacon listener on port 50053
      try {
        final socket = await RawDatagramSocket.bind(
            InternetAddress.anyIPv4, 50053,
            reuseAddress: true, reusePort: false);
        final sub = socket.listen((event) {
          if (event == RawSocketEvent.read) {
            final dg = socket.receive();
            if (dg != null) {
              try {
                final str = utf8.decode(dg.data);
                final map = jsonDecode(str) as Map<String, dynamic>;
                if (map['v'] == 'rpc7' &&
                    map['host'] != null &&
                    map['port'] != null) {
                  final w = ClusterWorker.fromMap(map);
                  if (!isAdded(w)) {
                    seen[w.endpoint] = w;
                    found.assignAll(seen.values);
                  }
                }
              } catch (_) {}
            }
          }
        });
        await Future.delayed(const Duration(milliseconds: 2000));
        await sub.cancel();
        socket.close();
      } catch (_) {}

      found.assignAll(seen.values);
    } finally {
      isScanning.value = false;
    }
  }

  Future<void> _setEnabled(bool value) async {
    enabled.value = value;
    await _hive.setSetting(AppConstants.keyClusterEnabled, value);
  }

  Future<void> _save() => _hive.setSetting(
      AppConstants.keyClusterWorkers, [for (final w in workers) w.toMap()]);

  // ── Master: loading ────────────────────────────────────────────────────

  /// Plans a GGUF load right before it starts and hands the plan to the
  /// native engine (or turns cluster mode off for it). Returns the number of
  /// workers used; throws with a readable message if the model won't fit
  /// or a worker can't be reached.
  ///
  /// If the model fits on this phone, asks whether to keep it here (faster:
  /// no Wi-Fi in the loop) or share it with the workers anyway. Otherwise
  /// the fastest phones are filled first, up to their free memory: phones
  /// take turns on every token, so the most work belongs on the fastest one,
  /// and phones that aren't needed are left out (one less Wi-Fi hop each).
  Future<int> applyForNextLoad(String modelPath, {String? modelName}) async {
    split.clear();
    if (!isSupported) return 0;
    if (!enabled.value || workers.isEmpty) {
      await LlamaCluster.clear();
      return 0;
    }

    // Workers that are off or away are left out, not fatal: the model may
    // fit without them.
    final offline = await relocateWorkers();
    final available = workers.where((w) => !offline.any((o) => o.endpoint == w.endpoint)).toList();
    if (offline.isNotEmpty) {
      final names = offline.map((w) => w.model ?? w.name).join(', ');
      showToast('Phone cluster', 'Not reachable, loading without: $names');
    }

    final modelMb = await File(modelPath).length() / (1024 * 1024);
    final needMb = modelMb * 1.1; // + KV cache and working buffers
    _localSpeed ??= await LlamaCluster.benchmark();
    // Free RAM: workers report theirs when they start. The master also runs
    // the app and the KV cache, so it offers less of its own.
    final master = _Phone(null, await LlamaCluster.localFreeRamMb() * 0.7, _localSpeed!);
    final phones = [
      master,
      for (final w in available)
        _Phone(w, (w.ramMb ?? 2000) * 0.75, w.bandwidth ?? _unknownSpeed),
    ];
    final totalMb = phones.fold<double>(0, (a, p) => a + p.capacityMb);
    if (needMb > totalMb) {
      throw Exception('This model (${_gb(modelMb)}) is too big for these '
          '${phones.length} phones: together they have about ${_gb(totalMb)} '
          'free.${offline.isEmpty ? '' : ' Not reachable: ${offline.map((w) => w.model ?? w.name).join(', ')}.'} '
          'Add another worker, close apps on the phones, or pick a '
          'smaller model.');
    }

    if (available.isEmpty) {
      await LlamaCluster.clear();
      return 0;
    }
    if (needMb <= master.capacityMb) {
      final share = await _askFitsHere(
          modelName ?? modelPath.split('/').last, modelMb, available.length);
      if (!share) {
        await LlamaCluster.clear();
        return 0;
      }
      _splitBySpeed(phones, needMb);
    } else {
      _fillFastestFirst(phones, needMb);
    }

    final used = phones.where((p) => p.worker != null && p.assignedMb > 0).toList();
    if (used.isEmpty) {
      await LlamaCluster.clear();
      return 0;
    }
    try {
      await LlamaCluster.setCluster(
        [for (final p in used) p.worker!],
        weights: [for (final p in used) p.assignedMb],
        localWeight: master.assignedMb,
      );
    } on PlatformException catch (e) {
      throw Exception(
          '${e.message ?? 'A worker is unreachable'}. Turn on Worker on that '
          'phone, or set this phone\'s Phone cluster to Off.');
    }
    return used.length;
  }

  /// Fastest first, each up to its free memory, until the model is placed.
  static void _fillFastestFirst(List<_Phone> phones, double needMb) {
    final bySpeed = [...phones]..sort((a, b) => b.speed.compareTo(a.speed));
    var left = needMb;
    for (final p in bySpeed) {
      p.assignedMb = math.min(p.capacityMb, left);
      left -= p.assignedMb;
    }
  }

  /// Shares in proportion to speed, capped at each phone's free memory
  /// (what a capped phone can't take goes to the others).
  static void _splitBySpeed(List<_Phone> phones, double needMb) {
    var open = [...phones];
    var left = needMb;
    while (open.isNotEmpty && left > 0.5) {
      final speedSum = open.fold<double>(0, (a, p) => a + p.speed);
      final capped = [
        for (final p in open)
          if (p.assignedMb + left * p.speed / speedSum >= p.capacityMb) p,
      ];
      if (capped.isEmpty) {
        for (final p in open) {
          p.assignedMb += left * p.speed / speedSum;
        }
        left = 0;
      } else {
        for (final p in capped) {
          left -= p.capacityMb - p.assignedMb;
          p.assignedMb = p.capacityMb;
        }
        open = open.where((p) => !capped.contains(p)).toList();
      }
    }
  }

  /// LiteRT models can't be split across phones (the cluster is llama.cpp).
  /// When this phone is a master with workers, asks before loading one here
  /// alone. True to go ahead.
  Future<bool> confirmLiteRtRunsHere(String model) async {
    final context = Get.context;
    if (role != ClusterRole.master || workers.isEmpty || context == null) return true;
    final go = await showAppSheet<bool>(
      context,
      title: 'This model can\'t use your workers',
      subtitle: '${model.replaceAll(RegExp(r'\.(litertlm|task)$'), '')} is a LiteRT '
          'model, and only GGUF models can be shared across phones. It will run '
          'on this phone only. To use your workers, load a GGUF model instead.',
      builder: (ctx) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          PrimaryButton(
            label: 'Load on this phone only',
            icon: Icons.smartphone_rounded,
            onPressed: () => Navigator.pop(ctx, true),
          ),
          const SizedBox(height: 10),
          TonalButton.neutral(ctx, label: 'Cancel', onTap: () => Navigator.pop(ctx, false)),
        ],
      ),
    );
    return go == true;
  }

  /// "Fits on this phone" sheet. True to share with workers anyway.
  Future<bool> _askFitsHere(String model, double modelMb, int workerCount) async {
    final context = Get.context;
    if (context == null) return false;
    final share = await showAppSheet<bool>(
      context,
      title: 'This model fits on this phone',
      subtitle: '${model.replaceAll(RegExp(r'\.gguf$'), '')} needs about '
          '${_gb(modelMb)}, and this phone has room for it. Running it here '
          'is faster, because sharing sends every word over Wi-Fi. Share it '
          'to free up memory on this phone.',
      builder: (ctx) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          PrimaryButton(
            label: 'Load on this phone only',
            icon: Icons.smartphone_rounded,
            onPressed: () => Navigator.pop(ctx, false),
          ),
          const SizedBox(height: 10),
          TonalButton.neutral(
            ctx,
            label: 'Share with $workerCount worker${workerCount == 1 ? '' : 's'}',
            onTap: () => Navigator.pop(ctx, true),
          ),
        ],
      ),
    );
    return share == true;
  }

  /// Reads where the model actually went after a cluster load. Workers'
  /// shares come from llama.cpp's per-device weight buffers; this phone's
  /// share is the rest of the file, because its own buffers overlap (the
  /// mapped file and the CPU's repacked copy of the same weights).
  Future<void> refreshSplit(String modelPath) async {
    final buffers = await LlamaCluster.lastLoadBuffers();
    final remote = <String, double>{};
    for (final b in buffers) {
      final open = b.name.indexOf('[');
      if (open < 0) continue;
      final endpoint = b.name.substring(open + 1, b.name.lastIndexOf(']'));
      final name = workers.firstWhereOrNull((w) => w.endpoint == endpoint)?.model ?? endpoint;
      remote[name] = (remote[name] ?? 0) + b.mb;
    }
    final fileMb = await File(modelPath).length() / (1024 * 1024);
    final remoteMb = remote.values.fold<double>(0, (a, b) => a + b);
    split.assignAll([
      ClusterShare(deviceName.isEmpty ? 'This phone' : deviceName,
          math.max(0, fileMb - remoteMb), isThisPhone: true),
      for (final e in remote.entries) ClusterShare(e.key, e.value),
    ]);
  }

  void clearSplit() => split.clear();

  static String _gb(double mb) => '${(mb / 1024).toStringAsFixed(1)} GB';
}
