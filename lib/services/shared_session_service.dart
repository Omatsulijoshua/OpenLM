import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';

import '../controllers/settings_controller.dart';
import '../models/chat_message.dart';
import '../models/chat_session.dart';
import 'app_log_service.dart';
import 'hive_service.dart';
import 'inference_service.dart';

/// Shares chat history and hands the active chat controls between trusted
/// devices on the same LAN. Model inference remains on the cluster host.
class SharedSessionService extends GetxService {
  SharedSessionService();

  static const int port = 50054;
  static const _scheme = 'openlm';
  static const _sessionsPath = '/openlm/sessions';
  static const _deviceIdKey = 'shared_session_device_id';
  static const _hostKey = 'shared_session_host';
  static const _portKey = 'shared_session_port';
  static const _tokenKey = 'shared_session_token';
  static const _sharedSessionIdsKey = 'shared_session_ids';

  final HiveService _hive = Get.find<HiveService>();
  final deviceId = ''.obs;
  final isHosting = false.obs;
  final isConnected = false.obs;
  final isPrimary = true.obs;
  final primaryName = 'This device'.obs;
  final stateVersion = 0.obs;
  final lastError = RxnString();

  final http.Client _client = http.Client();
  HttpServer? _server;
  Timer? _syncTimer;
  String? _token;
  String? _host;
  int _hostPort = port;
  String _localName = 'OpenLM device';
  String? _hostAddress;
  String _primaryDeviceId = '';
  String _primaryDeviceName = 'This device';
  int _revision = 0;
  bool _generating = false;
  bool _syncing = false;
  final Set<String> _sharedSessionIds = {};

  bool get hasControl => !isConnected.value || isPrimary.value;
  String? get inviteLink {
    final address = _hostAddress;
    final token = _token;
    if (!isHosting.value || address == null || token == null) return null;
    return Uri(scheme: _scheme, host: 'session', queryParameters: {
      'host': address,
      'port': '$port',
      'key': token,
      'name': _localName,
    }).toString();
  }

  Future<String?> _discoverLocalAddress() async {
    try {
      final addresses = await NetworkInterface.list(
        type: InternetAddressType.IPv4,
        includeLoopback: false,
      );
      for (final interface in addresses) {
        final name = interface.name.toLowerCase();
        if (name.contains('docker') ||
            name.contains('virtual') ||
            name.contains('wsl') ||
            name.contains('vmnet')) {
          continue;
        }
        for (final address in interface.addresses) {
          if (_isPrivateLanAddress(address.address)) {
            return address.address;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  bool _isPrivateLanAddress(String value) {
    final address = InternetAddress.tryParse(value);
    if (address == null) return false;
    if (address.type == InternetAddressType.IPv4) {
      final b = address.rawAddress;
      return b[0] == 10 ||
          (b[0] == 172 && b[1] >= 16 && b[1] <= 31) ||
          (b[0] == 192 && b[1] == 168) ||
          (b[0] == 169 && b[1] == 254) ||
          (b[0] == 100 && b[1] >= 64 && b[1] <= 127) ||
          b[0] == 127;
    }
    final b = address.rawAddress;
    return address.isLoopback ||
        (b[0] & 0xfe) == 0xfc ||
        (b[0] == 0xfe && (b[1] & 0xc0) == 0x80);
  }

  Future<SharedSessionService> init() async {
    var id = _hive.getSetting<String>(_deviceIdKey);
    if (id == null || id.isEmpty) {
      id = const Uuid().v4();
      await _hive.setSetting(_deviceIdKey, id);
    }
    deviceId.value = id;
    final savedSharedIds =
        _hive.getSetting<List>(_sharedSessionIdsKey) ?? const [];
    _sharedSessionIds.addAll(savedSharedIds.whereType<String>());
    return this;
  }

  @override
  void onClose() {
    _syncTimer?.cancel();
    unawaited(_server?.close(force: true));
    _client.close();
    super.onClose();
  }

  Future<void> startHost({String? name, String? address}) async {
    if (kIsWeb || _server != null) return;
    _syncTimer?.cancel();
    _syncTimer = null;
    _host = null;
    await _hive.setSetting(_hostKey, null);
    await _hive.setSetting(_portKey, null);
    _localName =
        name?.trim().isNotEmpty == true ? name!.trim() : Platform.localHostname;
    _hostAddress = address != null && _isPrivateLanAddress(address)
        ? address
        : await _discoverLocalAddress();
    if (_hostAddress == null) {
      throw StateError(
          'Connect to a private Wi-Fi or LAN before sharing chat.');
    }
    final random = Random.secure();
    _token = List.generate(
            32, (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0'))
        .join();
    try {
      _server = await HttpServer.bind(
        InternetAddress(_hostAddress!),
        port,
        shared: true,
      );
      _primaryDeviceId = deviceId.value;
      _primaryDeviceName = _localName;
      isHosting.value = true;
      isConnected.value = true;
      isPrimary.value = true;
      primaryName.value = _localName;
      lastError.value = null;
      _server!.listen(_handleRequest, onError: (Object error) {
        Get.find<AppLogService>()
            .error('Shared chat server failed', details: error);
      });
      _notifyChanged();
    } catch (e) {
      _server = null;
      lastError.value = '$e';
      Get.find<AppLogService>().error('Shared chat server failed', details: e);
      rethrow;
    }
  }

  Future<void> stopHost() async {
    _syncTimer?.cancel();
    _syncTimer = null;
    await _server?.close(force: true);
    _server = null;
    isHosting.value = false;
    _hostAddress = null;
    if (_host == null) {
      isConnected.value = false;
      isPrimary.value = true;
    }
  }

  Future<String?> joinFromLink(String link) async {
    final uri = Uri.tryParse(link.trim());
    final host = uri?.queryParameters['host'];
    final key = uri?.queryParameters['key'];
    final portNumber = int.tryParse(uri?.queryParameters['port'] ?? '');
    if (uri == null ||
        uri.scheme != _scheme ||
        uri.host != 'session' ||
        host == null ||
        host.isEmpty ||
        !_isPrivateLanAddress(host) ||
        key == null ||
        key.length < 32 ||
        portNumber == null ||
        portNumber < 1 ||
        portNumber > 65535) {
      return 'That is not a valid OpenLM shared chat link.';
    }
    _host = host;
    _hostPort = portNumber;
    _token = key;
    _localName = Platform.localHostname;
    final result = await _request('GET', _sessionsPath);
    if (result.statusCode != 200) {
      _host = null;
      return result.body['error'] as String? ??
          'Could not reach that OpenLM session on this network.';
    }
    await _hive.setSetting(_hostKey, _host);
    await _hive.setSetting(_portKey, _hostPort);
    await _hive.setSetting(_tokenKey, _token);
    isConnected.value = true;
    await _applySnapshot(result.body);
    _startSync();
    return null;
  }

  Future<void> resume() async {
    final host = _hive.getSetting<String>(_hostKey);
    final key = _hive.getSetting<String>(_tokenKey);
    final p = _hive.getSetting<int>(_portKey, defaultValue: port) ?? port;
    if (host == null || key == null || deviceId.value.isEmpty) return;
    _host = host;
    _token = key;
    _hostPort = p;
    final result = await _request('GET', _sessionsPath);
    if (result.statusCode == 200) {
      isConnected.value = true;
      await _applySnapshot(result.body);
      _startSync();
    }
  }

  Future<void> disconnect() async {
    _syncTimer?.cancel();
    _syncTimer = null;
    _host = null;
    if (!isHosting.value) {
      _token = null;
      isConnected.value = false;
      isPrimary.value = true;
      primaryName.value = 'This device';
      await _hive.setSetting(_hostKey, null);
      await _hive.setSetting(_portKey, null);
      // The host token is shared by peers. Retain it while hosting.
      await _hive.setSetting(_tokenKey, null);
    }
    _notifyChanged();
  }

  Future<String?> takePrimaryControl() async {
    if ((!isHosting.value && _host == null) || isPrimary.value) return null;
    final result = await _request('POST', '/openlm/takeover', body: {
      'deviceId': deviceId.value,
      'name': _localName,
    });
    if (result.statusCode != 200) {
      return result.body['error'] as String? ?? 'Could not take chat control.';
    }
    await _applySnapshot(result.body);
    return null;
  }

  Future<ChatMessage> generateOnHost({
    required ChatSession session,
    required ChatMessage userMessage,
  }) async {
    if (_host == null || !isPrimary.value) {
      throw StateError('Take primary chat control on this device first.');
    }
    final result = await _request('POST', '/openlm/chat',
        body: {
          'session': session.toMap(),
          'message': userMessage.toMap(),
        },
        timeout: const Duration(minutes: 10));
    if (result.statusCode != 200) {
      throw StateError(result.body['error'] as String? ??
          'The model host could not complete this message.');
    }
    final rawMessage = result.body['message'];
    if (rawMessage is! Map) throw StateError('Invalid shared chat response.');
    return ChatMessage.fromMap(rawMessage);
  }

  Future<void> publishSession(ChatSession session) async {
    if (_host == null || !isPrimary.value || isHosting.value) return;
    await _request('POST', '/openlm/session', body: session.toMap());
  }

  Future<String?> deleteSharedSession(String sessionId) async {
    if (!isConnected.value || !hasControl) return null;
    final result = await _request('POST', '/openlm/delete',
        body: {'sessionId': sessionId});
    if (result.statusCode != 200) {
      return result.body['error'] as String? ?? 'Could not delete shared chat.';
    }
    return null;
  }

  Future<void> markLocalDataChanged() async {
    if (isHosting.value) {
      _revision++;
      _notifyChanged();
    }
  }

  void _notifyChanged() => stateVersion.value++;

  void _startSync() {
    _syncTimer?.cancel();
    _syncTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      if (_host == null || _syncing) return;
      _syncing = true;
      try {
        final result = await _request(
          'GET',
          _sessionsPath,
          query: {'revision': '$_revision'},
        );
        if (result.statusCode == 200) await _applySnapshot(result.body);
      } finally {
        _syncing = false;
      }
    });
  }

  Future<void> _applySnapshot(Map<String, dynamic> body) async {
    final previousRevision = _revision;
    final previousPrimary = _primaryDeviceId;
    final primary = body['primary'];
    if (primary is Map) {
      _primaryDeviceId = primary['id']?.toString() ?? '';
      _primaryDeviceName = primary['name']?.toString() ?? 'OpenLM device';
      isPrimary.value = _primaryDeviceId == deviceId.value;
      primaryName.value = isPrimary.value ? 'This device' : _primaryDeviceName;
    }
    final sessions = body['sessions'];
    final incomingSessionIds = <String>{};
    if (sessions is List) {
      for (final entry in sessions.whereType<Map>()) {
        final map = Map<String, dynamic>.from(entry);
        final id = map['id']?.toString();
        if (id != null && id.isNotEmpty) {
          incomingSessionIds.add(id);
          await _hive.saveSession(id, map);
        }
      }
    }
    for (final id in _sharedSessionIds.difference(incomingSessionIds)) {
      await _hive.deleteSession(id);
    }
    _sharedSessionIds
      ..clear()
      ..addAll(incomingSessionIds);
    await _hive.setSetting(_sharedSessionIdsKey, _sharedSessionIds.toList());
    final messages = body['messages'];
    if (messages is List) {
      for (final entry in messages.whereType<Map>()) {
        final map = Map<String, dynamic>.from(entry);
        final id = map['id']?.toString();
        if (id != null && id.isNotEmpty) await _hive.saveMessage(id, map);
      }
    }
    _revision =
        body['revision'] is num ? (body['revision'] as num).toInt() : _revision;
    if (_revision != previousRevision || _primaryDeviceId != previousPrimary) {
      _notifyChanged();
    }
  }

  Future<_SharedResult> _request(String method, String path,
      {Map<String, dynamic>? body,
      Map<String, String>? query,
      Duration timeout = const Duration(seconds: 8)}) async {
    final host = _host ?? _hostAddress;
    if (host == null) {
      return const _SharedResult(503, {'error': 'No LAN address.'});
    }
    final uri = Uri(
      scheme: 'http',
      host: host,
      port: _hostPort,
      path: path,
      queryParameters: query,
    );
    try {
      final request = http.Request(method, uri)
        ..headers['Authorization'] = 'Bearer ${_token ?? ''}'
        ..headers['X-OpenLM-Device'] = deviceId.value
        ..headers['X-OpenLM-Name'] = _localName;
      if (body != null) {
        request.headers['Content-Type'] = 'application/json';
        request.body = jsonEncode(body);
      }
      final response = await _client.send(request).timeout(timeout);
      final text = await response.stream.bytesToString().timeout(timeout);
      final decoded = text.isEmpty ? <String, dynamic>{} : jsonDecode(text);
      return _SharedResult(
        response.statusCode,
        decoded is Map
            ? Map<String, dynamic>.from(decoded)
            : <String, dynamic>{},
      );
    } catch (e) {
      lastError.value = '$e';
      return _SharedResult(503, {'error': 'Session connection failed: $e'});
    }
  }

  Future<void> _handleRequest(HttpRequest request) async {
    request.response.headers.contentType = ContentType.json;
    if (request.headers.value(HttpHeaders.authorizationHeader) !=
        'Bearer $_token') {
      await _respond(request, 401, {'error': 'Invalid session link.'});
      return;
    }
    final device = request.headers.value('x-openlm-device') ?? '';
    final name = request.headers.value('x-openlm-name') ?? 'OpenLM device';
    try {
      if (request.method == 'GET' && request.uri.path == _sessionsPath) {
        if (request.uri.queryParameters['revision'] == '$_revision') {
          request.response.statusCode = HttpStatus.noContent;
          await request.response.close();
          return;
        }
        await _respond(request, 200, _snapshot());
        return;
      }
      if (request.method == 'POST' && request.uri.path == '/openlm/takeover') {
        final body = await _body(request);
        _primaryDeviceId = device;
        _primaryDeviceName = body['name']?.toString() ?? name;
        _revision++;
        await _respond(request, 200, _snapshot());
        return;
      }
      if (device != _primaryDeviceId) {
        await _respond(request, 409,
            {'error': 'Another device has primary chat control.'});
        return;
      }
      if (request.method == 'POST' && request.uri.path == '/openlm/session') {
        final body = await _body(request);
        final id = body['id']?.toString();
        if (id == null || id.isEmpty) {
          await _respond(request, 400, {'error': 'Invalid chat session.'});
          return;
        }
        await _hive.saveSession(id, body);
        _revision++;
        _notifyChanged();
        await _respond(request, 200, {'ok': true});
        return;
      }
      if (request.method == 'POST' && request.uri.path == '/openlm/delete') {
        if (_generating) {
          await _respond(request, 409, {
            'error': 'Wait for the current response before deleting a chat.'
          });
          return;
        }
        final body = await _body(request);
        final id = body['sessionId']?.toString() ?? '';
        if (id.isEmpty) {
          await _respond(request, 400, {'error': 'Invalid chat session.'});
          return;
        }
        await _hive.deleteSession(id);
        _revision++;
        _notifyChanged();
        await _respond(request, 200, {'ok': true});
        return;
      }
      if (request.method == 'POST' && request.uri.path == '/openlm/chat') {
        if (_generating) {
          await _respond(request, 409,
              {'error': 'The model is already answering a prompt.'});
          return;
        }
        final inference = Get.find<InferenceService>();
        if (!inference.isModelLoaded.value) {
          await _respond(request, 409,
              {'error': 'Load a local model on the model host first.'});
          return;
        }
        final body = await _body(request);
        // Check again after reading the body: simultaneous requests can both
        // pass the first check while their request bodies are being received.
        if (_generating) {
          await _respond(request, 409,
              {'error': 'The model is already answering a prompt.'});
          return;
        }
        final sessionRaw = body['session'];
        final messageRaw = body['message'];
        if (sessionRaw is! Map || messageRaw is! Map) {
          await _respond(request, 400, {'error': 'Invalid chat request.'});
          return;
        }
        final session = ChatSession.fromMap(sessionRaw);
        final message = ChatMessage.fromMap(messageRaw);
        if (session.id.isEmpty ||
            message.chatId != session.id ||
            message.role != 'user' ||
            message.content.trim().isEmpty) {
          await _respond(request, 400, {'error': 'Invalid chat message.'});
          return;
        }
        _generating = true;
        await _hive.saveSession(session.id, session.toMap());
        await _hive.saveMessage(message.id, message.toMap());
        _revision++;
        _notifyChanged();
        try {
          final history = _hive
              .getMessagesForChat(session.id)
              .map(ChatMessage.fromMap)
              .where((m) => m.role == 'user' || m.role == 'assistant')
              .toList()
            ..sort((a, b) => a.timestamp.compareTo(b.timestamp));
          final settings = Get.find<SettingsController>();
          final modelName = inference.loadedModelName.value;
          final response = await inference.generate(
            prompt: message.content,
            systemPrompt: settings.effectiveSystemPromptForModel(modelName),
            conversationHistory: history
                .map((m) => {
                      'role': m.role,
                      'content': m.content,
                    })
                .toList(),
            source: 'shared-chat',
            stopSequences: settings.stopSequences.toList(),
          );
          if (response.startsWith('ERROR:')) {
            await _respond(request, 409, {'error': response});
            return;
          }
          final ai = ChatMessage(
            id: const Uuid().v4(),
            chatId: session.id,
            role: 'assistant',
            content: response,
            tokensPerSec: inference.tokensPerSecond.value,
          );
          await _hive.saveMessage(ai.id, ai.toMap());
          final updated = session.copyWith(lastMessage: response);
          await _hive.saveSession(updated.id, updated.toMap());
          _revision++;
          _notifyChanged();
          await _respond(request, 200, {'message': ai.toMap()});
        } finally {
          _generating = false;
        }
        return;
      }
      await _respond(request, 404, {'error': 'Unknown shared chat action.'});
    } catch (e, stack) {
      Get.find<AppLogService>()
          .error('Shared chat request failed', details: '$e\n$stack');
      await _respond(request, 500, {'error': 'Shared chat request failed: $e'});
    }
  }

  Map<String, dynamic> _snapshot() => {
        'revision': _revision,
        'primary': {'id': _primaryDeviceId, 'name': _primaryDeviceName},
        'sessions': _hive.getAllSessions(),
        'messages': [
          for (final session in _hive.getAllSessions())
            for (final message in _hive.getMessagesForChat('${session['id']}'))
              _shareSafeMessage(message),
        ],
      };

  Map<String, dynamic> _shareSafeMessage(Map<dynamic, dynamic> message) {
    final copy = Map<String, dynamic>.from(message);
    copy.remove('imageBase64');
    copy.remove('imagePath');
    copy.remove('filePath');
    return copy;
  }

  Future<Map<String, dynamic>> _body(HttpRequest request) async {
    final bytes = await request.fold<List<int>>(<int>[], (all, chunk) {
      if (all.length + chunk.length > 1024 * 1024) {
        throw const FormatException('Request is too large.');
      }
      all.addAll(chunk);
      return all;
    });
    final value = jsonDecode(utf8.decode(bytes));
    if (value is! Map) throw const FormatException('Invalid request body.');
    return Map<String, dynamic>.from(value);
  }

  Future<void> _respond(
      HttpRequest request, int status, Map<String, dynamic> body) async {
    request.response.statusCode = status;
    request.response.headers.contentType = ContentType.json;
    request.response.write(jsonEncode(body));
    await request.response.close();
  }
}

class _SharedResult {
  final int statusCode;
  final Map<String, dynamic> body;
  const _SharedResult(this.statusCode, this.body);
}
