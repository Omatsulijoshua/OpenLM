import 'package:flutter_test/flutter_test.dart';
import 'package:openlm/services/cluster_service.dart';
import 'package:openlm/services/device_info_native.dart';
import 'package:openlm/models/ai_model.dart';

void main() {
  test('ClusterService platform support flags', () {
    expect(ClusterService.isSupported, isTrue);
    expect(ClusterService.canBeMaster, isTrue);
    expect(ClusterService.workerStopsWithApp, isTrue);
  });

  test('SocFamily display names and recommended quant', () {
    expect(SocFamily.apple.displayName, 'Apple Silicon');
    expect(SocFamily.intel.displayName, contains('Intel'));
    expect(SocFamily.amd.displayName, contains('AMD'));
    expect(SocFamily.intel.recommendedQuant, contains('Q4_K_M'));
  });

  test('AiModel runtime detection for GGUF and LiteRT', () {
    expect(AiModel.runtimeFromFilename('test.gguf'), AiModel.runtimeLlama);
    expect(AiModel.runtimeFromFilename('test.litertlm'), AiModel.runtimeLiteRt);
    expect(AiModel.runtimeFromFilename('model.safetensors'), AiModel.runtimeSd);
  });
}
