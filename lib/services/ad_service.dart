import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import 'app_log_service.dart';

class AdService extends GetxService {
  static const _appBannerUnitId = 'ca-app-pub-3049300561479903/9833101795';
  static const _testBannerUnitId = 'ca-app-pub-3940256099942544/9214589741';

  static String get bannerUnitId =>
      kDebugMode ? _testBannerUnitId : _appBannerUnitId;

  final canShowBanner = false.obs;
  final bannerDismissed = false.obs;
  final privacyOptionsRequired = false.obs;

  bool _adsSdkInitialized = false;

  Future<AdService> init() async {
    if (!Platform.isAndroid) return this;

    ConsentInformation.instance.requestConsentInfoUpdate(
      ConsentRequestParameters(),
      () => unawaited(_finishConsentFlow()),
      (error) {
        _log('Ad consent information could not be refreshed', error);
        unawaited(_refreshConsentState());
      },
    );
    return this;
  }

  Future<void> _finishConsentFlow() async {
    try {
      await ConsentForm.loadAndShowConsentFormIfRequired((error) {
        if (error != null) _log('Ad consent form failed', error);
      });
    } catch (error) {
      _log('Ad consent form could not be shown', error);
    }
    await _refreshConsentState();
  }

  Future<void> _refreshConsentState() async {
    try {
      final required = await ConsentInformation.instance
              .getPrivacyOptionsRequirementStatus() ==
          PrivacyOptionsRequirementStatus.required;
      privacyOptionsRequired.value = required;
      await _initializeAdsIfAllowed();
    } catch (error) {
      _log('Ad consent state could not be read', error);
    }
  }

  Future<void> _initializeAdsIfAllowed() async {
    try {
      final allowed = await ConsentInformation.instance.canRequestAds();
      if (!allowed) {
        canShowBanner.value = false;
        return;
      }
      if (!_adsSdkInitialized) {
        await MobileAds.instance.initialize();
        _adsSdkInitialized = true;
      }
      canShowBanner.value = true;
    } catch (error) {
      _log('Mobile ads could not be initialized', error);
    }
  }

  Future<void> showPrivacyOptions() async {
    canShowBanner.value = false;
    try {
      await ConsentForm.showPrivacyOptionsForm((error) {
        if (error != null) _log('Ad privacy options failed', error);
      });
    } catch (error) {
      _log('Ad privacy options could not be shown', error);
    }
    await _refreshConsentState();
  }

  void dismissBanner() => bannerDismissed.value = true;

  void _log(String message, Object error) {
    if (Get.isRegistered<AppLogService>()) {
      Get.find<AppLogService>().warning(message, details: error);
    }
  }
}
