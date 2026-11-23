import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import '../services/ad_service.dart';

/// A small, dismissible banner shown only in Android Settings → About.
class AndroidBannerAd extends StatelessWidget {
  const AndroidBannerAd({super.key});

  @override
  Widget build(BuildContext context) {
    if (defaultTargetPlatform != TargetPlatform.android) {
      return const SizedBox.shrink();
    }

    final ads = Get.find<AdService>();
    return Obx(() {
      if (!ads.canShowBanner.value || ads.bannerDismissed.value) {
        return const SizedBox.shrink();
      }
      return const _LoadedBannerSlot();
    });
  }
}

class _LoadedBannerSlot extends StatefulWidget {
  const _LoadedBannerSlot();

  @override
  State<_LoadedBannerSlot> createState() => _LoadedBannerSlotState();
}

class _LoadedBannerSlotState extends State<_LoadedBannerSlot> {
  BannerAd? _banner;
  bool _loaded = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadBanner());
  }

  Future<void> _loadBanner() async {
    if (!mounted) return;
    final banner = BannerAd(
      adUnitId: AdService.bannerUnitId,
      request: const AdRequest(),
      size: AdSize.banner,
      listener: BannerAdListener(
        onAdLoaded: (ad) {
          if (mounted) setState(() => _loaded = true);
        },
        onAdFailedToLoad: (ad, error) {
          ad.dispose();
          if (mounted) setState(() => _banner = null);
        },
      ),
    );
    _banner = banner;
    try {
      await banner.load();
    } catch (_) {
      banner.dispose();
      if (mounted) setState(() => _banner = null);
    }
  }

  @override
  void dispose() {
    _banner?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final banner = _banner;
    if (!_loaded || banner == null) return const SizedBox.shrink();

    final colors = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
      decoration: BoxDecoration(
        color: colors.surfaceContainerLow,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.5)),
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Row(children: [
          Expanded(
            child: Text(
              'ADVERTISEMENT',
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    letterSpacing: 0.8,
                    color: colors.onSurfaceVariant,
                  ),
            ),
          ),
          SizedBox(
            width: 32,
            height: 32,
            child: IconButton(
              padding: EdgeInsets.zero,
              tooltip: 'Hide ad',
              onPressed: Get.find<AdService>().dismissBanner,
              icon: const Icon(Icons.close_rounded, size: 18),
            ),
          ),
        ]),
        Center(
          child: SizedBox(
            width: banner.size.width.toDouble(),
            height: banner.size.height.toDouble(),
            child: AdWidget(ad: banner),
          ),
        ),
      ]),
    );
  }
}
