import 'dart:io';

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_fonts/google_fonts.dart';
import '../controllers/home_controller.dart';
import '../widgets/chat_drawer.dart';
import 'chat_view.dart';
import 'model_view.dart';
import 'server_view.dart';
import 'settings_view.dart';
import '../core/haptics.dart';

class HomeView extends GetView<HomeController> {
  const HomeView({super.key});

  static const _tabs = [
    _NavItem(
        icon: Icons.chat_bubble_outline_rounded,
        activeIcon: Icons.chat_bubble_rounded,
        label: 'Chat'),
    _NavItem(
        icon: Icons.layers_outlined,
        activeIcon: Icons.layers_rounded,
        label: 'Models'),
    _NavItem(
        icon: Icons.lan_outlined,
        activeIcon: Icons.lan_rounded,
        label: 'Server'),
    _NavItem(
        icon: Icons.settings_outlined,
        activeIcon: Icons.settings,
        label: 'Settings'),
  ];

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final isWide = MediaQuery.sizeOf(context).width >= 960;
    return Obx(() => Scaffold(
          key: controller.scaffoldKey,
          backgroundColor: isDark ? Colors.black : Colors.white,
          drawer: const ChatDrawer(),
          drawerScrimColor: Colors.black.withValues(alpha: isDark ? 0.6 : 0.32),
          // Edge-swipe only where the sidebar makes sense.
          drawerEnableOpenDragGesture: controller.currentTab.value == 0,
          drawerEdgeDragWidth: 28,
          body: Obx(() {
            final content = IndexedStack(
              index: controller.currentTab.value,
              children: const [
                ChatView(),
                ModelView(),
                ServerView(),
                SettingsView()
              ],
            );
            if (isWide) {
              return Row(children: [
                _buildSidebar(context, isDark),
                VerticalDivider(
                    width: 0.5,
                    thickness: 0.5,
                    color: isDark
                        ? Colors.white.withValues(alpha: 0.08)
                        : Colors.black.withValues(alpha: 0.08)),
                Expanded(child: content),
              ]);
            }
            return content;
          }),
          bottomNavigationBar: isWide ? null : _buildBottomNav(context, isDark),
        ));
  }

  Widget _buildBottomNav(BuildContext context, bool isDark) {
    final accent = isDark ? const Color(0xFF0B8278) : const Color(0xFF0F766E);
    final muted = isDark ? const Color(0xFF7C7C80) : const Color(0xFF8E8E93);
    final current = controller.currentTab.value;
    return Container(
      decoration: BoxDecoration(
        color: isDark ? Colors.black : Colors.white,
        border: Border(
            top: BorderSide(
          color: isDark
              ? Colors.white.withValues(alpha: 0.08)
              : Colors.black.withValues(alpha: 0.08),
          width: 0.5,
        )),
      ),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 54,
          child: Row(children: [
            for (var i = 0; i < _tabs.length; i++)
              Expanded(
                child: Semantics(
                  selected: current == i,
                  button: true,
                  label: _tabs[i].label,
                  child: GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onTap: () {
                      if (current != i) Haptics.selectionClick();
                      controller.changeTab(i);
                    },
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 160),
                          child: Icon(
                            current == i ? _tabs[i].activeIcon : _tabs[i].icon,
                            key: ValueKey(current == i),
                            size: 23,
                            color: current == i ? accent : muted,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(_tabs[i].label,
                            style: GoogleFonts.inter(
                                fontSize: 10.5,
                                letterSpacing: 0.1,
                                fontWeight: current == i
                                    ? FontWeight.w600
                                    : FontWeight.w500,
                                color: current == i ? accent : muted)),
                      ],
                    ),
                  ),
                ),
              ),
          ]),
        ),
      ),
    );
  }

  Widget _buildSidebar(BuildContext context, bool isDark) {
    final accent = isDark ? const Color(0xFF0B8278) : const Color(0xFF0F766E);
    final muted = Theme.of(context).hintColor;

    return Container(
      width: 236,
      color: isDark ? Colors.black : Colors.white,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 24, 16, 28),
          child: Row(children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(11),
              child: Image.asset(
                  Platform.isLinux
                      ? 'assets/icons/openlm-mark.png'
                      : 'assets/icons/appicon.png',
                  width: 36,
                  height: 36),
            ),
            const SizedBox(width: 12),
            Text('OpenLM',
                style: GoogleFonts.inter(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.4,
                  color: isDark ? Colors.white : Colors.black,
                )),
          ]),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 12, 10),
          child: Text('WORKSPACE',
              style: GoogleFonts.inter(
                fontSize: 10,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.1,
                color: muted,
              )),
        ),
        Expanded(child: Obx(() {
          final current = controller.currentTab.value;
          return ListView.builder(
            itemCount: _tabs.length,
            itemBuilder: (_, i) {
              final tab = _tabs[i];
              final sel = current == i;
              return Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 3),
                child: Material(
                  color:
                      sel ? accent.withValues(alpha: 0.12) : Colors.transparent,
                  borderRadius: BorderRadius.circular(12),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () => controller.changeTab(i),
                    child: SizedBox(
                      height: 46,
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: Row(children: [
                          Icon(sel ? tab.activeIcon : tab.icon,
                              color: sel ? accent : muted, size: 19),
                          const SizedBox(width: 12),
                          Text(tab.label,
                              style: GoogleFonts.inter(
                                  fontSize: 13,
                                  fontWeight:
                                      sel ? FontWeight.w600 : FontWeight.w500,
                                  color: sel ? accent : muted)),
                        ]),
                      ),
                    ),
                  ),
                ),
              );
            },
          );
        })),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 12, 18),
          child: Text('Your devices. One conversation.',
              style: GoogleFonts.inter(fontSize: 11, color: muted)),
        ),
      ]),
    );
  }
}

class _NavItem {
  final IconData icon;
  final IconData activeIcon;
  final String label;
  const _NavItem(
      {required this.icon, required this.activeIcon, required this.label});
}
