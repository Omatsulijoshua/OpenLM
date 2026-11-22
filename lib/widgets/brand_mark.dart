import 'package:flutter/material.dart';

/// A small, scalable identity mark that stays crisp at any display density.
class BrandMark extends StatelessWidget {
  final double size;
  const BrandMark({super.key, this.size = 64});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0F766E), Color(0xFF134E4A)],
        ),
        borderRadius: BorderRadius.circular(size * 0.32),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF0F766E).withValues(alpha: 0.18),
            blurRadius: size * 0.45,
            offset: Offset(0, size * 0.12),
          ),
        ],
      ),
      child: Icon(Icons.auto_awesome_rounded,
          color: const Color(0xFFCCFBF1), size: size * 0.5),
    );
  }
}
