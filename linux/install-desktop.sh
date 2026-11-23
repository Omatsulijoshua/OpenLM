#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${HOME}/.local/opt/openlm"
ICON_DIR="${HOME}/.local/share/icons/hicolor/512x512/apps"
DESKTOP_DIR="${HOME}/.local/share/applications"

mkdir -p "${APP_DIR}" "${ICON_DIR}" "${DESKTOP_DIR}"
if [[ "${SOURCE_DIR}" != "${APP_DIR}" ]]; then
  cp -a "${SOURCE_DIR}/." "${APP_DIR}/"
fi
install -m 0644 "${APP_DIR}/share/icons/hicolor/512x512/apps/openlm.png" \
  "${ICON_DIR}/openlm.png"

cat > "${DESKTOP_DIR}/openlm.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=OpenLM
Comment=One conversation across all your devices
Exec=${APP_DIR}/openlm
Icon=openlm
Terminal=false
Categories=Utility;Network;
StartupWMClass=com.omatsuli.openlm
EOF

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${DESKTOP_DIR}" || true
fi

printf 'OpenLM is installed. Find it in your desktop app launcher.\n'
