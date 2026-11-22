# Shared chat across devices

Shared chat lets invited devices view the same saved conversations and move the active chat control between screens. The device hosting inference keeps the model loaded; takeover changes which device can send prompts, not where the model runs.

## Start a shared session

1. Put the host and invited devices on a private local network.
2. On the model host, open **Device cluster → Master → Start shared chat**.
3. Scan the QR code from another phone or tablet, or copy the invite link to another computer and paste it in **Device cluster → Shared chat**.
4. Open the Chat screen on an invited device. Its history updates as messages are added.
5. Tap **Take over** in the Chat screen to make that device the only active chat controller. Another device can take control later.

Shared chat can be joined while a device's compute role is **Off**. Choose **Worker** separately when the device should also lend CPU/GPU and memory to the model.

## Boundaries and network security

- The model remains on the original host. If that host or its network connection stops, shared generation stops until it returns.
- Saved text conversations sync. Image attachments are not copied between devices.
- The host listens only on the selected private LAN address. The invite contains a random bearer token; anyone who has it can read synced history and take chat control.
- Session traffic currently uses unencrypted HTTP. A device on the same network could intercept or alter it. Share only over a trusted network, and stop sharing when finished. The invite token rotates each time the host starts sharing.
- Keep OpenLM open on the host while sharing. On Android, OpenLM keeps the screen awake so the model and session server remain available.
