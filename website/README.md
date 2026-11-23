# OpenLM field guide

Static public website for `openlm.thispages.xyz`. The design brief is in the repository root `prompt.md`.

Run `npm run dev` inside this directory and open http://localhost:3000.

Deploy with `vercel --prod` from this directory. In Vercel Git project settings, use `website` as the Root Directory, Other as the framework, and `public` as Output Directory. No build or dependency installation is required. Add `openlm.thispage.xyz` as the production domain and apply the DNS record Vercel reports.

The 30B walkthrough is a conceptual illustration, not live inference. The cluster Master hosts inference while Workers contribute compute. Opt-in Shared chat syncs saved text conversations to devices with the LAN invite link; one device at a time holds primary chat control. Taking over changes the active chat screen, while the model stays on the original host. Shared traffic uses unencrypted HTTP and the invite link is a bearer token, so only use it on a trusted local network. Image attachments are not synced yet. The website gallery includes a Windows desktop capture and Android phone screens from Joshua's Galaxy S8. No analytics, forms, cookies or inference API calls are included. Google Fonts is used with system font fallbacks.

## Store links

Edit public/store-links.js to replace the empty googlePlay and appStore values with published listing URLs. Until then the badges open a clearly labeled demo preview. Assets are the official Apple SVG and Google Play badge PNG, stored locally for sharp rendering without an external image request.


Windows and macOS direct download placeholders are also configured in public/store-links.js using the windows and mac keys. Use HTTPS URLs to your installers or archives. Empty values open a sample preview.

