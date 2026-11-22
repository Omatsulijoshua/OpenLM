# OpenLM field guide

Static public website for `openlm.thispages.xyz`. The design brief is in the repository root `prompt.md`.

Run `npm run dev` inside this directory and open http://localhost:3000.

Deploy with `vercel --prod` from this directory. In Vercel Git project settings, use `website` as the Root Directory, Other as the framework, and `public` as Output Directory. No build or dependency installation is required. Add `openlm.thispage.xyz` as the production domain and apply the DNS record Vercel reports.

The 30B walkthrough is a conceptual illustration, not live inference. The current cluster has a Master for chat and Workers for compute. Shared conversation history and chat/control from every paired device are shown as the product direction, not as shipped behavior. App screenshots are from Joshua's connected Galaxy S8. No analytics, forms, cookies or inference API calls are included. Google Fonts is used with system font fallbacks.

## Store links

Edit public/store-links.js to replace the empty googlePlay and appStore values with published listing URLs. Until then the badges open a clearly labeled demo preview. Assets are the official Apple SVG and Google Play badge PNG, stored locally for sharp rendering without an external image request.


Windows and macOS direct download placeholders are also configured in public/store-links.js using the windows and mac keys. Use HTTPS URLs to your installers or archives. Empty values open a sample preview.

