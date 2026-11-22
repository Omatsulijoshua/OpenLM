# OpenLM website brief

Build an original public website for Omatsuli Joshua Toritseju's OpenLM fork, hosted on Vercel at openlm.thispage.xyz. Implement in `website/` independently of Flutter's `web/` application.

## Creative direction

Make a field guide to sharing a model across everyday devices. Use warm paper, ink typography, vermilion signals, fine rules, generous space, numbered chapters and technical drawing details. Pair oversized editorial headlines with small monospace annotations. Avoid generic gradient backgrounds, glowing orbs, repeated rounded feature cards, invented testimonials and empty marketing claims. Draw the devices and network in native SVG/CSS. Make the network diagram the visual identity.

## Story

1. Lead with “One model. More room to think.” Explain that OpenLM distributes GGUF model layers over a trusted local network.
2. Explain what parameters and quantization mean. A dense 30B model at exactly 4 bits is about 15 GB of raw weights in decimal units, before quantization metadata, cache and runtime buffers. Illustrate an assumed 18 GB weight file. This is a teaching estimate, not a named model or benchmark.
3. An interactive 30B walkthrough: pair devices, place weights, pass intermediate activations across layer groups, return a token to the coordinator. One token repeats the computation. Never imply three full copies of the model or guaranteed linear speedup.
4. Show three illustrative memory budgets: laptop 12 GB, desktop 8 GB, phone 4 GB, totaling 24 GB available for this example after reserves. Use an illustrative 9/6/3 GB weight split. Explain remaining room is needed for runtime and cache, and actual fit varies.
5. Include real Galaxy S8 screenshots from this fork: Chat, Models, Server, Settings. Label them honestly; the phone has no model downloaded. Do not call them evidence of a 30B run. Provide accessible image enlargement with close button, Escape and focus restoration.
6. Explain setup: install OpenLM on participating devices, same trusted LAN, worker mode and pairing, coordinator/master mode, load compatible GGUF, begin chat. Mention mixed platforms, network latency, device temperature, memory headroom and that LiteRT models are not distributed by this GGUF cluster path.
7. Finish with direct links to the fork, source code and Joshua's profile. Keep upstream MIT attribution.

## Implementation requirements

Semantic responsive HTML, plain CSS and JavaScript. Zero runtime dependencies. A small Node static server for `npm run dev`, serving only public assets. Vercel static configuration. Keyboard navigation, visible focus, reduced motion, readable mobile layout, no horizontal page overflow. Use a pause/play control and manual phase buttons for the simulation; no fake live telemetry or real inference claims. Use CSS/SVG motion and actual screenshots rather than generated images.

## Accuracy sources

- Repository `lib/services/cluster_service.dart`: GGUF split, local/worker RAM reserves, speed-based allocation and master/worker roles.
- Repository `lib/views/cluster_view.dart`: Off/Master/Worker UI and pairing.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- https://vercel.com/docs/project-configuration/vercel-json

## Delivery

Check the static server, simulation transitions, screenshot paths and responsive layouts. Publish through Vercel, associate the custom domain if account access allows it, and report any DNS requirement accurately. Do not claim the custom hostname is live before checking it.
