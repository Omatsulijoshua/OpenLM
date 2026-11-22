# OpenLM website brief

Build an original public website for Omatsuli Joshua Toritseju's OpenLM fork, hosted on Vercel at openlm.thispages.xyz. Implement in `website/` independently of Flutter's `web/` application.

## Creative direction

Make a field guide to sharing a model across everyday devices. Use warm paper, ink typography, vermilion signals, fine rules, generous space, numbered chapters and technical drawing details. Pair oversized editorial headlines with small monospace annotations. Avoid generic gradient backgrounds, glowing orbs, repeated rounded feature cards, invented testimonials and empty marketing claims. Draw the devices and network in native SVG/CSS. Make the network diagram the visual identity. Emphasize that the product goal is a shared AI workspace: no device is merely a remote phone controller. Laptops, desktops, tablets and phones should all be able to chat, control the model and see the same conversation history, while contributing compute when appropriate. Identify synchronized history and cross-device handoff as product direction where not yet implemented.

## Story

1. Lead with “One model. More room to think.” Explain that OpenLM distributes GGUF model layers over a trusted local network.
2. Explain what parameters and quantization mean. A dense 30B model at exactly 4 bits is about 15 GB of raw weights in decimal units, before quantization metadata, cache and runtime buffers. Illustrate an assumed 18 GB weight file. This is a teaching estimate, not a named model or benchmark.
3. An interactive 30B walkthrough with four device types: laptop, desktop, tablet and phone. Illustrate pair, place weights, pass intermediate activations across layer groups, and return a token. Use one shared set of weights (illustrative 6/5/4/3 GB split); never imply full copies or guaranteed speedup.
4. Show illustrative usable budgets: laptop 12 GB, desktop 8 GB, tablet 6 GB, phone 4 GB (30 GB combined). Explain that reserves, cache and runtime affect real fit. Clearly label budgets and model weights as examples.
5. Explain the current architecture honestly: a Master coordinates GGUF inference, Workers contribute memory and compute. Present the user’s intended product direction as a distinct concept: every paired laptop, desktop, tablet or phone can open the same live conversation, view shared history, send prompts and take control; all devices can also contribute model memory and computation. Shared cross-device chat history/control is a roadmap concept until implemented. Do not state it is currently available.
6. Include real Galaxy S8 screenshots from this fork: Chat, Models, Server, Settings. Label them honestly; the phone has no model downloaded. Do not call them evidence of a 30B run. Provide accessible image enlargement with close button, Escape and focus restoration.
7. Explain setup: install OpenLM on participating devices, same trusted LAN, worker mode and pairing, coordinator/master mode, load compatible GGUF, begin chat. Mention mixed platforms, network latency, device temperature, memory headroom and that LiteRT models are not distributed by this GGUF cluster path.
8. Finish with direct links to the fork, source code and Joshua's profile. Keep upstream MIT attribution.

## Implementation requirements

Semantic responsive HTML, plain CSS and JavaScript. Zero runtime dependencies. A small Node static server for `npm run dev`, serving only public assets. Vercel static configuration. Keyboard navigation, visible focus, reduced motion, readable mobile layout, no horizontal page overflow. Use a pause/play control and manual phase buttons for the simulation; no fake live telemetry or real inference claims. Use CSS/SVG motion and actual screenshots rather than generated images.

## Accuracy sources

- Repository `lib/services/cluster_service.dart`: GGUF split, local/worker RAM reserves, speed-based allocation and master/worker roles.
- Repository `lib/views/cluster_view.dart`: Off/Master/Worker UI and pairing.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- https://vercel.com/docs/project-configuration/vercel-json

## Delivery

Check the static server, simulation transitions, screenshot paths and responsive layouts. Publish through Vercel, associate the custom domain if account access allows it, and report any DNS requirement accurately. Do not claim the custom hostname is live before checking it.
