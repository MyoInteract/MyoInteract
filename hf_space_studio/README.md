---
title: MyoInteract Studio
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Set up, train and evaluate musculoskeletal interaction tasks
---

# MyoInteract Studio (Space template)

[Studio](https://github.com/MyoInteract/MyoInteract) with its backend, as a Docker Space in **your own**
Hugging Face account, on the hardware you choose there.

1. Create a Space: *New Space → Docker → Blank*, visibility **Private**.
2. Upload this folder: `hf upload <user>/myointeract-studio hf_space_studio . --repo-type space`.
3. Pick the hardware in the Space's settings. Training needs a GPU (e.g. *Nvidia T4 small*); on CPU hardware
   Studio still configures tasks, previews them and plays the example policies.
4. Open the Space: Studio runs in the page, training on the Space's GPU.

Keep the Space private. Whoever can open it can start training runs on your hardware, and the server asks
for no pairing code (`--behind-proxy`), because Hugging Face already restricts a private Space to you.

Runs and presets are written to `/data` when the Space has persistent storage, otherwise to the container
(lost on restart).
