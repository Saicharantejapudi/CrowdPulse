---
title: CrowdCounter
emoji: 👥
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Count people in aerial/crowd photos with a dot overlay
---

# Crowd Counter

Upload an aerial or dense crowd photo; get back the image with a dot on every detected
head and the total count printed underneath.

- **Model:** fine-tuned PET (Point-quEry Transformer), point-localization crowd counter.
- **Inference:** native-resolution tiling for dense-scene recall; adjustable head-confidence
  threshold to recover packed-core heads.
- **Hardware:** ZeroGPU (GPU attached per request via `@spaces.GPU`).

Accuracy is best on higher-resolution stills; expect ±10–15% on very dense scenes. The dot
overlay is the audit trail — check whether dots track heads.
