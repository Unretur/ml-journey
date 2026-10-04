# Hyperframes Composition Brief: SupplyGuard AI

## Objective
Create a short launch-style brag video for SupplyGuard AI.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `c:\Users\yashb\ml-journey\SupplyGuardproject`
- Primary files read: `api.py`, `train.py`, `evaluate.py`, `Data.py`, `features.py`, `01_eda.ipynb`
- Product name: SupplyGuard AI
- Tagline / strongest claim: Shipping Mode and Days for shipment (scheduled) dominate the SHAP chart
- Key UI or visual moment to recreate: FastAPI `POST /predict` request + `predicted_delay_days` / `is_late` response
- Copy that must appear verbatim:
  - SupplyGuard AI
  - POST /predict
  - predicted_delay_days
  - is_late
  - shipping_mode
  - days_for_shipment_scheduled
  - Delivery Status
  - Late_delivery_risk
  - delay_days

## Creative Direction
- Tone preset: polished
- Creative direction: quiet ops-room late-shipment film
- Interpretation: restraint, long holds, no invented accuracy numbers
- Angle: Predict delay before the delivery-status column exists.
- Hook: The shipment left on time. The calendar didn’t.
- Outro / punchline: SupplyGuard AI / POST /predict → is_late
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Invented MAE / RMSE / fake predicted_delay_days values
  - Unrelated visual redesign

## Visual Identity
- Background: `#0B1F33`
- Text: `#F4F7FA`
- Accent: `#E8A838`
- Display font: system-ui
- Body font: system-ui
- Visual references from the project: FastAPI form fields, JSON keys from `api.py`, date-split and leakage notes from `Data.py`, 180519 from the notebook

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The calendar didn’t — 3.5s — two-line hook
2. POST /predict — 9.0s — form, click, JSON keys
3. What the model is allowed to see — 4.5s — three fact cards
4. is_late — 3.0s — wordmark

## Audio
- Audio role: sparse professional accents
- Audio arc: quiet bed throughout; UI clicks in scene 2; card drops in scene 3; one landing cue; fade
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3
- Music treatment: volume 0.18, fade under final logo
- Music cue guidance: detect at composition via hyperframes beats
- Audio-reactive treatment: subtle; panel/glow presence with RMS if extraction is available; skip if helper missing
- Audio-coupled moments:
  - Scene 1 line 2 — drop
  - Scene 2 Predict — click
  - Scene 2 JSON — bong
  - Scene 3 cards — drops
  - Scene 4 logo — soft bell
- Exact SFX choice: Hyperframes should choose filenames after animation exists; planned copies: `drop_001.ogg`, `click_001.ogg`, `bong_001.ogg`, `impactSoft_medium_000.ogg`
- Audio files: copy into `brag-output/composition/assets/`

## Hyperframes Instructions
Standalone `index.html`. GSAP paused timeline registered as `window.__timelines["supplyguard-brag"]`. Show the `/predict` UI. Keep 15–25s. Local render only. Run `hyperframes check` before render.
