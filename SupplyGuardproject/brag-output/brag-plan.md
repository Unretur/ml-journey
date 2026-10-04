# Brag Plan: SupplyGuard AI

## What is this app?
SupplyGuard AI predicts shipment `delay_days` from a DataCo order — real shipping days minus scheduled days — then serves that call as FastAPI `POST /predict`.

## The angle
Late delivery is not a mystery after the truck is late. The model’s job is the number you can ask *before* `Delivery Status` exists. Shipping Mode and scheduled days dominate the SHAP chart; the API makes you type those two on purpose.

## Hook (first 2-3 seconds)
Quiet ops-room type: “The shipment left on time.” Hold. Then: “The calendar didn’t.”

## Key moments (the middle)
- Recreate `/predict`: fill `shipping_mode` and `days_for_shipment_scheduled`, hit the call, watch `predicted_delay_days` and `is_late` appear as the contract (no invented MAE, no fake delay number — the trained `.joblib` is not in the repo).
- Three grounded facts, one by one: 180,519 orders; date split, not shuffle; leakage columns (`Delivery Status`, `Late_delivery_risk`, real shipping days) dropped.

## Outro / punchline
`SupplyGuard AI` then the line from the code: `POST /predict` returns `is_late`.

## User flow worth showing
Entry: FastAPI shipment form with the two SHAP-dominant fields required.
Key action: `POST /predict`.
Result: JSON with `predicted_delay_days` and `is_late`.

## Tone
- Preset: polished
- Creative direction: quiet ops-room late-shipment film
- Interpretation: few scenes, long holds, amber-on-navy, no hype metrics, no generic SaaS verbs.

## Format: landscape — 1920x1080
## Duration: 20 seconds

## Visual identity (from the project)
No product CSS exists (Python + FastAPI + notebook). Palette is derived for an ops console, not extracted from a stylesheet:
- Background: `#0B1F33`
- Accent: `#E8A838`
- Text: `#F4F7FA`
- Late mark: `#FFB4A8`
- Display font: system-ui (no webfont files in the project)
- Body font: system-ui
- Strongest visual element: the `/predict` request/response panel from `api.py`

## Share copy (draft)
SupplyGuard AI predicts `delay_days` before `Delivery Status` exists. `POST /predict` answers `is_late`.

## Audio direction
- Role: sparse professional accents over a low cinematic bed
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady/clean; polished)
- Music treatment: start at 0, volume ~0.18, fade under the final wordmark
- Music cue guidance: detect at composition time via `npx hyperframes beats`
- Audio-reactive treatment: subtle; navy glow / panel presence breathe with RMS. No waveform bars.
- SFX posture: sparse; motion-matched; professional restraint
- Audio-coupled moments: hook line settle, Predict click, JSON keys, three fact cards, logo
- Restraint rule: no chaotic hits, no stacked SFX, never duck into silence as a joke

## Storyboard

### Scene 1 — The calendar didn’t — 3.5s
Navy full-bleed. Line 1: “The shipment left on time.” Line 2: “The calendar didn’t.”
Sequential/interaction: yes — two lines, one after the other
Audio intent: quiet open
Audio-coupled idea: soft drop on line 2
Music: low bed
Transition mood: soft → Scene 2

### Scene 2 — POST /predict — 9.0s
Recreate the working call: labels `shipping_mode` and `days_for_shipment_scheduled` fill in (First Class / 1 — field names from the API, values are the Swagger-style demo inputs, not a claimed live prediction). Cursor clicks Predict. Response keys `predicted_delay_days` and `is_late` type in as the contract; values stay as an em dash because `final_model.joblib` is not in the repo.
Sequential/interaction: yes — fields, click, JSON keys
Audio intent: precise UI
Audio-coupled idea: click on Predict, soft bong when keys land
Transition mood: soft → Scene 3

### Scene 3 — What the model is allowed to see — 4.5s
Three cards in sequence:
1. 180,519 orders
2. Split by date, not shuffle
3. No `Delivery Status`. No `Late_delivery_risk`.
Sequential/interaction: yes — three cards
Audio intent: dry confirmations
Audio-coupled idea: card-place / drop on each card; hold full set after the third
Music: still low
Transition mood: soft → Scene 4

### Scene 4 — is_late — 3.0s
Wordmark SupplyGuard AI. Subline: `POST /predict` → `is_late`.
Sequential/interaction: none
Audio intent: quiet landing
Audio-coupled idea: one soft bell
Music: fade

**Music mood for this video:** cinematic / polished
**Audio summary:** A quiet vol-12 bed with a handful of UI clicks and one landing bell; nothing louder than the copy.

## Rubric answers
1. App: Delay predictor + FastAPI for DataCo shipments.
2. Strongest claim: Shipping Mode and scheduled days dominate SHAP, so the API refuses to default them.
3. Visual hook: amber `is_late` on navy console.
4. Show: `/predict` form + JSON contract from `api.py`.
5. Shortest satisfying: 20s.
6. Tone: polished / ops-room film.
7. Audio: low bed + sparse UI.
8. Share: see above.
9. Flow: form → `/predict` → `predicted_delay_days` / `is_late`.
