# Spike: real models on real speech (2026-11-02)

Can NVIDIA TitaNet and a trained anti-spoofing model replace the demo detectors? On a small public sample, yes. Both work on an Apple M1 Pro, and the three demo stories keep their outcomes on real audio.

## What was tested

| Piece | Choice | Licence |
|---|---|---|
| Voiceprint | NVIDIA TitaNet-Large (`nvidia/speakerverification_en_titanet_large`) via NeMo 3.0 | CC-BY-4.0 |
| Anti-spoofing | Spectra-AASIST (`lab260/Spectra-AASIST`, revision `eb65c266`) | Apache-2.0 / MIT (card states both) |
| Anti-spoofing, comparison | `MelodyMachine/Deepfake-audio-detection-V2` (community wav2vec2 classifier) | Apache-2.0 |
| Real voices | LibriSpeech dev-clean, 40 speakers | CC-BY-4.0 |
| Clone attack | Chatterbox zero-shot clones of 8 of those speakers | MIT |
| Bot attack | Kokoro-82M stock voices (8 voices) | Apache-2.0 |
| Older TTS | macOS `say` (6 voices) | — |

Sample: 120 real test clips, 82 fake clips (32 clones, 32 Kokoro, 18 macOS), 80 enrollment clips. Every test ran twice: on clean 16 kHz audio, and through a simulated phone line (8 kHz G.711).

## Results

**Anti-spoofing** (decision point 0.5):

| Model | Channel | EER | Real callers flagged | Clones caught | Kokoro caught | macOS caught |
|---|---|---|---|---|---|---|
| Spectra-AASIST | clean | 1.0% | 1 / 120 | 100% | 100% | 100% |
| Spectra-AASIST | phone | 0.0% | 0 / 120 | 100% | 100% | 100% |
| Community wav2vec2 | clean | 28.7% | 6 / 120 | 22% | 0% | 6% |
| Community wav2vec2 | phone | 43.0% | 13 / 120 | 53% | 3% | 11% |

**TitaNet voiceprint** (cosine similarity to the enrolled voice):

| Channel | Same speaker (mean / min) | Other speaker (mean / max) | Clone of the customer (mean) | Clones that pass at 0.70 |
|---|---|---|---|---|
| clean | 0.88 / 0.74 | 0.05 / 0.51 | 0.73 | 72% |
| phone | 0.89 / 0.72 | 0.25 / 0.62 | 0.79 | **100%** |

TitaNet separates real speakers well, and it also accepts most clones as the customer: on the phone line, every clone passed. A voiceprint-only check would have approved them. The anti-spoofing model is what catches them, which is the project's core claim, now shown on real audio.

**Thresholds set from these numbers:** TitaNet match ≥ 0.70 (NeMo's default) and mismatch ≤ 0.55. Spectra-AASIST uses its model-card threshold (bona fide logit −1.14), mapped to probability 0.5.

**Speed** (M1 Pro, MPS): Spectra-AASIST processes audio about 50× faster than real time. Both models together take about 0.1 s per call turn once loaded. First load takes about a minute, because weights download once.

## The three demo stories on real audio

`VSHIELD_DATA=data/samples/demo VSHIELD_VOICEPRINT=titanet VSHIELD_SPOOF_MODEL=spectra`

| Call | Risk path | Final | Evidence |
|---|---|---|---|
| Genuine (real LibriSpeech speaker) | 22% → 3% | PROCEED | 1% synthetic; voiceprint 0.96 |
| Voice clone (Chatterbox clone of the account holder) | 51% → 74% → 88% | ESCALATE | voiceprint **matches** (0.80), but 97% synthetic → "Likely voice clone" |
| AI bot (Kokoro) | 59% → 96% | ESCALATE | 98% synthetic; voiceprint mismatch (−0.04) |

For comparison, the old pseudo-speech demo audio scores 52–60% synthetic on Spectra-AASIST for all three calls, including the "genuine" one. That audio is not speech, and only the hand-tuned demo detectors could tell it apart.

## Limits: what this does NOT show

- **Small, easy sample.** 120 real clips can't measure a false-alarm rate in the 1% range. One more mistake moves it by almost a point.
- **Read audiobook speech, not phone calls.** TitaNet's training data includes LibriSpeech (different speakers, same domain), so its speaker separation here is flattering. Spectra-AASIST was trained on studio-recorded data too.
- **Phone line is G.711 only.** Mobile calls use AMR-NB, EVS or Opus, plus packet loss, which this test doesn't cover.
- **Known generators only.** Chatterbox and Kokoro are 2025-era models. Newer or adversarially tuned generators may evade the detector. Spectra-AASIST's own card reports 14% EER on ASVspoof5.
- **Timings are still scripted.** Reply delays and the liveness challenge timings come from the call scripts, not live audio.
- **The genuine call's words are audiobook text**, because the real speakers never said our script. Consented teammate recordings fix this.

## Reproduce

```bash
uv venv --python 3.12 .venv-gen && uv pip install --python .venv-gen/bin/python -r requirements-gen.txt
uv venv --python 3.12 .venv-ml  && uv pip install --python .venv-ml/bin/python  -r requirements-ml.txt
# LibriSpeech dev-clean -> data/external/librispeech/ (https://www.openslr.org/12, md5 42e2234ba48799c1f50f24a7926300a1)
.venv-gen/bin/python scripts/build_sample_set.py      # ~30 min on M1 (cloning dominates)
.venv-ml/bin/python scripts/evaluate.py               # -> reports/spike_eval.json
VSHIELD_DATA=data/samples/demo VSHIELD_VOICEPRINT=titanet VSHIELD_SPOOF_MODEL=spectra .venv-ml/bin/streamlit run app.py
```

Generated audio stays in git-ignored `data/samples/` and isn't redistributed. Chatterbox output carries Resemble AI's Perth watermark.
