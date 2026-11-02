# Deepfake Voice Shield

Real-time agent co-pilot that tells Customer Care agents how likely a caller is a **voice clone** or an **AI negotiation bot**. It explains why, suggests a liveness challenge, and writes a **transfer handoff card** so the next agent doesn't re-verify a caller who has already been verified.

AI Unlocked Innovation Sprint prototype · Category: AI & Analytics · **100% synthetic data, no customer information.**

---

## Quick start (VS Code)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_demo_data.py   # optional, data/ is already included
streamlit run app.py                   # opens the agent dashboard in your browser
```

Other entry points:

```bash
python -m vshield.cli                  # score all 3 demo calls
python -m vshield.cli voice_clone      # turn-by-turn risk + generated handoff card + fraud referral
pytest                                 # 11 tests
```

VS Code: open the folder, accept the recommended Python extension, then use **Run and Debug** → "Dashboard (Streamlit)", "CLI: all calls" or "Tests".

---

## The three demo calls

| Call | What happens | Shield result |
|---|---|---|
| **Genuine customer** (Jordan Ellis, duplicate TravelPass charge) | Natural voice, um/uh, variable timing, stops when the agent interrupts | **~4% → PROCEED.** Handoff card says no re-verification is needed (saves 60-90 s on transfer) |
| **Voice clone** (SIM-swap attempt on Alex Chen) | The cloned voice **matches the enrolled voiceprint**, asks to send the OTP to a different email, uses a spoofed ANI with C attestation | 51% at ring → **72% STEP-UP** after the first sentence (voice matches but has synthetic artefacts) → fails the phrase challenge → **88% ESCALATE** |
| **AI negotiation bot** (retention credit on Sam Rivera) | Polished scripted negotiation, constant ~0.9 s reply delay, talks over the agent's interruption, bot-farm ANI | **90%+ ESCALATE** by the second turn. The challenge confirms it (3.2× latency spike) |

The clone case is the key slide. **A voiceprint check alone would have approved this caller.** Only fusing it with the synthetic-voice detector catches the clone.

---

## How it works

```
               ┌─────────────────────────── live call ───────────────────────────┐
 caller audio ─┤ A. Synthetic-voice detector  (anti-spoofing: demo DSP | HF model) │
               │ B. Voiceprint match          (demo cepstral | NVIDIA TitaNet)     │
 transcript  ──┤ C. Conversation behaviour    (latency, disfluency, barge-in,      │──► Risk fusion ──► 0-100 % + verdict
 + timestamps  │                               bot playbook, social engineering)  │    (explainable)    + action + reasons
               │ + Liveness challenge result  (phrase-repeat latency spike)       │                      + suggested challenge
 CTI / ANI   ──┤ D. Telecom signals           (STIR/SHAKEN, VoIP, velocity, DTMF)  │                      + handoff card / fraud referral
 account     ──┤ E. Account context           (risk flags, intent, credit history) │
               └──────────────────────────────────────────────────────────────────┘
```

* **Fusion** is a weighted average of log-odds. The weight of each signal grows with evidence (more seconds of speech means more trust). A confidence factor keeps the score near the prior until enough evidence has arrived. Two explainable pattern rules sit on top:
  * *voice matches + synthetic artefacts → voice clone*
  * *synthetic + bot-like behaviour → AI agent*
* **The score is never 0% or 100%.** Actions are graded:
  * **PROCEED** (<40)
  * **STEP-UP** (40-75): liveness challenge or OTP to the device on file; hold credits and changes
  * **ESCALATE** (>75): warm transfer to Fraud
* **Disclosed AI assistants** (callers who say "I'm an AI assistant calling on behalf of…") are routed to an *authorised-agent* flow instead of being treated as fraud.

### Why the liveness challenge is a random phrase, not "what's your zip?"
A bot with breached data answers knowledge questions instantly. A random phrase forces a bot or clone pipeline to run ASR → LLM → TTS on the fly, which shows up as a latency spike compared with the caller's own baseline. A human simply repeats it.

---

## Swapping in production models

The demo backends need only numpy and scipy, so the prototype runs on any laptop. The real models plug into the same interfaces through environment variables:

| Signal | Env var | Production model |
|---|---|---|
| Voiceprint | `VSHIELD_VOICEPRINT=titanet` | NVIDIA TitaNet-Large (`nvidia/speakerverification_en_titanet_large`) via NeMo |
| Synthetic voice | `VSHIELD_SPOOF_MODEL=spectra` | Spectra-AASIST (wav2vec2 XLS-R + AASIST), pinned revision |
| Synthetic voice | `VSHIELD_SPOOF_MODEL=hf:<model_id>` | Any Hugging Face audio-classification anti-spoofing model |
| Data folder | `VSHIELD_DATA=data/samples/demo` | The same three calls with real speech (built by `scripts/build_sample_set.py`) |
| Transcript | (future) | Streaming ASR such as NVIDIA Parakeet / Riva or Whisper feeding `CallSession.add_turn()` |

The real models run in a separate venv (`requirements-ml.txt`), verified on an Apple M1 Pro. Spike results, reproduction steps and limits are in [reports/spike_eval.md](reports/spike_eval.md). In short: on public LibriSpeech speech against Chatterbox clones and Kokoro voices, Spectra-AASIST caught every fake with 0–1% EER, and TitaNet accepted most clones as the customer. The three demo stories keep their outcomes on real audio.

Notes:
* Recalibrate thresholds after swapping models: `THRESH` in `signals/voiceprint.py` and the logistic in `signals/acoustic.py`.
* Fine-tune or validate on **8 kHz telephony audio**. Detectors trained on clean 16 kHz audio degrade on phone lines.
* Check licences. Several anti-spoofing checkpoints are non-commercial (fine for a prototype, but flag it).
* Confirm with a Tech Coach that NeMo and Hugging Face are on the approved-tools list.

**About the demo detector:** the synthetic audio here is generated by a source-filter synthesizer. "Human" rendering has natural vocal micro-jitter, and "TTS" rendering is unnaturally steady, which mirrors a real cue that neural vocoders leave behind. The demo detector measures that cue transparently. It is a stand-in that shows the end-to-end flow. It is **not** a claim of real-world accuracy; that comes from the trained models above and an evaluation on ASVspoof / In-the-Wild data.

---

## Project layout

```
app.py                      Streamlit agent dashboard
vshield/
  session.py                Streaming call session (turn-by-turn risk snapshots)
  fusion.py                 Risk fusion, verdicts, actions, top reasons
  challenge.py              Liveness challenge suggestion + evaluation
  handoff.py                Generated transfer handoff card + fraud referral
  signals/acoustic.py       A. synthetic voice detection (demo | HF model)
  signals/voiceprint.py     B. voiceprint match (demo | NVIDIA TitaNet)
  signals/behavior.py       C. conversational behaviour
  signals/context.py        D. telecom + E. account context
  dsp.py, synth.py          audio utilities + synthetic voice generator
  cli.py                    command-line replay
scripts/generate_demo_data.py   builds data/ (accounts, enrolment audio, calls)
data/                       synthetic accounts, audio, transcripts, CTI metadata
tests/                      scenario + unit tests
```

## Next steps toward the final demo
1. Swap in TitaNet and a trained anti-spoofing model, then report a confusion matrix on ASVspoof plus your own 8 kHz recordings.
2. Live microphone mode: stream mic audio → ASR → `CallSession`.
3. Replace phrase lists with an LLM classifier for bot-playbook and social-engineering detection, and use the LLM to write the handoff card.
4. Looker-style baseline: chart transfers avoided and time saved (60-90 s × transfers) to back the ~76K agent-hours / ~$2.2M impact claim.
