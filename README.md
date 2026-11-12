# Deepfake Voice Shield

A real-time co-pilot for Customer Care agents. It tells them how likely a caller is to be a **voice clone** or an
**AI negotiation bot**, explains why, suggests a liveness challenge, and writes a **transfer handoff card** so the next
agent doesn't re-verify a caller who has already been verified.

It runs two real models on the caller's voice, **NVIDIA TitaNet** (voiceprint) and **Spectra-AASIST** (synthetic-speech
detection). It fuses them with conversation behaviour, caller-ID signals, account context and a liveness challenge into
one explainable 1-99 % risk score.

AI Unlocked Innovation Sprint · AI & Analytics · **No customer data: open data and fictional accounts only.**

---

## Five calls, five use cases

| Call | What happens | Shield's decision |
|---|---|---|
| **Genuine customer** - duplicate TravelPass charge | A real person (LibriSpeech voice) asks about a billing error | **PROCEED** (risk 3 %). The handoff card says *no re-verification needed*: 60-90 s saved on the transfer |
| **Genuine customer abroad** - lost phone, new SIM | A real person calls from a hotel line abroad and asks for a SIM change | **STEP UP** (risky context and a high-risk request), he **passes** the phrase challenge, then **PROCEED** |
| **Voice clone** - SIM swap on Alex Chen | A Chatterbox clone of the customer's own voice | The voiceprint **matches** (0.82), but the voice is synthetic (98 %): **STEP UP**, the clone **fails** the challenge (2.1x slower), then **ESCALATE** |
| **AI negotiation bot** - retention credit | An undisclosed AI agent impersonating the customer | **ESCALATE** after one sentence (96 %): synthetic voice, wrong voiceprint, machine-steady timing, talks over the agent |
| **AI assistant (disclosed)** - checks a bill for the customer | A legitimate assistant that says it is an AI | **Authorised AI agent**: consent request on the customer's phone, then information only - not treated as fraud |

The key point: **a voiceprint-only system would let the clone in.** Only combining the voiceprint with the
synthetic-voice model exposes it (*the voice matches, but it is synthetic*).

## Measured results

Held-out test set: LibriSpeech **test-clean**, 40 speakers never used for tuning. 120 real clips; 82 fakes (32 voice
clones, 32 neural TTS, 18 older TTS). Thresholds were fixed beforehand (on dev-clean, or from the model card).

| Through a simulated phone line (8 kHz G.711) | Result |
|---|---|
| Fake voices flagged as synthetic | **100 %** (clones, neural TTS and older TTS) |
| Real voices wrongly flagged | **0 of 120** (true rate below ~2.5 % with 95 % confidence) |
| Clones that fool a voiceprint-only check (TitaNet, match at 0.70) | **97 %** |
| ...of those, flagged by Shield | **31 of 31** |
| Voiceprint equal error rate on real speakers | 0.7 % |

These are small, read-speech sets. See the **Model evaluation** page or `reports/eval.json` for every split and
channel, and *Limitations* below.

---

## Quick start (any laptop, no GPU, no ML install)

```bash
python3 -m venv .venv                      # tested with Python 3.9 (macOS built-in) and 3.12
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                       # http://localhost:8501
```

The dashboard uses the scores that TitaNet and Spectra-AASIST produced on these exact recordings, stored in each call
file, so it shows real model output without downloading the models.

| Page | What it is for |
|---|---|
| **Call co-pilot** | Pick a call and press **Next** or **Play**: every turn plays its audio and the risk, verdict, reasons, suggested challenge and signals update. The handoff card and fraud referral are at the bottom. |
| **Try your voice** | Record, upload or pick a clip and run both models; enroll your own voice first. |
| **Model evaluation** | Results on the calibration and held-out sets, clean and over a phone line. |
| **Business impact** | The brief's 76,000 agent-hours / ~$2.2M model, with every assumption adjustable. |
| **How it works** | Architecture, signals, decision policy, and what is real vs simulated. |

Also: `python -m vshield.cli` (all calls in the terminal), `python -m vshield.cli voice_clone` (turn by turn plus the
generated documents) and `pytest` (tests for every use case, plus an audio quality gate).

### On a locked-down or office network

The dashboard never contacts Hugging Face, NVIDIA or any other model host. It installs only the five packages in
`requirements.txt` (from PyPI or your company's mirror), and usage telemetry is switched off in
`.streamlit/config.toml`, so it runs fully offline.

| Without the ML packages (e.g. Hugging Face blocked) | Needs `requirements-ml.txt` + access to Hugging Face / NVIDIA |
|---|---|
| All five calls, with real model scores, audio, decisions and generated documents | The **Run the models live** switch (re-computing the scores) |
| Try your voice on the five demo clips | Recording or uploading your own audio on Try your voice |
| Model evaluation, Business impact, How it works, CLI, tests | Rebuilding the data or the evaluation |

After `git pull`, run `pip install -r requirements.txt` again: it upgrades Streamlit if your copy is older than 1.50.

## Run the real models live (optional)

```bash
pip install uv                                          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-ml
uv pip install --python .venv-ml/bin/python -r requirements-ml.txt
.venv-ml/bin/streamlit run app.py
```

- **Call co-pilot:** switch on **Run the models live** in the sidebar.
- **Try your voice:** records and uploads now work. The first run downloads about 2.6 GB of weights.
- **Speed (measured on an Apple M1 Pro):** about 0.1 s per call turn on the GPU (MPS), about 1 s on the CPU.
- **Windows:** use WSL2, because NeMo is unreliable on native Windows.
- **Live CLI:** `.venv-ml/bin/python -m vshield.cli --live` re-runs the models from the command line.

**Demo idea, clone yourself (only your own voice, or a teammate's with written consent):**

```bash
uv venv --python 3.12 .venv-gen && uv pip install --python .venv-gen/bin/python -r requirements-gen.txt
.venv-gen/bin/python scripts/clone_voice.py my_voice.wav     # 20 s of you talking -> my_voice_clone.wav
```

On **Try your voice**, enroll yourself with the microphone, then upload `my_voice_clone.wav`. The voiceprint matches
you, and Shield flags the clone.

---

## How it works

```
caller audio  ─► A. Synthetic voice   Spectra-AASIST (wav2vec2 XLS-R + AASIST)    ─┐
              ─► B. Voiceprint        NVIDIA TitaNet-Large, cosine vs enrolled     │
transcript    ─► C. Behaviour         reply-time regularity, barge-in, scripted     ├─► risk fusion 1-99 % ─► policy ─► co-pilot
+ timestamps     + Liveness challenge  playbook, social engineering, AI disclosure  │   (weighted log-odds)   (proceed /    + handoff card
caller ID/CTI ─► D. Telecom           STIR/SHAKEN, VoIP, velocity, bot-farm ANI     │                          step up /     + fraud referral
account       ─► E. Account           risk flags, security events, high-risk ask   ─┘                          escalate /
                                                                                                                 AI agent)
```

- **Risk bands:** under 40 % **proceed**; 40-80 % **step up** (challenge or OTP; hold credits and SIM changes);
  80 % and over **escalate** to Fraud. The score is never 0 % or 100 %.
- **Pattern rules:** *voice matches + synthetic speech* means a clone, and *synthetic + bot-like behaviour* means a bot.
  A failed liveness challenge sets a floor of 88 %.
- **Policy rules:**
  - A high-risk request (SIM change, port-out, new line) always needs one liveness check.
  - Nothing counts as verified before the caller has spoken.
  - A caller that discloses it is an AI goes to the **authorised-agent** flow (consent on the customer's phone).
- **Never auto-hang-up:** Shield steps up, so a genuine customer on a bad line just answers one quick question.
- **Why a random phrase:** a bot with breached data answers "what's your zip code?" instantly. Repeating a random
  phrase forces a clone or bot pipeline to generate new audio on the fly, which shows up as a delay compared with the
  caller's own baseline.

Code map:

```
app.py                     Streamlit entry point (5 pages in ui/)
ui/                        copilot, try_voice, evaluation, impact, about, shared style
vshield/session.py         CallSession: replays a call turn by turn, one risk snapshot per turn
vshield/detectors.py       the two ML models (lazy imports; optional)
vshield/signals/           synthetic_voice, voiceprint, behavior, context (telecom + account)
vshield/challenge.py       liveness challenge: suggest and evaluate
vshield/fusion.py          risk fusion + action policy
vshield/handoff.py         transfer handoff card + fraud referral
vshield/impact.py          business-impact model
vshield/audio.py           WAV I/O, resampling, phone-line simulation, audio quality check
vshield/cli.py             command-line replay
data/                      5 calls (JSON + audio), accounts, enrolled voiceprints - see data/NOTICE.md
reports/eval.json          evaluation results (read by the Model evaluation page)
scripts/                   rebuild data, score calls, build the evaluation sets, evaluate, clone a voice
tests/                     use cases, signals and policy, audio quality, live models (when installed)
```

## Rebuild everything (reproducible)

```bash
.venv-gen/bin/python scripts/get_librispeech.py      # public speech, ~680 MB, checksummed
.venv-gen/bin/python scripts/build_demo_calls.py     # the 5 calls (~5 min on an M1)
.venv-ml/bin/python  scripts/score_calls.py          # store the model scores in the call files
.venv-gen/bin/python scripts/build_eval_set.py       # labelled real/fake sets, both splits (~1 h on an M1)
.venv-ml/bin/python  scripts/evaluate.py             # -> reports/eval.json
pytest                                               # includes the audio quality gate
```

Three environments keep each install small: `requirements.txt` (dashboard), `requirements-ml.txt` (models) and
`requirements-gen.txt` (voice generators: Chatterbox pins an older torch than NeMo).

## Data and licences

- **Real voices:** LibriSpeech (CC BY 4.0).
- **Voice clones:** Chatterbox (MIT; watermarked).
- **TTS voices and agents:** Kokoro-82M (Apache-2.0).
- **Models:** NVIDIA TitaNet-Large (CC BY 4.0) and Spectra-AASIST (Apache-2.0 / MIT, pinned revision).
- **Mock data:** accounts, numbers and call metadata are fictional.

Details are in [data/NOTICE.md](data/NOTICE.md).

## Limitations

- **Read speech:** evaluation uses read audiobook speech, not real phone conversations. Mobile codecs (AMR, EVS, Opus)
  and packet loss are not covered yet.
- **Small sets:** 120 real clips per split can't pin down a 1 % false-alarm rate.
- **Arms race:** newer or adversarially tuned generators may evade the detector, which reports 14 % EER on the harder
  ASVspoof5 benchmark. Production needs monitoring and retraining on confirmed fraud.
- **Simulated parts:** call timing (reply delays, barge-ins), caller ID / CTI data and accounts are simulated, and
  transcripts are the known text. Production would use streaming ASR such as NVIDIA Parakeet.
- **Confirm tools:** check NVIDIA NeMo, the Hugging Face models and any LLM with a Tech Coach (sprint rule 2).
