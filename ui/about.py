"""Page 5 - How it works: architecture, signals, decision policy, and what is real vs simulated."""
from __future__ import annotations

import streamlit as st

from ui import style as S

DIAGRAM = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#f4f4f2", color="#c3c2b7", fontname="Helvetica", fontsize=11];
  edge [color="#898781", arrowsize=0.6];
  audio [label="Caller audio", fillcolor="#ffffff"];
  text  [label="Transcript +\\ntimestamps", fillcolor="#ffffff"];
  cti   [label="Caller ID / CTI", fillcolor="#ffffff"];
  acct  [label="Account record", fillcolor="#ffffff"];
  sv [label="A. Synthetic voice\\nSpectra-AASIST", fillcolor="#e8f0fb"];
  vp [label="B. Voiceprint\\nNVIDIA TitaNet", fillcolor="#e8f0fb"];
  bh [label="C. Conversation\\nbehaviour"];
  ch [label="Liveness\\nchallenge"];
  id [label="Identity check\\naccount history,\\nphone on file"];
  tc [label="D. Telecom\\nsignals"];
  ac [label="E. Account\\ncontext"];
  fu [label="Risk fusion\\n1-99 %", fillcolor="#fde9e9", color="#cd040b"];
  po [label="Policy\\nproceed / step up /\\nescalate / AI agent", fillcolor="#fde9e9", color="#cd040b"];
  ui [label="Agent co-pilot\\nverdict, reasons,\\ncheck to ask", fillcolor="#ffffff"];
  doc [label="Handoff card /\\nfraud referral", fillcolor="#ffffff"];
  audio -> sv; audio -> vp; text -> bh; text -> ch; text -> id; acct -> id; cti -> tc; acct -> ac;
  sv -> fu; vp -> fu; bh -> fu; ch -> fu; id -> fu; tc -> fu; ac -> fu; fu -> po; po -> ui; po -> doc;
}
"""


def render() -> None:
    S.header("How <span>it works</span>",
             "Seven pieces of evidence, one explainable risk score, and a policy that steps up instead of hanging up.")
    st.graphviz_chart(DIAGRAM, width="stretch")

    st.markdown("""
#### The evidence

| Signal | What it looks at | How |
|---|---|---|
| **A. Synthetic voice** | Is this a real human voice, or TTS / a voice clone? | **Spectra-AASIST** anti-spoofing model (wav2vec2 XLS-R + AASIST) on the caller's audio |
| **B. Voiceprint** | Does the voice match the account holder's enrolled voice? | **NVIDIA TitaNet-Large** speaker embeddings, cosine similarity |
| **C. Conversation behaviour** | Machine-steady reply timing, talking over an interruption, scripted negotiation phrases, social-engineering cues, AI self-disclosure | Transcript and timestamps |
| **Liveness challenge** | A random phrase to repeat. A bot or clone pipeline needs fresh ASR, LLM and TTS, so its answer comes late | Reply delay vs the caller's own baseline, plus the words |
| **Identity check** | For a real human who is not the enrolled voice: a question from the account history (not printed on the bill), and a confirmation on the phone on file | The answer is matched against the account record; the customer taps approve or *Not me* |
| **D. Telecom** | STIR/SHAKEN attestation, VoIP, number not on file, call velocity, one number hitting many accounts, machine-even keypresses | CTI metadata |
| **E. Account** | Risk flags, recent security events, high-risk requests (SIM change, port-out), serial credit requests | Account record |

**The key idea:** each model alone has a blind spot, and each covers the other's.
- A good voice clone *matches* the voiceprint, so a voiceprint-only system lets it in. The synthetic-voice model
  exposes it: *the voice matches, but it is synthetic*.
- A human impersonator, or a bill-negotiation service calling as the customer, *is* a real human, so a deepfake
  detector alone lets them in. The voiceprint exposes them: *a real voice, but not the account holder's*. Shield
  then asks what only the customer knows (account history) and has (their phone), because a liveness check proves
  nothing against a live human.

#### The decision

| Risk | Action | What the agent does |
|---|---|---|
| under 40 % | ✅ **Proceed** | Serve the customer. On transfer, the handoff card says *no re-verification needed* (60-90 s saved). |
| 40-80 % | ⚠️ **Step up** | Ask the suggested check (a liveness phrase, or an account-history question when the voice is a real person but not the account holder) or send a confirmation to the phone on file; hold credits, SIM and account changes. |
| 80 % and over | ⛔ **Escalate** | No credits or account changes; warm-transfer to Fraud with the fraud referral. |
| AI says it is an AI | 🤖 **Authorised AI agent** | Consent request on the customer's phone; then information only, never credits or SIM changes. |

Three policy rules sit on top of the score:
- **High-risk requests always get one liveness check** (SIM change, port-out, new line), even when the voice looks genuine.
- **Nothing is "verified" before the caller has spoken**: caller ID and account data alone never skip verification.
- **Never "genuine" on someone else's voice**: a live human voice lowers the risk only when it is also the account
  holder's. A real person who doesn't match the voiceprint is stepped up (risk at least 50 %), and escalated (88 %)
  if they can't answer from the account history.

Shield never hangs up on anyone: it steps up, so a genuine customer on a bad line just answers one quick question.

#### What is real, and what is simulated

| Real | Simulated for the demo |
|---|---|
| Caller voices: real people (LibriSpeech, CC BY 4.0), a Chatterbox voice clone, Kokoro TTS voices | Call timing (reply delays, barge-ins) is scripted |
| Matching an answer against the account record (keywords) | Account histories, and the customer's tap on their phone, are scripted |
| Both models: NVIDIA TitaNet and Spectra-AASIST, with their real outputs | Caller ID, CTI data and accounts are mock (555 numbers, fictional names) |
| Measured accuracy on a held-out set (Model evaluation page) | Transcripts are the known text; production would use streaming ASR such as NVIDIA Parakeet |

#### Toward production

1. Streaming ASR (NVIDIA Parakeet / Riva) feeding the session turn by turn, and a live audio fork from the contact-centre platform.
2. Fine-tune and monitor on 8 kHz call audio with mobile codecs; retrain on confirmed fraud cases (the arms race is real).
3. An approved LLM to replace the phrase lists for playbook and social-engineering cues, and to write the handoff card (the template stays as a fallback).
4. Real CTI / ANI reputation feeds and the account risk store, and a baseline from the Looker *chip_cs Handoffs* dashboard.
""")
