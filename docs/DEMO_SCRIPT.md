# Demo script (3 minutes)

Before you start, open the dashboard (`streamlit run app.py`), turn the volume up, and keep **Hear each turn** on.
Pick each call in the sidebar and press **Play**; press **Pause** whenever you want to talk over it.

| Time | Screen | What to say |
|---|---|---|
| 0:00 | Call co-pilot, ringing | "AI agents now call care lines to negotiate bills, and cloned voices try SIM swaps. Our agents can't hear the difference, and when a call is transferred the next agent starts verification from scratch: about 76,000 agent hours a year." |
| 0:20 | **Genuine customer**, Play | "A real customer. Before anyone speaks, Shield has already scored the caller ID and the account. Once she talks, NVIDIA TitaNet matches her voice and the anti-spoofing model hears a live human: risk drops to 3 %." Open **Transfer handoff card**: "On transfer, the next agent sees *verified, no re-verification needed*. That's 60 to 90 seconds saved per transfer." |
| 0:50 | **Voice clone**, Play | "Now someone has cloned Alex's voice to steal their number. Look at the voiceprint: it **matches**. A voiceprint-only system lets this caller in. But the anti-spoofing model says 98 % synthetic: the voice matches, but it isn't a person." Point at the suggested challenge: "Shield doesn't hang up, it steps up: the agent asks the caller to repeat a random phrase. The clone needs a fresh text-to-speech round trip and answers 2.1 times slower than its own baseline. Escalate, with a fraud referral already written." |
| 1:40 | **AI negotiation bot**, Play to turn 3 | "An undisclosed AI agent negotiating credits. Synthetic voice, the wrong voiceprint, machine-steady reply timing, and it keeps talking when the agent interrupts. Escalated after its first sentence." |
| 2:05 | **AI assistant** | "Not every AI caller is fraud. This assistant says what it is, so Shield routes it to consent on the customer's phone: information only, never credits or SIM changes." |
| 2:20 | **Model evaluation** (held-out, phone line) | "These are real models measured on a held-out set: every fake voice caught, none of the 120 real voices flagged, through a simulated phone line. And the key number: the voiceprint alone accepted 97 % of the clones, and Shield caught every one of them." |
| 2:40 | **Business impact** | "With the brief's numbers that's about 76,000 agent hours, or 2.3 million dollars a year, before counting the fraud it stops. Next step: a pilot on one transfer queue with live ASR." |

**Optional live moment (30 s; needs the live models):** On **Try your voice**, enroll yourself with the microphone,
then upload a clone of yourself made beforehand with `scripts/clone_voice.py`. "That's my voice, cloned in 30 seconds
with an open-source model. The voiceprint says it's me. Shield says it's a clone."

## Likely questions

- **"Won't it flag real customers?"** On the held-out set, 0 of 120 real voices were flagged (95 % upper bound about
  2.5 %). And Shield never blocks anyone: the worst case for a real customer is one 10-second phrase to repeat.
- **"What about new deepfake generators?"** That's the arms race. The detector is one of six signals, the liveness
  challenge works regardless of how good the voice is, and confirmed fraud audio feeds retraining. The anti-spoofing
  model reports 14 % EER on the hardest public benchmark (ASVspoof5), so we're honest that it isn't perfect alone.
- **"Is it real time?"** On an M1 laptop, about 0.1 s per turn on its GPU and about 1 s on the CPU, running on audio as it arrives.
- **"What's real here?"** The voices (real people, a real clone, real TTS), both models and their outputs, and the
  evaluation. Simulated: call timing, caller-ID data and accounts. Transcripts would come from streaming ASR such as
  NVIDIA Parakeet.
- **"Which models, and can we use them?"** NVIDIA TitaNet-Large (CC BY 4.0) and Spectra-AASIST (Apache-2.0), both
  open. To be confirmed with a Tech Coach per the sprint rules.
- **"Privacy?"** No customer data in the prototype. In production, voiceprints are enrolled with consent and stored as
  embeddings, not audio.
