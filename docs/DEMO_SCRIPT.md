# Demo script (3 minutes)

Before you start, open the dashboard (`streamlit run app.py`), turn the volume up, and keep **Hear each turn** on.
Pick each call in the sidebar and press **Play**; press **Pause** whenever you want to talk over it.

| Time | Screen | What to say |
|---|---|---|
| 0:00 | Call co-pilot, ringing | "AI agents now call care lines to negotiate bills, cloned voices try SIM swaps, and people still pose as customers. Our agents can't hear the difference, and when a call is transferred the next agent starts verification from scratch: about 76,000 agent hours a year." |
| 0:20 | **Genuine customer**, Play | "A real customer. Before anyone speaks, Shield has already scored the caller ID and the account. Once the customer talks, NVIDIA TitaNet matches the voice and the anti-spoofing model hears a live human: risk drops to 3 %." Open **Transfer handoff card**: "On transfer, the next agent sees *verified, no re-verification needed*. That's 60 to 90 seconds saved per transfer." |
| 0:45 | **Voice clone**, Play | "Someone has cloned Alex's voice to steal their number. The voiceprint **matches**: a voiceprint-only system lets this caller in. But the anti-spoofing model says 98 % synthetic." Point at the suggested challenge: "Shield doesn't hang up, it steps up: a random phrase to repeat. The clone needs a fresh text-to-speech round trip and answers 2.1 times slower than its own baseline. Escalate, with a fraud referral already written." |
| 1:20 | **Human impersonator**, Next to turn 4, then Play | "The opposite trap: a real person with stolen details. The anti-spoofing model says *human*, and it's right, so a deepfake detector alone lets this caller in. TitaNet says it isn't Marcus." Point at the card: "A liveness phrase can't help against a live human, so Shield switches to a question from the account history, something that isn't on the bill. The caller can't answer: 88 %. Then the real Marcus taps *Not me* on the phone on file: 95 %." |
| 1:55 | **AI negotiation bot**, then **Bill negotiator** | "Bill negotiation, two ways. An undisclosed AI agent: synthetic voice, machine-steady timing, talks over the agent, escalated after one sentence. And a human negotiation service calling as Taylor: human voice, but the wrong voiceprint, a number that touched 37 accounts this week, and no answer to the account question. No credits for either." |
| 2:20 | **AI assistant** | "Not every third party is fraud. This assistant says what it is, so Shield routes it to consent on the customer's phone: information only, never credits or SIM changes." |
| 2:30 | **Model evaluation** (held-out, phone line) | "Real models on a held-out set, through a phone line: every fake voice caught, none of the 120 real voices flagged, the voiceprint alone fooled by 97 % of clones and Shield catching all of them, and 0 of 2,280 other people accepted as the account holder." |
| 2:45 | **Business impact** | "With the brief's numbers that's about 76,000 agent hours, or 2.3 million dollars a year, before counting the fraud it stops. Next step: a pilot on one transfer queue with live ASR." |

**Optional live moment (30 s; needs the live models):** On **Try your voice**, enroll yourself with the microphone,
then upload a clone of yourself made beforehand with `scripts/clone_voice.py`. "That's my voice, cloned in 30 seconds
with an open-source model. The voiceprint says it's me. Shield says it's a clone."

## Likely questions

- **"What if the customer's partner calls about the account?"** Their voice won't match, so Shield steps up instead
  of approving: one question from the account history, or a confirmation on the account holder's phone. It never
  hangs up, and it never calls someone else's voice "verified".
- **"Can't an impersonator answer the questions with breached data?"** Static answers (zip code, last four digits)
  can be. That's why Shield asks about account history that isn't printed on the bill, and why the phone on file is
  the second check: the impersonator doesn't have it, and the real customer sees the request.
- **"Won't it flag real customers?"** On the held-out set, 0 of 120 real voices were flagged (95 % upper bound about
  2.5 %). And Shield never blocks anyone: the worst case for a real customer is one 10-second phrase to repeat.
- **"What about new deepfake generators?"** That's the arms race. The detector is one of seven signals, the liveness
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
