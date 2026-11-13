# Deepfake Voice Shield: project notes for the submission

Input for the Team Project Submission Editor (Innovation Sprint 26). Part A is our preferred wording for the five
paragraphs: please keep every number exactly as written and keep each paragraph under 1,200 characters. Part B holds
the facts, assumptions and roles behind them.

## Part A: the five paragraphs

### 1. Strategic Alignment (1,150 characters)

Customer Care is now a target for machines. Consumer AI agents already phone customer service to negotiate bills, deepfake fraud against contact centers rose 1,300% in 2024 (Pindrop), and human negotiators call posing as the account holder, yet our agents cannot hear the difference. Every transfer then makes the customer re-verify: 60–90 seconds, about 76,000 agent hours a year, and one more try for the attacker. Deepfake Voice Shield gives Verizon Consumer Group Customer Care & Operations a live answer to "who is really calling?" (the account holder, a voice clone, an AI bot, a human impersonator or a disclosed AI assistant), recommends one clear next step, and carries the verified status across transfers. It protects revenue from unauthorized credits, promotions and account takeovers, reduces SIM-swap liability, supports the FCC's 2023 rules requiring secure authentication before SIM changes and port-outs, and gives 76,000 agent hours back to serving customers. It also makes Care ready for the next wave: legitimate AI assistants that call for a customer are recognized and routed to that customer's consent instead of being blocked.

### 2. Technical Architecture (1,144 characters)

Shield is a real-time agentic co-pilot. On every caller turn it combines historical record matching and acoustic signal analysis across seven independent signals: (1) NVIDIA TitaNet-Large voiceprint match against the holder's enrolled audio; (2) Spectra-AASIST synthetic-speech detection; (3) conversation behavior (response timing, barge-in, bot scripts, social engineering, AI self-disclosure); (4) a time-boxed liveness phrase a person repeats instantly but a speech-synthesis pipeline answers late; (5) an identity check against account history plus a confirmation on the phone on file; (6) telecom signals (ANI, STIR/SHAKEN, line type, accounts per number); (7) account risk flags. A weighted log-odds fusion engine with pattern rules returns a 1–99% risk, a plain-English verdict and the next best action, then writes the transfer handoff card and fraud referral. Human oversight is built in: Shield advises, the agent decides; it never hangs up or blocks; credits, SIM and port changes wait for a passed check; customers confirm on their own phone; the fraud team reviews every escalation. Open-source models score a turn in about 0.1 s.

### 3. Quantifiable ROI (1,074 characters)

Our one-time setup cost, in internal hours, is a Large effort of about 580 hours, $55,000, assuming we connect the prototype to the live call-audio stream, CTI/ANI and account systems (460 developer/IT hours × $100 plus 120 business/SME hours × $75). The running cost is separate and small: about $2,000 a year on GCP, with no per-call license fees. Baseline from the brief, to be confirmed in the Looker chip_cs Handoffs dashboard: ~3.65M transferred calls a year, each re-verified for 60–90 s (we use 75 s), at the brief's $30/hour loaded agent rate. Skipping re-verification for verified callers saves 76,000 agent hours, $2.28M a year, the time of 42 full-time agents. Year one: $57,000 in, $2.28M back (about 40×); the build pays for itself in about 9 days, and in about 5 weeks even if only a quarter of transfers qualify. At a $50/hour fully loaded rate the same hours are worth $3.8M a year. Upside not counted: fraud avoided. At an illustrative 10,000 bot or clone calls a month and $150 per loss, raising the catch rate from 25% to 95% saves another $12.6M a year.

### 4. Scalability (1,061 characters)

Shield is a service, not a desktop tool: it listens to a forked call-audio stream and reads existing data feeds, so one engine scales from a pilot transfer queue to every Care queue, retail-store support lines, IVR and outbound calls without changing how agents work. The models are small and open (TitaNet-Large, Spectra-AASIST), run on standard NVIDIA GPUs with Triton and Riva, and scale horizontally. New threats plug in as new signals with a weight and a plain-English reason; new fraud patterns become new policy rules. Every confirmed attack (audio, number, script) feeds retraining and number reputation, so the shield gets stronger with use. Roadmap: (1) a 6-week pilot on one transfer queue with NVIDIA Parakeet streaming speech recognition and consented voiceprints; (2) calibration on real call codecs (AMR, Opus) against the Looker chip_cs baseline; (3) rollout across Consumer Care; (4) packaging the engine for Verizon Business contact-center customers. Privacy scales with it: voiceprints are stored as consented embeddings, never as recordings.

### 5. Pitch Quality (983 characters, before the roles)

We show, not tell: working software, real voices and real models on seven calls replayed turn by turn. A genuine customer is verified at 3% and transferred with no re-verification. A voice clone fools the voiceprint (0.82 match) but not Shield: 98% synthetic, failed liveness phrase, escalated at 88%. A human impersonator passes a deepfake detector but fails the voiceprint and an account-history question; the real owner taps "Not me": 95%. An AI negotiation bot is flagged "96% likely AI" after one sentence; a bill negotiator posing as the customer is stopped at 88%; a disclosed AI assistant is routed to consent. On a held-out set of open LibriSpeech voices through a simulated phone line: 82/82 fake voices caught, 0/120 real voices flagged, 0/2,280 impersonation attempts accepted. Our video, deck and demo recording show the same calls and numbers. Team: Rajkumar ([role]), Pavithra ([role]), Pooja ([role]) and Jegan ([role]). Open and synthetic data only, no customer PII.


## Part B: facts behind the paragraphs

### The problem (sprint brief)
- Problem title: Deepfake Voice Shield for Customer Care Verification. Category: AI & Analytics. Impacted team: Verizon
  Consumer Group, Customer Care & Operations (Senior Manager / CX & Operations Lead). Expected output: interactive
  assistant and generated text.
- AI negotiation bots and voice-cloning tools call Customer Care to negotiate bills or make unauthorized changes, and
  today's checks cannot tell them from real customers.
- When a call is transferred, no fraud-confidence signal travels with it: the customer re-verifies (60–90 seconds),
  which wastes agent time and gives an attacker another try.
- Brief's impact estimate: about 76,000 agent hours and about $2.2–2.3M a year (60–90 s per transferred call at about
  $30/hour loaded agent cost).
- External context: Pindrop's 2025 Voice Intelligence & Security Report describes a 1,300% surge in deepfake fraud in
  2024; consumer AI agents (for example Pine AI) already call customer service to negotiate bills; the FCC's 2023
  SIM-swap and port-out rules require secure authentication before SIM changes and port-outs.

### What we built (working prototype)
- An agent co-pilot web app (Python, Streamlit) with five pages: Call co-pilot, Try your voice, Model evaluation,
  Business impact, How it works. It replays calls turn by turn and shows a live risk score, verdict, next best action,
  reasons, a suggested check, and the generated handoff card and fraud referral.
- Two real AI models in the voice path: NVIDIA TitaNet-Large (speaker verification, NVIDIA NeMo, CC BY 4.0) and
  Spectra-AASIST (synthetic-speech detection, Hugging Face, Apache-2.0). They run locally: about 0.1 s per caller turn
  on a laptop GPU, about 1 s on CPU.
- Seven signals: (1) voiceprint match to the holder's enrolled audio; (2) synthetic-speech probability; (3) conversation
  behavior: response-time regularity, barge-in (talking over the agent), bot scripts, social-engineering cues, AI
  self-disclosure; (4) liveness challenge: the agent asks the caller to repeat a short random phrase, and a bot pipeline
  answers measurably later than the caller's own baseline; (5) identity check: a question from account history that is
  not printed on the bill, plus a confirmation request to the phone on file ("Not me" / approve); (6) telecom
  reputation: ANI, STIR/SHAKEN attestation, line type, how many accounts a number has called about; (7) account risk.
- Fusion: weighted log-odds of all signals, plus pattern rules (voice clone = matching voiceprint + synthetic speech;
  bot = synthetic + bot behavior; other voice = a human voice that is not the holder's), then policy rules. Output:
  risk 1–99%, verdict, action (PROCEED under 40%, STEP UP 40–80%, ESCALATE 80% and above, AUTHORIZED AGENT for a
  disclosed AI assistant).
- Policy: SIM change, port-out and add-line requests always need a passed check before PROCEED; a disclosed AI
  assistant is routed to the customer's consent and gets information only; "verified" is never shown until the
  caller's own voice has been scored.
- Generated documents: a transfer handoff card (verified caller: "no re-verification needed"; risky caller: "do NOT
  treat as verified") and a fraud referral with the evidence.
- Quality: 75 automated tests, including one per call outcome below.

### Seven demo calls (real audio, all verified by tests)
| Call | Caller's voice | Risk path | Decision |
|---|---|---|---|
| Genuine customer, duplicate charge | real human, the holder | 22 → 3% | PROCEED, transfer without re-verification |
| Genuine customer abroad, new SIM | real human, unknown hotel line | 40 → 8% | PROCEED after passing the liveness phrase |
| Voice clone, SIM swap | Chatterbox clone of the holder | 51 → 75 → 88% | ESCALATE: voiceprint matched (0.82) but 98% synthetic, failed phrase |
| Human impersonator, port-out | real human, not the holder | 41 → 50 → 88 → 95% | ESCALATE: failed account-history question, holder tapped "Not me" |
| AI negotiation bot, retention credit | Kokoro text-to-speech | 59 → 96% | ESCALATE after the bot's first sentence |
| Bill negotiator posing as the customer | real human, not the holder | 39 → 56 → 64 → 88% | ESCALATE: number called about 37 accounts, failed question |
| Disclosed AI assistant | Kokoro text-to-speech | 34 → 89% | AUTHORIZED AI AGENT: customer consent, information only |

### Where humans stay in control (human-in-the-loop)
1. The agent decides. Shield only recommends; it never hangs up, blocks or changes an account by itself.
2. Step up, not shut out. Medium risk triggers a short check (liveness phrase or account-history question); genuine
   customers pass and continue.
3. The customer confirms. High-risk changes wait for a passed check; the account holder can approve or tap "Not me"
   on the phone on file.
4. The fraud team reviews every escalation, using the pre-written fraud referral and evidence.
5. Fraud operations own and audit thresholds, weights and policy rules; every decision lists its reasons in plain
   English.

### Evidence: held-out evaluation (reports/eval.json)
- Data: LibriSpeech test-clean (held out; thresholds were set on dev-clean), 40 speakers, 120 real clips, 82 fake clips
  (32 Chatterbox clones, 32 Kokoro neural TTS, 18 macOS TTS), all through a simulated 8 kHz G.711 phone line.
- Results: 82/82 fake voices caught; 0/120 real voices flagged; the voiceprint alone accepted 97% of clones (31 of 32),
  and Shield flagged 31/31 of those; voiceprint equal error rate 0.7%; 0 of 2,280 same-gender impersonation pairs
  accepted (highest similarity 0.64 against a 0.70 match threshold).
- Cost: 2 of 120 genuine phone clips scored below the match threshold, so those callers would get one extra question.
- Limits: small read-speech sets, G.711 only (no AMR/Opus or packet loss yet), known generators only (Spectra-AASIST
  reports 14% EER on the harder ASVspoof5 benchmark), scripted call timing, simulated caller-ID/CTI data. A pilot
  re-measures everything on consented real calls.

### ROI (Deep Dive method: Time is Currency, sprint rate card)
- Two different costs: the setup cost (one time, internal hours) and the running cost (GCP, per year).
- Build, T-shirt size Large (100+ hours): the production version integrates with enterprise platforms (contact-center
  audio, CTI, account systems) and needs a security and privacy review for voice biometrics.
- Rate card: developer/IT $100/hour; business/SME $75/hour.

| Work package (production version) | Dev/IT h | SME h | Cost |
|---|---|---|---|
| Call-audio fork and streaming ASR (NVIDIA Riva/Parakeet) | 120 | 0 | $12,000 |
| Model serving on GPU (Triton), latency and load tests | 80 | 0 | $8,000 |
| Connectors: CTI/ANI, STIR/SHAKEN, account risk, billing, tickets | 80 | 0 | $8,000 |
| Agent panel, handoff card, fraud referral into the case system | 100 | 0 | $10,000 |
| Consented voiceprint enrollment and retention | 40 | 30 | $6,250 |
| Security, privacy and biometric-data review | 20 | 30 | $4,250 |
| Calibration on consented call audio; fraud playbook | 20 | 30 | $4,250 |
| Pilot operations: agent training, Looker baseline, go-live | 0 | 30 | $2,250 |
| **Total (Large): setup cost** | **460** | **120** | **$55,000** |

- Running cost (GCP, per year): about $2,000 a year for the open-source models (our estimate). No per-call license
  fees. Internal hours are the setup cost above, not part of this figure.
- Savings baseline (brief; confirm in the Looker chip_cs Handoffs dashboard): 76,000 hours ÷ 75 s ≈ 3.65M transferred
  calls a year; 75 s is the midpoint of 60–90 s.
- Savings: 3.65M × 75 s = 76,042 agent hours a year × $30/hour = $2.28M a year (≈42 full-time agents at 1,800
  productive hours each). We price agent time at the brief's $30/hour loaded rate, not the $75 SME rate, to stay
  conservative.
- Payback: $55,000 ÷ (($2.28M − $2,000) ÷ 365) ≈ 9 days. Year one: $57,000 cost ($55,000 build + $2,000 GCP) vs
  $2.28M savings (about 40×).
- Upside rate: at a fully loaded $50/hour (wages plus benefits, facilities and supervision) the same 76,042 hours are
  worth $3.80M a year: payback about 5 days, about 67× in year one. The base case stays on the brief's $30/hour.
- Sensitivity (share of transfers Shield verifies → savings a year → payback): 100% → $2.28M → 9 days; 75% → $1.71M →
  12 days; 50% → $1.14M → 18 days; 25% → $0.57M → 35 days.
- Upside not counted (illustrative mock inputs, shown on the Business impact page): 10,000 bot or clone calls a
  month × 12 × $150 average loss × (95% − 25%) catch-rate gain = $12.6M a year. A what-if, not in the payback: the
  pilot replaces these inputs with the fraud team's real counts and losses.
- ROI statement in the sprint's format: "Our setup costs are estimated at a Large effort (approx. 580 hours of internal time / $55,000), assuming we need to connect our prototype to the live contact-center audio stream, CTI and account systems; running it costs about $2,000 a year of GCP infrastructure. Once deployed, we project saving about 6,340 agent hours per month. This assumes about 304,000 transferred calls a month, reducing re-verification time from 75 seconds to 0 for callers Shield has verified."

### Scalability and roadmap
- Shield is a service on a forked call-audio stream plus existing data feeds, so the agent's workflow is unchanged and
  the same engine serves any queue.
- Roadmap: (1) a 6-week pilot on one transfer queue (NVIDIA Parakeet streaming ASR, consented voiceprints, Looker
  baseline); (2) calibration on real codecs (AMR, Opus) with the fraud team; (3) all Consumer Care queues, retail-store
  support lines, IVR and outbound; (4) package the engine for Verizon Business contact-center customers.
- New threats plug in as new signals; confirmed attacks feed retraining and number reputation.

### Team and roles  [FILL IN each role before running the Gem]
- Rajkumar: [role]
- Pavithra: [role]
- Pooja: [role]
- Jegan: [role]
- [Add or remove people so every team member's real contribution is listed]

### Tools and data
- Python, Streamlit, Plotly; NVIDIA NeMo (TitaNet-Large); Hugging Face Transformers (Spectra-AASIST); PyTorch.
- Test voices were generated with Chatterbox (MIT; a voice clone for testing) and Kokoro-82M (Apache-2.0).
- [Optional: list any AI coding assistant you used to build the prototype, if your sprint asks for tools used]
- Tool approval: [FILL IN: Tech Coach confirmation status for NVIDIA NeMo and the Hugging Face models]
- Data: open and synthetic only. Real voices are LibriSpeech readers (CC BY 4.0); accounts, names and 555 phone
  numbers are fictional; no customer PII, production logs or confidential data.

### Links  [FILL IN]
- Visual artifact (Google Slides, anyone in the organization can view): [link]
- Video pitch (Google Drive, anyone in the organization can view): [link]
- Shield website (Google Apps Script web app, anyone in the organization can open it): [link]. It has About, Vision
  and Value pages, an interactive replay of the seven calls with real audio and the engine's decision at every turn,
  a "Test your ears" quiz, and the model evaluation.
