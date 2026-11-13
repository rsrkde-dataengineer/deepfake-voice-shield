"""The product story on the About, Vision and Value pages.

One source for the dashboard (ui/product.py) and the web replay (exported to its page), so the claims match the deck
and the written submission. Plain data, no Streamlit; the money figures come from vshield.impact.
"""

TAGLINE = ("Verify the caller once.", "Protect every transfer.")
TEAM = [("Rajkumar", ""), ("Pavithra", ""), ("Pooja", ""), ("Jegan", "")]   # (name, role); a role shows when filled in

ABOUT_KICKER = "Real-time caller verification for Customer Care"
ABOUT_INTRO = ("Deepfake Voice Shield is a real-time co-pilot for Customer Care agents. On every caller turn it tells "
               "the agent who is really calling (the customer, a voice clone, an AI bot or a human impersonator), "
               "explains why, suggests the one check to ask, and carries a verified caller through every transfer.")

THREATS = [   # (material icon, title, text)
    ("smart_toy", "AI negotiation bots", "Consumer AI agents already phone customer service to negotiate bills. "
     "Undisclosed, they talk their way into credits and promotions."),
    ("theater_comedy", "Voice clones", "Deepfake fraud on contact centers rose 1,300% in 2024. An open-source clone "
     "fooled our own voiceprint check in 31 of 32 tests."),
    ("no_accounts", "Human impersonators", "People with stolen details, and negotiators posing as the account holder. "
     "A deepfake detector passes them: they are human."),
]
TRANSFER_PROBLEM = ("And every transfer makes the customer prove who they are again: 60-90 s per transferred call, about "
                    "76,000 agent hours a year. No fraud-confidence signal travels with the call, so the attacker gets "
                    "a fresh try with every new agent.")

FLOW = [   # how a call moves through Shield
    ("Listen", "Call audio, the live transcript, caller ID and telecom data, account and security context."),
    ("Score seven signals", "Two real AI models on the voice, plus behavior, a liveness check, account history, "
     "telecom and account risk."),
    ("Decide", "Weighted fusion with pattern rules: a 1-99% risk, a plain-English verdict and the next best action."),
    ("Hand off", "The agent decides. Shield writes the transfer handoff card, the fraud referral or a consent request "
     "to the customer's phone."),
]
SIGNALS = [   # (signal, what it asks, how)
    ("Voiceprint", "Does the voice match the account holder's enrolled voice?", "NVIDIA TitaNet-Large"),
    ("Synthetic voice", "Is it a real human, or text-to-speech or a voice clone?", "Spectra-AASIST"),
    ("Conversation behavior", "Machine-steady replies, talking over an interruption, bot scripts, social engineering, "
     "AI self-disclosure.", "Transcript and timing"),
    ("Liveness challenge", "A random phrase to repeat: a person answers at once, a bot pipeline answers late.",
     "Reply delay vs the caller's own baseline"),
    ("Identity check", "A question from the account history, and a confirmation on the phone on file.",
     "Account records, customer's phone"),
    ("Telecom reputation", "STIR/SHAKEN, line type, number not on file, one number calling many accounts.", "CTI / ANI"),
    ("Account risk", "Risk flags, recent security events, high-risk requests such as SIM changes and port-outs.",
     "Account record"),
]
BLIND_SPOTS = [   # (title, text)
    ("A voiceprint alone lets clones in", "The voice clone matched the account holder's voiceprint (0.82). Shield's "
     "anti-spoofing model hears 98% synthetic speech, and the clone fails the liveness phrase: escalated at 88%."),
    ("A deepfake detector alone lets impostors in", "A real human (1-6% synthetic) passes any deepfake detector. The "
     "voiceprint doesn't match, the caller can't answer from the account history, and the real owner taps \"Not me\": "
     "95%."),
]
HUMAN_IN_THE_LOOP = [   # (material icon, title, text)
    ("support_agent", "The agent decides", "Shield shows the risk, the reasons and one recommended action. The agent "
     "chooses. It never hangs up or blocks a caller."),
    ("quiz", "Step up, not shut out", "Doubtful callers get a 10-second check: a phrase to repeat or an "
     "account-history question. Real customers pass."),
    ("phone_iphone", "The customer confirms", "SIM changes, port-outs and new lines wait for a passed check. The holder "
     "can approve, or tap \"Not me\", on their phone."),
    ("gavel", "Fraud team reviews", "Every escalation arrives with a pre-written referral and the evidence. A fraud "
     "analyst makes the final call."),
    ("tune", "Ops owns the rules", "Thresholds, weights and policies are set and audited by fraud operations. Every "
     "decision lists its reasons."),
]
AI_ASSISTANTS = ("Disclosed AI assistants are welcome: Shield routes them to the customer's consent, for information "
                 "only, never credits or SIM changes.")
PROOF = [   # held-out evaluation, phone line (reports/eval.json, test-clean, telephone)
    ("82 / 82", "fake voices caught: clones, neural TTS, older TTS"),
    ("0 / 120", "real voices wrongly flagged as fake"),
    ("31 / 31", "clones that fooled the voiceprint, caught by Shield"),
    ("0 / 2,280", "same-gender impersonation attempts accepted"),
]
PROOF_LIMITS = ("Held-out LibriSpeech voices through a simulated 8 kHz G.711 phone line, thresholds fixed in advance. "
                "Small read-speech sets and known generators only: a pilot re-measures on real calls.")
BUILT_WITH = ("NVIDIA TitaNet-Large (NeMo) and Spectra-AASIST (Hugging Face), open-source models that score a turn in "
              "about 0.1 s; a Python fusion engine with 75 automated tests. Production path: NVIDIA Riva/Parakeet "
              "streaming ASR and Triton GPU serving.")
DATA_ETHICS = ("Open and synthetic data only: LibriSpeech readers (CC BY 4.0), a Chatterbox clone (MIT) and Kokoro "
               "voices (Apache-2.0); fictional accounts and 555 numbers; no customer data. In production, voiceprints "
               "are enrolled with consent and kept as embeddings, never as recordings.")

VISION_HEADLINE = "A trust layer for every call"
VISION_INTRO = ("Today an agent has to guess who is calling, and every transfer starts the guessing again. Shield makes "
                "every call arrive with its verification attached: the customer verified once, a bot caught on its "
                "first sentence, and the customer's own AI assistant welcomed through a front door built for it.")
PILLARS = [   # (material icon, title, text)
    ("verified_user", "Verified once, trusted everywhere", "The verification travels with the call through every "
     "transfer, queue and channel: Care, retail-store support, IVR and outbound. No customer proves who they are "
     "twice."),
    ("smart_toy", "A front door for AI agents", "Customers will soon send AI assistants to call for them. Shield turns "
     "\"is it a bot?\" into \"is it your bot?\": consent on the customer's phone, information-only access, every "
     "action logged."),
    ("autorenew", "Stronger with every attack", "Each confirmed attack (the audio, the number, the script) feeds "
     "retraining and number reputation. New threats plug in as new signals, each with a plain-English reason."),
    ("lock", "Private and accountable by design", "Voiceprints enrolled with consent and kept as embeddings, never "
     "recordings. People make every final call, and every decision lists its reasons."),
]
ROADMAP = [   # (step, text)
    ("Pilot · 6 weeks", "One transfer queue, live call audio, consented voiceprints, NVIDIA Parakeet streaming ASR and "
     "a Looker baseline."),
    ("Calibrate", "Real phone codecs (AMR, Opus); thresholds set with the fraud team and agents."),
    ("Scale", "Every Consumer Care queue, retail-store support, IVR and outbound calls."),
    ("Productize", "Package the engine for Verizon Business contact-center customers."),
]
PILOT_NEEDS = ["One transfer queue and its call-audio fork", "Read-only CTI/ANI, account and risk feeds",
               "A fraud-team partner who owns the thresholds", "Privacy review for consented voiceprints"]
NEXT_IN_PRODUCT = [   # (title, text)
    ("Streaming co-pilot", "NVIDIA Parakeet / Riva speech recognition feeding Shield live, turn by turn."),
    ("Phone-line robustness", "Mobile codecs and packet loss in training and evaluation."),
    ("Language understanding", "An approved LLM for social-engineering cues and a free-text handoff card."),
    ("Connected signals", "Number-reputation feeds, the account risk store and the case system."),
]

VALUE_HEADLINE = "$55k to build. $2.28M a year back."
VALUE_INTRO = ("Verified once, never re-asked: the handoff card travels with every transfer. The base case uses the "
               "brief's own numbers; everything else is upside.")
VALUE_STREAMS = [   # (material icon, title, text)
    ("schedule", "Agent time", "A verified caller skips the 60-90 s re-verification on every transfer: about 76,000 "
     "agent hours a year, the time of 42 full-time agents."),
    ("shield", "Fraud and liability", "Bots and clones stopped before they get credits, promotions, SIM swaps or "
     "port-outs. Escalations reach Fraud with the evidence attached."),
    ("sentiment_satisfied", "Customer experience", "Genuine customers aren't asked the same security questions twice. "
     "A doubtful caller gets one quick check, never a hang-up."),
    ("policy", "Compliance and trust", "Secure authentication before SIM changes and port-outs, as the FCC's 2023 "
     "rules call for, with a reason and an audit trail for every decision."),
]
VALUE_NOTES = ("Base case: the brief's ~76,000 agent hours (~3.65M transferred calls × 75 s) at its $30/hour loaded "
               "rate; confirm in the Looker chip_cs Handoffs dashboard. Build: the sprint's T-shirt sizing (Large) and "
               "rate card ($100/h developer/IT, $75/h business/SME). Fraud: a what-if with mock inputs, not in the "
               "payback; the pilot replaces them with the fraud team's real counts.")
