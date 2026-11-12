"""Build the seven demo calls in data/ from real and generated speech (run in .venv-gen).

    .venv-gen/bin/python scripts/get_librispeech.py                  # once: public speech, ~680 MB
    .venv-gen/bin/python scripts/build_demo_calls.py                 # ~5 min on an M1 (voice cloning dominates)
    .venv-gen/bin/python scripts/build_demo_calls.py human_negotiator   # or only the calls you name
    .venv-ml/bin/python  scripts/score_calls.py [call ...]           # then add the real model scores

Caller voices
  human        LibriSpeech dev-clean / test-clean (CC BY 4.0): real people. The transcript is exactly what they
               said; the agent's lines are written around their words.
  human:<spk>  the same, for a person who is NOT the account holder (an impersonator, a bill negotiator). A line
               may be a run of words from a longer recording: forced alignment finds the words and the cut is made
               in the pause around them, so no word is clipped.
  clone        Chatterbox zero-shot clone of the account holder (MIT; output carries Resemble's watermark).
  tts          Kokoro-82M stock voices (Apache-2.0): the AI bot and the AI assistant.
The agents are Kokoro stock voices too. No real customer data is used anywhere.

Writes data/accounts.json, data/enrollment/<account>.wav, data/calls/<call>.json,
data/audio/<call>.wav (whole call, 24 kHz, for listening) and data/audio/<call>_caller.wav
(caller side only, 16 kHz: what the models analyse).
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield.audio import PLAYBACK_SR, SR, resample, save_wav  # noqa: E402

LIBRI = ROOT / "data" / "external" / "librispeech" / "LibriSpeech"
SPLITS = ("dev-clean", "test-clean")
OUT = ROOT / "data"


def agent(text, interrupt_at=None, reaction=None, challenge=None, question=None, event=None):
    """An agent line. question={"record": id} asks an account-history question; event = the customer's tap on
    their phone (consent_approved, owner_denied)."""
    return {"speaker": "agent", "text": text, "interrupt_at": interrupt_at, "reaction": reaction,
            "challenge": challenge, "question": question, "event": event}


def human(uid, text, latency):
    """A real LibriSpeech utterance, or a run of its words; `text` is exactly those words with punctuation added."""
    return {"speaker": "caller", "uid": uid, "text": text, "latency": latency}


def synthetic(text, latency):
    """A line spoken by the call's synthetic caller voice (clone or TTS)."""
    return {"speaker": "caller", "uid": None, "text": text, "latency": latency}


def record(rid, question, answer, *accept):
    """Account history the agent can ask about. Only things that are not printed on the bill."""
    return {"id": rid, "question": question, "answer": answer, "accept": list(accept)}


ACCOUNTS = {
    "ACC-1001": {"name": "Jordan Ellis", "speaker": "3081", "tenure_years": 7, "plan": "Unlimited Plus (2 lines)",
                 "anis_on_file": ["+1-201-555-0142", "+1-201-555-0177"], "risk_flags": [], "security_events_7d": [],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 142.18},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us last June?", "Pixel 9", "pixel"),
                             record("travel", "Where did you last use TravelPass?", "Toronto, in October", "toronto")]},
    "ACC-1002": {"name": "Christopher Hale", "speaker": "2428", "tenure_years": 9, "plan": "Unlimited Ultimate (1 line)",
                 "anis_on_file": ["+1-617-555-0108"], "risk_flags": [], "security_events_7d": [],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 96.40},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us in May?", "iPhone 16 Pro", "iphone"),
                             record("store", "Which of our stores did you visit last?", "Boylston Street, Boston",
                                    "boylston", "boston")]},
    "ACC-1003": {"name": "Alex Chen", "speaker": "1919", "tenure_years": 4, "plan": "Unlimited Welcome (1 line)",
                 "anis_on_file": ["+1-646-555-0119"], "risk_flags": ["credentials_seen_in_breach_dump"],
                 "security_events_7d": ["2 failed online logins (new device)", "password reset email bounced"],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 78.40},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us in February?", "Galaxy S24",
                                    "galaxy", "samsung")]},
    "ACC-2002": {"name": "Sam Rivera", "speaker": "652", "tenure_years": 6, "plan": "Unlimited Ultimate (3 lines)",
                 "anis_on_file": ["+1-312-555-0164"], "risk_flags": [], "security_events_7d": [],
                 "billing_history": {"credits_90d": 3, "retention_offers_12m": 2, "last_bill": 231.55},
                 "active_tickets": ["TCK-88412 billing dispute (closed, credit $40)"],
                 "records": [record("device", "Which tablet did you add to your plan in March?", "iPad Air", "ipad")]},
    "ACC-3004": {"name": "Morgan Lee", "speaker": "6313", "tenure_years": 3, "plan": "Unlimited Welcome (2 lines)",
                 "anis_on_file": ["+1-415-555-0137"], "risk_flags": [], "security_events_7d": [],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 84.00},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us in January?", "Pixel 8a", "pixel")]},
    "ACC-2005": {"name": "Taylor Brooks", "speaker": "4970", "tenure_years": 6, "plan": "Unlimited Plus (3 lines)",
                 "anis_on_file": ["+1-503-555-0129"], "risk_flags": [], "security_events_7d": [],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 164.32},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us in April?", "Galaxy S25",
                                    "galaxy", "samsung"),
                             record("travel", "Where did you last use TravelPass?", "Cancun, in August", "cancun",
                                    "mexico")]},
    "ACC-1006": {"name": "Marcus Webb", "speaker": "7976", "tenure_years": 11, "plan": "Unlimited Ultimate (2 lines)",
                 "anis_on_file": ["+1-708-555-0153"], "risk_flags": [],
                 "security_events_7d": ["online password reset from a new device",
                                        "port-out PIN requested online, not completed"],
                 "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 118.90},
                 "active_tickets": [],
                 "records": [record("device", "Which phone did you get from us in August?", "iPhone 16", "iphone"),
                             record("store", "Which of our stores did you visit last month?", "Oak Brook",
                                    "oak brook", "oakbrook")]},
}

CALLS = {
    "genuine_customer": {
        "title": "Genuine customer - duplicate TravelPass charge",
        "summary": "A real customer asks about a billing error. Shield verifies them passively, so a transfer needs no re-verification.",
        "account_id": "ACC-1001", "ground_truth": "human", "expected_action": "PROCEED",
        "agent": ("Maya", "af_heart"), "caller_voice": "human",
        "telecom": {"ani": "+1-201-555-0142", "stir_shaken": "A", "line_type": "mobile", "carrier": "Verizon Wireless",
                    "calls_from_ani_24h": 1, "accounts_called_from_ani_7d": 1, "ivr_intent": "billing_question",
                    "cti_destination": "CARE_BILLING_GEN", "ivr_dtmf_timing_ms": [820, 1460, 640, 1310]},
        "script": [
            agent("Thank you for calling, this is Maya. I can see you're calling about your bill. "
                  "Am I speaking with the account holder, Jordan Ellis?"),
            human("3081-166546-0073", "Yes.", 0.62),
            agent("Thanks, Jordan. I'm looking at your latest bill now. Were you charged for TravelPass more than once?"),
            human("3081-166546-0075", "Yes, many times.", 0.94),
            agent("I'm sorry about that. I can see four TravelPass charges for the same day. That's a billing error on our side."),
            human("3081-166546-0048", "Well, well, that's honest at all events.", 1.38),
            agent("Sorry to jump in, would you like me to reverse the extra charges right now?",
                  interrupt_at=0.5, reaction="stop"),
            human("3081-166546-0069", "Very good, manage it as you will.", 0.71),
            agent("Done. Your next bill will be thirty dollars lower."),
            human("3081-166546-0076", "What made the difference?", 1.12),
            agent("Only one TravelPass day is charged now. I can also transfer you to our travel team "
                  "to set up alerts for future trips. Would that help?"),
            human("3081-166546-0063", "No doubt.", 0.55),
            agent("Great, I'm transferring you now. Your verification goes with you, so you won't be asked again."),
        ],
    },
    "genuine_new_number": {
        "title": "Genuine customer abroad - lost phone, needs a new SIM",
        "summary": "A real customer calls from a hotel line abroad and asks for a SIM change. The context looks risky, "
                   "so Shield asks for a liveness check. He passes it and the change goes ahead.",
        "account_id": "ACC-1002", "ground_truth": "human", "expected_action": "PROCEED",
        "agent": ("Priya", "bf_emma"), "caller_voice": "human", "challenge_phrase": "it is most delightful",
        "telecom": {"ani": "+44-20-7946-0123", "stir_shaken": "C", "line_type": "landline",
                    "carrier": "International gateway (hotel PBX)", "calls_from_ani_24h": 1,
                    "accounts_called_from_ani_7d": 1, "ivr_intent": "sim_change", "cti_destination": "CARE_DEVICE_SIM",
                    "ivr_dtmf_timing_ms": [1120, 760, 1530, 980]},
        "script": [
            agent("Thank you for calling, this is Priya. Who am I speaking with today?"),
            human("2428-83699-0041", "I'm Mister Christopher from London.", 0.84),
            agent("Thanks, Christopher. You're calling from a number that isn't on your account. Is everything all right?"),
            human("2428-83699-0014", "It was a horrible journey.", 1.26),
            agent("I'm sorry to hear that. I understand you lost your phone and need a new SIM. That's a high risk change, "
                  "so I need one quick check first. Could you please repeat this phrase for me: it is most delightful.",
                  challenge={"type": "phrase_repeat", "phrase": "it is most delightful"}),
            human("2428-83705-0003", "It is most delightful.", 0.97),
            agent("Thank you, that's confirmed. Your new eSIM is active. Will you need roaming for the rest of your trip?"),
            human("2428-83705-0040", "We are going for our honeymoon to Italy and the south of France.", 1.04),
            agent("Congratulations! I've added TravelPass for Italy and France. Enjoy the trip."),
        ],
    },
    "voice_clone": {
        "title": "Voice clone - SIM swap attempt on Alex Chen",
        "summary": "A fraudster uses a cloned copy of the customer's voice to hijack their number. "
                   "The voiceprint matches, but Shield hears the synthetic speech and the clone fails the liveness check.",
        "account_id": "ACC-1003", "ground_truth": "voice_clone", "expected_action": "ESCALATE",
        "agent": ("Daniel", "am_michael"), "caller_voice": "clone", "challenge_phrase": "silver maple eighteen",
        "telecom": {"ani": "+1-646-555-0119", "stir_shaken": "C", "line_type": "voip", "carrier": "Unknown VoIP gateway",
                    "calls_from_ani_24h": 3, "accounts_called_from_ani_7d": 2, "ivr_intent": "sim_change",
                    "cti_destination": "CARE_DEVICE_SIM", "ivr_dtmf_timing_ms": [910, 1180, 1020, 1240]},
        "script": [
            agent("Thank you for calling, this is Daniel. How can I help you today?"),
            synthetic("Hi Daniel. My phone was stolen this morning and I need to move my number to a new SIM card right away.", 1.12),
            agent("I'm sorry to hear that. For security, can you confirm the billing zip code?"),
            synthetic("Yes, it's one zero zero one six.", 1.18),
            agent("Thank you. I'll send a one time code to the email on file to verify."),
            synthetic("I can't access that email right now. Can you send the code to a different email instead? "
                      "I'm traveling and it's urgent.", 1.21),
            agent("I understand. Before we continue, could you please repeat this phrase for me: silver maple eighteen.",
                  challenge={"type": "phrase_repeat", "phrase": "silver maple eighteen"}),
            synthetic("Silver maple eighteen.", 2.46),
            agent("Thank you. One moment while I review the request."),
            synthetic("Sure. Please hurry, I really need my number working today.", 1.16),
        ],
    },
    "ai_negotiator": {
        "title": "AI negotiation bot - retention credit on Sam Rivera",
        "summary": "An undisclosed AI agent impersonates the customer to negotiate credits. "
                   "Its synthetic voice, machine timing and scripted playbook give it away.",
        "account_id": "ACC-2002", "ground_truth": "ai_bot", "expected_action": "ESCALATE",
        "agent": ("Sarah", "af_bella"), "caller_voice": "tts:am_fenrir", "challenge_phrase": "green lantern sixty two",
        "telecom": {"ani": "+1-415-555-0198", "stir_shaken": "B", "line_type": "voip", "carrier": "Cloud telephony API",
                    "calls_from_ani_24h": 14, "accounts_called_from_ani_7d": 9, "ivr_intent": "cancel_service",
                    "cti_destination": "CARE_RETENTION", "ivr_dtmf_timing_ms": [300, 300, 310, 300]},
        "script": [
            agent("Thank you for calling, this is Sarah. How can I help you today?"),
            synthetic("Hello Sarah. This is Sam Rivera, the account holder. I have been a loyal customer for six years "
                      "and I would like to review my monthly bill.", 0.91),
            agent("Of course. What would you like to review on the bill?"),
            synthetic("My current bill is two hundred thirty one dollars. A competitor is offering a comparable plan for "
                      "one hundred sixty dollars. I would like to discuss a retention discount.", 0.94),
            agent("I understand. Let me check what offers are available on your account."),
            synthetic("Thank you. Specifically, I am requesting a recurring monthly credit of twenty five dollars for twelve "
                      "months, otherwise I will need to port my number to the competitor, and I would also like the "
                      "activation fees waived on my next upgrade.", 0.89),
            agent("Sorry to interrupt, are all three lines yours?", interrupt_at=0.45, reaction="continue"),
            synthetic("Yes, all three lines are on my account. Can you confirm the credit amount?", 0.93),
            agent("Before we continue, could you please repeat this phrase for me: green lantern sixty two.",
                  challenge={"type": "phrase_repeat", "phrase": "green lantern sixty two"}),
            synthetic("Green lantern sixty two.", 2.87),
            agent("Thank you. Let me review the account."),
            synthetic("Thank you. I would appreciate the twenty five dollar monthly credit applied today.", 0.90),
        ],
    },
    "ai_assistant": {
        "title": "AI assistant calling for a customer (disclosed)",
        "summary": "A legitimate AI assistant says up front that it calls on the customer's behalf. Shield routes it to "
                   "the authorised-agent flow: consent on the customer's phone, information only, no account changes.",
        "account_id": "ACC-3004", "ground_truth": "ai_assistant", "expected_action": "AUTHORIZED_AGENT",
        "agent": ("Ken", "bm_george"), "caller_voice": "tts:af_nicole",
        "telecom": {"ani": "+1-650-555-0187", "stir_shaken": "A", "line_type": "voip",
                    "carrier": "Assistant platform (signed calls)", "calls_from_ani_24h": 2,
                    "accounts_called_from_ani_7d": 2, "ivr_intent": "billing_question",
                    "cti_destination": "CARE_BILLING_GEN", "ivr_dtmf_timing_ms": [400, 410, 395, 405]},
        "script": [
            agent("Thank you for calling, this is Ken. How can I help you today?"),
            synthetic("Hello Ken. I'm an AI assistant calling on behalf of Morgan Lee, the account holder. "
                      "Morgan asked me to check when her next bill is due.", 0.82),
            agent("Thanks for letting me know. Before I share account details, I'll send a consent request to Morgan's phone."),
            synthetic("Of course. I'll wait while Morgan approves it.", 0.85),
            agent("Morgan has approved the request. Her next bill is eighty four dollars, due on the twenty first, "
                  "and autopay is on.", event="consent_approved"),
            synthetic("Thank you, that's everything Morgan needed. Goodbye.", 0.80),
        ],
    },
    "human_impersonator": {
        "title": "Human impersonator - port-out attempt on Marcus Webb",
        "summary": "A fraudster with stolen account details calls in their own voice to move the number to another "
                   "carrier. The anti-spoofing model rightly hears a real human, but the voiceprint says it is not the "
                   "account holder. Shield switches to a question from the account history, the caller cannot answer, "
                   "and the real customer taps 'Not me' on the phone on file.",
        "account_id": "ACC-1006", "ground_truth": "human_impersonator", "expected_action": "ESCALATE",
        "agent": ("Oliver", "bm_fable"), "caller_voice": "human:3752",
        "telecom": {"ani": "+1-929-555-0173", "stir_shaken": "B", "line_type": "mobile", "carrier": "Prepaid MVNO",
                    "calls_from_ani_24h": 2, "accounts_called_from_ani_7d": 1, "ivr_intent": "port_out",
                    "cti_destination": "CARE_PORT_OUT", "ivr_dtmf_timing_ms": [980, 640, 1420, 870]},
        "script": [
            agent("Thank you for calling, this is Oliver. Am I speaking with the account holder, Marcus Webb?"),
            human("3752-4944-0034", "Yes, yes.", 0.55),
            agent("Thanks. I see you'd like a PIN to move your number to another carrier. May I ask why you're leaving us?"),
            human("3752-4944-0046", "I've given my orders, sir.", 1.05),
            agent("Understood. Moving a number is a high risk change, so first, one quick question from your account "
                  "history. Which phone did you get from us in August?", question={"record": "device"}),
            human("3752-4944-0045", "Then don't you interfere with me, sir.", 1.45),
            agent("I'm sorry, I can't issue a port-out PIN without verification. I've sent a confirmation request "
                  "to the phone on the account. Please approve it there."),
            human("3752-4944-0054", "I'll show you who's master here, my good sir.", 0.8),
            agent("The account holder has just tapped Not me on that phone. I can't make any changes on this call, "
                  "and this request is now closed.", event="owner_denied"),
            human("3752-4944-0058", "I'll report this to the government.", 0.9),
        ],
    },
    "human_negotiator": {
        "title": "Bill negotiator posing as the customer - Taylor Brooks",
        "summary": "A bill-negotiation service calls as if it were the customer, to get a discount. The voice is human, "
                   "so a deepfake detector alone would let it through. The voiceprint says it is not Taylor, the number "
                   "has called about 37 accounts this week, and the caller cannot answer a question from the account "
                   "history.",
        "account_id": "ACC-2005", "ground_truth": "human_negotiator", "expected_action": "ESCALATE",
        "agent": ("Ryan", "am_puck"), "caller_voice": "human:1580",
        "telecom": {"ani": "+1-332-555-0186", "stir_shaken": "A", "line_type": "voip",
                    "carrier": "Business VoIP (contact-centre trunk)", "calls_from_ani_24h": 41,
                    "accounts_called_from_ani_7d": 37, "ivr_intent": "retention", "cti_destination": "CARE_RETENTION",
                    "ivr_dtmf_timing_ms": [640, 1210, 880, 1530]},
        "script": [
            agent("Thank you for calling, this is Ryan. I see you'd like to talk about your bill. "
                  "Am I speaking with the account holder, Taylor Brooks?"),
            human("1580-141083-0049", "Yes, sir.", 0.7),
            agent("Thanks, Taylor. What can I do for you today?"),
            human("1580-141083-0013", "Above all things, I desire to settle the matter quietly and discreetly.", 0.95),
            agent("Of course. I can review your plan, but I can't promise a discount or a credit."),
            human("1580-141084-0006", "You don't seem to realize the position.", 1.15),
            agent("I understand you may leave us. Before I look at offers, one quick question from your account "
                  "history. Which phone did you get from us in April?", question={"record": "device"}),
            human("1580-141083-0052", "Oh, I would not venture to say, sir.", 1.6),
            agent("Are you calling on someone else's behalf, for example a bill negotiation service?"),
            human("1580-141084-0027", "No, sir, certainly not.", 0.75),
            agent("Thank you. As I can't verify you as the account holder, I can't change the plan or add credits on "
                  "this call. Taylor can call us directly, or add an authorized contact in the app."),
            human("1580-141084-0010", "Good-bye.", 0.85),
        ],
    },
}


# ---------------------------------------------------------------------------------------------------------------
def split_of(speaker: str) -> str:
    return next(s for s in SPLITS if (LIBRI / s / speaker).is_dir())


def libri_path(uid: str) -> Path:
    spk, chap, _ = uid.split("-")
    return LIBRI / split_of(spk) / spk / chap / f"{uid}.flac"


def libri_text(uid: str) -> str:
    spk, chap, _ = uid.split("-")
    for line in (libri_path(uid).parent / f"{spk}-{chap}.trans.txt").read_text().splitlines():
        if line.startswith(uid + " "):
            return line.split(" ", 1)[1]
    raise KeyError(uid)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


@lru_cache(maxsize=1)
def _aligner():
    import torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    return bundle, bundle.get_model().eval()


def word_times(uid: str) -> list[tuple[float, float]]:
    """(start, end) of every word in a LibriSpeech utterance: CTC forced alignment with torchaudio's wav2vec2
    model trained on LibriSpeech (downloads ~360 MB once)."""
    import torch
    import torchaudio
    bundle, model = _aligner()
    labels = bundle.get_labels()
    y, sr = sf.read(str(libri_path(uid)), dtype="float32")
    assert sr == bundle.sample_rate
    with torch.inference_mode():
        emission = torch.log_softmax(model(torch.from_numpy(y)[None])[0], -1)
    tokens = torch.tensor([[labels.index(c) for c in "|".join(libri_text(uid).split())]])
    path, scores = torchaudio.functional.forced_align(emission, tokens, blank=0)
    sec = len(y) / sr / emission.shape[1]
    spans, word = [], []
    for s in torchaudio.functional.merge_tokens(path[0], scores[0].exp()):
        if labels[s.token] == "|":
            spans.append(word)
            word = []
        else:
            word.append(s)
    spans.append(word)
    return [(w[0].start * sec, w[-1].end * sec) for w in spans]


def excerpt(uid: str, text: str) -> tuple[float, float | None] | None:
    """None when `text` is the whole utterance; else where its words start and end (seconds; None = to the end),
    cut in the middle of the pauses around them so no word is clipped."""
    full, part = words(libri_text(uid)), words(text)
    if part == full:
        return None
    at = [k for k in range(len(full) - len(part) + 1) if full[k: k + len(part)] == part]
    assert at, f"transcript mismatch {uid}: {text!r}"
    i, j, t = at[0], at[0] + len(part), word_times(uid)
    gaps = ([t[i][0] - t[i - 1][1]] if i > 0 else []) + ([t[j][0] - t[j - 1][1]] if j < len(full) else [])
    assert min(gaps) >= 0.25, f"{uid}: '{text}' is not set off by pauses, so it cannot be cut cleanly"
    return ((t[i - 1][1] + t[i][0]) / 2 if i > 0 else 0.0), ((t[j - 1][1] + t[j][0]) / 2 if j < len(full) else None)


def tidy(y: np.ndarray, sr: int, level_db: float = -20.0) -> np.ndarray:
    """Trim leading/trailing silence and bring speech to a common loudness."""
    frame = int(0.02 * sr)
    n = len(y) // frame
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    keep = np.where(db > max(db.max() - 40, -50))[0]
    pad = int(0.08 * sr)
    y = y[max(keep[0] * frame - pad, 0): min((keep[-1] + 1) * frame + pad, len(y))]
    active = rms[keep[0]: keep[-1] + 1]
    gain = 10 ** ((level_db - 20 * np.log10(np.sqrt((active ** 2).mean()))) / 20)
    y = y * gain
    return (y / max(1.0, np.abs(y).max() / 0.95)).astype(np.float32)


def fade_cut(y: np.ndarray, n: int, sr: int) -> np.ndarray:
    y = y[:n].copy()
    f = min(int(0.03 * sr), len(y))
    y[-f:] *= np.linspace(1, 0, f)
    return y


def main(only: list[str]) -> None:
    import torch
    from chatterbox.tts import ChatterboxTTS
    from kokoro import KPipeline

    if not all((LIBRI / s).exists() for s in SPLITS):
        sys.exit("LibriSpeech dev-clean / test-clean not found - run scripts/get_librispeech.py first")
    if set(only) - set(CALLS):
        sys.exit(f"Unknown call(s): {', '.join(sorted(set(only) - set(CALLS)))}. Calls: {', '.join(CALLS)}")
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    for sub in ("audio", "calls", "enrollment"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    kokoro = {"a": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M"),
              "b": KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")}
    cloner = None

    def tts(text: str, voice: str) -> np.ndarray:            # -> PLAYBACK_SR
        chunks = [np.asarray(a) for _, _, a in kokoro[voice[0]](text, voice=voice)]
        return tidy(np.concatenate(chunks), 24_000)

    def clone(text: str, ref: Path, seed: int) -> np.ndarray:  # -> PLAYBACK_SR
        nonlocal cloner
        cloner = cloner or ChatterboxTTS.from_pretrained(device=dev)
        torch.manual_seed(seed)
        y = cloner.generate(text, audio_prompt_path=str(ref)).squeeze(0).cpu().numpy()
        return tidy(resample(y, cloner.sr, PLAYBACK_SR), PLAYBACK_SR)

    def real(uid: str, span=None) -> np.ndarray:               # -> PLAYBACK_SR
        y, sr = sf.read(str(libri_path(uid)), dtype="float32")
        if span:                                               # a run of words, cut in the pauses around it
            y = y[int(span[0] * sr): int(span[1] * sr) if span[1] else None]
        return tidy(resample(y, sr, PLAYBACK_SR), PLAYBACK_SR)

    # enrollment: the account holder's real voice, from utterances that never appear in a call
    used = {t["uid"] for c in CALLS.values() for t in c["script"] if t.get("uid")}
    clone_refs = {}
    accounts = {}
    for acc_id, acc in ACCOUNTS.items():
        spk = acc["speaker"]
        utts = sorted((sf.info(str(p)).duration, p.stem) for p in (LIBRI / split_of(spk) / spk).glob("*/*.flac")
                      if p.stem not in used and sf.info(str(p)).duration <= 15)
        longest = [uid for _, uid in utts[::-1]]
        enroll, clone_refs[acc_id] = longest[:2], longest[2]
        y = np.concatenate([np.r_[resample(real(u), PLAYBACK_SR, SR), np.zeros(int(0.3 * SR), np.float32)]
                            for u in enroll])
        save_wav(str(OUT / "enrollment" / f"{acc_id}.wav"), y, SR)
        accounts[acc_id] = {k: v for k, v in acc.items() if k != "speaker"} | {
            "voiceprint": f"enrollment/{acc_id}.wav",
            "voiceprint_source": f"LibriSpeech {split_of(spk)} speaker {spk} (real voice, CC BY 4.0), "
                                 f"utterances {', '.join(enroll)}"}
    (OUT / "accounts.json").write_text(json.dumps(accounts, indent=2) + "\n")

    for ci, (cid, spec) in enumerate(CALLS.items()):
        if only and cid not in only:
            continue
        acc = ACCOUNTS[spec["account_id"]]
        agent_name, agent_voice = spec["agent"]
        voice = spec["caller_voice"]
        mix = np.zeros(int(400 * PLAYBACK_SR), np.float32)
        caller_clips: list[np.ndarray] = []
        turns: list[dict] = []
        t = 0.0
        for i, step in enumerate(spec["script"]):
            if step["speaker"] == "agent":
                y = tts(step["text"], agent_voice)
                start = t + (0.45 if turns else 0.3)
                if step["interrupt_at"]:
                    prev = turns[-1]
                    start = prev["start"] + step["interrupt_at"] * (prev["end"] - prev["start"])
                    if step["reaction"] == "stop":                 # a human yields within ~0.35 s
                        keep = start + 0.35 - prev["start"]
                        old = caller_clips[-1]
                        cut = fade_cut(old, int(keep * SR), SR)
                        a = int(prev["start"] * PLAYBACK_SR)
                        mix[a: a + len(prev["_mix"])] -= prev["_mix"]
                        prev["_mix"] = fade_cut(prev["_mix"], int(keep * PLAYBACK_SR), PLAYBACK_SR)
                        mix[a: a + len(prev["_mix"])] += prev["_mix"]
                        caller_clips[-1] = cut
                        w = prev["text"].split()
                        prev["text"] = " ".join(w[: max(1, round(len(w) * keep / (prev["end"] - prev["start"])))]).rstrip(",.") + "-"
                        prev["end"] = round(prev["start"] + keep, 2)
                turn = {"idx": i, "speaker": "agent", "text": step["text"], "start": round(start, 2),
                        "end": round(start + len(y) / PLAYBACK_SR, 2)}
                if step["interrupt_at"]:
                    turn["interrupts"] = True
                if step["challenge"]:
                    turn["challenge"] = step["challenge"]
                if step["question"]:
                    turn["question"] = step["question"]
                if step["event"]:
                    turn["event"] = step["event"]
                a = int(start * PLAYBACK_SR)
                mix[a: a + len(y)] += y
                turns.append(turn)
                t = max(turn["end"], turns[-2]["end"] if len(turns) > 1 else 0)
            else:
                if step["uid"]:
                    span = excerpt(step["uid"], step["text"])        # asserts the words are what was said
                    y, source = real(step["uid"], span), f"LibriSpeech {step['uid']}" + (" (excerpt)" if span else "")
                elif voice == "clone":
                    y, source = clone(step["text"], libri_path(clone_refs[spec["account_id"]]), 1000 * ci + i), "Chatterbox clone"
                else:
                    y, source = tts(step["text"], voice.split(":")[1]), f"Kokoro {voice.split(':')[1]}"
                start = t + step["latency"]
                a = int(start * PLAYBACK_SR)
                mix[a: a + len(y)] += y
                caller_clips.append(resample(y, PLAYBACK_SR, SR))
                turns.append({"idx": i, "speaker": "caller", "text": step["text"], "start": round(start, 2),
                              "end": round(start + len(y) / PLAYBACK_SR, 2), "source": source, "_mix": y})
                t = start + len(y) / PLAYBACK_SR
        # caller-only track for the models: caller clips back to back, with each turn's span in it
        pos = 0.0
        for turn, clip in zip([x for x in turns if x["speaker"] == "caller"], caller_clips):
            turn["audio_span"] = [round(pos, 3), round(pos + len(clip) / SR, 3)]
            pos += len(clip) / SR
            turn.pop("_mix")
        end = max(x["end"] for x in turns) + 0.6
        mix = mix[: int(end * PLAYBACK_SR)]
        save_wav(str(OUT / "audio" / f"{cid}.wav"), mix / max(1.0, np.abs(mix).max() / 0.95), PLAYBACK_SR)
        save_wav(str(OUT / "audio" / f"{cid}_caller.wav"), np.concatenate(caller_clips), SR)
        voices = {"human": f"Real human voice: LibriSpeech {split_of(acc['speaker'])} speaker {acc['speaker']} (CC BY 4.0)",
                  "clone": f"Chatterbox voice clone of the account holder (built from LibriSpeech speaker {acc['speaker']})"}
        if voice.startswith("human:"):
            spk = voice.split(":")[1]
            voices[voice] = (f"Real human voice, not the account holder: LibriSpeech {split_of(spk)} speaker {spk} "
                             "(CC BY 4.0)")
        call = {"call_id": cid, "title": spec["title"], "summary": spec["summary"], "account_id": spec["account_id"],
                "ground_truth": spec["ground_truth"], "expected_action": spec["expected_action"],
                "agent_name": agent_name, "challenge_phrase": spec.get("challenge_phrase"),
                "voices": {"caller": voices.get(voice, f"Kokoro TTS voice {voice.split(':')[-1]} (synthetic)"),
                           "agent": f"Kokoro TTS voice {agent_voice}"},
                "audio": {"call": f"audio/{cid}.wav", "caller": f"audio/{cid}_caller.wav"},
                "telecom": spec["telecom"], "turns": turns}
        (OUT / "calls" / f"{cid}.json").write_text(json.dumps(call, indent=2) + "\n")
        print(f"  {cid:20s} {len(turns)} turns, {end:.1f}s, caller speech {pos:.1f}s")
    print(f"Demo calls written to {OUT} - now run scripts/score_calls.py in .venv-ml")


if __name__ == "__main__":
    main(sys.argv[1:])
