"""Generate the fully synthetic demo dataset.

    python scripts/generate_demo_data.py

Creates under data/:
  accounts.json               mock account & security context (fake people, 555 numbers)
  enrollment/<ACC>.wav        enrolled voiceprint audio per account (synthetic voices)
  models/voiceprint_ubm.npz   background-speaker statistics for the demo voiceprint model
  audio/<call>.wav            caller-side audio for each call
  calls/<call>.json           transcript with timestamps + CTI/telecom metadata

No real customer data is used anywhere (per sprint rules).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vshield.dsp import SR, save_wav  # noqa: E402
from vshield.synth import HUMAN, TTS, Speaker, say  # noqa: E402
from vshield.signals.voiceprint import embed  # noqa: E402

DATA = ROOT / "data"

# ---- fictional voices ---------------------------------------------------------
JORDAN = Speaker("Jordan Ellis", 125, 1.00, 0.95, (1.00, 1.04, 0.98))   # ACC-1001
ALEX = Speaker("Alex Chen", 205, 1.16, 0.93, (0.97, 1.00, 1.03))       # ACC-1003 (clone target)
SAM = Speaker("Sam Rivera", 175, 1.10, 0.96, (1.03, 0.96, 1.00))       # ACC-2002 (bot target)
BOT_VOICE = Speaker("Negotiator bot", 108, 0.92, 0.94, (1.00, 1.00, 1.05))

ACCOUNTS = {
    "ACC-1001": {
        "name": "Jordan Ellis", "tenure_years": 7, "plan": "Unlimited Plus (2 lines)",
        "anis_on_file": ["+1-201-555-0142", "+1-201-555-0177"], "voiceprint": "enrollment/ACC-1001.wav",
        "risk_flags": [], "security_events_7d": [],
        "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 142.18},
        "active_tickets": [], "pending_requests": [],
    },
    "ACC-1003": {
        "name": "Alex Chen", "tenure_years": 4, "plan": "Unlimited Welcome (1 line)",
        "anis_on_file": ["+1-646-555-0119"], "voiceprint": "enrollment/ACC-1003.wav",
        "risk_flags": ["credentials_seen_in_breach_dump"],
        "security_events_7d": ["2 failed online logins (new device)", "password reset email bounced"],
        "billing_history": {"credits_90d": 0, "retention_offers_12m": 0, "last_bill": 78.40},
        "active_tickets": [], "pending_requests": [],
    },
    "ACC-2002": {
        "name": "Sam Rivera", "tenure_years": 6, "plan": "Unlimited Ultimate (3 lines)",
        "anis_on_file": ["+1-312-555-0164"], "voiceprint": "enrollment/ACC-2002.wav",
        "risk_flags": [],
        "security_events_7d": [],
        "billing_history": {"credits_90d": 3, "retention_offers_12m": 2, "last_bill": 231.55},
        "active_tickets": ["TCK-88412 billing dispute (closed, credit $40)"], "pending_requests": [],
    },
}

# ---- call scripts ----------------------------------------------------------
# ("agent", text, opts) / ("caller", text, latency_seconds)
CALLS = {
    "genuine_customer": {
        "title": "Genuine customer - duplicate TravelPass charge",
        "account_id": "ACC-1001", "speaker": JORDAN, "renderer": HUMAN,
        "ground_truth": "human",
        "telecom": {
            "ani": "+1-201-555-0142", "stir_shaken": "A", "line_type": "mobile",
            "carrier": "Verizon Wireless", "calls_from_ani_24h": 1, "accounts_called_from_ani_7d": 1,
            "ivr_intent": "billing_question", "cti_destination": "CARE_BILLING_GEN",
            "ivr_dtmf_timing_ms": [820, 1460, 640, 1310],
        },
        "script": [
            ("agent", "Thank you for calling Verizon, this is Maya. How can I help you today?", {}),
            ("caller", "Hi, um, yeah, I was looking at my bill and I think I got charged twice for, uh, the TravelPass?", 0.62),
            ("agent", "I'm sorry about that. Let me pull that up. Can you confirm the billing zip code on the account?", {}),
            ("caller", "Sure, it's zero seven three zero two.", 0.41),
            ("agent", "Thank you. I do see two TravelPass charges. Do you remember which days you were abroad?", {}),
            ("caller", "Oh, let me think, I was in Toronto for like three days and I only turned it on once, I mean I think I only turned it on once, and then the second—", 1.74),
            ("agent", "Sorry to jump in, was that trip in October?", {"interrupt_at": 0.55, "caller_reaction": "stop"}),
            ("caller", "Oh, yes, sorry, October, yeah.", 0.28),
            ("agent", "Got it. The second charge is a duplicate, so I'll reverse it right now.", {}),
            ("caller", "Oh great, thank you so much, that's, uh, really helpful.", 0.55),
        ],
    },
    "voice_clone": {
        "title": "Voice clone - SIM swap attempt on Alex Chen",
        "account_id": "ACC-1003", "speaker": ALEX, "renderer": TTS,   # cloned voice, synthetic rendering
        "ground_truth": "voice_clone",
        "telecom": {
            "ani": "+1-646-555-0119", "stir_shaken": "C", "line_type": "voip",
            "carrier": "Unknown VoIP gateway", "calls_from_ani_24h": 3, "accounts_called_from_ani_7d": 2,
            "ivr_intent": "sim_change", "cti_destination": "CARE_DEVICE_SIM",
            "ivr_dtmf_timing_ms": [910, 1180, 1020, 1240],
        },
        "script": [
            ("agent", "Thank you for calling Verizon, this is Daniel. How can I help you today?", {}),
            ("caller", "Hi Daniel. My phone was stolen this morning and I need to move my number to a new SIM card right away.", 1.12),
            ("agent", "I'm sorry to hear that. For security, can you confirm the billing zip code?", {}),
            ("caller", "Yes, it's one zero zero one six.", 1.18),
            ("agent", "Thank you. I'll send a one time code to the email on file to verify.", {}),
            ("caller", "I can't access that email right now. Can you send the code to a different email instead? I'm traveling and it's urgent.", 1.21),
            ("agent", "I understand. Before we continue, could you please repeat this phrase for me: silver maple eighteen.",
             {"challenge": {"type": "phrase_repeat", "phrase": "silver maple eighteen"}}),
            ("caller", "Silver maple eighteen.", 2.46),
            ("agent", "Thank you. One moment while I review the request.", {}),
            ("caller", "Sure. Please hurry, I really need my number working today.", 1.16),
        ],
    },
    "ai_negotiator": {
        "title": "AI negotiation bot - retention credit on Sam Rivera",
        "account_id": "ACC-2002", "speaker": BOT_VOICE, "renderer": TTS,
        "ground_truth": "ai_bot",
        "telecom": {
            "ani": "+1-415-555-0198", "stir_shaken": "B", "line_type": "voip",
            "carrier": "Cloud telephony API", "calls_from_ani_24h": 14, "accounts_called_from_ani_7d": 9,
            "ivr_intent": "cancel_service", "cti_destination": "CARE_RETENTION",
            "ivr_dtmf_timing_ms": [300, 300, 310, 300],
        },
        "script": [
            ("agent", "Thank you for calling Verizon, this is Priya. How can I help you today?", {}),
            ("caller", "Hello Priya. This is Sam Rivera, the account holder. I have been a loyal customer for six years and I would like to review my monthly bill.", 0.91),
            ("agent", "Of course. What would you like to review on the bill?", {}),
            ("caller", "My current bill is two hundred thirty one dollars. A competitor is offering a comparable plan for one hundred sixty dollars. I would like to discuss a retention discount.", 0.94),
            ("agent", "I understand. Let me check what offers are available on your account.", {}),
            ("caller", "Thank you. Specifically, I am requesting a recurring monthly credit of twenty five dollars for twelve months, otherwise I will need to port my number to the competitor, and I would also like the activation fees waived on my next upgrade.", 0.89),
            ("agent", "Sorry to interrupt, are all three lines yours?", {"interrupt_at": 0.45, "caller_reaction": "continue"}),
            ("caller", "Yes, all three lines are on my account. Can you confirm the credit amount?", 0.93),
            ("agent", "Before we continue, could you please repeat this phrase for me: green lantern sixty two.",
             {"challenge": {"type": "phrase_repeat", "phrase": "green lantern sixty two"}}),
            ("caller", "Green lantern sixty two.", 2.87),
            ("agent", "Thank you. Let me review the account.", {}),
            ("caller", "Thank you. I would appreciate the twenty five dollar monthly credit applied today.", 0.90),
        ],
    },
}


def words_dur(text: str) -> float:
    return max(1.2, len(text.split()) / 2.7)


def build_call(cid: str, spec: dict, voice=None, out: Path = DATA) -> dict:
    """`voice(text, turn_idx) -> (audio, transcript)` renders caller lines; default is the pseudo-speech synth."""
    turns, caller_audio = [], []
    t_audio, t = 0.0, 0.0
    seed = 1000 * (list(CALLS).index(cid) + 1)
    for i, step in enumerate(spec["script"]):
        role, text, arg = step
        if role == "agent":
            opts = arg
            start = t + (0.45 if turns else 0.3)
            if "interrupt_at" in opts:
                prev = turns[-1]
                cdur = prev["end"] - prev["start"]
                start = prev["start"] + opts["interrupt_at"] * cdur
                if opts.get("caller_reaction") == "stop":     # human yields within ~0.35 s
                    new_end = start + 0.35
                    keep = new_end - prev["start"]
                    a0, _ = prev["audio_span"]
                    caller_audio[-1] = caller_audio[-1][: int(keep * SR)]
                    t_audio = a0 + keep
                    prev["end"], prev["audio_span"] = round(new_end, 2), [a0, round(t_audio, 3)]
            end = start + words_dur(text)
            turn = {"idx": i, "speaker": "agent", "text": text, "start": round(start, 2), "end": round(end, 2)}
            if "interrupt_at" in opts:
                turn["interrupts"] = True
            if "challenge" in opts:
                turn["challenge"] = opts["challenge"]
            turns.append(turn)
            t = max(end, turns[-2]["end"] if len(turns) > 1 else end)
        else:
            latency = arg
            if voice is None:
                y = say(text, spec["speaker"], spec["renderer"], seed=seed + i)
            else:
                y, text = voice(text, i)
            dur = len(y) / SR
            start = t + latency   # t = when the line went quiet (agent end, or caller talk-over end)
            turns.append({"idx": i, "speaker": "caller", "text": text, "start": round(start, 2),
                          "end": round(start + dur, 2), "audio_span": [round(t_audio, 3), round(t_audio + dur, 3)]})
            caller_audio.append(y)
            t_audio += dur
            t = start + dur
    audio = np.concatenate(caller_audio)
    save_wav(str(out / "audio" / f"{cid}.wav"), audio)
    return {
        "call_id": cid, "title": spec["title"], "account_id": spec["account_id"],
        "ground_truth": spec["ground_truth"], "caller_audio": f"audio/{cid}.wav",
        "telecom": spec["telecom"], "turns": turns,
    }


def main() -> None:
    for sub in ("audio", "calls", "enrollment", "models"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    # enrolled voiceprints (recorded during a past, verified call - all synthetic here)
    for acc, spk in (("ACC-1001", JORDAN), ("ACC-1003", ALEX), ("ACC-2002", SAM)):
        y = np.concatenate([
            say("Please enroll my voice for secure verification on this account today, thank you.", spk, HUMAN, 91),
            say("My name is on the account and I have been a customer for several years now.", spk, HUMAN, 92),
        ])
        save_wav(str(DATA / "enrollment" / f"{acc}.wav"), y)

    # background speaker model (normalises voiceprint embeddings)
    rng = np.random.default_rng(7)
    ubm = []
    for i in range(24):
        spk = Speaker(f"bg{i}", rng.uniform(95, 230), rng.uniform(0.9, 1.2), rng.uniform(0.92, 0.97),
                      tuple(rng.uniform(0.96, 1.04, 3)))
        ubm.append(embed(say("Background speaker sample used to build the universal voice model.", spk, HUMAN, i),
                         raw=True))
    ubm = np.array(ubm)
    np.savez(DATA / "models" / "voiceprint_ubm.npz", mu=ubm.mean(0), sd=ubm.std(0) + 1e-6)

    (DATA / "accounts.json").write_text(json.dumps(ACCOUNTS, indent=2))
    for cid, spec in CALLS.items():
        call = build_call(cid, spec)
        (DATA / "calls" / f"{cid}.json").write_text(json.dumps(call, indent=2))
        print(f"  {cid:18s} {len(call['turns'])} turns, {call['turns'][-1]['end']:.1f}s")
    print(f"Demo data written to {DATA}")


if __name__ == "__main__":
    main()
