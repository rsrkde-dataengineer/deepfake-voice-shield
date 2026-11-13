"""Business impact model behind the brief's numbers. Every input is an assumption the user can change.

Transfer savings: when Shield has verified a caller, the next agent skips re-verification.
    3.65M transferred calls/year x 75 s saved = ~76,000 agent hours; x $30/hour = ~$2.3M per year
(the brief's "~76,000 agent hours, ~$2.2M per year" at 60-90 s per transferred call).

Upside: the same hours at a fully loaded $50/hour (wages plus benefits, facilities and supervision) = ~$3.8M per year.

Fraud losses avoided: bot/clone calls that would otherwise get credits or a SIM swap. Illustrative what-if with mock
inputs (10,000 calls a month x $150, caught 25% -> 95% = $12.6M per year), to be replaced by the fraud team's data.

Setup cost (one time, internal hours): the production version is a Large effort (the sprint's T-shirt sizing) priced
with its rate card, $100/hour developer/IT and $75/hour business/SME: 580 hours = $55,000.
Running cost (GCP, per year): about $2,000 a year on GCP for the open-source models; no license fees.
"""
from __future__ import annotations

BRIEF = {"transfers_per_year": 3_650_000, "seconds_saved": 75, "share_verified": 1.0, "hourly_cost": 30.0}
FULLY_LOADED_HOURLY_COST = 50.0
FRAUD_WHAT_IF = {"attempts_per_month": 10_000, "loss_per_attempt": 150, "catch_rate": 0.95,
                 "baseline_catch_rate": 0.25}

RATE_CARD = {"developer": 100.0, "business": 75.0}       # $/hour, the sprint's standard rate card
BUILD_SIZE = "Large"                                       # the sprint's T-shirt sizing: Large = 100+ hours
BUILD_PACKAGES = [                                         # (work package, developer/IT hours, business/SME hours)
    ("Call-audio fork and streaming ASR (NVIDIA Riva/Parakeet)", 120, 0),
    ("Model serving on GPU (Triton), latency and load tests", 80, 0),
    ("Connectors: CTI/ANI, STIR/SHAKEN, account risk, billing, tickets", 80, 0),
    ("Agent panel, handoff card, fraud referral into the case system", 100, 0),
    ("Consented voiceprint enrollment and retention", 40, 30),
    ("Security, privacy and biometric-data review", 20, 30),
    ("Calibration on consented call audio; fraud playbook", 20, 30),
    ("Pilot operations: agent training, Looker baseline, go-live", 0, 30),
]
RUN_COST_PER_YEAR = 2_000.0    # GCP infrastructure to run the models, per year (the team's estimate); internal hours are the setup cost


def build_cost() -> tuple[int, int, float]:
    """(developer hours, business hours, dollars): the setup cost, in internal hours, of the production version."""
    dev = sum(p[1] for p in BUILD_PACKAGES)
    biz = sum(p[2] for p in BUILD_PACKAGES)
    return dev, biz, dev * RATE_CARD["developer"] + biz * RATE_CARD["business"]


def payback_days(savings_per_year: float) -> float:
    """Days of net savings (after the running cost) that pay back the build."""
    net_per_day = (savings_per_year - RUN_COST_PER_YEAR) / 365
    return build_cost()[2] / net_per_day if net_per_day > 0 else float("inf")


def payback_label(days: float) -> str:
    """Payback in words: hours under two days, days after that."""
    if days == float("inf"):
        return "never"
    return f"about {days * 24:.0f} hours" if days < 2 else f"about {days:.0f} days"


def transfer_savings(transfers_per_year: float, seconds_saved: float, share_verified: float,
                     hourly_cost: float) -> tuple[float, float]:
    """(agent hours, dollars) saved per year by skipping re-verification on verified transfers."""
    hours = transfers_per_year * share_verified * seconds_saved / 3600
    return hours, hours * hourly_cost


def fraud_avoided(attempts_per_month: float, loss_per_attempt: float, catch_rate: float,
                  baseline_catch_rate: float) -> float:
    """Dollars per year no longer lost, versus today's catch rate."""
    return attempts_per_month * 12 * loss_per_attempt * max(catch_rate - baseline_catch_rate, 0.0)
