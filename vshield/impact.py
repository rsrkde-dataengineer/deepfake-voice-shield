"""Business impact model behind the brief's numbers. Every input is an assumption the user can change.

Transfer savings: when Shield has verified a caller, the next agent skips re-verification.
    3.65M transferred calls/year x 75 s saved = ~76,000 agent hours; x $30/hour = ~$2.3M per year
(the brief's "~76,000 agent hours, ~$2.2M per year" at 60-90 s per transferred call).

Upside: the same hours at a fully loaded $50/hour (wages plus benefits, facilities and supervision) = ~$3.8M per year.

Fraud losses avoided: bot/clone calls that would otherwise get credits or a SIM swap. Illustrative what-if with mock
inputs (10,000 calls a month x $150, caught 25% -> 95% = $12.6M per year), to be replaced by the fraud team's data.
"""
from __future__ import annotations

BRIEF = {"transfers_per_year": 3_650_000, "seconds_saved": 75, "share_verified": 1.0, "hourly_cost": 30.0}
FULLY_LOADED_HOURLY_COST = 50.0
FRAUD_WHAT_IF = {"attempts_per_month": 10_000, "loss_per_attempt": 150, "catch_rate": 0.95,
                 "baseline_catch_rate": 0.25}


def transfer_savings(transfers_per_year: float, seconds_saved: float, share_verified: float,
                     hourly_cost: float) -> tuple[float, float]:
    """(agent hours, dollars) saved per year by skipping re-verification on verified transfers."""
    hours = transfers_per_year * share_verified * seconds_saved / 3600
    return hours, hours * hourly_cost


def fraud_avoided(attempts_per_month: float, loss_per_attempt: float, catch_rate: float,
                  baseline_catch_rate: float) -> float:
    """Dollars per year no longer lost, versus today's catch rate."""
    return attempts_per_month * 12 * loss_per_attempt * max(catch_rate - baseline_catch_rate, 0.0)
