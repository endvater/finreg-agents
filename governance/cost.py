"""
Block J – Kostenmodell inkl. Self-Hosting + Cost per valid completed task.

Erweitert die bisherige reine per-Token-Schätzung (pipeline._estimate_costs) um:
  - ein Self-Hosting-Kostenmodell (Fixkosten/Monat + Idle statt nur per-Token),
  - Cost per valid completed task (CPVCT) – die Leitmetrik des Buches,
  - eine Break-even-Hilfe self-hosted vs. fremdgehostet über das Monatsvolumen,
  - ein *weiches* Budget-Warnsignal pro Lauf (kein Safety-Hard-Stop – bewusst, da
    internes Batch-QS-Tool ohne Cost-Exhaustion-Vektor).

Preise in governance/cost_model.json (versioniert, pro Provider/Modell).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_COST_MODEL_PATH = Path(__file__).parent / "cost_model.json"

_DEFAULT_COST_MODEL = {
    "currency": "USD",
    "hosted": {
        "default": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    },
    "self_hosted": {
        # Beispiel-GPU-Knoten; Werte müssen je Setup kalibriert werden ("Baseline first").
        "gpu_node_monthly_eur": 1500.0,
        "assumed_monthly_runs": 200,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
    },
}


def load_cost_model(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _COST_MODEL_PATH
    if not p.exists():
        return _DEFAULT_COST_MODEL
    return json.loads(p.read_text(encoding="utf-8"))


def hosted_cost(token_stats_nach_agent: dict, model: Optional[dict] = None) -> dict:
    """Per-Token-Kosten (fremdgehostet) je Agent."""
    cm = model or load_cost_model()
    rates = cm["hosted"]["default"]
    details, total = {}, 0.0
    for agent, usage in (token_stats_nach_agent or {}).items():
        in_c = (usage.get("input", 0) / 1000.0) * rates["input_per_1k"]
        out_c = (usage.get("output", 0) / 1000.0) * rates["output_per_1k"]
        c = round(in_c + out_c, 6)
        details[agent] = {
            "input_cost": round(in_c, 6),
            "output_cost": round(out_c, 6),
            "total_cost": c,
        }
        total += c
    return {
        "mode": "hosted",
        "currency": cm.get("currency", "USD"),
        "total_cost": round(total, 6),
        "nach_agent": details,
    }


def self_hosted_cost_per_run(model: Optional[dict] = None) -> dict:
    """Amortisierte Self-Hosting-Kosten pro Lauf (Fix/Monat ÷ angenommene Läufe)."""
    cm = model or load_cost_model()
    sh = cm["self_hosted"]
    runs = max(1, int(sh.get("assumed_monthly_runs", 1)))
    per_run = round(sh.get("gpu_node_monthly_eur", 0.0) / runs, 4)
    return {
        "mode": "self_hosted",
        "currency": "EUR",
        "monthly_fixed": sh.get("gpu_node_monthly_eur", 0.0),
        "assumed_monthly_runs": runs,
        "amortized_per_run": per_run,
        "marginal_token_cost": 0.0,
    }


def cost_per_valid_task(total_cost: float, valid_tasks: int) -> float:
    """CPVCT – Leitmetrik. valid_tasks = abgeschlossen ODER korrekt eskaliert."""
    if valid_tasks <= 0:
        return 0.0
    return round(total_cost / valid_tasks, 6)


def breakeven_runs(
    model: Optional[dict] = None, hosted_cost_per_run: float = 0.0
) -> Optional[float]:
    """Ab wie vielen Läufen/Monat amortisiert Self-Hosting gegenüber fremdgehostet?

    Vereinfachte Heuristik: monatliche Fixkosten / hosted-Kosten-pro-Lauf.
    Ergebnis = Lauf-Schwelle; darüber lohnt Self-Hosting rein ökonomisch.
    (Datenhoheit ist separat zu bewerten und überstimmt die reine Ökonomie häufig.)
    """
    cm = model or load_cost_model()
    fixed = cm["self_hosted"].get("gpu_node_monthly_eur", 0.0)
    if hosted_cost_per_run <= 0:
        return None
    return round(fixed / hosted_cost_per_run, 1)


def soft_budget_check(
    total_cost: float, *, budget: Optional[float] = None, warn_ratio: float = 0.8
) -> dict:
    """Weiches Budget-Signal pro Lauf (Warnung, KEIN Abbruch)."""
    if budget is None or budget <= 0:
        return {"status": "no_budget", "ratio": None}
    ratio = round(total_cost / budget, 4)
    if ratio >= 1.0:
        status = "exceeded"
    elif ratio >= warn_ratio:
        status = "warn"
    else:
        status = "ok"
    return {
        "status": status,
        "ratio": ratio,
        "budget": budget,
        "cost": round(total_cost, 6),
    }


def estimate_run_cost(
    token_stats: dict,
    *,
    route_is_local: bool = False,
    valid_tasks: int = 0,
    budget: Optional[float] = None,
    model: Optional[dict] = None,
) -> dict:
    """Vollständige Kostenauskunft für einen Lauf (hosted oder self-hosted)."""
    cm = model or load_cost_model()
    nach_agent = (token_stats or {}).get("nach_agent", {})
    if route_is_local:
        base = self_hosted_cost_per_run(cm)
        total = base["amortized_per_run"]
        base["nach_agent"] = nach_agent
    else:
        base = hosted_cost(nach_agent, cm)
        total = base["total_cost"]
    base["cost_per_valid_task"] = cost_per_valid_task(total, valid_tasks)
    base["budget_check"] = soft_budget_check(total, budget=budget)
    return base
