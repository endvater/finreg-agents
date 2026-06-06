"""
FinRegAgents – Governance-Monitoring-Dashboard.

Überwacht die QS-/Epistemik-Eigenschaften der Simulationsläufe (siehe
docs/anforderungen-thinking-agentic.md). Liest ausschließlich abgelegte Artefakte
(governance_summary_*.json, decision_trace_*.jsonl) sowie die Register/Agent-Cards.

Start:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --output-dir ./reports/output
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from governance import GOVERNANCE_VERSION
from governance import monitoring, registry
from governance.agent_card import load_cards, cards_summary
from governance.routing import load_policy
from governance.evaluation import load_golden
from governance.eval_sets import eval_sets_summary


def _output_dir() -> str:
    args = sys.argv
    if "--output-dir" in args:
        return args[args.index("--output-dir") + 1]
    return "./reports/output"


st.set_page_config(
    page_title="FinRegAgents – Governance-Monitor", page_icon="🛡️", layout="wide"
)

OUT = _output_dir()

st.title("🛡️ FinRegAgents – Governance-Monitor")
st.caption(
    f"Governance v{GOVERNANCE_VERSION} · QS-/Epistemik-Überwachung für ein internes "
    f"Simulationswerkzeug (Gattung G2/G3) · Quelle: `{OUT}`"
)

snap = monitoring.fleet_snapshot(OUT)
runs = monitoring.collect_runs(OUT)

tab_over, tab_quality, tab_evid, tab_route, tab_eval, tab_reg = st.tabs(
    [
        "Übersicht",
        "Qualität & Confidence",
        "Evidenz & Drift",
        "Routing & Datenhoheit",
        "Eval & Release-Gate",
        "Register & Agent Cards",
    ]
)

# ── Übersicht ──────────────────────────────────────────────────────────────
with tab_over:
    if not snap.get("runs"):
        st.info(
            "Noch keine Läufe gefunden. Starte eine Prüfung – das Dashboard liest "
            "`governance_summary_*.json` aus dem Output-Verzeichnis."
        )
    else:
        c = st.columns(4)
        c[0].metric("Läufe", snap["runs"])
        c[1].metric("Ø Schema-Compliance", f"{snap['mean_schema_compliance']:.0%}")
        c[2].metric("Ø Confidence", f"{snap['mean_confidence']:.2f}")
        c[3].metric("Ø Review-Quote", f"{snap['mean_review_rate']:.0%}")
        c2 = st.columns(4)
        c2[0].metric("Disputed gesamt", snap["total_disputed"])
        c2[1].metric("Term-Drift gesamt", snap["total_term_drift"])
        gpr = snap.get("gate_pass_rate")
        c2[2].metric("Release-Gate Pass", "—" if gpr is None else f"{gpr:.0%}")
        c2[3].metric("Regulatoriken", len(snap["regulatoriken"]))
        st.subheader("Läufe")
        st.dataframe(
            [
                {
                    "run_id": r["run_id"],
                    "regulatorik": r["regulatorik"],
                    "modell": r.get("model"),
                    "befunde": r.get("befunde_total"),
                    "review": f"{r.get('review_rate', 0):.0%}",
                    "schema": f"{r.get('schema_compliance', 0):.0%}",
                    "conf": r.get("confidence_mean"),
                    "disputed": r.get("disputed_count"),
                    "gate": (r.get("gate") or {}).get("passed"),
                    "ts": r.get("timestamp"),
                }
                for r in runs
            ],
            use_container_width=True,
        )

# ── Qualität & Confidence ───────────────────────────────────────────────────
with tab_quality:
    if runs:
        latest = runs[0]
        st.subheader(f"Letzter Lauf: {latest['run_id']} ({latest['regulatorik']})")
        bc = latest.get("bewertung_counts", {})
        if bc:
            st.bar_chart(bc)
        st.write(
            "**Confidence:** Ø "
            f"{latest.get('confidence_mean')} · min {latest.get('confidence_min')}"
        )
        st.write(
            f"**Schema-Verstöße:** {latest.get('schema_violations')} von "
            f"{latest.get('befunde_total')} Befunden"
        )
        st.line_chart(
            {
                "review_rate": [r.get("review_rate", 0) for r in reversed(runs)],
                "confidence": [r.get("confidence_mean", 0) for r in reversed(runs)],
            }
        )
    else:
        st.info("Keine Läufe.")

# ── Evidenz & Drift ─────────────────────────────────────────────────────────
with tab_evid:
    st.subheader("Term-/Context-Drift & Disputed")
    if runs:
        st.bar_chart(
            {
                "term_drift": [r.get("term_drift_count", 0) for r in reversed(runs)],
                "disputed": [r.get("disputed_count", 0) for r in reversed(runs)],
            }
        )
        st.caption(
            "Hohe Werte = Hinweis auf Phantom-Zitate / widersprüchliche Befunde – "
            "kein Fehler, sondern QS-Signal (Block L)."
        )
        traces = (
            sorted(Path(OUT).glob("decision_trace_*.jsonl"))
            if Path(OUT).exists()
            else []
        )
        if len(traces) >= 2:
            from governance.trace import what_changed

            st.subheader("what_changed (zwei jüngste Traces)")
            st.dataframe(what_changed(traces[-2], traces[-1]), use_container_width=True)
    else:
        st.info("Keine Läufe.")

# ── Routing & Datenhoheit ───────────────────────────────────────────────────
with tab_route:
    st.subheader("Routing-Policy (Datenklasse → lokal/fremdgehostet)")
    pol = load_policy()
    st.json(pol.get("data_classes", {}))
    st.caption(
        f"Policy-Version {pol.get('policy_version')} · vertrauliche Daten → lokal "
        "(Datenhoheit/DSGVO), öffentlich/Katalog → fremdgehostet zulässig."
    )
    if runs:
        st.subheader("Tatsächliches Routing je Lauf")
        st.dataframe(
            [{"run_id": r["run_id"], **(r.get("route") or {})} for r in runs],
            use_container_width=True,
        )

# ── Eval & Release-Gate ─────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Release-Gate je Lauf")
    if runs and any(r.get("gate") for r in runs):
        for r in runs:
            gate = r.get("gate") or {}
            if not gate:
                continue
            status = "✅ PASS" if gate.get("passed") else "⛔ BLOCKED"
            with st.expander(f"{r['run_id']} – {status}"):
                st.dataframe(gate.get("checks", []), use_container_width=True)
                if r.get("eval"):
                    st.json(r["eval"])
    else:
        st.info(
            "Noch keine Eval-/Gate-Ergebnisse. Golden Datasets liegen in "
            "`governance/golden/` (aktuell nur Seed)."
        )
    st.subheader("Golden-Datasets je Verordnung")
    st.dataframe(
        [
            {
                "regulatorik": reg,
                "dataset": (g.dataset_id if (g := load_golden(reg)) else "—"),
                "version": (g.version if g else "—"),
                "fälle": (len(g.cases) if g else 0),
            }
            for reg in ["gwg", "amlr", "micar", "macomp", "kwg_crr"]
        ],
        use_container_width=True,
    )
    st.subheader("Eval-Splits (Security / Chaos / Drift)")
    st.json(eval_sets_summary())
    st.caption(
        "Security/Chaos/Drift sind Verhaltens-Sets (Block H/I/L): Angriff/Störung/Drift "
        "mit erwartetem Abwehr-/Degradationsverhalten. Doctored Eingabedokumente erzeugen: "
        "`python -m governance.eval_docs ./eval_runtime` (Generator in governance/eval_docs.py)."
    )

# ── Register & Agent Cards ──────────────────────────────────────────────────
with tab_reg:
    st.subheader("Agent Cards")
    st.json(cards_summary())
    for card in load_cards():
        with st.expander(
            f"{card.agent_name} (Gattung {card.gattung}, v{card.version})"
        ):
            st.json(card.model_dump())
            if card.issues():
                st.warning("Hinweise: " + "; ".join(card.issues()))
    st.subheader("Modell-/Quellen-Register")
    st.json(registry.registry_summary())
    st.write("**Provider-Konzentration (Anti-Lock-in):**")
    st.bar_chart(registry.provider_concentration())
