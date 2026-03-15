import streamlit as st
import os
import json
import shutil
import base64
from pathlib import Path
from datetime import datetime
import streamlit.components.v1 as components
from pipeline import AuditPipeline, KATALOG_REGISTRY, KATALOG_LABELS

# --- Streamlit config ---
st.set_page_config(
    page_title="FinRegAgents v2",
    page_icon="🏦",
    layout="wide",
)

# --- Constants & State ---
TMP_DOC_DIR = Path("./streamlit_tmp_docs")
OUTPUT_DIR = Path("./reports/output")


def init_session_state():
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "reports" not in st.session_state:
        st.session_state.reports = None
    if "input_dir" not in st.session_state:
        st.session_state.input_dir = ""
    if "review_actions" not in st.session_state:
        st.session_state.review_actions = {}
    if "reprompt_notes" not in st.session_state:
        st.session_state.reprompt_notes = {}
    if "previous_run_stats" not in st.session_state:
        st.session_state.previous_run_stats = None


init_session_state()


def _read_json_file(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_report_payload() -> dict | None:
    reports = st.session_state.reports
    if not reports:
        return None
    json_path = reports.get("json")
    if not json_path or not Path(json_path).exists():
        return None
    return _read_json_file(json_path)


def _flatten_befunde(report_payload: dict | None) -> list[dict]:
    if not report_payload:
        return []
    rows = []
    for sektion in report_payload.get("sektionen", []):
        for befund in sektion.get("befunde", []):
            rows.append(
                {
                    "sektion_id": sektion.get("id", ""),
                    "sektion_titel": sektion.get("titel", ""),
                    **befund,
                }
            )
    return rows


def _load_run_stats(report_payload: dict | None) -> dict | None:
    if not report_payload:
        return None
    direct = report_payload.get("token_stats")
    if direct:
        return {
            "token_stats": direct,
            "stats_file": report_payload.get("stats_file"),
        }

    stats_file = report_payload.get("stats_file")
    if not stats_file:
        return None
    stats_path = Path(stats_file)
    if not stats_path.exists():
        return None
    return _read_json_file(str(stats_path))


def _current_run_key(report_payload: dict | None) -> str:
    if not report_payload:
        return "run-unbekannt"
    meta = report_payload.get("meta", {})
    ts = (
        meta.get("audit_trail", {}).get("timestamp")
        or meta.get("pruefungsdatum")
        or "run-unbekannt"
    )
    return str(ts)


def _review_action_key(run_key: str, befund_id: str) -> str:
    return f"{run_key}:{befund_id}"


def _summarize_actions(run_key: str) -> dict:
    decisions = {"approved": 0, "rejected": 0, "reprompt": 0}
    for key, val in st.session_state.review_actions.items():
        if not key.startswith(f"{run_key}:"):
            continue
        decision = val.get("decision")
        if decision in decisions:
            decisions[decision] += 1
    return decisions


def _to_number(value):
    return value if isinstance(value, (int, float)) else 0


def _agent_delta_rows(
    current_token_stats: dict, previous_token_stats: dict
) -> list[dict]:
    curr_agents = current_token_stats.get("nach_agent", {}) or {}
    prev_agents = previous_token_stats.get("nach_agent", {}) or {}
    agent_names = sorted(set(curr_agents.keys()) | set(prev_agents.keys()))
    rows = []
    for name in agent_names:
        curr = curr_agents.get(name, {})
        prev = prev_agents.get(name, {})
        rows.append(
            {
                "agent": name,
                "curr_total": _to_number(curr.get("total")),
                "prev_total": _to_number(prev.get("total")),
                "delta_total": _to_number(curr.get("total"))
                - _to_number(prev.get("total")),
                "curr_input": _to_number(curr.get("input")),
                "curr_output": _to_number(curr.get("output")),
            }
        )
    return rows


def _find_source_path(source_name: str) -> str | None:
    if not source_name:
        return None
    input_dir = st.session_state.get("input_dir") or ""
    if not input_dir or not Path(input_dir).exists():
        return None
    for file in Path(input_dir).rglob("*"):
        if file.is_file() and file.name == source_name:
            return str(file)
    return None


def _render_pdf_preview(path: str, height: int = 650):
    pdf_bytes = Path(path).read_bytes()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe = (
        f'<iframe src="data:application/pdf;base64,{encoded}" '
        f'width="100%" height="{height}" type="application/pdf"></iframe>'
    )
    components.html(iframe, height=height + 10, scrolling=True)


def _extract_timeline(logs: list[str]) -> list[dict]:
    events = []
    for line in logs:
        msg = line.strip()
        icon = "ℹ️"
        if "Schritt 1/4" in msg:
            icon = "📂"
        elif "Schritt 2/4" in msg:
            icon = "🔍"
        elif "Schritt 3/4" in msg:
            icon = "📋"
        elif "Schritt 4/4" in msg:
            icon = "📝"
        elif "REVIEW" in msg:
            icon = "🔎"
        elif "Skeptiker" in msg:
            icon = "⚔️"
        elif "abgeschlossen" in msg:
            icon = "✅"
        events.append({"icon": icon, "message": msg})
    return events[-40:]


# --- UI Sidebar ---
with st.sidebar:
    st.title("🏦 FinRegAgents v2")
    st.markdown("KI-Agenten für regulatorische Prüfungen")

    st.header("Konfiguration")
    api_key_anthropic = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    api_key_openai = st.text_input(
        "OpenAI API Key (Embeddings)",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )

    if api_key_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = api_key_anthropic
    if api_key_openai:
        os.environ["OPENAI_API_KEY"] = api_key_openai

    st.divider()

    institution = st.text_input("Institution", "Musterbank AG")
    regulatorik = st.selectbox(
        "Regulatorik",
        options=list(KATALOG_REGISTRY.keys()),
        format_func=lambda x: KATALOG_LABELS.get(x, x.upper()),
    )

    with st.expander("Erweiterte Einstellungen"):
        model = st.selectbox(
            "Modell", ["claude-sonnet-4-5-20250514", "claude-opus-4-5"]
        )
        top_k = st.slider("RAG Chunks (Top-K)", min_value=3, max_value=20, value=8)
        skeptiker = st.checkbox(
            "Skeptiker-Agent aktivieren ⚔️",
            value=False,
            help="Adversariales Review der Ergebnisse",
        )
        skeptiker_only_konform = (
            st.checkbox("Skeptiker nur für 'konform'", value=False)
            if skeptiker
            else False
        )

# --- UI Main Area ---
st.title("Prüfungszentrale")

setup_tab, result_tab = st.tabs(["📄 Dokumente & Start", "📊 Ergebnisse"])

with setup_tab:
    st.subheader("Dokumenten-Input")
    input_mode = st.radio("Input-Methode", ["Lokales Verzeichnis", "Upload (temporär)"])

    input_dir = ""
    if input_mode == "Lokales Verzeichnis":
        input_dir_str = st.text_input("Pfad zum Dokumentenverzeichnis", "./docs")
        if os.path.isdir(input_dir_str):
            input_dir = input_dir_str
            st.session_state.input_dir = input_dir
            st.success(f"Verzeichnis gefunden: {input_dir}")
        else:
            st.warning("Verzeichnis existiert nicht oder Pfad ist ungültig.")
    else:
        # File uploader logic
        uploaded_files = st.file_uploader(
            "Dokumente hochladen (PDF, Excel, JSON/YAML, Bilder)",
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("Dateien für Prüfung vorbereiten"):
                if TMP_DOC_DIR.exists():
                    shutil.rmtree(TMP_DOC_DIR)

                for subdir in ["pdfs", "excel", "interviews", "screenshots", "logs"]:
                    (TMP_DOC_DIR / subdir).mkdir(parents=True, exist_ok=True)

                for f in uploaded_files:
                    ext = f.name.split(".")[-1].lower()
                    target_dir = "pdfs"
                    if ext in ["xlsx", "csv"]:
                        target_dir = "excel"
                    elif ext in ["json", "yaml"]:
                        target_dir = "interviews"
                    elif ext in ["png", "jpg", "jpeg"]:
                        target_dir = "screenshots"
                    elif ext in ["txt", "log"]:
                        target_dir = "logs"

                    file_path = TMP_DOC_DIR / target_dir / f.name
                    with open(file_path, "wb") as out_f:
                        out_f.write(f.getbuffer())

                st.success(
                    f"{len(uploaded_files)} Dateien erfolgreich in {TMP_DOC_DIR} abgelegt."
                )
                input_dir = str(TMP_DOC_DIR)
                st.session_state.input_dir = input_dir

    if not input_dir and st.session_state.input_dir:
        input_dir = st.session_state.input_dir
        st.info(f"Aktives Input-Verzeichnis: {input_dir}")

    st.divider()

    can_run = bool(input_dir) and bool(api_key_anthropic) and bool(api_key_openai)
    if not can_run:
        st.info("Bitte API Keys setzen und gültiges Dokumentenverzeichnis wählen.")

    if st.button(
        "🚀 Prüfung starten",
        disabled=os.environ.get("PIPELINE_RUNNING") == "1" or not can_run,
        type="primary",
    ):
        st.session_state.logs = []
        st.session_state.reports = None
        st.session_state.review_actions = {}
        st.session_state.reprompt_notes = {}
        existing_stats = OUTPUT_DIR / "run_stats.json"
        st.session_state.previous_run_stats = (
            _read_json_file(str(existing_stats)) if existing_stats.exists() else None
        )
        os.environ["PIPELINE_RUNNING"] = "1"

        log_placeholder = st.empty()

        with st.spinner(f"Führe {KATALOG_LABELS.get(regulatorik, regulatorik)} aus..."):
            pipeline = AuditPipeline(
                input_dir=input_dir,
                institution=institution,
                regulatorik=regulatorik,
                output_dir=str(OUTPUT_DIR),
                model=model,
                top_k=top_k,
                skeptiker=skeptiker,
                skeptiker_only_konform=skeptiker_only_konform,
                verbose=True,
            )

            original_log = pipeline._log

            def st_log(msg):
                original_log(msg)
                if msg.strip():
                    st.session_state.logs.append(msg.strip())
                    formatted_logs = "\n".join(st.session_state.logs[-30:])
                    log_placeholder.code(formatted_logs, language="text")

            pipeline._log = st_log

            try:
                report_paths = pipeline.run()
                st.session_state.reports = report_paths
                st.success("Prüfung erfolgreich abgeschlossen!")
            except Exception as e:
                st.error(f"Fehler während der Prüfung: {str(e)}")
            finally:
                os.environ["PIPELINE_RUNNING"] = "0"

with result_tab:
    st.subheader("Ergebnisse & Review")
    reports = st.session_state.reports
    report_payload = _load_report_payload()
    flat_befunde = _flatten_befunde(report_payload)

    if not reports:
        st.info("Noch keine Ergebnisse vorhanden. Starte eine Prüfung im ersten Tab.")
    else:
        run_key = _current_run_key(report_payload)
        (
            overview_subtab,
            queue_subtab,
            explain_subtab,
            run_subtab,
            report_subtab,
        ) = st.tabs(
            [
                "🧭 Übersicht",
                "🧑‍⚖️ Review-Queue",
                "🔎 Explainability",
                "📈 Run-Transparenz",
                "📄 Reports",
            ]
        )

        with overview_subtab:
            summary = (report_payload or {}).get("zusammenfassung", {})
            total = summary.get("total_prueffelder", 0)
            review_required = summary.get("review_erforderlich", 0)
            avg_conf = summary.get("avg_confidence", 0.0)
            disputed = summary.get("disputed", 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prüffelder", total)
            c2.metric("Review erforderlich", review_required)
            c3.metric("Ø Confidence", f"{avg_conf:.0%}" if avg_conf else "0%")
            c4.metric("Strittig", disputed)

            if report_payload:
                st.markdown("### Laufzusammenfassung")
                st.json(summary)
            else:
                st.warning("JSON-Report konnte nicht geladen werden.")
            st.caption(f"Run-ID: {run_key}")

        with queue_subtab:
            queue_items = [
                b for b in flat_befunde if b.get("review_erforderlich", False) is True
            ]
            queue_items.sort(
                key=lambda x: (
                    x.get("confidence", 0.0),
                    x.get("sektion_id", ""),
                    x.get("id", ""),
                )
            )
            st.caption(
                f"{len(queue_items)} Befunde in der Review-Queue (confidence-basiert sortiert)."
            )
            action_stats = _summarize_actions(run_key)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Queue gesamt", len(queue_items))
            q2.metric("Approved", action_stats["approved"])
            q3.metric("Rejected", action_stats["rejected"])
            q4.metric("Re-Prompt", action_stats["reprompt"])

            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
            levels = sorted(
                {str(i.get("confidence_level", "low")) for i in queue_items}
            ) or ["low", "medium", "high"]
            selected_levels = filter_col1.multiselect(
                "Confidence-Level",
                options=levels,
                default=levels,
            )
            sektionen = sorted({str(i.get("sektion_id", "")) for i in queue_items})
            selected_sektion = filter_col2.selectbox(
                "Sektion",
                options=["Alle"] + sektionen,
                index=0,
            )
            decision_filter = filter_col3.selectbox(
                "Review-Status",
                options=["Alle", "Offen", "Approved", "Rejected", "Re-Prompt"],
                index=0,
            )

            filtered_queue = []
            for item in queue_items:
                if item.get("confidence_level", "low") not in selected_levels:
                    continue
                if (
                    selected_sektion != "Alle"
                    and item.get("sektion_id") != selected_sektion
                ):
                    continue
                befund_id = item.get("id", "unbekannt")
                action_key = _review_action_key(run_key, befund_id)
                decision = (
                    st.session_state.review_actions.get(action_key, {}).get("decision")
                    or "offen"
                )
                if decision_filter == "Offen" and decision != "offen":
                    continue
                if decision_filter == "Approved" and decision != "approved":
                    continue
                if decision_filter == "Rejected" and decision != "rejected":
                    continue
                if decision_filter == "Re-Prompt" and decision != "reprompt":
                    continue
                filtered_queue.append(item)

            st.caption(f"Gefilterte Befunde: {len(filtered_queue)}")
            if st.button("Review-Entscheidungen zurücksetzen"):
                st.session_state.review_actions = {
                    k: v
                    for k, v in st.session_state.review_actions.items()
                    if not k.startswith(f"{run_key}:")
                }
                st.session_state.reprompt_notes = {}
                st.rerun()

            import_file = st.file_uploader(
                "Review-Entscheidungen importieren (JSON)",
                type=["json"],
                key=f"import_review_{run_key}",
            )
            if import_file and st.button(
                "Import anwenden", key=f"apply_import_{run_key}"
            ):
                try:
                    payload = json.loads(import_file.getvalue().decode("utf-8"))
                    imported = 0
                    for row in payload.get("decisions", []):
                        befund_id = row.get("befund_id")
                        if not befund_id:
                            continue
                        action_key = _review_action_key(run_key, befund_id)
                        st.session_state.review_actions[action_key] = {
                            "decision": row.get("decision", "approved"),
                            "timestamp": row.get(
                                "timestamp",
                                datetime.now().isoformat(timespec="seconds"),
                            ),
                            **({"note": row.get("note")} if row.get("note") else {}),
                        }
                        imported += 1
                    st.success(f"{imported} Entscheidungen importiert.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import fehlgeschlagen: {e}")

            export_payload = {
                "run_key": run_key,
                "decisions": [
                    {"befund_id": k.split(":", 1)[1], **v}
                    for k, v in st.session_state.review_actions.items()
                    if k.startswith(f"{run_key}:")
                ],
            }
            st.download_button(
                "Review-Entscheidungen exportieren (JSON)",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"review_queue_{run_key.replace(':', '_')}.json",
                mime="application/json",
            )

            if not queue_items:
                st.success("Keine offenen Review-Befunde im aktuellen Run.")
            for item in filtered_queue:
                befund_id = item.get("id", "unbekannt")
                action_key = _review_action_key(run_key, befund_id)
                action = st.session_state.review_actions.get(action_key)
                with st.expander(
                    f"{befund_id} · {item.get('frage', '')[:90]}",
                    expanded=False,
                ):
                    st.markdown(
                        f"**Sektion:** `{item.get('sektion_id', '')}`  \n"
                        f"**Bewertung:** `{item.get('bewertung', '')}`  \n"
                        f"**Confidence:** `{item.get('confidence_level', 'low')}` ({item.get('confidence', 0.0):.0%})"
                    )
                    reasons = item.get("low_confidence_reasons", [])
                    if reasons:
                        st.markdown(
                            f"**Low-Confidence-Reasons:** `{', '.join(reasons)}`"
                        )
                    if action:
                        st.info(
                            f"Aktueller Review-Status: {action['decision']} ({action['timestamp']})"
                        )

                    c1, c2, c3 = st.columns(3)
                    if c1.button("Approve", key=f"approve_{befund_id}"):
                        st.session_state.review_actions[action_key] = {
                            "decision": "approved",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    if c2.button("Reject", key=f"reject_{befund_id}"):
                        st.session_state.review_actions[action_key] = {
                            "decision": "rejected",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    reprompt_key = f"reprompt_{befund_id}"
                    st.session_state.reprompt_notes.setdefault(
                        reprompt_key, "Bitte prüfe den Befund erneut mit Fokus auf ..."
                    )
                    note = st.text_area(
                        "Re-Prompt-Anweisung",
                        key=reprompt_key,
                        height=90,
                    )
                    if c3.button(
                        "Re-Prompt markieren", key=f"reprompt_btn_{befund_id}"
                    ):
                        st.session_state.review_actions[action_key] = {
                            "decision": "reprompt",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "note": note,
                        }

            by_section = {}
            for item in queue_items:
                section = item.get("sektion_id", "UNBEKANNT")
                by_section.setdefault(
                    section,
                    {
                        "sektion": section,
                        "queue_total": 0,
                        "approved": 0,
                        "rejected": 0,
                        "reprompt": 0,
                    },
                )
                by_section[section]["queue_total"] += 1
                action_key = _review_action_key(run_key, item.get("id", ""))
                decision = st.session_state.review_actions.get(action_key, {}).get(
                    "decision"
                )
                if decision in ("approved", "rejected", "reprompt"):
                    by_section[section][decision] += 1
            if by_section:
                st.markdown("### Fortschritt pro Sektion")
                st.dataframe(list(by_section.values()), use_container_width=True)

        with explain_subtab:
            if not flat_befunde:
                st.info("Noch keine Befunddaten verfügbar.")
            else:
                preview_rows = [
                    {
                        "id": b.get("id"),
                        "sektion": b.get("sektion_id"),
                        "bewertung": b.get("bewertung"),
                        "confidence": round(float(b.get("confidence", 0.0)), 3),
                        "review": b.get("review_erforderlich"),
                    }
                    for b in flat_befunde
                ]
                st.dataframe(preview_rows, use_container_width=True, height=220)
                labels = [
                    f"{b.get('id', '?')} · {b.get('sektion_id', '')} · {b.get('frage', '')[:70]}"
                    for b in flat_befunde
                ]
                selected_label = st.selectbox("Befund auswählen", labels)
                selected_idx = labels.index(selected_label)
                b = flat_befunde[selected_idx]

                left, right = st.columns([2, 3])
                with left:
                    st.markdown("### Bewertung & Guardrails")
                    st.markdown(
                        f"**Bewertung:** `{b.get('bewertung', '')}`  \n"
                        f"**Confidence:** `{b.get('confidence_level', 'low')}` ({b.get('confidence', 0.0):.0%})  \n"
                        f"**Review erforderlich:** `{b.get('review_erforderlich', False)}`"
                    )
                    st.markdown("### Confidence-Guards")
                    st.json(b.get("confidence_guards", {}))
                    st.markdown("### Validierungshinweise")
                    hints = b.get("validierungshinweise", [])
                    if hints:
                        for hint in hints:
                            st.markdown(f"- {hint}")
                    else:
                        st.caption("Keine Validierungshinweise.")

                with right:
                    st.markdown("### Quellen & Evidenz")
                    sources = b.get("quellen", [])
                    if sources:
                        st.markdown("**Quellen:**")
                        for src in sources:
                            st.markdown(f"- `{src}`")
                        selected_source = st.selectbox(
                            "Quelle für Detailansicht",
                            options=sources,
                            key=f"source_select_{b.get('id', 'x')}",
                        )
                        source_path = _find_source_path(selected_source)
                        if source_path:
                            st.caption(f"Datei gefunden: {source_path}")
                            p = Path(source_path)
                            if p.suffix.lower() == ".pdf":
                                with st.expander("PDF-Vorschau öffnen", expanded=False):
                                    _render_pdf_preview(source_path, height=520)
                            else:
                                with st.expander(
                                    "Datei-Inhalt (Textvorschau)", expanded=False
                                ):
                                    if p.suffix.lower() in {
                                        ".txt",
                                        ".log",
                                        ".json",
                                        ".yaml",
                                        ".yml",
                                        ".md",
                                        ".csv",
                                    }:
                                        content = p.read_text(
                                            encoding="utf-8", errors="ignore"
                                        )
                                        st.code(content[:6000], language="text")
                                    else:
                                        st.info(
                                            "Kein Inline-Preview für diesen Dateityp verfügbar."
                                        )
                        else:
                            st.caption(
                                "Quelle aktuell nicht im aktiven Input-Verzeichnis gefunden."
                            )
                    textstellen = b.get("belegte_textstellen", [])
                    if textstellen:
                        st.markdown("**Belegte Textstellen:**")
                        for t in textstellen:
                            st.markdown(f"> {t}")
                    st.markdown("### Claim-Provenance")
                    claims = b.get("claim_list", [])
                    if claims:
                        st.json(claims)
                    else:
                        st.caption("Keine Claim-Liste vorhanden.")

        with run_subtab:
            stats_payload = _load_run_stats(report_payload)
            st.markdown("### Token & Kosten")
            if not stats_payload:
                st.info("Keine run_stats/token_stats verfügbar.")
            else:
                token_stats = stats_payload.get("token_stats", {})
                costs = token_stats.get("kosten_schaetzung", {})
                gesamt = token_stats.get("gesamt", {})
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Total Tokens", gesamt.get("total", 0))
                r2.metric("Input Tokens", gesamt.get("input", 0))
                r3.metric("Output Tokens", gesamt.get("output", 0))
                r4.metric(
                    "Kosten (USD)",
                    f"{costs.get('total_cost', 0):.4f}"
                    if isinstance(costs.get("total_cost"), (int, float))
                    else costs.get("total_cost", "n/a"),
                )
                st.markdown("#### Tokens nach Agent")
                st.json(token_stats.get("nach_agent", {}))
                stats_file = stats_payload.get("stats_file")
                if stats_file:
                    st.caption(f"Stats-Datei: {stats_file}")

                previous = st.session_state.previous_run_stats or {}
                prev_token_stats = previous.get("token_stats", {})
                if prev_token_stats:
                    prev_total = _to_number(
                        prev_token_stats.get("gesamt", {}).get("total")
                    )
                    curr_total = _to_number(gesamt.get("total"))
                    prev_cost = _to_number(
                        prev_token_stats.get("kosten_schaetzung", {}).get("total_cost")
                    )
                    curr_cost = _to_number(costs.get("total_cost"))
                    st.markdown("#### Delta zum vorherigen Run")
                    d1, d2 = st.columns(2)
                    d1.metric("Δ Total Tokens", curr_total - prev_total)
                    d2.metric("Δ Kosten (USD)", f"{curr_cost - prev_cost:.4f}")
                    delta_rows = _agent_delta_rows(token_stats, prev_token_stats)
                    if delta_rows:
                        st.markdown("#### Delta pro Agent")
                        st.dataframe(delta_rows, use_container_width=True)
                else:
                    st.caption("Kein vorheriger Run für Delta-Vergleich verfügbar.")

            st.markdown("### Live-Logs (letzte 30 Zeilen)")
            if st.session_state.logs:
                st.code("\n".join(st.session_state.logs[-30:]), language="text")
                st.markdown("### Ereignis-Timeline")
                for ev in _extract_timeline(st.session_state.logs):
                    st.markdown(f"- {ev['icon']} {ev['message']}")
            else:
                st.caption("Noch keine Logs verfügbar.")

        with report_subtab:
            col1, col2, col3 = st.columns(3)

            if "json" in reports and os.path.exists(reports["json"]):
                with open(reports["json"], "r", encoding="utf-8") as f:
                    col1.download_button(
                        "🔽 JSON herunterladen",
                        f.read(),
                        file_name=Path(reports["json"]).name,
                        mime="application/json",
                    )

            md_content = ""
            if "markdown" in reports and os.path.exists(reports["markdown"]):
                with open(reports["markdown"], "r", encoding="utf-8") as f:
                    md_content = f.read()
                    col2.download_button(
                        "🔽 Markdown herunterladen",
                        md_content,
                        file_name=Path(reports["markdown"]).name,
                        mime="text/markdown",
                    )

            if "html" in reports and os.path.exists(reports["html"]):
                with open(reports["html"], "r", encoding="utf-8") as f:
                    col3.download_button(
                        "🔽 HTML herunterladen",
                        f.read(),
                        file_name=Path(reports["html"]).name,
                        mime="text/html",
                    )

            st.divider()
            st.markdown("### Report Vorschau (Markdown)")
            if md_content:
                with st.container(height=600):
                    st.markdown(md_content)
            else:
                st.info("Kein Markdown-Report für Vorschau verfügbar.")
