import streamlit as st
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
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
            if not queue_items:
                st.success("Keine offenen Review-Befunde im aktuellen Run.")
            for item in queue_items:
                befund_id = item.get("id", "unbekannt")
                action = st.session_state.review_actions.get(befund_id)
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
                        st.session_state.review_actions[befund_id] = {
                            "decision": "approved",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    if c2.button("Reject", key=f"reject_{befund_id}"):
                        st.session_state.review_actions[befund_id] = {
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
                        st.session_state.review_actions[befund_id] = {
                            "decision": "reprompt",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "note": note,
                        }

        with explain_subtab:
            if not flat_befunde:
                st.info("Noch keine Befunddaten verfügbar.")
            else:
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

            st.markdown("### Live-Logs (letzte 30 Zeilen)")
            if st.session_state.logs:
                st.code("\n".join(st.session_state.logs[-30:]), language="text")
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
