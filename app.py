import streamlit as st
import os
import time
import io
import contextlib
import json
from pathlib import Path
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
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False
    if "reports" not in st.session_state:
        st.session_state.reports = None

init_session_state()

# --- Utility: Capture Output ---
class StreamlitLogCapture(io.StringIO):
    def __init__(self, log_placeholder):
        super().__init__()
        self.log_placeholder = log_placeholder
    
    def write(self, s):
        super().write(s)
        if s.strip():
            st.session_state.logs.append(s.strip())
            # Format logs for display
            formatted_logs = "\n".join(st.session_state.logs[-30:]) # Show last 30 lines
            self.log_placeholder.code(formatted_logs, language="text")

# --- UI Sidebar ---
with st.sidebar:
    st.title("🏦 FinRegAgents v2")
    st.markdown("KI-Agenten für regulatorische Prüfungen")
    
    st.header("Konfiguration")
    api_key_anthropic = st.text_input("Anthropic API Key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""))
    api_key_openai = st.text_input("OpenAI API Key (Embeddings)", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    
    if api_key_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = api_key_anthropic
    if api_key_openai:
        os.environ["OPENAI_API_KEY"] = api_key_openai
        
    st.divider()
    
    institution = st.text_input("Institution", "Musterbank AG")
    regulatorik = st.selectbox(
        "Regulatorik",
        options=list(KATALOG_REGISTRY.keys()),
        format_func=lambda x: KATALOG_LABELS.get(x, x.upper())
    )
    
    with st.expander("Erweiterte Einstellungen"):
        model = st.selectbox("Modell", ["claude-sonnet-4-5-20250514", "claude-opus-4-5"])
        top_k = st.slider("RAG Chunks (Top-K)", min_value=3, max_value=20, value=8)
        skeptiker = st.checkbox("Skeptiker-Agent aktivieren ⚔️", value=False, help="Adversariales Review der Ergebnisse")
        skeptiker_only_konform = st.checkbox("Skeptiker nur für 'konform'", value=False) if skeptiker else False

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
            st.success(f"Verzeichnis gefunden: {input_dir}")
        else:
            st.warning("Verzeichnis existiert nicht oder Pfad ist ungültig.")
    else:
        # File uploader logic
        uploaded_files = st.file_uploader("Dokumente hochladen (PDF, Excel, JSON/YAML, Bilder)", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Dateien für Prüfung vorbereiten"):
                # Clean TMP dir
                if TMP_DOC_DIR.exists():
                    import shutil
                    shutil.rmtree(TMP_DOC_DIR)
                
                # Create subdirs
                for subdir in ["pdfs", "excel", "interviews", "screenshots", "logs"]:
                    (TMP_DOC_DIR / subdir).mkdir(parents=True, exist_ok=True)
                
                for f in uploaded_files:
                    # Simple heuristic for folder distribution
                    ext = f.name.split('.')[-1].lower()
                    target_dir = "pdfs"
                    if ext in ["xlsx", "csv"]: target_dir = "excel"
                    elif ext in ["json", "yaml"]: target_dir = "interviews"
                    elif ext in ["png", "jpg", "jpeg"]: target_dir = "screenshots"
                    elif ext in ["txt", "log"]: target_dir = "logs"
                    
                    file_path = TMP_DOC_DIR / target_dir / f.name
                    with open(file_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                
                st.success(f"{len(uploaded_files)} Dateien erfolgreich in {TMP_DOC_DIR} abgelegt.")
                input_dir = str(TMP_DOC_DIR)

    st.divider()
    
    # Run Button
    can_run = bool(input_dir) and bool(api_key_anthropic) and bool(api_key_openai)
    if not can_run:
        st.info("Bitte API Keys setzen und gültiges Dokumentenverzeichnis wählen.")
    
    if st.button("🚀 Prüfung starten", disabled=os.environ.get("PIPELINE_RUNNING") == "1" or not can_run, type="primary"):
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
                verbose=True
            )
            
            # Subclass or monkeypatch _log to capture output
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
    st.subheader("Berichte")
    if st.session_state.reports:
        reports = st.session_state.reports
        
        col1, col2, col3 = st.columns(3)
        
        # Download buttons
        if "json" in reports and os.path.exists(reports["json"]):
            with open(reports["json"], "r") as f:
                col1.download_button("🔽 JSON herunterladen", f.read(), file_name=Path(reports["json"]).name, mime="application/json")
                
        if "markdown" in reports and os.path.exists(reports["markdown"]):
            with open(reports["markdown"], "r") as f:
                md_content = f.read()
                col2.download_button("🔽 Markdown herunterladen", md_content, file_name=Path(reports["markdown"]).name, mime="text/markdown")
                
        if "html" in reports and os.path.exists(reports["html"]):
            with open(reports["html"], "r") as f:
                col3.download_button("🔽 HTML herunterladen", f.read(), file_name=Path(reports["html"]).name, mime="text/html")
        
        st.divider()
        st.markdown("### Report Vorschau (Markdown)")
        if "markdown" in reports and os.path.exists(reports["markdown"]):
            with st.container(height=600):
                st.markdown(md_content)
    else:
        st.info("Noch keine Ergebnisse vorhanden. Starte eine Prüfung im ersten Tab.")
