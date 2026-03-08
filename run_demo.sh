#!/usr/bin/env bash
# FinRegAgents – Demo-Runner
# Erzeugt synthetische Prüfungsdokumente und startet die GwG-Prüfung.
#
# Verwendung:
#   ./run_demo.sh                     # Standard-Lauf mit Skeptiker
#   ./run_demo.sh --no-skeptiker      # Ohne adversariales Review
#   ./run_demo.sh --regulatorik dora  # DORA statt GwG
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$SCRIPT_DIR/demo"
OUTPUT_DIR="$SCRIPT_DIR/reports/output"
INSTITUTION="Musterbank AG"
REGULATORIK="gwg"
SKEPTIKER="--skeptiker"
MODEL="claude-sonnet-4-5-20250514"

# Argumente auswerten
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-skeptiker)   SKEPTIKER=""; shift ;;
        --regulatorik)    REGULATORIK="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --output)         OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unbekannter Parameter: $1" >&2; exit 1 ;;
    esac
done

echo "========================================================"
echo " FinRegAgents Demo – Musterbank AG"
echo "========================================================"
echo " Regulatorik : $REGULATORIK"
echo " Modell      : $MODEL"
echo " Skeptiker   : ${SKEPTIKER:-deaktiviert}"
echo " Ausgabe     : $OUTPUT_DIR"
echo "========================================================"
echo ""

# Schritt 1: Demo-Dokumente erzeugen (idempotent)
if [ -d "$DEMO_DIR/pdfs" ] && [ "$(ls -A "$DEMO_DIR/pdfs" 2>/dev/null)" ]; then
    echo "ℹ️  Demo-Dokumente vorhanden, überspringe Generierung."
    echo "   (Neu erzeugen: rm -rf demo && ./run_demo.sh)"
else
    echo "📄 Erzeuge Demo-Dokumente …"
    python3 "$SCRIPT_DIR/tools/create_demo_docs.py" --output "$DEMO_DIR"
fi

echo ""
echo "🚀 Starte FinRegAgents Pipeline …"
echo ""

# Schritt 2: Pipeline starten
python3 "$SCRIPT_DIR/pipeline.py" \
    --input       "$DEMO_DIR" \
    --institution "$INSTITUTION" \
    --regulatorik "$REGULATORIK" \
    --output      "$OUTPUT_DIR" \
    --model       "$MODEL" \
    $SKEPTIKER

echo ""
echo "========================================================"
echo "✅ Demo abgeschlossen. Berichte:"
find "$OUTPUT_DIR" -maxdepth 1 -name "*.html" -o -name "*.md" -o -name "*.json" \
    2>/dev/null | sort | while read -r f; do echo "   $f"; done
echo "========================================================"
