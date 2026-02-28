"""
GwG Audit Pipeline – Prüfbericht-Generator
Erzeugt einen formellen BaFin-Prüfbericht aus den Agenten-Befunden.
Output: JSON (maschinenlesbar) + Markdown (lesbar) + HTML (präsentierbar)
"""

import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.pruef_agent import Befund, Bewertung


# ------------------------------------------------------------------ #
# Bewertungs-Farbcodes für HTML
# ------------------------------------------------------------------ #
BEWERTUNG_STYLE = {
    "konform":        {"emoji": "✅", "color": "#27ae60", "bg": "#eafaf1"},
    "teilkonform":    {"emoji": "⚠️",  "color": "#e67e22", "bg": "#fef9e7"},
    "nicht_konform":  {"emoji": "🔴", "color": "#c0392b", "bg": "#fdedec"},
    "nicht_prüfbar":  {"emoji": "❓", "color": "#7f8c8d", "bg": "#f2f3f4"},
}

SCHWEREGRAD_STYLE = {
    "wesentlich": {"label": "WESENTLICH", "color": "#c0392b"},
    "bedeutsam":  {"label": "BEDEUTSAM",  "color": "#e67e22"},
    "gering":     {"label": "GERING",     "color": "#27ae60"},
}


class GwGBerichtGenerator:

    def __init__(self, institution: str = "Prüfinstitut", pruefer: str = "KI-Prüfungssystem"):
        self.institution = institution
        self.pruefer = pruefer
        self.pruefungsdatum = datetime.now().strftime("%d.%m.%Y")

    # ------------------------------------------------------------------ #
    # Public: Alle Formate auf einmal generieren
    # ------------------------------------------------------------------ #
    def generiere_alle_berichte(
        self,
        sektionsergebnisse: list,
        output_dir: str = "./reports/output"
    ) -> dict[str, str]:
        """
        Generiert JSON, Markdown und HTML-Berichte.
        Returns dict mit Pfaden zu den erzeugten Dateien.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Zusammenfassung berechnen
        zusammenfassung = self._berechne_zusammenfassung(sektionsergebnisse)

        # JSON
        json_path = f"{output_dir}/gwg_pruefbericht_{ts}.json"
        self._schreibe_json(sektionsergebnisse, zusammenfassung, json_path)

        # Markdown
        md_path = f"{output_dir}/gwg_pruefbericht_{ts}.md"
        self._schreibe_markdown(sektionsergebnisse, zusammenfassung, md_path)

        # HTML
        html_path = f"{output_dir}/gwg_pruefbericht_{ts}.html"
        self._schreibe_html(sektionsergebnisse, zusammenfassung, html_path)

        return {"json": json_path, "markdown": md_path, "html": html_path}

    # ------------------------------------------------------------------ #
    # Zusammenfassung berechnen
    # ------------------------------------------------------------------ #
    def _berechne_zusammenfassung(self, sektionsergebnisse: list) -> dict:
        alle_befunde = [b for s in sektionsergebnisse for b in s.befunde]
        bewertungs_zähler = Counter(b.bewertung.value for b in alle_befunde)
        mängel = [b for b in alle_befunde if b.bewertung.value == "nicht_konform"]
        teilkonform = [b for b in alle_befunde if b.bewertung.value == "teilkonform"]

        wesentliche_mängel = [b for b in mängel if b.schweregrad == "wesentlich"]

        # Gesamtbewertung
        if wesentliche_mängel:
            gesamtbewertung = "ERHEBLICHE MÄNGEL"
            gesamtfarbe = "#c0392b"
        elif mängel or len(teilkonform) >= 3:
            gesamtbewertung = "MÄNGEL FESTGESTELLT"
            gesamtfarbe = "#e67e22"
        elif teilkonform:
            gesamtbewertung = "TEILKONFORM – NACHBESSERUNG ERFORDERLICH"
            gesamtfarbe = "#f39c12"
        else:
            gesamtbewertung = "KONFORM"
            gesamtfarbe = "#27ae60"

        return {
            "gesamtbewertung": gesamtbewertung,
            "gesamtfarbe": gesamtfarbe,
            "total_prueffelder": len(alle_befunde),
            "konform": bewertungs_zähler.get("konform", 0),
            "teilkonform": bewertungs_zähler.get("teilkonform", 0),
            "nicht_konform": bewertungs_zähler.get("nicht_konform", 0),
            "nicht_pruefbar": bewertungs_zähler.get("nicht_prüfbar", 0),
            "anzahl_mängel": len(mängel),
            "anzahl_wesentliche_mängel": len(wesentliche_mängel),
            "kritische_befunde": [
                {"id": b.prueffeld_id, "frage": b.frage, "schweregrad": b.schweregrad,
                 "mangel": b.mangel_text}
                for b in sorted(mängel + teilkonform,
                                key=lambda x: 0 if x.schweregrad == "wesentlich" else 1)
            ],
        }

    # ------------------------------------------------------------------ #
    # JSON-Report
    # ------------------------------------------------------------------ #
    def _schreibe_json(self, sektionsergebnisse, zusammenfassung, path):
        report = {
            "meta": {
                "institution": self.institution,
                "pruefer": self.pruefer,
                "pruefungsdatum": self.pruefungsdatum,
                "pruefungstyp": "GwG-Sonderprüfung (simuliert)",
                "basis": ["GwG 2017 i.d.F. 2024", "§25h KWG", "BaFin AuA GwG"],
            },
            "zusammenfassung": zusammenfassung,
            "sektionen": [
                {
                    "id": s.sektion_id,
                    "titel": s.titel,
                    "befunde": [
                        {
                            "id": b.prueffeld_id,
                            "frage": b.frage,
                            "bewertung": b.bewertung.value,
                            "schweregrad": b.schweregrad,
                            "begruendung": b.begruendung,
                            "belegte_textstellen": b.belegte_textstellen,
                            "mangel_text": b.mangel_text,
                            "empfehlungen": b.empfehlungen,
                            "quellen": b.quellen,
                        }
                        for b in s.befunde
                    ],
                }
                for s in sektionsergebnisse
            ],
        }
        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  📄 JSON-Bericht: {path}")

    # ------------------------------------------------------------------ #
    # Markdown-Report
    # ------------------------------------------------------------------ #
    def _schreibe_markdown(self, sektionsergebnisse, zusammenfassung, path):
        lines = []

        # Header
        lines += [
            f"# GwG-Sonderprüfungsbericht (simuliert)",
            f"",
            f"| | |",
            f"|---|---|",
            f"| **Institut** | {self.institution} |",
            f"| **Prüfer** | {self.pruefer} |",
            f"| **Prüfungsdatum** | {self.pruefungsdatum} |",
            f"| **Prüfungsgrundlage** | GwG, §25h KWG, BaFin AuA |",
            f"",
            f"---",
            f"",
            f"## Gesamtergebnis: {zusammenfassung['gesamtbewertung']}",
            f"",
            f"| Bewertung | Anzahl |",
            f"|---|---|",
            f"| ✅ Konform | {zusammenfassung['konform']} |",
            f"| ⚠️ Teilkonform | {zusammenfassung['teilkonform']} |",
            f"| 🔴 Nicht konform | {zusammenfassung['nicht_konform']} |",
            f"| ❓ Nicht prüfbar | {zusammenfassung['nicht_pruefbar']} |",
            f"| **Gesamt** | **{zusammenfassung['total_prueffelder']}** |",
            f"",
        ]

        # Mängelübersicht
        if zusammenfassung["kritische_befunde"]:
            lines += [
                f"## Mängelkatalog ({zusammenfassung['anzahl_mängel']} Mängel, "
                f"davon {zusammenfassung['anzahl_wesentliche_mängel']} wesentlich)",
                f"",
            ]
            for m in zusammenfassung["kritische_befunde"]:
                sg = m.get("schweregrad", "").upper()
                lines.append(f"- **[{m['id']}] [{sg}]** {m['mangel'] or m['frage']}")
            lines.append("")

        lines.append("---")

        # Detailbefunde
        lines.append("## Detailbefunde")
        for sektion in sektionsergebnisse:
            lines += [f"", f"### {sektion.sektion_id}: {sektion.titel}", ""]
            for b in sektion.befunde:
                style = BEWERTUNG_STYLE.get(b.bewertung.value, {})
                emoji = style.get("emoji", "")
                lines += [
                    f"#### {b.prueffeld_id}: {b.frage}",
                    f"",
                    f"**Bewertung:** {emoji} `{b.bewertung.value.upper()}`  ",
                    f"**Schweregrad:** {b.schweregrad}  ",
                    f"",
                    f"{b.begruendung}",
                    f"",
                ]
                if b.belegte_textstellen:
                    lines.append("**Belegte Textstellen:**")
                    for t in b.belegte_textstellen:
                        lines.append(f"> {t}")
                    lines.append("")
                if b.mangel_text:
                    lines += [f"**⚠ Mangel:** {b.mangel_text}", ""]
                if b.empfehlungen:
                    lines.append("**Empfehlungen:**")
                    for e in b.empfehlungen:
                        lines.append(f"- {e}")
                    lines.append("")
                if b.quellen:
                    lines.append(f"*Quellen: {', '.join(b.quellen)}*")
                lines.append("---")

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  📄 Markdown-Bericht: {path}")

    # ------------------------------------------------------------------ #
    # HTML-Report (druckfähig, BaFin-Stil)
    # ------------------------------------------------------------------ #
    def _schreibe_html(self, sektionsergebnisse, zusammenfassung, path):
        html = self._html_header(zusammenfassung)
        html += self._html_zusammenfassung(zusammenfassung)
        if zusammenfassung["kritische_befunde"]:
            html += self._html_mangelkatalog(zusammenfassung)
        html += self._html_detailbefunde(sektionsergebnisse)
        html += self._html_footer()
        Path(path).write_text(html, encoding="utf-8")
        print(f"  📄 HTML-Bericht: {path}")

    def _html_header(self, z) -> str:
        return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GwG-Prüfbericht – {self.institution}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
          color: #2c3e50; background: #f5f6fa; line-height: 1.6; }}
  .page {{ max-width: 1100px; margin: 0 auto; background: white;
           box-shadow: 0 2px 20px rgba(0,0,0,0.08); }}
  .header {{ background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
             color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .header .subtitle {{ opacity: 0.8; font-size: 13px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
                border-bottom: 1px solid #ecf0f1; }}
  .meta-item {{ padding: 12px 20px; border-right: 1px solid #ecf0f1; }}
  .meta-item:last-child {{ border-right: none; }}
  .meta-label {{ font-size: 10px; text-transform: uppercase; color: #95a5a6;
                 font-weight: 600; letter-spacing: 0.5px; }}
  .meta-value {{ font-weight: 600; color: #2c3e50; margin-top: 2px; }}
  .content {{ padding: 32px 40px; }}
  .gesamtbewertung {{ padding: 16px 20px; border-radius: 8px; margin-bottom: 24px;
                      font-weight: 700; font-size: 16px; border-left: 5px solid {z['gesamtfarbe']};
                      background: {z['gesamtfarbe']}15; color: {z['gesamtfarbe']}; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                 margin-bottom: 32px; }}
  .stat-card {{ padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #ecf0f1; }}
  .stat-number {{ font-size: 28px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #7f8c8d; text-transform: uppercase; margin-top: 4px; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ font-size: 15px; font-weight: 700; color: #1a3a5c;
                    padding: 10px 0; border-bottom: 2px solid #3498db;
                    margin-bottom: 16px; }}
  .befund-card {{ border: 1px solid #ecf0f1; border-radius: 8px; margin-bottom: 12px;
                  overflow: hidden; }}
  .befund-header {{ display: flex; align-items: center; gap: 12px;
                    padding: 12px 16px; background: #f8f9fa; }}
  .befund-id {{ font-size: 11px; font-weight: 700; color: #7f8c8d;
                font-family: monospace; white-space: nowrap; }}
  .befund-frage {{ flex: 1; font-weight: 600; color: #2c3e50; font-size: 13px; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 11px;
            font-weight: 700; white-space: nowrap; }}
  .befund-body {{ padding: 12px 16px; }}
  .begruendung {{ color: #34495e; margin-bottom: 10px; }}
  .textstellen {{ background: #f8f9fa; border-left: 3px solid #3498db;
                  padding: 8px 12px; margin: 8px 0; font-size: 12px;
                  color: #555; border-radius: 0 4px 4px 0; }}
  .mangel-box {{ background: #fdedec; border: 1px solid #fadbd8;
                 border-radius: 6px; padding: 10px 14px; margin: 8px 0;
                 color: #922b21; font-size: 12px; }}
  .empfehlungen {{ background: #eafaf1; border: 1px solid #a9dfbf;
                   border-radius: 6px; padding: 10px 14px; margin: 8px 0; }}
  .empfehlung-item {{ color: #1e8449; font-size: 12px; margin-bottom: 4px; }}
  .quellen {{ font-size: 11px; color: #95a5a6; margin-top: 8px; }}
  .mangelkatalog {{ background: #fdedec; border-radius: 8px; padding: 20px;
                    margin-bottom: 32px; border: 1px solid #fadbd8; }}
  .mangelkatalog h2 {{ color: #922b21; margin-bottom: 12px; font-size: 15px; }}
  .mangel-item {{ padding: 6px 0; border-bottom: 1px solid #fadbd8;
                  font-size: 12px; color: #34495e; }}
  .mangel-item:last-child {{ border-bottom: none; }}
  .footer {{ background: #f8f9fa; padding: 16px 40px; font-size: 11px;
             color: #95a5a6; border-top: 1px solid #ecf0f1; text-align: center; }}
  @media print {{ body {{ background: white; }} .page {{ box-shadow: none; }} }}
</style>
</head>
<body>
<div class="page">
<div class="header">
  <h1>GwG-Sonderprüfungsbericht</h1>
  <div class="subtitle">Simulierte Sonderprüfung gemäß §25h KWG · GwG · BaFin-Auslegungshinweise</div>
</div>
<div class="meta-grid">
  <div class="meta-item"><div class="meta-label">Institut</div>
    <div class="meta-value">{self.institution}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfer</div>
    <div class="meta-value">{self.pruefer}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfungsdatum</div>
    <div class="meta-value">{self.pruefungsdatum}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfungstyp</div>
    <div class="meta-value">GwG-Sonderprüfung</div></div>
</div>
<div class="content">
"""

    def _html_zusammenfassung(self, z) -> str:
        return f"""
<div class="gesamtbewertung">Gesamtergebnis: {z['gesamtbewertung']}</div>
<div class="stats-grid">
  <div class="stat-card" style="background:#eafaf1;border-color:#a9dfbf">
    <div class="stat-number" style="color:#27ae60">{z['konform']}</div>
    <div class="stat-label">✅ Konform</div></div>
  <div class="stat-card" style="background:#fef9e7;border-color:#f9e79f">
    <div class="stat-number" style="color:#e67e22">{z['teilkonform']}</div>
    <div class="stat-label">⚠️ Teilkonform</div></div>
  <div class="stat-card" style="background:#fdedec;border-color:#fadbd8">
    <div class="stat-number" style="color:#c0392b">{z['nicht_konform']}</div>
    <div class="stat-label">🔴 Nicht konform</div></div>
  <div class="stat-card" style="background:#f2f3f4;border-color:#d5d8dc">
    <div class="stat-number" style="color:#7f8c8d">{z['nicht_pruefbar']}</div>
    <div class="stat-label">❓ Nicht prüfbar</div></div>
</div>
"""

    def _html_mangelkatalog(self, z) -> str:
        items = "".join(
            f'<div class="mangel-item">'
            f'<strong>[{m["id"]}] [{(m.get("schweregrad") or "").upper()}]</strong> '
            f'{m.get("mangel") or m["frage"]}</div>'
            for m in z["kritische_befunde"]
        )
        return f"""
<div class="mangelkatalog">
  <h2>⚠ Mängelkatalog ({z['anzahl_mängel']} Mängel,
      davon {z['anzahl_wesentliche_mängel']} wesentlich)</h2>
  {items}
</div>
"""

    def _html_detailbefunde(self, sektionsergebnisse) -> str:
        html = '<h2 style="font-size:17px;color:#1a3a5c;margin-bottom:24px">Detailbefunde</h2>'
        for sektion in sektionsergebnisse:
            html += f'<div class="section">'
            html += f'<div class="section-title">{sektion.sektion_id}: {sektion.titel}</div>'
            for b in sektion.befunde:
                style = BEWERTUNG_STYLE.get(b.bewertung.value, {"emoji":"?","color":"#7f8c8d","bg":"#f2f3f4"})
                sg_style = SCHWEREGRAD_STYLE.get(b.schweregrad or "", {"label":b.schweregrad or "","color":"#7f8c8d"})

                textstellen_html = ""
                if b.belegte_textstellen:
                    for t in b.belegte_textstellen:
                        textstellen_html += f'<div class="textstellen">📎 {t}</div>'

                mangel_html = f'<div class="mangel-box">⚠ <strong>Mangel:</strong> {b.mangel_text}</div>' if b.mangel_text else ""

                empf_html = ""
                if b.empfehlungen:
                    items = "".join(f'<div class="empfehlung-item">→ {e}</div>' for e in b.empfehlungen)
                    empf_html = f'<div class="empfehlungen"><strong>Empfehlungen:</strong>{items}</div>'

                quellen_html = f'<div class="quellen">Quellen: {", ".join(b.quellen)}</div>' if b.quellen else ""

                html += f"""
<div class="befund-card">
  <div class="befund-header">
    <span class="befund-id">{b.prueffeld_id}</span>
    <span class="befund-frage">{b.frage}</span>
    <span class="badge" style="background:{style['bg']};color:{style['color']}">
      {style['emoji']} {b.bewertung.value.upper()}</span>
    <span class="badge" style="background:#f2f3f4;color:{sg_style['color']}">
      {sg_style['label']}</span>
  </div>
  <div class="befund-body">
    <div class="begruendung">{b.begruendung}</div>
    {textstellen_html}
    {mangel_html}
    {empf_html}
    {quellen_html}
  </div>
</div>"""
            html += '</div>'
        return html

    def _html_footer(self) -> str:
        return f"""
</div>
<div class="footer">
  Dieser Bericht wurde von einem KI-gestützten Prüfsystem generiert. Er dient ausschließlich
  internen Simulationszwecken und ersetzt keine offizielle BaFin-Prüfung. Stand: {self.pruefungsdatum}
</div>
</div></body></html>"""
