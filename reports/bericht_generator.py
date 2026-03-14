"""
FinRegAgents – Prüfbericht-Generator (v2)
Erzeugt einen formellen Prüfbericht aus den Agenten-Befunden.
Output: JSON (maschinenlesbar) + Markdown (lesbar) + HTML (präsentierbar)

Änderungen gegenüber v1:
  - XSS-Schutz: html.escape() für alle Befund-Texte im HTML-Report
  - Dynamische Regulatorik-Labels (nicht mehr GwG-hardcoded)
  - Audit-Trail: Modell, Katalog-Version, Confidence-Statistiken
  - Gesamtbewertung berücksichtigt nicht_prüfbar-Anteil
  - Confidence-Indikatoren und Review-Markierungen im Bericht
"""

import json
import html
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING, Optional

from agents.provenance import CorroborationStatus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------ #
# Bewertungs-Farbcodes
# ------------------------------------------------------------------ #
BEWERTUNG_STYLE = {
    "konform": {"emoji": "✅", "color": "#27ae60", "bg": "#eafaf1"},
    "teilkonform": {"emoji": "⚠️", "color": "#e67e22", "bg": "#fef9e7"},
    "nicht_konform": {"emoji": "🔴", "color": "#c0392b", "bg": "#fdedec"},
    "nicht_prüfbar": {"emoji": "❓", "color": "#7f8c8d", "bg": "#f2f3f4"},
    "disputed": {"emoji": "⚖️", "color": "#8e44ad", "bg": "#f4ecf7"},
}

SCHWEREGRAD_STYLE = {
    "wesentlich": {"label": "WESENTLICH", "color": "#c0392b"},
    "bedeutsam": {"label": "BEDEUTSAM", "color": "#e67e22"},
    "gering": {"label": "GERING", "color": "#27ae60"},
}

REGULATORIK_LABELS = {
    "gwg": (
        "GwG-Sonderprüfungsbericht",
        "Simulierte Sonderprüfung gemäß §25h KWG · GwG · BaFin-Auslegungshinweise",
        "GwG-Sonderprüfung",
        ["GwG 2017 i.d.F. 2024", "§25h KWG", "BaFin AuA GwG"],
    ),
    "dora": (
        "DORA-Prüfungsbericht",
        "Prüfung der digitalen operationalen Resilienz gemäß DORA (EU) 2022/2554",
        "DORA-Prüfung",
        ["DORA (EU) 2022/2554", "RTS ICT Risk", "RTS Incident Reporting"],
    ),
    "marisk": (
        "MaRisk-Prüfungsbericht",
        "Prüfung gemäß MaRisk und §25a KWG",
        "MaRisk-Prüfung",
        ["MaRisk 2023 AT/BT", "§25a KWG", "EBA-Leitlinien"],
    ),
    "wphg": (
        "WpHG/MaComp-Prüfungsbericht",
        "Prüfung gemäß WpHG, MaComp und MiFID II",
        "WpHG/MaComp-Prüfung",
        ["WpHG", "MaComp", "MAR", "MiFID II"],
    ),
}


def _esc(text: str) -> str:
    """HTML-Escape für sichere Einbettung in HTML-Reports."""
    if text is None:
        return ""
    return html.escape(str(text))


# Provenance status display mappings
_PROV_EMOJI = {
    CorroborationStatus.CORROBORATED: "✅",
    CorroborationStatus.SINGLE_SOURCED: "⚠️",
    CorroborationStatus.UNVERIFIED: "❓",
}
_PROV_LABEL_MD = {
    CorroborationStatus.CORROBORATED: "belegt (2+ Quellen)",
    CorroborationStatus.SINGLE_SOURCED: "Einzelquelle",
    CorroborationStatus.UNVERIFIED: "unbelegt",
}
_PROV_COLOR_HTML = {
    CorroborationStatus.CORROBORATED: "#27ae60",
    CorroborationStatus.SINGLE_SOURCED: "#e67e22",
    CorroborationStatus.UNVERIFIED: "#7f8c8d",
}
_PROV_BG_HTML = {
    CorroborationStatus.CORROBORATED: "#eafaf1",
    CorroborationStatus.SINGLE_SOURCED: "#fef9e7",
    CorroborationStatus.UNVERIFIED: "#f2f3f4",
}


def _esc_md(text: str) -> str:
    """Escape pipe chars in Markdown table cells."""
    return str(text).replace("|", "\\|")


def _render_provenance_markdown(claim_provenance: list) -> list[str]:
    """Render a compact provenance table in Markdown."""
    if not claim_provenance:
        return []
    lines = [
        "**Aussagen-Provenance:**",
        "",
        "| ID | Aussage | Status |",
        "|---|---|---|",
    ]
    for cp in claim_provenance:
        emoji = _PROV_EMOJI.get(cp.status, "❓")
        label = _PROV_LABEL_MD.get(cp.status, cp.status.value)
        claim_short = (
            cp.claim_text[:80] + "…" if len(cp.claim_text) > 80 else cp.claim_text
        )
        lines.append(
            f'| `{cp.provenance_id}` | "{_esc_md(claim_short)}" | {emoji} {label} |'
        )
    lines.append("")
    return lines


def _render_provenance_html(claim_provenance: list) -> str:
    """Render a compact provenance table in HTML."""
    if not claim_provenance:
        return ""
    rows = ""
    for cp in claim_provenance:
        emoji = _PROV_EMOJI.get(cp.status, "❓")
        label = _PROV_LABEL_MD.get(cp.status, cp.status.value)
        color = _PROV_COLOR_HTML.get(cp.status, "#7f8c8d")
        bg = _PROV_BG_HTML.get(cp.status, "#f2f3f4")
        claim_short = (
            cp.claim_text[:80] + "…" if len(cp.claim_text) > 80 else cp.claim_text
        )
        rows += (
            f"<tr>"
            f'<td style="font-family:monospace;font-size:11px;white-space:nowrap">'
            f"{_esc(cp.provenance_id)}</td>"
            f'<td style="font-size:12px">&ldquo;{_esc(claim_short)}&rdquo;</td>'
            f'<td><span style="background:{bg};color:{color};padding:2px 8px;'
            f'border-radius:10px;font-size:11px;font-weight:700;white-space:nowrap">'
            f"{emoji} {_esc(label)}</span></td>"
            f"</tr>"
        )
    return (
        '<div style="margin:8px 0">'
        '<strong style="font-size:12px;color:#555">Aussagen-Provenance:</strong>'
        '<table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:12px">'
        "<thead><tr>"
        '<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #ecf0f1;'
        'color:#7f8c8d;font-size:10px">ID</th>'
        '<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #ecf0f1;'
        'color:#7f8c8d;font-size:10px">Aussage</th>'
        '<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #ecf0f1;'
        'color:#7f8c8d;font-size:10px">Status</th>'
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


class BerichtGenerator:
    def __init__(
        self,
        institution: str = "Prüfinstitut",
        pruefer: str = "KI-Prüfungssystem",
        regulatorik: str = "gwg",
        model: str = "unbekannt",
        katalog_version: str = "unbekannt",
    ):
        self.institution = institution
        self.pruefer = pruefer
        self.regulatorik = regulatorik
        self.model = model
        self.katalog_version = katalog_version
        self.pruefungsdatum = datetime.now().strftime("%d.%m.%Y")

        labels = REGULATORIK_LABELS.get(regulatorik)
        if labels is None:
            raise ValueError(
                f"Unbekannte Regulatorik: '{regulatorik}'. "
                f"Verfügbar: {list(REGULATORIK_LABELS.keys())}"
            )
        self.report_title = labels[0]
        self.report_subtitle = labels[1]
        self.report_typ = labels[2]
        self.report_basis = labels[3]

    # ------------------------------------------------------------------ #
    # Public: Alle Formate auf einmal generieren
    # ------------------------------------------------------------------ #
    def generiere_alle_berichte(
        self,
        sektionsergebnisse: list,
        output_dir: str = "./reports/output",
        token_stats: Optional[dict] = None,
        stats_file: Optional[str] = None,
        verbose: bool = False,
    ) -> dict[str, str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.regulatorik}_pruefbericht_{ts}"

        zusammenfassung = self._berechne_zusammenfassung(sektionsergebnisse)

        json_path = f"{output_dir}/{prefix}.json"
        self._schreibe_json(
            sektionsergebnisse,
            zusammenfassung,
            json_path,
            token_stats,
            stats_file,
            verbose,
        )

        md_path = f"{output_dir}/{prefix}.md"
        self._schreibe_markdown(
            sektionsergebnisse,
            zusammenfassung,
            md_path,
            token_stats,
            stats_file,
            verbose,
        )

        html_path = f"{output_dir}/{prefix}.html"
        self._schreibe_html(
            sektionsergebnisse,
            zusammenfassung,
            html_path,
            token_stats,
            stats_file,
            verbose,
        )

        return {"json": json_path, "markdown": md_path, "html": html_path}

    # ------------------------------------------------------------------ #
    # Zusammenfassung berechnen
    # ------------------------------------------------------------------ #
    def _berechne_zusammenfassung(self, sektionsergebnisse: list) -> dict:
        alle_befunde = [b for s in sektionsergebnisse for b in s.befunde]
        bewertungs_zähler = Counter(b.bewertung.value for b in alle_befunde)
        mängel = [b for b in alle_befunde if b.bewertung.value == "nicht_konform"]
        teilkonform = [b for b in alle_befunde if b.bewertung.value == "teilkonform"]
        nicht_pruefbar = [
            b for b in alle_befunde if b.bewertung.value == "nicht_prüfbar"
        ]
        strittig = [b for b in alle_befunde if b.bewertung.value == "disputed"]
        review_nötig = [b for b in alle_befunde if b.review_erforderlich]

        wesentliche_mängel = [b for b in mängel if b.schweregrad == "wesentlich"]

        total = len(alle_befunde)
        gewertet = max(0, total - len(strittig))
        unresolved = len(nicht_pruefbar) + len(strittig)
        np_quote = unresolved / gewertet if gewertet else 1.0

        # Confidence-Statistiken
        confidences = [b.confidence for b in alle_befunde]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Gesamtbewertung – JETZT mit nicht_prüfbar-Berücksichtigung
        if wesentliche_mängel:
            gesamtbewertung = "ERHEBLICHE MÄNGEL"
            gesamtfarbe = "#c0392b"
        elif np_quote >= 0.5:
            gesamtbewertung = "UNZUREICHENDE EVIDENZ – PRÜFUNG NICHT BELASTBAR"
            gesamtfarbe = "#7f8c8d"
        elif mängel or len(teilkonform) >= 3:
            gesamtbewertung = "MÄNGEL FESTGESTELLT"
            gesamtfarbe = "#e67e22"
        elif np_quote >= 0.3:
            gesamtbewertung = "EINGESCHRÄNKT BELASTBAR – HOHER ANTEIL NICHT PRÜFBAR"
            gesamtfarbe = "#f39c12"
        elif teilkonform:
            gesamtbewertung = "TEILKONFORM – NACHBESSERUNG ERFORDERLICH"
            gesamtfarbe = "#f39c12"
        else:
            gesamtbewertung = "KONFORM"
            gesamtfarbe = "#27ae60"

        return {
            "gesamtbewertung": gesamtbewertung,
            "gesamtfarbe": gesamtfarbe,
            "total_prueffelder": total,
            "gewertete_prueffelder": gewertet,
            "konform": bewertungs_zähler.get("konform", 0),
            "teilkonform": bewertungs_zähler.get("teilkonform", 0),
            "nicht_konform": bewertungs_zähler.get("nicht_konform", 0),
            "nicht_pruefbar": bewertungs_zähler.get("nicht_prüfbar", 0),
            "disputed": bewertungs_zähler.get("disputed", 0),
            "nicht_pruefbar_quote": round(np_quote * 100, 1),
            "review_erforderlich": len(review_nötig),
            "avg_confidence": round(avg_confidence, 3),
            "anzahl_mängel": len(mängel),
            "anzahl_wesentliche_mängel": len(wesentliche_mängel),
            "kritische_befunde": [
                {
                    "id": b.prueffeld_id,
                    "frage": b.frage,
                    "schweregrad": b.schweregrad,
                    "mangel": b.mangel_text,
                }
                for b in sorted(
                    mängel + teilkonform,
                    key=lambda x: 0 if x.schweregrad == "wesentlich" else 1,
                )
            ],
            "strittige_befunde": [
                {
                    "id": b.prueffeld_id,
                    "frage": b.frage,
                    "empfehlung": next(
                        (
                            h.split("'")[1]
                            for h in b.validierungshinweise
                            if "Skeptiker widerspricht" in h and "'" in h
                        ),
                        "unklar",
                    ),
                }
                for b in strittig
            ],
            "audit_trail": {
                "modell": self.model,
                "katalog_version": self.katalog_version,
                "regulatorik": self.regulatorik,
                "generator_version": "finreg-agents v2.0",
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------ #
    # JSON-Report
    # ------------------------------------------------------------------ #
    def _schreibe_json(
        self,
        sektionsergebnisse,
        zusammenfassung,
        path,
        token_stats=None,
        stats_file=None,
        verbose=False,
    ):
        report = {
            "meta": {
                "institution": self.institution,
                "pruefer": self.pruefer,
                "pruefungsdatum": self.pruefungsdatum,
                "pruefungstyp": f"{self.report_typ} (simuliert)",
                "basis": self.report_basis,
                "audit_trail": zusammenfassung["audit_trail"],
            },
            "zusammenfassung": zusammenfassung,
            "token_stats": token_stats or {},
            "stats_file": stats_file,
            "sektionen": [
                {
                    "id": s.sektion_id,
                    "titel": s.titel,
                    "review_quote": round(s.review_quote, 2),
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
                            "confidence": b.confidence,
                            "confidence_level": b.confidence_level,
                            "confidence_guards": b.confidence_guards,
                            "low_confidence_reasons": b.low_confidence_reasons,
                            "claim_list": b.claim_list,
                            "review_erforderlich": b.review_erforderlich,
                            "validierungshinweise": b.validierungshinweise,
                            "term_drift_warnings": getattr(
                                b, "term_drift_warnings", []
                            ),
                            "claim_provenance": [
                                {
                                    "provenance_id": cp.provenance_id,
                                    "claim_text": cp.claim_text,
                                    "status": cp.status.value,
                                    "source_chunk_ids": cp.source_chunk_ids,
                                }
                                for cp in getattr(b, "claim_provenance", [])
                            ],
                            **({"token_usage": b.token_usage} if verbose else {}),
                        }
                        for b in s.befunde
                    ],
                }
                for s in sektionsergebnisse
            ],
        }
        Path(path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON-Bericht: %s", path)

    # ------------------------------------------------------------------ #
    # Markdown-Report
    # ------------------------------------------------------------------ #
    def _schreibe_markdown(
        self,
        sektionsergebnisse,
        zusammenfassung,
        path,
        token_stats=None,
        stats_file=None,
        verbose=False,
    ):
        z = zusammenfassung
        lines = [
            f"# {self.report_title} (simuliert)",
            "",
            "| | |",
            "|---|---|",
            f"| **Institut** | {self.institution} |",
            f"| **Prüfer** | {self.pruefer} |",
            f"| **Prüfungsdatum** | {self.pruefungsdatum} |",
            f"| **Prüfungsgrundlage** | {', '.join(self.report_basis)} |",
            f"| **Modell** | {self.model} |",
            f"| **Ø Confidence** | {z['avg_confidence']:.1%} |",
            "",
            "---",
            "",
            f"## Gesamtergebnis: {z['gesamtbewertung']}",
            "",
            "| Bewertung | Anzahl |",
            "|---|---|",
            f"| ✅ Konform | {z['konform']} |",
            f"| ⚠️ Teilkonform | {z['teilkonform']} |",
            f"| 🔴 Nicht konform | {z['nicht_konform']} |",
            f"| ❓ Nicht prüfbar | {z['nicht_pruefbar']} ({z['nicht_pruefbar_quote']}%) |",
            f"| ⚖️ Strittig (disputed) | {z['disputed']} |",
            f"| **Gesamt** | **{z['total_prueffelder']}** |",
            f"| **Gewertet** | **{z['gewertete_prueffelder']}** |",
            f"| 🔍 Review erforderlich | {z['review_erforderlich']} |",
            "",
        ]

        if token_stats:
            kosten = token_stats.get("kosten_schaetzung", {})
            lines += [
                "## Token-Stats",
                "",
                f"- Gesamt Tokens: `{token_stats.get('gesamt', {}).get('total', 0)}`",
                f"- Input/Output: `{token_stats.get('gesamt', {}).get('input', 0)}` / `{token_stats.get('gesamt', {}).get('output', 0)}`",
                f"- Kostenschätzung: `{kosten.get('total_cost', 0)} {kosten.get('currency', 'USD')}`",
                f"- Pricing-Stand: `{kosten.get('pricing_timestamp', 'n/a')}`",
                f"- Stats-Datei: `{stats_file or token_stats.get('stats_file', '')}`",
                "",
            ]

        # Nicht-prüfbar-Warnung
        if z["nicht_pruefbar_quote"] >= 30:
            lines += [
                f"> ⚠️ **Hinweis:** {z['nicht_pruefbar_quote']}% der Prüffelder konnten nicht "
                f"bewertet werden. Die Prüfungsergebnisse sind eingeschränkt belastbar.",
                "",
            ]

        # Mängelübersicht
        if z["kritische_befunde"]:
            lines += [
                f"## Mängelkatalog ({z['anzahl_mängel']} Mängel, "
                f"davon {z['anzahl_wesentliche_mängel']} wesentlich)",
                "",
            ]
            for m in z["kritische_befunde"]:
                sg = (m.get("schweregrad") or "").upper()
                lines.append(f"- **[{m['id']}] [{sg}]** {m['mangel'] or m['frage']}")
            lines.append("")

        if z["strittige_befunde"]:
            lines += [
                f"## Strittige Befunde ({len(z['strittige_befunde'])})",
                "",
            ]
            for s in z["strittige_befunde"]:
                lines.append(
                    f"- **[{s['id']}]** {s['frage']} *(Skeptiker-Empfehlung: {s['empfehlung']})*"
                )
            lines.append("")

        lines.append("---")

        # Detailbefunde
        lines.append("## Detailbefunde")
        for sektion in sektionsergebnisse:
            lines += ["", f"### {sektion.sektion_id}: {sektion.titel}", ""]
            if sektion.review_quote > 0.3:
                lines.append(
                    f"> ⚠️ {sektion.review_quote:.0%} der Befunde in dieser Sektion erfordern manuelles Review.\n"
                )

            for b in sektion.befunde:
                style = BEWERTUNG_STYLE.get(b.bewertung.value, {})
                emoji = style.get("emoji", "")
                conf_str = f" | Confidence: {b.confidence:.0%}" if b.confidence else ""
                review_str = (
                    " | 🔍 REVIEW ERFORDERLICH" if b.review_erforderlich else ""
                )

                lines += [
                    f"#### {b.prueffeld_id}: {b.frage}",
                    "",
                    f"**Bewertung:** {emoji} `{b.bewertung.value.upper()}`{conf_str}{review_str}  ",
                    f"**Schweregrad:** {b.schweregrad}  ",
                    f"**Confidence-Level:** `{b.confidence_level}`  ",
                    "",
                    f"{b.begruendung}",
                    "",
                ]
                if b.low_confidence_reasons:
                    lines += [
                        f"**Low-Confidence-Reasons:** `{', '.join(b.low_confidence_reasons)}`",
                        "",
                    ]
                if b.claim_list:
                    lines.append("**Claim-Provenance:**")
                    for c in b.claim_list:
                        status = c.get("status", "unverified")
                        tag = c.get("skeptiker_tag", "none")
                        pid = ", ".join(c.get("provenance_ids", [])) or "n/a"
                        lines.append(
                            f"- `{c.get('claim_id', 'C?')}` [{status}] [{tag}] "
                            f"{c.get('text', '')} _(prov: {pid})_"
                        )
                    lines.append("")
                if b.confidence_guards:
                    lines += [
                        f"**Confidence-Guards:** `passed={b.confidence_guards.get('passed')}` | "
                        f"`violations={', '.join(b.confidence_guards.get('violations', [])) or 'none'}`",
                        "",
                    ]
                if verbose and b.token_usage:
                    lines += [
                        f"**Token-Usage:** input `{b.token_usage.get('input', 0)}` | "
                        f"output `{b.token_usage.get('output', 0)}` | total `{b.token_usage.get('total', 0)}`",
                        "",
                    ]
                claim_prov = getattr(b, "claim_provenance", [])
                if claim_prov:
                    lines += _render_provenance_markdown(claim_prov)
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
                if b.validierungshinweise:
                    lines.append("**Validierungshinweise:**")
                    for v in b.validierungshinweise:
                        lines.append(f"- ⚡ {v}")
                    lines.append("")
                term_drift = getattr(b, "term_drift_warnings", [])
                if term_drift:
                    lines.append("**Term-Drift-Warnungen:**")
                    for w in term_drift:
                        lines.append(f"- 🌊 {w}")
                    lines.append("")
                if b.quellen:
                    lines.append(f"*Quellen: {', '.join(b.quellen)}*")
                lines.append("---")

        # Audit Trail
        lines += [
            "",
            "## Audit Trail",
            "",
            "| Parameter | Wert |",
            "|---|---|",
            f"| Modell | {self.model} |",
            f"| Katalog-Version | {self.katalog_version} |",
            "| Generator | finreg-agents v2.0 |",
            f"| Zeitstempel | {datetime.now().isoformat()} |",
        ]

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown-Bericht: %s", path)

    # ------------------------------------------------------------------ #
    # HTML-Report (druckfähig)
    # ------------------------------------------------------------------ #
    def _schreibe_html(
        self,
        sektionsergebnisse,
        zusammenfassung,
        path,
        token_stats=None,
        stats_file=None,
        verbose=False,
    ):
        z = zusammenfassung
        parts = [
            self._html_header(z),
            self._html_zusammenfassung(z),
        ]
        if token_stats:
            parts.append(self._html_token_stats(token_stats, stats_file))
        if z["kritische_befunde"]:
            parts.append(self._html_mangelkatalog(z))
        if z["nicht_pruefbar_quote"] >= 30:
            parts.append(self._html_evidenz_warnung(z))
        parts.append(self._html_detailbefunde(sektionsergebnisse))
        parts.append(self._html_audit_trail(z))
        parts.append(self._html_footer())
        Path(path).write_text("".join(parts), encoding="utf-8")
        logger.info("HTML-Bericht: %s", path)

    # ------------------------------------------------------------------ #
    # HTML Building Blocks
    # ------------------------------------------------------------------ #

    def _html_header(self, z) -> str:
        return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(self.report_title)} – {_esc(self.institution)}</title>
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
                      font-weight: 700; font-size: 16px; border-left: 5px solid {z["gesamtfarbe"]};
                      background: {z["gesamtfarbe"]}15; color: {z["gesamtfarbe"]}; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
                 margin-bottom: 32px; }}
  .stat-card {{ padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #ecf0f1; }}
  .stat-number {{ font-size: 28px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #7f8c8d; text-transform: uppercase; margin-top: 4px; }}
  .warning-box {{ background: #fef9e7; border: 1px solid #f9e79f; border-radius: 8px;
                  padding: 14px 20px; margin-bottom: 24px; color: #7d6608; font-size: 13px; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ font-size: 15px; font-weight: 700; color: #1a3a5c;
                    padding: 10px 0; border-bottom: 2px solid #3498db;
                    margin-bottom: 16px; }}
  .befund-card {{ border: 1px solid #ecf0f1; border-radius: 8px; margin-bottom: 12px;
                  overflow: hidden; }}
  .befund-header {{ display: flex; align-items: center; gap: 12px;
                    padding: 12px 16px; background: #f8f9fa; flex-wrap: wrap; }}
  .befund-id {{ font-size: 11px; font-weight: 700; color: #7f8c8d;
                font-family: monospace; white-space: nowrap; }}
  .befund-frage {{ flex: 1; font-weight: 600; color: #2c3e50; font-size: 13px; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 11px;
            font-weight: 700; white-space: nowrap; }}
  .confidence-bar {{ width: 60px; height: 6px; background: #ecf0f1; border-radius: 3px;
                     overflow: hidden; display: inline-block; vertical-align: middle; }}
  .confidence-fill {{ height: 100%; border-radius: 3px; }}
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
  .validation-hints {{ background: #fef9e7; border: 1px solid #f9e79f;
                       border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 12px; }}
  .quellen {{ font-size: 11px; color: #95a5a6; margin-top: 8px; }}
  .mangelkatalog {{ background: #fdedec; border-radius: 8px; padding: 20px;
                    margin-bottom: 32px; border: 1px solid #fadbd8; }}
  .mangelkatalog h2 {{ color: #922b21; margin-bottom: 12px; font-size: 15px; }}
  .mangel-item {{ padding: 6px 0; border-bottom: 1px solid #fadbd8;
                  font-size: 12px; color: #34495e; }}
  .mangel-item:last-child {{ border-bottom: none; }}
  .token-box {{ background: #eef6ff; border: 1px solid #d1e7ff; border-radius: 8px;
                padding: 16px 20px; margin-bottom: 24px; }}
  .token-box h2 {{ color: #1a3a5c; margin-bottom: 10px; font-size: 14px; }}
  .token-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 14px; }}
  .token-item {{ font-size: 12px; color: #2c3e50; }}
  .token-label {{ color: #5d6d7e; font-weight: 600; }}
  .audit-trail {{ background: #f8f9fa; border-radius: 8px; padding: 16px 20px;
                  margin-top: 32px; font-size: 12px; }}
  .audit-trail h2 {{ font-size: 14px; color: #1a3a5c; margin-bottom: 8px; }}
  .audit-row {{ display: flex; padding: 4px 0; border-bottom: 1px solid #ecf0f1; }}
  .audit-label {{ width: 180px; color: #95a5a6; font-weight: 600; }}
  .audit-value {{ color: #2c3e50; }}
  .footer {{ background: #f8f9fa; padding: 16px 40px; font-size: 11px;
             color: #95a5a6; border-top: 1px solid #ecf0f1; text-align: center; }}
  @media print {{ body {{ background: white; }} .page {{ box-shadow: none; }} }}
</style>
</head>
<body>
<div class="page">
<div class="header">
  <h1>{_esc(self.report_title)}</h1>
  <div class="subtitle">{_esc(self.report_subtitle)}</div>
</div>
<div class="meta-grid">
  <div class="meta-item"><div class="meta-label">Institut</div>
    <div class="meta-value">{_esc(self.institution)}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfer</div>
    <div class="meta-value">{_esc(self.pruefer)}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfungsdatum</div>
    <div class="meta-value">{_esc(self.pruefungsdatum)}</div></div>
  <div class="meta-item"><div class="meta-label">Prüfungstyp</div>
    <div class="meta-value">{_esc(self.report_typ)}</div></div>
</div>
<div class="content">
"""

    def _html_zusammenfassung(self, z) -> str:
        return f"""
<div class="gesamtbewertung">Gesamtergebnis: {_esc(z["gesamtbewertung"])}</div>
<div class="stats-grid">
  <div class="stat-card" style="background:#eafaf1;border-color:#a9dfbf">
    <div class="stat-number" style="color:#27ae60">{z["konform"]}</div>
    <div class="stat-label">✅ Konform</div></div>
  <div class="stat-card" style="background:#fef9e7;border-color:#f9e79f">
    <div class="stat-number" style="color:#e67e22">{z["teilkonform"]}</div>
    <div class="stat-label">⚠️ Teilkonform</div></div>
  <div class="stat-card" style="background:#fdedec;border-color:#fadbd8">
    <div class="stat-number" style="color:#c0392b">{z["nicht_konform"]}</div>
    <div class="stat-label">🔴 Nicht konform</div></div>
  <div class="stat-card" style="background:#f4ecf7;border-color:#e8daef">
    <div class="stat-number" style="color:#8e44ad">{z["disputed"]}</div>
    <div class="stat-label">⚖️ Strittig</div></div>
  <div class="stat-card" style="background:#f2f3f4;border-color:#d5d8dc">
    <div class="stat-number" style="color:#7f8c8d">{z["nicht_pruefbar"]}</div>
    <div class="stat-label">❓ Nicht prüfbar ({z["nicht_pruefbar_quote"]}%)</div></div>
</div>
"""

    def _html_token_stats(self, token_stats: dict, stats_file: Optional[str]) -> str:
        gesamt = token_stats.get("gesamt", {})
        kosten = token_stats.get("kosten_schaetzung", {})
        stats_ref = stats_file or token_stats.get("stats_file", "")
        return f"""
<div class="token-box">
  <h2>Token-Stats</h2>
  <div class="token-grid">
    <div class="token-item"><span class="token-label">Gesamt Tokens:</span> {_esc(gesamt.get("total", 0))}</div>
    <div class="token-item"><span class="token-label">Input / Output:</span> {_esc(gesamt.get("input", 0))} / {_esc(gesamt.get("output", 0))}</div>
    <div class="token-item"><span class="token-label">Kostenschätzung:</span> {_esc(kosten.get("total_cost", 0))} {_esc(kosten.get("currency", "USD"))}</div>
    <div class="token-item"><span class="token-label">Pricing-Stand:</span> {_esc(kosten.get("pricing_timestamp", "n/a"))}</div>
    <div class="token-item"><span class="token-label">Stats-Datei:</span> {_esc(stats_ref)}</div>
  </div>
</div>
"""

    def _html_evidenz_warnung(self, z) -> str:
        return f"""
<div class="warning-box">
  ⚠️ <strong>Evidenz-Warnung:</strong> {z["nicht_pruefbar_quote"]}% der Prüffelder konnten nicht bewertet werden.
  Die Prüfungsergebnisse sind eingeschränkt belastbar. Bitte stellen Sie sicher, dass alle relevanten
  Dokumente im Prüfungskorpus enthalten sind.
</div>
"""

    def _html_mangelkatalog(self, z) -> str:
        items = "".join(
            f'<div class="mangel-item">'
            f"<strong>[{_esc(m['id'])}] [{_esc((m.get('schweregrad') or '').upper())}]</strong> "
            f"{_esc(m.get('mangel') or m['frage'])}</div>"
            for m in z["kritische_befunde"]
        )
        return f"""
<div class="mangelkatalog">
  <h2>⚠ Mängelkatalog ({z["anzahl_mängel"]} Mängel,
      davon {z["anzahl_wesentliche_mängel"]} wesentlich)</h2>
  {items}
</div>
"""

    def _html_detailbefunde(self, sektionsergebnisse) -> str:
        out = '<h2 style="font-size:17px;color:#1a3a5c;margin-bottom:24px">Detailbefunde</h2>'
        for sektion in sektionsergebnisse:
            out += '<div class="section">'
            out += f'<div class="section-title">{_esc(sektion.sektion_id)}: {_esc(sektion.titel)}</div>'

            if sektion.review_quote > 0.3:
                out += (
                    f'<div class="warning-box">⚠️ {sektion.review_quote:.0%} der Befunde '
                    f"in dieser Sektion erfordern manuelles Review.</div>"
                )

            for b in sektion.befunde:
                style = BEWERTUNG_STYLE.get(
                    b.bewertung.value,
                    {"emoji": "?", "color": "#7f8c8d", "bg": "#f2f3f4"},
                )
                sg_style = SCHWEREGRAD_STYLE.get(
                    b.schweregrad or "",
                    {"label": b.schweregrad or "", "color": "#7f8c8d"},
                )

                # Confidence-Bar
                conf_pct = int(b.confidence * 100)
                conf_color = (
                    "#27ae60"
                    if b.confidence >= 0.7
                    else "#e67e22"
                    if b.confidence >= 0.4
                    else "#c0392b"
                )
                conf_html = (
                    f'<span style="font-size:11px;color:#7f8c8d;margin-left:8px">'
                    f'<span class="confidence-bar"><span class="confidence-fill" '
                    f'style="width:{conf_pct}%;background:{conf_color}"></span></span> '
                    f"{conf_pct}%</span>"
                )

                review_html = (
                    (
                        ' <span class="badge" style="background:#fef9e7;color:#7d6608">'
                        "🔍 REVIEW</span>"
                    )
                    if b.review_erforderlich
                    else ""
                )

                textstellen_html = ""
                if b.belegte_textstellen:
                    for t in b.belegte_textstellen:
                        textstellen_html += (
                            f'<div class="textstellen">📎 {_esc(t)}</div>'
                        )

                mangel_html = (
                    f'<div class="mangel-box">⚠ <strong>Mangel:</strong> {_esc(b.mangel_text)}</div>'
                    if b.mangel_text
                    else ""
                )

                empf_html = ""
                if b.empfehlungen:
                    items = "".join(
                        f'<div class="empfehlung-item">→ {_esc(e)}</div>'
                        for e in b.empfehlungen
                    )
                    empf_html = f'<div class="empfehlungen"><strong>Empfehlungen:</strong>{items}</div>'

                val_html = ""
                if b.validierungshinweise:
                    items = "".join(
                        f"<div>⚡ {_esc(v)}</div>" for v in b.validierungshinweise
                    )
                    val_html = f'<div class="validation-hints"><strong>Validierung:</strong>{items}</div>'

                drift_html = ""
                term_drift = getattr(b, "term_drift_warnings", [])
                if term_drift:
                    items = "".join(f"<div>🌊 {_esc(w)}</div>" for w in term_drift)
                    drift_html = f'<div class="validation-hints"><strong>Term-Drift-Warnungen:</strong>{items}</div>'

                quellen_html = (
                    f'<div class="quellen">Quellen: {_esc(", ".join(b.quellen))}</div>'
                    if b.quellen
                    else ""
                )

                prov_html = _render_provenance_html(getattr(b, "claim_provenance", []))

                out += f"""
<div class="befund-card">
  <div class="befund-header">
    <span class="befund-id">{_esc(b.prueffeld_id)}</span>
    <span class="befund-frage">{_esc(b.frage)}</span>
    <span class="badge" style="background:{style["bg"]};color:{style["color"]}">
      {style["emoji"]} {_esc(b.bewertung.value.upper())}</span>
    <span class="badge" style="background:#f2f3f4;color:{sg_style["color"]}">
      {_esc(sg_style["label"])}</span>
    {conf_html}{review_html}
  </div>
  <div class="befund-body">
    <div class="begruendung">{_esc(b.begruendung)}</div>
    {prov_html}
    {textstellen_html}
    {mangel_html}
    {empf_html}
    {val_html}
    {drift_html}
    {quellen_html}
  </div>
</div>"""
            out += "</div>"
        return out

    def _html_audit_trail(self, z) -> str:
        at = z.get("audit_trail", {})
        rows = "".join(
            f'<div class="audit-row"><span class="audit-label">{_esc(k)}</span>'
            f'<span class="audit-value">{_esc(str(v))}</span></div>'
            for k, v in at.items()
        )
        return f"""
<div class="audit-trail">
  <h2>Audit Trail</h2>
  {rows}
  <div class="audit-row"><span class="audit-label">Ø Confidence</span>
    <span class="audit-value">{z["avg_confidence"]:.1%}</span></div>
  <div class="audit-row"><span class="audit-label">Review erforderlich</span>
    <span class="audit-value">{z["review_erforderlich"]} / {z["total_prueffelder"]} Befunde</span></div>
</div>
"""

    def _html_footer(self) -> str:
        return f"""
</div>
<div class="footer">
  Dieser Bericht wurde von einem KI-gestützten Prüfsystem generiert. Er dient ausschließlich
  internen Simulationszwecken und ersetzt keine offizielle Aufsichtsprüfung.
  Stand: {_esc(self.pruefungsdatum)} | {_esc(self.model)}
</div>
</div></body></html>"""


# Rückwärtskompatibilität
GwGBerichtGenerator = BerichtGenerator
