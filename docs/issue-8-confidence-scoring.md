# Issue #8: v2 Confidence-Scoring (4 Signale)

Quelle: https://github.com/endvater/finreg-agents/issues/8

## Acceptance Criteria

- [ ] Das System berechnet pro Antwort einen `confidence_score` von `0-100`.
- [ ] Der Score basiert auf genau 4 Teil-Signalen mit klarer Definition: `source_reliability`, `evidence_coverage`, `consistency_check`, `regulatory_alignment`.
- [ ] Jedes Signal liefert einen normierten Wert `0.0-1.0` und eine kurze Begründung.
- [ ] Die Aggregation ist deterministisch dokumentiert (gleiche Inputs => gleicher Score).
- [ ] Die Ausgabe enthält ein standardisiertes Objekt mit: `confidence_score`, `confidence_level` (`low|medium|high`), `signals` (4 Werte + Rationale), `low_confidence_reasons` (Liste, ggf. leer).
- [ ] Schwellenwerte sind fest definiert: `low: 0-49`, `medium: 50-74`, `high: 75-100`.
- [ ] Bei `confidence_level=low` wird automatisch ein Review-Hinweis erzeugt (Human-in-the-loop Flag).
- [ ] Mindestens 10 Referenz-Testfälle decken ab: hohe Evidenz/niedrige Evidenz, widersprüchliche Quellen, fehlende regulatorische Grundlage, Grenzfälle an Schwellenwerten.
- [ ] Für alle Referenzfälle liegen erwartete Score-Bänder vor und Tests laufen grün.
- [ ] Dokumentation beschreibt Signallogik, Gewichtung, Schwellen und bekannte Grenzen.

## Definition of Done

- [ ] Confidence-Scoring ist im Hauptpfad integriert und für relevante Antworten aktiv.
- [ ] Unit- und Integrations-Tests für Score-Berechnung und Schwellen-Logik sind grün.
- [ ] Mindestens 10 Referenzfälle sind versioniert dokumentiert (Input, erwartetes Band, Ergebnis).
- [ ] Ausgabeformat ist stabil und in der API/Schema-Doku festgehalten.
- [ ] Low-Confidence-Flag triggert den vorgesehenen Human-Review-Mechanismus.
- [ ] Monitoring/Logging erfasst Score-Verteilung und Low-Confidence-Rate.
- [ ] README/Architektur-Doku ist aktualisiert.
- [ ] Abnahme durch Maintainer ist erfolgt.
