# Spike: Evidence Relevance Classifier (Issue #29)

## Ziel
Machbarkeitsprüfung eines auditierbaren Pre-Filters vor der LLM-Bewertung.

## Implementierter Spike
- Feature-Flag: `--evidence-relevance-filter`
- Chunk-Klassen:
  - `regulatory_requirement`
  - `control_evidence`
  - `context_noise`
- Gedroppt wird nur `context_noise`.
- Guardrail: Chunk mit offensichtlichen Reg-Referenzen (`§`, `Art.`, `MaRisk`, `DORA`, `GwG`, `KWG`, `WpHG`, `MaComp`) wird nie gedroppt.
- Drop-Reasons:
  - `NO_REG_REF`
  - `MARKETING_PHRASE`
  - `NON_CONTROL_CONTEXT`

## Output-Artefakt
Pro Run wird unter `<output_dir>/relevance_filter_samples.json` geschrieben:
- aggregierte Filter-Stats
- zufälliges Sample von max. 20 gedroppten Chunks (Snippet + Reason)

## Risiken / False Positives
- Fachlich valide Kontrolltexte ohne explizite Schlüsselwörter könnten als `context_noise` klassifiziert werden.
- Marketing-nahe Sprache in Governance-Dokumenten kann fälschlich droppen.
- Sprachvarianz (de/en Mischtexte) kann die Heuristik verzerren.

## Entscheidungsvorlage
- `adopt`: wenn Token-Verbrauch deutlich sinkt und Befundqualität stabil bleibt.
- `adjust`: wenn Drop-Rate hoch ist oder relevante Claims verloren gehen.
- `reject`: wenn reproduzierbar wesentliche Evidenz entfernt wird.
