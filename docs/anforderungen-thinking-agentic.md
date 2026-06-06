# Anforderungen aus „Thinking Agentic – das Playbook" an FinRegAgents

> Abgeleitet aus dem Manuskript *Thinking Agentic – das Playbook* (J. Schiller García,
> Stand 2026-06, 377 S.). Der Anspruch des Buches: KI-Agenten im regulierten Banken-Umfeld
> **von der Machbarkeit zur Freigabefähigkeit** bringen — sicher bauen, steuern, betreiben,
> mit Governance und Kostenkontrolle.
>
> Dieses Dokument spiegelt die normativen Buch-Anforderungen gegen den **Ist-Zustand**
> von FinRegAgents und benennt die Lücken. FinRegAgents ist selbst ein Agentensystem
> (PrüferAgent/SkeptikerAgent) — die Buch-Anforderungen gelten daher **rekursiv für das
> Tool selbst**, nicht nur für die geprüften Institute.

> **Einordnung / Augenmaß (Zweckbestimmung):** FinRegAgents ist ein **rein internes
> Simulations- und QS-Werkzeug**. Es ersetzt **keinen** echten Prüfer, sondern simuliert eine
> Sonderprüfung, damit die Bank für eine *reale* Prüfung gewappnet ist. Das System trifft
> **keine bindenden Entscheidungen**, ruft **keine Tools** auf, schreibt **nicht** in
> Bank-Systeme; sein Output wird **menschlich reviewt**. Nach dem Agent-vs-Automatisierung-Test
> des Buches (Einleitung: Agent = wählt Pfad zur Laufzeit **und** löst operative Schritte aus)
> ist es damit näher an **Decision-Support/Automatisierung** als an einem autonomen Agenten und
> im Tier-Modell (Kap. 10) bei **~Tier 1–2**, nicht Tier 4.
>
> Konsequenz für den Maßstab — gestützt auf die **MVCS-Wachstumslogik** des Buches selbst
> (*„Die Kontrollfläche muss mit Datenrisiko, Entscheidungsnähe und Autonomie wachsen; nicht
> alle Piloten brauchen alle Kontrollen"*, Kap. 1): Das relevante Risiko ist hier **nicht**
> „handelt das System unsicher autonom?", sondern **„ist die Simulation wahrheitsfähig?"** —
> eine falsche/unbelegte Aussage, die der Bank trügerische Sicherheit gibt. Deshalb gelten die
> **epistemischen** Anforderungen (Provenienz, Schema-Vertrag, Eval/Groundedness, Confidence,
> Drift, adversariale QS) **voll**, die **Wirk-/Laufzeit-Sicherheits**-Anforderungen
> (Tool-Governance, Budget-Circuit-Breaker als Safety, Kill-Switch, Three-Lines-Control-Plane)
> nur **reduziert**.
>
> ⚠️ **Lesehinweis:** Die Gesamtbild-Tabelle und die Roadmap weiter unten sind ursprünglich am
> **strengen Vollmaßstab eines produktiven Wirk-/Entscheidungsagenten (Gattung G1)**
> kalibriert. Dieses G1 **existiert in FinRegAgents nicht und soll es nicht.** Die für die
> tatsächlich vorhandenen Gattungen **proportional gültige** Bewertung steht im Abschnitt
> **„Gattungen von Agenten & proportionale Anforderungs-Matrix"** unten — dieser hat für die
> Priorisierung Vorrang.

## Der Kern-Anspruch (Bewertungsmaßstab)

> Produktionsreife agentischer KI entsteht nicht aus Modellstärke, sondern aus
> **Kontrollübersetzung**: Was in Governance, Risiko und Compliance gefordert ist, muss in
> Architektur, Laufzeitmetriken, Tool-Rechte und Audit Trails übersetzt werden. Entscheidend
> ist nicht die sichtbare Fähigkeit (Demo, Chatfenster), sondern ob die Entscheidungen des
> Systems **lesbar, beweisbar und freigabefähig** werden. Ein Agent ohne fachliche Grenzen,
> Beobachtbarkeit, Kontrollübersetzung und explizite Limitierung ist keine Automation, sondern
> Schatten-IT mit moderner Oberfläche.

### Minimum Viable Control Surface (Kap. 1) — die sieben Pflicht-Artefakte je Agent
1. **Domain-Karte** (Bounded Context, erlaubte Events)
2. **Tool-Matrix** (zustandsgebundene Rechte)
3. **Zustandsmodell** (erlaubte Zustände/Übergänge)
4. **Kostenmodell** (Budget pro Task)
5. **Eval-Set** (Qualität, Drift, Sicherheit, Kosten)
6. **Trace-Modell** (auditierbare Ereignisse)
7. **Freigabemodell** (welche Rolle gibt welche Änderung frei)

> FinRegAgents besitzt heute keines dieser sieben Artefakte vollständig und explizit.
> Reifegrad-Einordnung: zwischen **„Pilotfähig"** und **„Produktionsnah kontrolliert"**,
> noch nicht **„Kritischer Produktionsbetrieb"** (Kap. 1, drei Reifegrade).

---

## Gesamtbild (Ist-Reife je Anforderungsblock)

> Diese Tabelle bewertet am **G1-Vollmaßstab** (produktiver Wirk-/Entscheidungsagent).
> Für die tatsächlich vorhandenen Gattungen gilt die **proportionale Matrix** weiter unten.

| Block | Anforderung | Ist | Reife |
|---|---|---|---|
| A | Domain-Driven Design / Bounded Context | teilweise | ◐ |
| B | Schema-as-Contract (Pydantic, `extra=forbid`) | **fehlt** | ○ |
| C | EvidencePackage / Claim-Provenienz | teilweise | ◐ |
| D | Confidence-Scoring & Schwellen | vorhanden | ● |
| E | Zustandsmodell (FSM statt freie Schleife) | teilweise | ◔ |
| F | Tool-Governance / Least Privilege / MCP-Proxy | n/a (keine Tools) | — |
| G | Human-in-the-Loop / Maker-Checker / Review-Queue | teilweise | ◑ |
| H | Resilienz / Chaos / Dead-Letter / Circuit Breaker | **fehlt** | ○ |
| I | Observability / Decision Trace / OpenTelemetry | teilweise | ◔ |
| J | Agentic FinOps / CPVCT / Budget-Breaker / Routing | teilweise | ◐ |
| K | Evaluation / Golden Dataset / Release-Gate | **fehlt** | ○ |
| L | Adversarial Security / Threat Model / Red Team | teilweise | ◐ |
| M | Daten-/RAG-/Quellen-Governance / Lineage / Hash | teilweise | ◐ |
| N | Lifecycle / Agent Card / Versionierung / Kill Switch | teilweise | ◔ |
| O | Three Lines of Defense / Control-Plane-Register | **fehlt** | ○ |
| P | Regulatorik-Mapping des Systems selbst (DORA/AI Act/MaRisk/DSGVO) | **fehlt** | ○ |
| Q | Vendor-Governance / Exit / Anti-Lock-in | teilweise | ◔ |
| R | Befähigung / AI Literacy | n/a (Tooling) | — |

● vorhanden · ◑ überwiegend · ◐ ansatzweise · ◔ rudimentär · ○ fehlt

---

## A. Domain-Driven Agent Design (Kap. 1)
**Soll:** Agent auf scharfen Bounded Context begrenzen; fünf explizite Grenzen je Agent
(fachlich, Daten, Tool, Entscheidung, Kosten); Ubiquitous Language (Signal/Alert/Claim/
Hypothese/Evidenz/Entscheidung getrennt); kein Universalagent.
**Ist:** Implizite Domänentrennung über Regulatorik-Kataloge (`catalog/*.json`); PrüferAgent
ist fachlich begrenzt. Keine explizite Context Map, keine fünf dokumentierten Grenzen.
**Gap:** Context Map + Agent Card mit den fünf Grenzen je Agent als Pflichtartefakt.

## B. Schema-as-Contract (Kap. 2, Anhang A) — **kritische Lücke**
**Soll:** Jeder Agenten-Output gegen Pydantic-/JSON-Schema validiert; Pflichtfelder,
Wertebereiche, `extra="forbid"`; Business-Regeln maschinell durchgesetzt (z. B.
`confidence < 0.75 → requires_human_review`); freier Text ist kein Bankschnittstellenformat.
**Ist:** Outputs sind `@dataclass` (`agents/pruef_agent.py`), JSON wird via `extract_json()` +
Regex geparst; nur Light-Validierung in `validate_befund_structure()`. Kein Schema-Vertrag,
kein `extra=forbid`.
**Gap:** Pydantic-Modelle als verbindlicher Output-Vertrag mit Validator-Gate, das die
Weiterverarbeitung bei Verstoß blockiert.

## C. EvidencePackage / Claim-Provenienz (Kap. 2, Anhang A)
**Soll:** Output als EvidencePackage = Claims mit Provenienz (`source_id`, `source_kind`,
`source_version`, `retrieved_at`, `method`, `quote_hash`, `confidence`), Hypothesen und
**offene Fragen** strukturiert getrennt; `prohibited_final_actions`; Edge Claims reifiziert
(Graph-Kanten als belegte Claims mit `is_admissible()`).
**Ist:** `agents/provenance.py` hat `ClaimProvenance` (`claim_text`, `status`,
`source_chunk_ids`) und Korroborations-Status. Es **fehlen** `source_version`, `retrieved_at`,
`method`, `quote_hash`, `confidence`-Feld und die strukturierten offenen Fragen.
**Gap:** Provenienz-Modell um die fehlenden Felder + Quote-Hash-Integrität erweitern;
EvidencePackage als Top-Level-Output etablieren.

## D. Confidence-Scoring & Schwellen (Kap. 5) — **vorhanden**
**Soll:** Mehrsignaliges Confidence; ab Schwelle Human-Review erzwingen.
**Ist:** `compute_confidence()` (4 Signale), `CONFIDENCE_REVIEW_THRESHOLD=0.7`,
`CONFIDENCE_AUTO_REJECT=0.4`, drei Confidence-Guards. Gut umgesetzt.
**Gap:** Confidence-Kalibrierung gegen Golden Set (siehe K); Inter-Rater-Reliability messen.

## E. Zustandsmodell statt freier Schleifen (Kap. 1/2, Anhang A)
**Soll:** Explizite Zustände/Übergänge; Eskalation als Zustandsübergang; AMLA-Kriterien als
Zustände modelliert; keine produktiven ReAct-Freischleifen; Cost-Interrupt als Zustand.
**Ist:** Sequenzielle Pipeline-Schleife (`pipeline.py` `run()`), Zustand implizit in Variablen;
Checkpoints für Resume. Kein expliziter FSM/Workflow-Graph.
**Gap:** Prüf-Workflow als expliziten Zustandsgraph (z. B. LangGraph) mit
Fehler-/Eskalations-/Budget-Interrupt-Zuständen.

## F. Tool-Governance / Least Privilege / MCP-Proxy (Kap. 2/10)
**Soll:** Least-Privilege-Toolset, zustandsgebundene Tool-Matrix, Tool-Gateway mit
Pre-Execution-Checks, MCP als Proxy (nicht Adapter), Tool-Rechte als versionierter Code.
**Ist:** System ist reines RAG, **keine Tool-/Function-Calls, kein MCP**. Damit aktuell n/a.
**Gap:** Sobald Tools eingeführt werden (z. B. Register-Abfragen, Sanktionslisten): Tool-Matrix
+ Gateway + Audit von Beginn an mitbauen. Bis dahin: bewusst als „kein Tool-Zugriff"
dokumentieren (reduziert Angriffsfläche — Stärke, nicht nur Lücke).

## G. Human-in-the-Loop / Maker-Checker (Kap. 2/11/12)
**Soll:** HITL = technische Funktionstrennung (Maker=Agent, Checker=Mensch) als
Workflow-Zustand; Human-Gate-Record mit 10 Feldern (Zustand, Output, Unsicherheiten,
Reviewer-Rolle, Versionen, Kosten, Vier-Augen-Flag, Zeitstempel, nachträgliche Änderung);
Review-Queue nach Unsicherheit×Tragweite priorisiert.
**Ist:** Streamlit-Review-Queue (`app.py`) mit Confidence-Sortierung, Approve/Reject,
`review_decisions_*.json`; `disputed`-Status vorhanden. **Kein** formales Vier-Augen-Prinzip,
Human-Gate kein erzwungener Workflow-Zustand.
**Gap:** Human-Gate als nicht-überspringbaren Zustand; vollständiges Gate-Protokoll;
Maker-Checker mit Reviewer-Rolle ≠ Autor.

## H. Resilienz / Chaos / Dead-Letter / Circuit Breaker (Kap. 3) — **kritische Lücke**
**Soll:** Reproduzierbare Fault-Injection; Max-Step-/Rekursions-/Budget-Limits mit
kontrolliertem Stop; Dead-Letter-Queue mit Pflichtfeldern; Retry-Limits + Backoff;
kontrollierter Abbruch statt Halluzination bei fehlender Evidenz; Bulkhead-Isolation;
Chaos-Experiment-Log als Revisionsnachweis.
**Ist:** Keine Fault-Injection, keine Dead-Letter-Queue, kein Step-/Budget-Circuit-Breaker.
**Gap:** Step-/Budget-Limits + Dead-Letter-Record (Template in Anhang D) + Chaos-Eval-Set.

## I. Observability / Decision Trace / OpenTelemetry (Kap. 4) — **kritische Lücke**
**Soll:** Strukturierter Decision Trace je Lauf (Modell-/Prompt-/Schema-Version, Tool-Calls,
Zustände, Kosten, Freigaben); OpenTelemetry-GenAI-Semantik; **append-only/immutable**
Event Ledger; AML-spezifisch: `why_flagged` / `why_not_flagged` / `what_changed`; Trace-Modell
**vor** Produktivfreigabe freigeben.
**Ist:** Checkpoints (`checkpoint_latest.json`, überschreibend — **nicht** append-only),
`run_stats.json` (Token/Kosten/Timestamp). Keine Modell-/Prompt-Version im Trace, kein
OpenTelemetry, keine drei Diagnose-Queries.
**Gap:** Immutable Trace-Store mit Pflichtfeldern + OTel-Spans; `why_not_flagged`/`what_changed`
als Abfragen (passt exzellent zum Prüfungskontext).

## J. Agentic FinOps (Kap. 6–9) 
**Soll:** **Cost per valid completed task** als Leitmetrik (inkl. Review/Retry/Fehlerfolge/
Governance); Budget-Circuit-Breaker pro Task (Hard Stop, kein Alarm); Task-Budget-Matrix
(Warn-/Hartwert); Cost State im Workflow; Policy-getriebenes Model-Routing (Zulässigkeit vor
Kosten); Semantic/Prompt-Caching mit Governance; FinOps-Gate in CI; Tail-Kosten (P95/P99).
**Ist:** Token-Tracking pro Agent + Kostenschätzung (hardcodierte Preise, `pipeline.py`);
`review_budget`-Kadenz vorhanden. **Kein** Budget-Circuit-Breaker für Kosten, kein
Cost-per-valid-task, kein Cost-Routing, kein Cache.
**Gap:** CPVCT-Rechner (Template Anhang D), Budget-Hard-Stop, Routing-Policy mit
Datenklassen-Vorfilter.

### J-bis. Kostenkontrolle & Self-Hosting (Self-hosted vs. fremdgehostet) — präzisiert
Für dieses Tool ist „Kostenkontrolle" **zwei Probleme in einem**; die Self-Hosting-Entscheidung
sitzt an deren Schnittpunkt:

- **(a) Reine Ökonomie — niedrige Stakes.** Batch, intern, kein Echtzeit-SLA, **kein
  adversarialer Cost-Exhaustion-Vektor** → Budget ist Betriebsthema, **kein Safety-Hard-Stop**.
- **(b) Datenhoheit — hohe Stakes.** FinRegAgents **ingestiert Bankdokumente** (interne
  Compliance-Unterlagen, ggf. Kundendaten) und schickt „viele viele Tokens" davon an das LLM.
  **Fremdgehostet = Auslagerung/DSGVO-Frage** (Kap. 13: Vendor-Governance, Compute Boundaries —
  *„Daten liegen bei der Abfrage"*). Das ist materiell Block **M/P/Q**, getarnt als Kostenfrage.

**Self-hosted vs. fremdgehostet löst beide gleichzeitig** — daher der höchste Werthebel der
FinOps-Dimension für dieses Tool.

**Token-Treiber hier:** Ein Voll-Lauf = Σ über Sektionen von (Retrieval-Kontext + Prompt) ×
Prüffelder, **×2 bei aktivem Skeptiker**, plus Retries. Bei 8 Katalogen × ~20 Prüffeldern und
großem Korpus entsteht hier das Token-Volumen. Wirksamste Stellschrauben = **Kontext-Ökonomie**
(`top_k`, `chunk_size`, Skeptiker selektiv), nicht der Modellpreis.

**Ökonomie-Vergleich (Kap. 7, lokale vs. Frontier als Portfolioentscheidung):**

| | Fremdgehostet (per Token) | Self-hosted (GPU) |
|---|---|---|
| Kostenstruktur | 0 Fix, **lineare** Grenzkosten | hohe **Fix**, ~0 Grenzkosten/Token |
| Günstig bei | niedrigem/sporadischem Volumen | hohem, dauerhaftem Volumen |
| Qualität | Frontier-Niveau | lokale Modelle schwächer (Reasoning) |
| Daten | verlassen das Haus → Auslagerung | bleiben im Haus |
| Ops-Last | keine | GPU-Betrieb, auch im Leerlauf bezahlt |

Crossover hängt am tatsächlichen Volumen → **messen statt raten** (*„Baseline first"*). Für
burstweisen Simulationsbetrieb ist fremdgehostet oft billiger — **es sei denn, die Datenhoheit
erzwingt lokal**, was hier wahrscheinlich dominiert.

**Self-Hosting-Pfad ist bereits angelegt:** `agents/llm_factory.py` (**Ollama**),
`agents/embedding_factory.py` (**fastembed**, lokal). Die Schiene existiert — sie wird nur nicht
*gesteuert*.

**Proportionale Maßnahmen (Augenmaß — nicht der volle FinOps-Control-Plane):**
1. **Routing nach Datenklasse** (Schlüsselzug, löst Kosten + Hoheit): vertrauliche Bankdokumente
   → lokales Modell (Ollama/fastembed); öffentlicher Regulatorik-/Kataloginhalt → ggf.
   fremdgehostet. Buch-Regel *„Zulässigkeit vor Kostenoptimierung"*.
2. **Tiered Execution:** lokales Klein-Modell für Extraktion/Klassifikation, Frontier nur für
   schwierige Fälle/Skeptiker.
3. **Kontext-Ökonomie:** `top_k`/`chunk_size` tunen, Skeptiker selektiv (Hochrisiko).
4. **Cost-per-Run-Metrik + weiches Budget-Warnsignal** pro Lauf (kein Circuit-Breaker-Safety);
   Token-Attribution vorhanden, Preis-Modell um Self-Hosting-Kostenmodell ergänzen (Fix+Idle
   statt nur per-Token).
5. **Caching** für stabilen System-Prompt/Katalog — **kein** Cache über vertrauliche
   Dokumentinhalte (Kap. 7: Kundendaten nicht cachebar).

## K. Evaluation / Golden Dataset / Release-Gate (Kap. 5) — **kritische Lücke**
**Soll:** Versioniertes Golden Dataset (Falltypologie, Grenz-/Hochrisiko-/Fehlerfälle,
10 Metadatenfelder, PII-Redaction-Gate); neun Eval-Splits (inkl. Adversarial/Chaos/Security/
Drift); Eval-Pipeline in CI/CD; Release-Gate mit Qualitäts-/Kosten-/Governance-Schwellen
(Schema ≥99 %, Groundedness ≥97 %, High-Risk 100 % Eskalation, CPVCT-Cap); Release-Zertifikat;
LLM-as-Judge nur als Sensor, nicht als Freigabe.
**Ist:** Unit-Tests (`tests/`), Drift-Vergleich (`ui_drift.py`). **Kein** Golden Dataset,
keine Metriken (Task Success/Groundedness), keine Release-Gates.
**Gap:** Golden Dataset je Regulatorik + Eval-Pipeline + Release-Gate + Release-Zertifikat.
(Höchster Hebel für Freigabefähigkeit.)

## L. Adversarial Security (Kap. 10)
**Soll:** Threat Model (8 Fragen); Untrusted-Data-Default für externe Dokumente;
Schutz gegen indirekte Prompt Injection, RAG-Poisoning (Hash-Integrität, Quellenregister),
Data Exfiltration, Cost Exhaustion; Security-Eval-Set als Release-Blocker; PenTest quartalsweise.
**Ist:** SkeptikerAgent (adversariales Review, optional), Term-Drift-Checker (Phantom-Zitate).
**Kein** Prompt-Injection-Schutz, kein RAG-Poisoning-Schutz, kein Security-Eval-Set.
**Gap:** Untrusted-Data-Behandlung für ingestierte Dokumente; Security-Eval-Set;
SkeptikerAgent für Hochrisiko verpflichtend statt optional.

## M. Daten-/RAG-/Quellen-Governance (Kap. 13)
**Soll:** Freigegebene Wissenskorpora; Quellenregister mit Owner/Freigabe/Version; Datenklasse
je Dokument; Lineage Output→Quelle; Hash-Integrität; Retrieval-Evals; Attribution (richtige
Quelle stützt Claim?); BCBS 239: Lineage vor Reasoning.
**Ist:** Strukturierte Ingestion (`ingestion/ingestor.py`, File-Hash-Dedup, Metadaten);
Relevanz-Filter (Feature-Flag). **Kein** Quellenregister/Freigabe-Gate, kein Lineage-Tracking,
keine Quote-Hash-Integrität.
**Gap:** Source-Registry + Freigabestatus + Quote-Hash + Lineage Chunk→Befund.

## N. Lifecycle / Agent Card / Versionierung / Kill Switch (Kap. 1/11/12)
**Soll:** Agent als versioniertes Betriebsobjekt; Agent Card (Zweck, Nicht-Zweck, Risikoklasse,
Owner, Versionen, Budgets, Evals, Freigabe); Prompts/Policies versioniert; Re-Abnahme bei
Modell-/Policy-Änderung/Drift; Kill-Switch (Soft/Degradation/Human-Fallback/Hard) vor
Produktivsetzung; Agent als Composite Configuration Item.
**Ist:** Katalog-Versionierung (`katalog_version`); statische `generator_version`. Prompts sind
String-Literale ohne Versionierung. Kein Agent Card, kein Kill Switch, keine Re-Abnahme-Trigger.
**Gap:** Agent Card (Template Anhang D), Prompt-/Policy-Versionierung im Repo, Kill-Switch-Stufen.

## O. Three Lines of Defense / Control-Plane-Register (Kap. 11/12) — **fehlt**
**Soll:** Gemeinsame Trace-Evidenz für alle drei Linien (operativ/Risk/Revision); zentrale
Register (Use-Case, Agent, Model, Tool, Prompt/Policy, Daten/RAG, Provider, Eval/Release,
Kosten, Human-Gate, Incident/Dead-Letter, Audit-Export); Control Evidence Pack je Agent.
**Ist:** Nur Katalog-Registry. Keine Agent/Model/Source-Register, kein Evidence Pack.
**Gap:** Minimal-Control-Plane: Agent-Registry + Model-Registry + Release-/Eval-Historie.

## P. Regulatorik-Mapping des Systems selbst (Kap. 11/13) — **fehlt**
**Soll:** Explizites Mapping „Governance-Anforderung → technische Kontrolle → Nachweis" für
DORA (Resilienz, Provider, Exit, Incident), EU AI Act (Tiering, Doku, Human Oversight, Logging),
MaRisk (Funktionstrennung, Freigabe, Mgmt-Info), DSGVO (Datenfluss, Redaction, Retention),
BCBS 239 (Lineage). Compliance als Laufzeitfähigkeit, nicht nachträgliche Doku.
**Ist:** FinRegAgents *prüft* diese Regulatorik bei anderen, hat aber **kein Mapping für sich
selbst** als KI-System (besondere Ironie für ein RegTech-Tool).
**Gap:** Governance-Mapping-Template (Anhang C) für FinRegAgents selbst ausfüllen; ggf. als
`docs/governance-mapping.md`.

## Q. Vendor-Governance / Exit / Anti-Lock-in (Kap. 10/13)
**Soll:** Provider-Register + Konzentrationsmessung; Exit-Plan getestet; Bank-owned: Policies,
Golden Dataset, Trace-Schema, EvidencePackage; austauschbare Modelladapter.
**Ist:** Multi-Provider-Abstraktion (`agents/llm_factory.py`, 7 Provider) — gute Anti-Lock-in-
Basis. Kein Provider-Register, kein dokumentierter/getesteter Exit, keine Konzentrationsmetrik.
**Gap:** Provider-Register + Exit-Testnachweis; die vorhandene Adapter-Abstraktion dokumentiert
als bewusste Anti-Lock-in-Architektur.

## R. Befähigung / AI Literacy (Kap. 14)
**Soll:** Rollenbasierte Befähigung, Artefakt-Lesefähigkeit der Kontrollfunktionen.
**Ist/Gap:** Für ein Tool nur eingeschränkt einschlägig; relevant wird die **Lesbarkeit** der
FinRegAgents-Artefakte (Trace, EvidencePackage, Report) für Compliance/Revision (→ Blöcke C, I).

---

## Gattungen von Agenten & proportionale Anforderungs-Matrix

> Maßgeblich für die Priorisierung. Die Buch-Anforderungen treffen **nicht jede Komponente
> gleich**. Anhand zweier Achsen des Buches — **Wirkmacht/Entscheidungsnähe** (löst das System
> operative Schritte aus?) und **Funktion** (erkennend vs. kontrollierend vs. deterministisch) —
> ergeben sich fünf Gattungen. Die einzige, die „alles" erfüllen muss (**G1**), existiert in
> FinRegAgents **bewusst nicht.**

| Gattung | Definition | Anforderungsintensität | In FinRegAgents |
|---|---|---|---|
| **G1 Wirk-/Entscheidungsagent** | wählt Pfad zur Laufzeit **und** ruft Tools / verändert Daten / löst Folgeschritte aus | **Muss ALLES** (volle Kontrollfläche, Tier 4) | **Nicht vorhanden — bewusst nicht** |
| **G2 Erkenntnis-/Analyseagent** | LLM erzeugt Befunde/Evidenz, keine bindende Wirkung, mündet in menschlich reviewtes Ergebnis | **Epistemik voll**, Wirk-Kontrollen gering | **PrüferAgent** (`agents/pruef_agent.py`) |
| **G3 Kontroll-/QS-/Adversarial-Agent** | prüft andere Outputs, ist selbst eine Kontrolle; nie alleiniges Gate (LLM-as-Judge = Sensor) | **Verlässlichkeit der Kontrolle** (Kalibrierung/Eval/Unabhängigkeit) | **SkeptikerAgent** (`agents/skeptiker_agent.py`) |
| **G4 Deterministische QS-/Hilfskomponente** | regelbasiert, keine Laufzeit-Pfadwahl, keine LLM-Autonomie (per Buch **kein „Agent"**) | Minimal: Determinismus + Tests + Versionierung | **TermDriftChecker** (`agents/term_checker.py`), **Provenance-Annotator** (`agents/provenance.py`) |
| **G5 Infrastruktur/Adapter** | kein Agent; Modell-/Embedding-/Ingest-Schicht | Anti-Lock-in/Exit + Tests | **llm_factory, embedding_factory, Ingestor** |

**Kernbefund:** Die Gattung, die den Vollmaßstab („alles") erfüllen müsste (G1), fehlt und soll
fehlen. Für die real vorhandenen Gattungen (G2–G5) reduziert sich der Anspruch auf den
**QS-/Epistemik-Kern**.

### Matrix: Gattung × Anforderungsblock
Legende: ● Muss · ◑ Soll · ○ gering/optional · – nicht anwendbar.
*(Stufen kalibriert auf „intern, simulierend, human-reviewed".)*

| Block (Kurz) | G1 Wirkagent | G2 Prüfer | G3 Skeptiker (QS) | G4 Determin. QS | G5 Infra |
|---|:--:|:--:|:--:|:--:|:--:|
| A Domain/Bounded Context | ● | ● | ◑ | ○ | – |
| B Schema-as-Contract | ● | ● | ◑ | ◑ | – |
| C EvidencePackage/Provenienz | ● | ● | ○ | ◑ | – |
| D Confidence & Schwellen | ● | ● | ◑ | ○ | – |
| E Zustandsmodell (FSM) | ● | ◑ | ○ | – | – |
| F Tool-Governance/MCP | ● | – | – | – | – |
| G HITL / Maker-Checker | ● | ● | ● | ○ | – |
| H Resilienz/Dead-Letter | ● | ◑¹ | ○ | ○ | – |
| I Decision Trace/Observability | ● | ◑ | ◑ | ○ | – |
| J FinOps — Budget-Hard-Stop *(Safety)* | ● | ○² | ○ | – | ○ |
| J FinOps — Routing nach Datenklasse *(Kosten+Hoheit)* | ● | ◑⁸ | ○ | – | ●⁸ |
| **K Eval/Golden Dataset** | ● | **●** | **●**³ | **●** | ◑ |
| L Adversarial-Security | ● | ◑⁴ | ●⁵ | ○ | – |
| M RAG-/Quellen-Governance | ● | ◑ | ○ | ◑ | ◑ |
| N Lifecycle/Agent Card/Versionierung | ● | ◑ | ◑ | ◑ | ◑ |
| O Three-Lines/Control-Plane-Register | ● | ○ | ○ | – | – |
| P Regulatorik-Mapping (System selbst) | ● | ○⁶ | ○ | – | – |
| Q Vendor/Exit/Anti-Lock-in | ● | ◑ | ○ | – | ●⁷ |
| R Befähigung/Lesbarkeit der Artefakte | ● | ◑ | ◑ | ○ | – |

¹ Nur „kontrollierter Abbruch statt Halluzination bei fehlender Evidenz" — Dead-Letter-Queue selbst gering.
² Kosten = Betriebsthema (Batch), **kein Safety-Hard-Stop** — kein Angriff, der Geld verbrennt.
³ Eine QS-Kontrolle ohne Kalibrierung gegen Ground Truth ist Kontrolltheater — für G3 die **wichtigste** Anforderung.
⁴ Relevant, weil der Prüfer **Bankdokumente ingestiert** → indirekte Prompt-Injection / RAG-Poisoning aus Dokumenten ist auch intern real.
⁵ Der Skeptiker **ist** die adversariale Kontrolle — für Hochrisiko-Sektionen sollte er von optional auf verpflichtend.
⁶ Nur als leichte Selbst-Klassifizierung: „internes Vorbereitungswerkzeug, kein Hochrisiko-System nach AI Act Anhang III, keine bindende Entscheidung" — ein Absatz, kein Programm.
⁷ Anti-Lock-in lebt genau hier (7-Provider-Abstraktion ist bereits stark).
⁸ Aufgespalten: Budget *als Safety-Hard-Stop* bleibt gering (○, Batch/intern, kein Exhaustion-Vektor). Das **Routing nach Datenklasse** (vertrauliche Bankdokumente → lokal/Ollama, öffentlicher Inhalt → ggf. fremdgehostet) ist dagegen wichtig (◑/●), weil es Kosten **und** Datenhoheit/Auslagerung zugleich löst — siehe Block J-bis. Für G5 (Infra) ●, da Routing/Adapter dort verankert ist.

### Proportionale Priorität (für die real vorhandenen Gattungen)
**Was für dieses Tool wirklich zählt — Vertrauenswürdigkeit der Simulation:**
- **K** Golden Dataset + Eval gegen Prüfer-Urteil (größter Hebel).
- **C/B** Provenienz-Felder + Schema-Vertrag (belegt vs. inferiert sauber trennen).
- **D** Confidence (bereits stark) + **L** Term-Drift/Skeptiker (Schutz vor Phantom-Befunden).
- **M** Quellen-Integrität beim Ingest (Bankdokumente).
- **J-bis** Routing nach Datenklasse (vertrauliche Dokumente → lokal/Ollama) — löst Kosten bei
  hohem Token-Volumen **und** Datenhoheit zugleich; plus Kontext-Ökonomie (`top_k`/`chunk_size`,
  Skeptiker selektiv) als direkter Token-Hebel.

**Was hier Über-Engineering wäre — Wirk-/Laufzeit-Sicherheit eines autonomen Systems:**
- **O** Three-Lines-Control-Plane-Register, **F** Tool-Governance, **J** Budget-Circuit-Breaker
  als Safety, Kill-Switch-Stufen, vollständiges Release-Zertifikat-Programm.

> Die folgende „Freigabefähigkeits"-Roadmap gilt **vollständig nur für ein hypothetisches G1**.
> Für G2–G5 ist allein der oben markierte QS-/Epistemik-Kern verbindlich.

---

## Priorisierte Roadmap zur Freigabefähigkeit (G1-Vollmaßstab — Referenz)

**P0 — Fundament für „freigabefähig" (höchster Hebel):**
- **B** Schema-as-Contract (Pydantic + Validator-Gate)
- **K** Golden Dataset + Release-Gate + Release-Zertifikat
- **I** Immutable Decision Trace mit Versions-/Kostenfeldern (+ `why_not_flagged`/`what_changed`)
- **C** EvidencePackage-Felder vervollständigen (Quote-Hash, Version, retrieved_at, method)

**P1 — Kontrolle & Betrieb:**
- **J** CPVCT + Budget-Circuit-Breaker
- **H** Dead-Letter-Queue + Step-/Budget-Limits
- **E** Expliziter Zustandsgraph mit Eskalations-/Interrupt-Zuständen
- **G** Human-Gate als erzwungener Zustand + Vier-Augen für Hochrisiko
- **N** Agent Card + Prompt-Versionierung

**P2 — Governance-Reife:**
- **O** Minimal-Control-Plane-Register (Agent/Model/Release/Eval)
- **P** Governance-Mapping (DORA/AI Act/MaRisk/DSGVO) für das System selbst
- **M** Source-Registry + Lineage + Hash-Integrität
- **L** Security-Eval-Set + Untrusted-Data-Default; Skeptiker für Hochrisiko verpflichtend
- **Q** Provider-Register + getesteter Exit

**Bereits stark (halten):** D (Confidence), Multi-Provider-Abstraktion (Q-Basis),
Term-/Context-Drift, SkeptikerAgent, Review-Queue mit `disputed`.
