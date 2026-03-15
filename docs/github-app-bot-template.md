# GitHub App Template fuer Multi-Bot-Setup

Dieses Template richtet ein label-gesteuertes Routing fuer `codex`, `claude` und `gemini` ein.

## Enthaltene Workflows

- `.github/workflows/bot-router.yml`
- `.github/workflows/bot-worker-template.yml`

## Prinzip

1. Auf einem PR wird ein Label gesetzt: `bot:codex`, `bot:claude` oder `bot:gemini`.
2. `bot-router.yml` erstellt ein GitHub-App-Token und dispatcht `repository_dispatch` mit Payload.
3. `bot-worker-template.yml` nimmt das Event entgegen und fuehrt den passenden Bot-Zweig aus.

## GitHub App Setup

1. GitHub App in der Organisation/User erstellen.
2. Berechtigungen setzen:
- `Contents`: Read/Write (nur wenn Bots Commits pushen sollen)
- `Pull requests`: Read/Write
- `Issues`: Read/Write
- `Metadata`: Read
3. App im Repo `endvater/finreg-agents` installieren.
4. Secrets im Repo hinterlegen:
- `FINREG_APP_ID`
- `FINREG_APP_PRIVATE_KEY`

## Labels

Einmalig im Repo anlegen:

- `bot:codex`
- `bot:claude`
- `bot:gemini`

## Manuelle Ausloesung

`bot-router.yml` kann ueber `workflow_dispatch` gestartet werden:

- `pr_number`: PR-Nummer
- `bot`: `codex|claude|gemini`

## Naechster Schritt

Im Worker-Template die drei TODO-Bloecke durch echte Bot-Aufrufe ersetzen (z. B. ueber CLI/MCP/HTTP).
