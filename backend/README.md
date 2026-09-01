# MAGIFF backend

The backend has four independent entry points:

- `python main.py` runs the interactive terminal agent.
- `uvicorn api.app:app --reload` runs the HTTP agent API.
- `python -m jobs.refresh_reports ...` runs report ingestion. Production
  ingestion remains in GitHub Actions and is not started by the API server.
- `python -m jobs.register_discord_commands` registers private Discord command
  profiles. The test guild gets `/ask`, `/news`, `/stats`, and `/game`; UAI gets
  `/news`, `/stats`, and `/game`.

The read-only draft advisor is intentionally separate from the general agent.
Use `python -m drafting.cli simulate ...` for a reproducible local mock or
`python -m drafting.cli live ...` for one public Sleeper draft snapshot. See
[`drafting/README.md`](drafting/README.md).

After the draft, `python -m league_management.cli ...` builds a deterministic,
read-only Sleeper league snapshot for future waiver, lineup, trade, and
news-response policies. It verifies the managed roster by Sleeper user ID and
joins public league state to MAGIFF player identities and current ECR without
invoking an LLM. See
[`league_management/README.md`](league_management/README.md).

`python -m waivers.cli --context-only` inspects the compact default waiver
packet without spending model tokens. Running `python -m waivers.cli "..."`
starts the separate read-only waiver advisor, which searches bounded available-
player slices, verifies latest news for every proposed add and drop, and returns
a structured recommendation without executing it. See
[`waivers/README.md`](waivers/README.md).

## Local API

Install dependencies from `backend/`, then start one Uvicorn worker:

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

The API reads the project-root `.env`. It uses these variables:

```text
OPENAI_API_KEY=...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=...
MAGIFF_API_KEY=a-separate-random-private-token
MAGIFF_ENV=development
MAGIFF_CORS_ORIGINS=http://localhost:5173
```

`MAGIFF_API_KEY` authenticates callers to this API; it is not an OpenAI key.
Generate a local value with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Test the server without spending model tokens:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Send an agent request:

```bash
export MAGIFF_API_KEY="the-same-value-used-in-your-.env"
curl -X POST http://127.0.0.1:8000/v1/agent/query \
  -H "Authorization: Bearer $MAGIFF_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Who led running backs in PPR scoring in 2025?"}'
```

## Render

The repository-root `render.yaml` defines one free Python web service with
`backend/` as its root directory. Create a Render Blueprint from this repo and
provide the prompted runtime values:

- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `MAGIFF_API_KEY`
- `DISCORD_APPLICATION_ID`
- `DISCORD_PUBLIC_KEY`
- `DISCORD_TEST_GUILD_ID`
- `DISCORD_UAI_GUILD_ID`
- `DISCORD_UAI_ENABLED`

The OpenAI project key needs Responses write access and Embeddings access
because report retrieval embeds incoming queries. The FantasyPros key is not
needed on Render because GitHub Actions owns report ingestion.

Render checks `/health`, which deliberately performs no OpenAI or Supabase
requests. `/ready` confirms that required configuration is present. The server
uses one worker because the planner and reranker currently use local SQLite
caches; Render's free filesystem is ephemeral, so those caches improve only
within the lifetime of one instance.

When a browser frontend is added, set `MAGIFF_CORS_ORIGINS` to its exact origin
(for example, the Vercel production URL). Never put `MAGIFF_API_KEY` in Vite or
other browser-visible code. Replace this temporary private bearer token with
real user authentication before exposing the query endpoint to untrusted users.

## Private Discord command

Discord calls `POST /v1/discord/interactions` directly. That route does not use
`MAGIFF_API_KEY`; it verifies Discord's Ed25519 request signature, restricts
commands to the enabled test/UAI guild allowlist, immediately shows the quoted
question and a thinking indicator, and edits that message after MAGIFF finishes.
The test guild can use every command; UAI gets `/news`, `/stats`, and `/game`
and can be disabled independently. `/ask` runs
the agent; `/news` performs a deterministic newest-first report read, while
`/stats` performs deterministic structured lookup and safe formula analytics
with native field autocomplete. `/game challenge` runs the deterministic,
button-driven 17-0 Challenge in either season-total or PPR-PPG mode and persists
users, sessions, picks, and actions in Supabase. None of the direct commands
invokes an LLM.
See the complete parameter and retry reference in
[`integrations/README.md`](integrations/README.md).

Add these runtime variables to Render alongside the existing API variables:

```text
DISCORD_APPLICATION_ID=the-application-id-from-General-Information
DISCORD_PUBLIC_KEY=the-public-key-from-General-Information
DISCORD_TEST_GUILD_ID=the-private-test-server-id
DISCORD_UAI_GUILD_ID=the-friends-server-id
DISCORD_UAI_ENABLED=true
```

Keep the bot token only in the project-root local `.env`; it is needed for
command registration but not by the running interaction endpoint:

```text
DISCORD_BOT_TOKEN=the-private-bot-token
```

After Render deploys the Discord code:

1. Open `https://YOUR-RENDER-HOST/health` so the free service is awake.
2. In Discord's Developer Portal, set the Interactions Endpoint URL to
   `https://YOUR-RENDER-HOST/v1/discord/interactions` and save it.
3. From `backend/` with the virtual environment active, run:

   ```bash
   python -m jobs.register_discord_commands
   ```

4. Install the application in the configured server if it is not already
   installed, then run `/ask`, `/news`, `/stats`, or `/game challenge` there.

Commands are registered at guild scope, so changes appear quickly and they are
not published globally. The registration job installs `/ask`, `/news`, `/stats`,
and `/game` in the test guild, but `/news`, `/stats`, and `/game` in UAI. Setting
`DISCORD_UAI_ENABLED=false` blocks execution there without deleting its visible
commands. Render's free service can sleep; a cold start may miss
Discord's three-second acknowledgement deadline. Wake `/health` before a demo.
Do not place `DISCORD_BOT_TOKEN`, `MAGIFF_API_KEY`, or any provider secret in
Discord URLs, source control, or browser code.
