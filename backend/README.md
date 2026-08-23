# MAGIFF backend

The backend has three independent entry points:

- `python main.py` runs the interactive terminal agent.
- `uvicorn api.app:app --reload` runs the HTTP agent API.
- `python -m jobs.refresh_reports ...` runs report ingestion. Production
  ingestion remains in GitHub Actions and is not started by the API server.

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
provide the four prompted secrets:

- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `MAGIFF_API_KEY`

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
