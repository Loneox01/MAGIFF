# MAGIFF frontend

The frontend is a small Vite/React client for the private MAGIFF agent API.
Browser requests use `/api/agent`; the bearer token is injected outside the
browser so it is never included in the JavaScript bundle.

## Local development

Keep the backend variables in the project-root `.env`, including:

```text
MAGIFF_API_URL=http://127.0.0.1:8000
MAGIFF_API_KEY=the-private-backend-token
```

Run the backend on port 8000, then run `npm run dev` from `frontend/`. Vite
proxies `/api/agent` to the backend and injects the bearer token locally.

## Vercel

Create a Vercel project with `frontend` as its root directory and add these
server-side environment variables:

```text
MAGIFF_API_URL=https://your-render-service.onrender.com
MAGIFF_API_KEY=the-same-private-token-configured-on-render
MAGIFF_WEB_ACCESS_KEY=an-optional-separate-passphrase-for-web-users
```

`MAGIFF_WEB_ACCESS_KEY` is optional but strongly recommended for any public
deployment. Visitors enter it through the **Access** control and it is retained
only in that tab's `sessionStorage`. Do not create a `VITE_MAGIFF_API_KEY`;
variables with the `VITE_` prefix are shipped to every browser.
