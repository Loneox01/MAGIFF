const MAX_PROMPT_LENGTH = 4_000;

export const config = {
    maxDuration: 120,
};

function send(response, status, payload) {
    response.status(status).json(payload);
}

export default async function handler(request, response) {
    if (request.method !== "POST") {
        response.setHeader("Allow", "POST");
        send(response, 405, { detail: "Method not allowed" });
        return;
    }

    const backendUrl = process.env.MAGIFF_API_URL?.replace(/\/+$/, "");
    const backendKey = process.env.MAGIFF_API_KEY;
    const webAccessKey = process.env.MAGIFF_WEB_ACCESS_KEY;

    if (!backendUrl || !backendKey) {
        send(response, 503, { detail: "MAGIFF web service is not configured" });
        return;
    }

    if (
        webAccessKey
        && request.headers["x-magiff-web-key"] !== webAccessKey
    ) {
        send(response, 401, { detail: "Invalid web access key" });
        return;
    }

    let requestBody = request.body;
    if (typeof requestBody === "string") {
        try {
            requestBody = JSON.parse(requestBody);
        } catch {
            send(response, 400, { detail: "Request body must be valid JSON" });
            return;
        }
    }

    const prompt = typeof requestBody?.prompt === "string"
        ? requestBody.prompt.trim()
        : "";
    if (!prompt || prompt.length > MAX_PROMPT_LENGTH) {
        send(response, 400, {
            detail: `Prompt must contain between 1 and ${MAX_PROMPT_LENGTH} characters`,
        });
        return;
    }

    try {
        const upstream = await fetch(`${backendUrl}/v1/agent/query`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${backendKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt }),
            signal: AbortSignal.timeout(118_000),
        });
        const body = await upstream.text();
        const requestId = upstream.headers.get("x-request-id");
        if (requestId) response.setHeader("X-Request-ID", requestId);
        response.status(upstream.status);
        response.setHeader("Content-Type", "application/json");
        response.send(body);
    } catch (error) {
        const timedOut = error instanceof Error && error.name === "TimeoutError";
        send(response, timedOut ? 504 : 502, {
            detail: timedOut
                ? "MAGIFF took too long to respond"
                : "MAGIFF backend is unavailable",
        });
    }
}
