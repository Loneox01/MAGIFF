export type AgentQueryResponse = {
    request_id: string;
    answer: string;
    model: string;
    latency_seconds: number;
    tool_rounds: number;
    usage: {
        input_tokens: number;
        cached_input_tokens: number;
        output_tokens: number;
    };
    route: {
        intent: string;
        capabilities: string[];
        structured_domains: string[];
    };
    tool_calls: Array<{
        name: string;
        succeeded: boolean;
        error: string | null;
    }>;
    estimated_cost_usd: number | null;
    web_search_calls: number;
    web_sources: Array<{ title: string; url: string }>;
};

export class AgentApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
        super(message);
        this.name = "AgentApiError";
        this.status = status;
    }
}

function errorMessage(status: number, detail?: string): string {
    if (status === 401) {
        return "This MAGIFF instance requires a valid access key. Add it under Access and retry.";
    }
    if (status === 429) {
        return "MAGIFF is handling too many requests right now. Please wait a moment and retry.";
    }
    if (status === 502 || status === 503 || status === 504) {
        return "MAGIFF is waking up or temporarily unavailable. Please retry in a moment.";
    }
    return detail || "MAGIFF could not finish that request. Please try again.";
}

export async function queryAgent(
    prompt: string,
    accessKey?: string,
): Promise<AgentQueryResponse> {
    const response = await fetch("/api/agent", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            ...(accessKey ? { "X-Magiff-Web-Key": accessKey } : {}),
        },
        body: JSON.stringify({ prompt }),
    });

    const payload = await response.json().catch(() => null) as
        | AgentQueryResponse
        | { detail?: string; error?: string }
        | null;

    if (!response.ok) {
        const detail = payload && "detail" in payload
            ? payload.detail
            : payload && "error" in payload
                ? payload.error
                : undefined;
        throw new AgentApiError(errorMessage(response.status, detail), response.status);
    }

    return payload as AgentQueryResponse;
}
