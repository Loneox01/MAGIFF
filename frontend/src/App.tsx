import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { AgentApiError, queryAgent, type AgentQueryResponse } from "./api";
import "./App.css";

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string;
    response?: AgentQueryResponse;
    failed?: boolean;
};

const STARTER_PROMPTS = [
    "Who were the top five PPR running backs in 2025?",
    "What is the latest news affecting Drake London's outlook?",
    "Which 2025 offense had the most yards per play?",
];

function messageId(): string {
    return crypto.randomUUID();
}

function renderInline(text: string): ReactNode[] {
    const parts = text.split(
        /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)]+\))/g,
    );

    return parts.map((part, index) => {
        const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
        if (link) {
            return (
                <a key={index} href={link[2]} target="_blank" rel="noreferrer">
                    {link[1]}
                </a>
            );
        }
        if (part.startsWith("**") && part.endsWith("**")) {
            return <strong key={index}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("*") && part.endsWith("*")) {
            return <em key={index}>{part.slice(1, -1)}</em>;
        }
        if (part.startsWith("`") && part.endsWith("`")) {
            return <code key={index}>{part.slice(1, -1)}</code>;
        }
        return part;
    });
}

function MarkdownAnswer({ content }: { content: string }) {
    const lines = content.split("\n");
    const blocks: ReactNode[] = [];

    for (let index = 0; index < lines.length;) {
        const line = lines[index].trimEnd();
        if (!line.trim()) {
            index += 1;
            continue;
        }

        if (/^[-*]\s+/.test(line)) {
            const items: string[] = [];
            while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
                items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
                index += 1;
            }
            blocks.push(
                <ul key={`ul-${index}`}>
                    {items.map((item, itemIndex) => (
                        <li key={itemIndex}>{renderInline(item)}</li>
                    ))}
                </ul>,
            );
            continue;
        }

        if (/^\d+\.\s+/.test(line)) {
            const items: string[] = [];
            while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
                items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
                index += 1;
            }
            blocks.push(
                <ol key={`ol-${index}`}>
                    {items.map((item, itemIndex) => (
                        <li key={itemIndex}>{renderInline(item)}</li>
                    ))}
                </ol>,
            );
            continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
            blocks.push(
                <h3 key={`heading-${index}`}>{renderInline(heading[2])}</h3>,
            );
        } else {
            blocks.push(<p key={`paragraph-${index}`}>{renderInline(line)}</p>);
        }
        index += 1;
    }

    return <div className="answer-copy">{blocks}</div>;
}

function ResponseMeta({ response }: { response: AgentQueryResponse }) {
    const toolCount = response.tool_calls.length;
    const totalTokens = response.usage.input_tokens + response.usage.output_tokens;
    const cost = response.estimated_cost_usd;

    return (
        <div className="response-meta" aria-label="Response details">
            <span>{response.model.replace("gpt-5.6-", "")}</span>
            <span>{response.latency_seconds.toFixed(1)}s</span>
            <span>{toolCount} {toolCount === 1 ? "tool" : "tools"}</span>
            <span>{totalTokens.toLocaleString()} tokens</span>
            {cost != null && <span>${cost.toFixed(4)}</span>}
        </div>
    );
}

function App() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [prompt, setPrompt] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [showAccess, setShowAccess] = useState(false);
    const [accessKey, setAccessKey] = useState(
        () => sessionStorage.getItem("magiff-web-access-key") ?? "",
    );
    const conversationEnd = useRef<HTMLDivElement>(null);
    const textarea = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        conversationEnd.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    const saveAccessKey = () => {
        const normalized = accessKey.trim();
        if (normalized) {
            sessionStorage.setItem("magiff-web-access-key", normalized);
        } else {
            sessionStorage.removeItem("magiff-web-access-key");
        }
        setShowAccess(false);
        textarea.current?.focus();
    };

    const submitPrompt = async (event?: FormEvent) => {
        event?.preventDefault();
        const question = prompt.trim();
        if (!question || isLoading) return;

        setMessages((current) => [
            ...current,
            { id: messageId(), role: "user", content: question },
        ]);
        setPrompt("");
        setIsLoading(true);

        try {
            const response = await queryAgent(
                question,
                sessionStorage.getItem("magiff-web-access-key") ?? undefined,
            );
            setMessages((current) => [
                ...current,
                {
                    id: messageId(),
                    role: "assistant",
                    content: response.answer,
                    response,
                },
            ]);
        } catch (error) {
            const apiError = error instanceof AgentApiError ? error : null;
            if (apiError?.status === 401) setShowAccess(true);
            setMessages((current) => [
                ...current,
                {
                    id: messageId(),
                    role: "assistant",
                    failed: true,
                    content: apiError?.message
                        ?? "MAGIFF could not finish that request. Please try again.",
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const chooseStarter = (starter: string) => {
        setPrompt(starter);
        textarea.current?.focus();
    };

    return (
        <div className="app-shell">
            <header className="site-header">
                <a className="brand" href="/" aria-label="MAGIFF home">
                    <span className="brand-mark" aria-hidden="true">M</span>
                    <span>
                        <strong>MAGIFF</strong>
                        <small>Fantasy football intelligence</small>
                    </span>
                </a>
                <button
                    className="access-button"
                    type="button"
                    onClick={() => setShowAccess((current) => !current)}
                    aria-expanded={showAccess}
                >
                    <span aria-hidden="true">⌁</span>
                    Access
                </button>
            </header>

            {showAccess && (
                <section className="access-panel" aria-label="Private access settings">
                    <div>
                        <strong>Private access</strong>
                        <p>Stored only for this browser session.</p>
                    </div>
                    <input
                        type="password"
                        value={accessKey}
                        onChange={(event) => setAccessKey(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === "Enter") saveAccessKey();
                        }}
                        placeholder="Web access key"
                        autoComplete="off"
                    />
                    <button type="button" onClick={saveAccessKey}>Save</button>
                </section>
            )}

            <main className="conversation" aria-live="polite">
                {messages.length === 0 ? (
                    <section className="welcome">
                        <p className="eyebrow">DATA + REPORTS + REASONING</p>
                        <h1>Your fantasy edge,<br />without the guesswork.</h1>
                        <p className="welcome-copy">
                            Ask about player value, historical performance, matchups,
                            depth charts, ECR, or the latest NFL reports.
                        </p>
                        <div className="starter-grid">
                            {STARTER_PROMPTS.map((starter) => (
                                <button
                                    key={starter}
                                    type="button"
                                    onClick={() => chooseStarter(starter)}
                                >
                                    <span>{starter}</span>
                                    <span aria-hidden="true">↗</span>
                                </button>
                            ))}
                        </div>
                    </section>
                ) : (
                    <section className="message-list" aria-label="Conversation">
                        {messages.map((message) => (
                            <article
                                className={`message ${message.role} ${message.failed ? "failed" : ""}`}
                                key={message.id}
                            >
                                <div className="message-label">
                                    {message.role === "user" ? "You" : "MAGIFF"}
                                </div>
                                {message.role === "assistant"
                                    ? <MarkdownAnswer content={message.content} />
                                    : <p>{message.content}</p>}
                                {message.response && <ResponseMeta response={message.response} />}
                            </article>
                        ))}
                        {isLoading && (
                            <article className="message assistant loading-message">
                                <div className="message-label">MAGIFF</div>
                                <div className="thinking-dots" aria-label="MAGIFF is thinking">
                                    <span />
                                    <span />
                                    <span />
                                </div>
                                <p className="cold-start-note">
                                    Searching stats and reports. A sleeping server may take a minute to wake.
                                </p>
                            </article>
                        )}
                        <div ref={conversationEnd} />
                    </section>
                )}
            </main>

            <footer className="composer-wrap">
                <form className="composer" onSubmit={submitPrompt}>
                    <textarea
                        ref={textarea}
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value.slice(0, 4_000))}
                        onKeyDown={(event) => {
                            if (event.key === "Enter" && !event.shiftKey) {
                                event.preventDefault();
                                void submitPrompt();
                            }
                        }}
                        placeholder="Ask MAGIFF about your draft, lineup, players, or the latest news..."
                        rows={1}
                        disabled={isLoading}
                        aria-label="Ask MAGIFF"
                    />
                    <button
                        className="send-button"
                        type="submit"
                        disabled={!prompt.trim() || isLoading}
                        aria-label="Send question"
                    >
                        <span>Ask</span>
                        <span aria-hidden="true">↑</span>
                    </button>
                </form>
                <p>MAGIFF can make mistakes. Verify important lineup and injury decisions.</p>
            </footer>
        </div>
    );
}

export default App;
