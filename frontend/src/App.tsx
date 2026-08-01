import { FormEvent, FormEventHandler, useEffect, useMemo, useRef, useState } from "react";
import {
  ChatMode,
  Message,
  fetchSession,
  resetSession,
  sendMessage,
} from "./api";

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

function renderContent(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return <span key={i}>{part}</span>;
  });
}

export default function App() {
  const [sessionId] = useState(() => newSessionId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [mode, setMode] = useState<ChatMode>("bot");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const snap = await fetchSession(sessionId);
        if (cancelled) return;
        setMessages(snap.messages);
        setSuggestions(snap.suggestions);
        setMode(snap.mode);
      } catch {
        if (!cancelled) {
          setError("Can't reach the backend. Is it running on port 8000?");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const title = useMemo(
    () => (mode === "live_agent" ? "Live Agent" : "North Star Support Bot"),
    [mode],
  );

  async function handleSend(raw: string) {
    const message = raw.trim();
    if (!message || loading) return;
    setError(null);
    setLoading(true);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    try {
      const res = await sendMessage(sessionId, message);
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
      setSuggestions(res.suggestions);
      setMode(res.mode);
    } catch {
      setError("Something went wrong sending that message.");
      setMessages((prev) => prev.slice(0, -1));
      setInput(message);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void handleSend(input);
  }

  async function onReset() {
    setLoading(true);
    setError(null);
    try {
      const snap = await resetSession(sessionId);
      setMessages(snap.messages);
      setSuggestions(snap.suggestions);
      setMode(snap.mode);
      setInput("");
    } catch {
      setError("Could not reset the chat.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="glow glow-a" aria-hidden />
      <div className="glow glow-b" aria-hidden />
      <main className="shell">
        <header className="topbar">
          <div className="brand-block">
            <p className="brand">North Star</p>
            <h1>{title}</h1>
            <p className="tagline">Outdoor gear support  -  track, return, or get geared up.</p>
          </div>
          <button type="button" className="ghost-btn" onClick={() => void onReset()} disabled={loading}>
            New chat
          </button>
        </header>

        {mode === "live_agent" && (
          <div className="live-banner" role="status">
            <span className="pulse" />
            Live Agent connected (simulated). Say &quot;main menu&quot; to return to the bot.
          </div>
        )}

        <section className="transcript" aria-live="polite">
          {messages.map((m, idx) => (
            <article key={`${m.role}-${idx}`} className={`bubble ${m.role}`}>
              <p className="who">{m.role === "assistant" ? (mode === "live_agent" && idx === messages.length - 1 ? "Live Agent" : "North Star") : "You"}</p>
              <div className="body">
                {m.content.split("\n").map((line, i) => (
                  <p key={i}>{renderContent(line) || <br />}</p>
                ))}
              </div>
            </article>
          ))}
          {loading && (
            <article className="bubble assistant typing">
              <p className="who">North Star</p>
              <div className="body"><span className="dot" /><span className="dot" /><span className="dot" /></div>
            </article>
          )}
          <div ref={bottomRef} />
        </section>

        {error && <p className="error">{error}</p>}

        {suggestions.length > 0 && (
          <div className="chips" aria-label="Quick replies">
            {suggestions.map((s) => (
              <button key={s} type="button" className="chip" disabled={loading} onClick={() => void handleSend(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        <form className="composer" onSubmit={onSubmit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={mode === "live_agent" ? "Message the live agent..." : "Ask about an order, return, or gear..."}
            disabled={loading}
            aria-label="Message"
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
