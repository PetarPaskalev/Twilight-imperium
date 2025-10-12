"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Message = { role: "user" | "assistant"; content: string };

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const sid = window.localStorage.getItem("ti_session_id");
    if (sid) setSessionId(sid);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function sendMessage() {
    if (!canSend) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.content, session_id: sessionId ?? undefined }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        window.localStorage.setItem("ti_session_id", data.session_id);
      }
      const botMsg: Message = { role: "assistant", content: data.response };
      setMessages((prev) => [...prev, botMsg]);
    } catch (e: any) {
      const err: Message = { role: "assistant", content: `Error: ${e.message || e}` };
      setMessages((prev) => [...prev, err]);
    } finally {
      setLoading(false);
    }
  }

  async function clearChat() {
    const id = sessionId || window.localStorage.getItem("ti_session_id");
    if (id) {
      try {
        await fetch(`${API_URL}/clear/${id}`, { method: "POST" });
      } catch {}
    }
    setMessages([]);
    setSessionId(null);
    window.localStorage.removeItem("ti_session_id");
  }

  return (
    <main style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <header style={{ padding: "12px 16px", borderBottom: "1px solid #eee" }}>
        <h1 style={{ margin: 0 }}>Twilight Imperium Assistant</h1>
        <p style={{ margin: 0, color: "#555" }}>Ask about rules, strategy cards, and faction abilities.</p>
      </header>

      <section style={{ flex: 1, padding: 16, overflowY: "auto" }}>
        {messages.length === 0 && (
          <div style={{ color: "#777", marginTop: 24 }}>
            Try: "What does the Leadership strategy card do?"
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ margin: "8px 0", whiteSpace: "pre-wrap" }}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong> {m.content}
          </div>
        ))}
        <div ref={endRef} />
      </section>

      <footer style={{ padding: 16, borderTop: "1px solid #eee", display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about Twilight Imperium..."
          style={{ flex: 1, padding: 12, border: "1px solid #ddd", borderRadius: 8 }}
        />
        <button onClick={sendMessage} disabled={!canSend} style={{ padding: "12px 16px" }}>
          {loading ? "Thinking..." : "Send"}
        </button>
        <button onClick={clearChat} style={{ padding: "12px 16px" }}>Clear</button>
      </footer>
    </main>
  );
}


