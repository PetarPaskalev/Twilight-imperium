"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from '../contexts/AuthContext';
import AuthModal from '../components/AuthModal';
import UserProfile from '../components/UserProfile';

type Message = { role: "user" | "assistant"; content: string };

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Page() {
  const { user, session, loading: authLoading } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);
  const STORAGE_KEY = "ti_messages";

  // Show auth modal if user is not authenticated after loading
  useEffect(() => {
    if (!authLoading && !user) {
      setShowAuthModal(true);
    }
  }, [authLoading, user]);

  useEffect(() => {
    const sid = window.localStorage.getItem("ti_session_id");
    if (sid) setSessionId(sid);
    // Load any saved messages from previous visits
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: Message[] = JSON.parse(saved);
        if (Array.isArray(parsed)) setMessages(parsed);
      }
    } catch {}
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  // Persist messages so users can see previous answers on reload
  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {}
  }, [messages]);

  async function sendMessage() {
    if (!canSend) return;
    
    // Check if user is authenticated
    if (!user || !session) {
      setShowAuthModal(true);
      return;
    }

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    
    try {
      // Get JWT token from session
      const token = session.access_token;
      
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ message: userMsg.content, session_id: sessionId ?? undefined }),
      });
      
      if (!res.ok) {
        if (res.status === 401) {
          throw new Error("Authentication failed. Please log in again.");
        }
        if (res.status === 429) {
          throw new Error("Daily message limit reached. Upgrade to paid tier for more messages.");
        }
        throw new Error(`HTTP ${res.status}`);
      }
      
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
    window.localStorage.removeItem(STORAGE_KEY);
  }

  return (
    <main style={{ display: "flex", flexDirection: "column", minHeight: "100vh", background: "#0b0f14", color: "#e5e7eb", position: "relative" }}>
      {/* Authentication Modal */}
      <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
      
      {/* User Profile Display */}
      {user && <UserProfile />}
      
      <header style={{ padding: "24px 20px", borderBottom: "1px solid #1f2430", background: "#0b0f14", position: "sticky", top: 0, zIndex: 10 }}>
        <div style={{ maxWidth: 900, margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 style={{ margin: 0, color: "#ffffff", fontSize: "1.75rem" }}>Twilight Imperium Assistant</h1>
            <p style={{ margin: "8px 0 0 0", color: "#9aa0a6" }}>Ask about rules, strategy cards, and faction abilities.</p>
          </div>
          {!user && (
            <button
              onClick={() => setShowAuthModal(true)}
              style={{
                padding: "10px 20px",
                background: "#2563eb",
                color: "#fff",
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                fontSize: "0.95rem",
                fontWeight: "bold",
              }}
            >
              Sign In
            </button>
          )}
        </div>
      </header>

      <section style={{ flex: 1, padding: "24px 20px", overflowY: "auto" }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          {messages.length === 0 && (
            <div style={{ color: "#777", marginTop: 32, fontSize: "0.95rem" }}>
              Try: "What does the Leadership strategy card do?"
            </div>
          )}
          <div
            style={{
              background: "#151821",
              border: "1px solid #222832",
              borderRadius: 12,
              padding: "24px",
              boxShadow: "0 1px 2px rgba(0,0,0,0.15)",
              display: "flex",
              flexDirection: "column",
              gap: 20
            }}
          >
            {messages.map((m, i) => (
              <div 
                key={i} 
                style={{ 
                  whiteSpace: "pre-wrap",
                  animation: "fadeIn 0.4s ease-in-out"
                }}
              >
                {m.role === "user" ? (
                  <div
                    style={{
                      background: "#101521",
                      color: "#e6e9ef",
                      border: "1px solid #232a36",
                      borderRadius: 10,
                      padding: "14px 16px",
                      lineHeight: 1.6,
                      width: "fit-content",
                      maxWidth: "100%"
                    }}
                  >
                    {m.content}
                  </div>
                ) : (
                  <div style={{ color: "#e6e9ef", lineHeight: 1.7, padding: "4px 0" }}>
                    {m.content}
                  </div>
                )}
              </div>
            ))}
            <div ref={endRef} />

            {/* Input area inside the same box */}
            <div style={{ display: "flex", gap: 10, paddingTop: 16 }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask about Twilight Imperium..."
                style={{
                  flex: 1,
                  padding: "14px 16px",
                  border: "1px solid #2a2f3a",
                  borderRadius: 8,
                  background: "#0f131a",
                  color: "#e5e7eb",
                  fontSize: "0.95rem"
                }}
              />
              <button 
                onClick={sendMessage} 
                disabled={!canSend} 
                style={{ 
                  padding: "14px 20px", 
                  background: "#2563eb", 
                  color: "#fff", 
                  border: "none", 
                  borderRadius: 8,
                  cursor: canSend ? "pointer" : "not-allowed",
                  opacity: canSend ? 1 : 0.6,
                  transition: "all 0.2s ease"
                }}
              >
                {loading ? "Thinking..." : "Send"}
              </button>
              <button 
                onClick={clearChat} 
                style={{ 
                  padding: "14px 20px", 
                  background: "#374151", 
                  color: "#e5e7eb", 
                  border: "none", 
                  borderRadius: 8,
                  cursor: "pointer",
                  transition: "all 0.2s ease"
                }}
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </section>

    </main>
  );
}


