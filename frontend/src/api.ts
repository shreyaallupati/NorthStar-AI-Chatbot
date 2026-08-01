const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type ChatMode = "bot" | "live_agent";

export type Message = {
  role: "user" | "assistant";
  content: string;
};

export type SessionSnapshot = {
  session_id: string;
  mode: ChatMode;
  messages: Message[];
  suggestions: string[];
};

export type ChatResponse = {
  reply: string;
  mode: ChatMode;
  intent: string;
  suggestions: string[];
  session_id: string;
};

export async function fetchSession(sessionId: string): Promise<SessionSnapshot> {
  const res = await fetch(`${API_URL}/chat/${sessionId}`);
  if (!res.ok) throw new Error("Failed to load session");
  return res.json();
}

export async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error("Failed to send message");
  return res.json();
}

export async function resetSession(sessionId: string): Promise<SessionSnapshot> {
  const res = await fetch(`${API_URL}/chat/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error("Failed to reset session");
  return res.json();
}
