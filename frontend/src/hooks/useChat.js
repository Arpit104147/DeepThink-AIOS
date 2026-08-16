import { useState, useRef } from "react";

export function useChat({
  serverUrl,
  routingMode,
  contextLength,
  maxTokens,
  temperature,
  deviceMode,
  searchMode,
  isConnected,
  isEvmActive,
}) {
  const [history, setHistory] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentLogs, setCurrentLogs] = useState([]);
  const [currentStream, setCurrentStream] = useState("");
  const [isPreloading, setIsPreloading] = useState(false);
  const [abortController, setAbortController] = useState(null);
  const currentLogsRef = useRef([]);

  const handleOffload = async () => {
    try {
      await fetch(`${serverUrl}/api/offload`, { method: "POST" });
      alert("All models offloaded from VRAM");
    } catch {
      alert("Failed to offload models.");
    }
  };

  const handleLoadAll = async () => {
    if (!isConnected || !isEvmActive || isPreloading) return;
    setIsPreloading(true);
    try {
      const res = await fetch(`${serverUrl}/api/load_all`, { method: "POST" });
      alert(res.ok ? "All models successfully loaded into System RAM" : "Error: Failed to load models.");
    } catch {
      alert("Error: Failed to load models.");
    } finally {
      setIsPreloading(false);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${serverUrl}/api/cancel`, { method: "POST" });
    } catch {}
    if (abortController) {
      abortController.abort();
      setIsGenerating(false);
      setAbortController(null);
      setCurrentStream("");
    }
  };

  const handleSend = async (userPrompt, attachedImage, setPrompt, setAttachedImage, textareaRef) => {
    if (!userPrompt.trim() && !attachedImage) return;

    const userText = userPrompt.trim();
    const img = attachedImage;
    if (setPrompt) setPrompt("");
    if (setAttachedImage) setAttachedImage(null);
    if (textareaRef?.current) textareaRef.current.style.height = "auto";

    setHistory((prev) => [...prev, { type: "user", text: userText || "📎 Image attached", image: img }]);
    setIsGenerating(true);
    setCurrentStream("");
    setCurrentLogs([]);
    currentLogsRef.current = [];

    const controller = new AbortController();
    setAbortController(controller);
    let fullText = "";

    try {
      const initLog = "Cold start: loading AI models into GPU memory (2-4 min on first run, instant after)...";
      setCurrentLogs([initLog]);
      currentLogsRef.current = [initLog];

      const res = await fetch(`${serverUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true",
          "bypass-tunnel-reminder": "true",
        },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: userText,
          image: img,
          mode: routingMode,
          context_length: contextLength,
          max_tokens: maxTokens,
          temperature,
          device_mode: deviceMode,
          gpu_layers: -1,
          search_mode: searchMode,
        }),
      });

      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try {
          const d = await res.json();
          if (d.detail) msg = d.detail;
        } catch {}
        throw new Error(msg);
      }

      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        throw new Error("Tunnel is blocking request (returned HTML). Open backend URL in browser first.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const combined = lineBuffer + chunk;
        const lines = combined.split("\n");
        lineBuffer = lines.pop() || "";

        for (let line of lines) {
          line = line.trim();
          if (line.startsWith("data: ")) line = line.substring(6);
          if (!line || line.includes('"keep_alive"')) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === "status") {
              currentLogsRef.current = [...currentLogsRef.current, data.message];
              setCurrentLogs((prev) => [...prev, data.message]);
            } else if (data.type === "chunk") {
              fullText += data.content || data.text || "";
              setCurrentStream(fullText);
            } else if (data.type === "final_response") {
              fullText = data.text;
              setCurrentStream(fullText);
              setHistory((prev) => [...prev, { type: "ai", text: fullText, logs: [] }]);
              setIsGenerating(false);
            } else if (data.type === "error") {
              setHistory((prev) => [...prev, { type: "ai", text: "Error: " + data.message }]);
              setIsGenerating(false);
            }
          } catch (parseErr) {
            console.warn("SSE parse error:", line.substring(0, 100));
          }
        }
      }

      if (lineBuffer.trim()) {
        try {
          const data = JSON.parse(lineBuffer.trim());
          if (data.type === "final_response") {
            fullText = data.text;
            setCurrentStream(fullText);
            setHistory((prev) => [...prev, { type: "ai", text: fullText, logs: [] }]);
            setIsGenerating(false);
          } else if (data.type === "status") {
            currentLogsRef.current = [...currentLogsRef.current, data.message];
            setCurrentLogs((prev) => [...prev, data.message]);
          } else if (data.type === "error") {
            setHistory((prev) => [...prev, { type: "ai", text: "Error: " + data.message }]);
            setIsGenerating(false);
          }
        } catch {}
      }

      if (fullText) {
        setHistory((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.type !== "ai" || last.text !== fullText) {
            return [...prev, { type: "ai", text: fullText }];
          }
          return prev;
        });
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setHistory((prev) => [...prev, { type: "ai", text: fullText || "Cancelled.", logs: currentLogsRef.current }]);
      } else if (err.message && (err.message.toLowerCase().includes("networkerror") || err.message.toLowerCase().includes("failed to fetch"))) {
        setHistory((prev) => [...prev, { type: "ai", text: `❌ **Cannot reach backend.**\n\nOpen a terminal and run:\n\`\`\`bash\n./venv/bin/python backend/app.py\n\`\`\`` }]);
      } else {
        setHistory((prev) => [...prev, { type: "ai", text: `Error: ${err.message}` }]);
      }
    } finally {
      setIsGenerating(false);
      setAbortController(null);
      setCurrentStream("");
      setHistory((prev) => {
        const copy = [...prev];
        const lastAi = [...copy].reverse().find((m) => m.type === "ai");
        if (lastAi && (!lastAi.logs || lastAi.logs.length === 0)) lastAi.logs = currentLogsRef.current;
        return copy;
      });
      setCurrentLogs([]);
    }
  };

  return {
    history,
    setHistory,
    isGenerating,
    currentLogs,
    currentStream,
    isPreloading,
    handleSend,
    handleStop,
    handleOffload,
    handleLoadAll,
  };
}
