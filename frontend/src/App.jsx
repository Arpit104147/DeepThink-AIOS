import React, { useState, useEffect, useRef } from "react";
import { Menu } from "lucide-react";
import "./index.css";

import Sidebar from "./components/layout/Sidebar";
import SettingsModal from "./components/layout/SettingsModal";
import ModelHubModal from "./components/layout/ModelHubModal";
import BenchmarkModal from "./components/layout/BenchmarkModal";
import InputArea from "./components/input/InputArea";
import MessageList from "./components/chat/MessageList";
import { useChat } from "./hooks/useChat";

export default function App() {
  // Server connection state
  const [serverUrl, setServerUrl] = useState(() =>
    localStorage.getItem("server_url") || "http://127.0.0.1:8080"
  );
  const [isConnected, setIsConnected] = useState(false);
  const [isEvmActive, setIsEvmActive] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${serverUrl}/api/status`);
        if (res.ok) {
          const data = await res.json();
          setIsConnected(true);
          setIsEvmActive(!!data.system?.evm_active);
        } else {
          setIsConnected(false);
          setIsEvmActive(false);
        }
      } catch {
        setIsConnected(false);
        setIsEvmActive(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  // Session management
  const [sessions, setSessions] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    } catch {
      return [];
    }
  });
  const [currentSessionId, setCurrentSessionId] = useState(Date.now());

  // UI Modals & Panels
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelHubOpen, setModelHubOpen] = useState(false);
  const [benchmarkOpen, setBenchmarkOpen] = useState(false);

  // Settings
  const [searchMode, setSearchMode] = useState("off");
  const [contextLength, setContextLength] = useState(0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [deviceMode, setDeviceMode] = useState("gpu");
  const [routingMode, setRoutingMode] = useState(() =>
    localStorage.getItem("routing_mode") || "auto"
  );

  // Prompt input state
  const [prompt, setPrompt] = useState("");
  const [attachedImage, setAttachedImage] = useState(null);
  const [displayText, setDisplayText] = useState("");
  const textareaRef = useRef(null);

  // Custom Chat Hook (SSE Streaming Controller)
  const {
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
  } = useChat({
    serverUrl,
    routingMode,
    contextLength,
    maxTokens,
    temperature,
    deviceMode,
    searchMode,
    isConnected,
    isEvmActive,
  });

  // Empty-state typing effect
  useEffect(() => {
    const target = "What is on your mind today?";
    let i = 0;
    setDisplayText("");
    const iv = setInterval(() => {
      setDisplayText(target.substring(0, i + 1));
      i++;
      if (i >= target.length) clearInterval(iv);
    }, 50);
    return () => clearInterval(iv);
  }, []);

  // Auto-focus textarea on session switch
  useEffect(() => {
    textareaRef.current?.focus();
  }, [currentSessionId]);

  // Persist chat sessions
  useEffect(() => {
    if (history.length === 0 && sessions.length === 0) return;
    setSessions((prev) => {
      const existing = prev.find((s) => s.id === currentSessionId);
      let title = "New Chat";
      const first = history.find((m) => m.type === "user");
      if (first) title = first.text.substring(0, 35) + (first.text.length > 35 ? "..." : "");

      let next;
      if (existing) {
        next = prev.map((s) => (s.id === currentSessionId ? { ...s, history, title } : s));
      } else {
        if (history.length === 0) return prev;
        next = [{ id: currentSessionId, title, history }, ...prev];
      }
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
  }, [history, currentSessionId]);

  // Session actions
  const createNewChat = () => {
    setCurrentSessionId(Date.now());
    setHistory([]);
  };

  const loadSession = (id) => {
    const s = sessions.find((x) => x.id === id);
    if (s) {
      setCurrentSessionId(id);
      setHistory(s.history);
      setSidebarOpen(false);
    }
  };

  const deleteSession = (id, e) => {
    e.stopPropagation();
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
    if (id === currentSessionId) createNewChat();
  };

  const onSendPrompt = (e) => {
    setMenuOpen(false);
    handleSend(prompt, attachedImage, setPrompt, setAttachedImage, textareaRef);
  };

  return (
    <div className="app">
      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        createNewChat={createNewChat}
        loadSession={loadSession}
        deleteSession={deleteSession}
        handleOffload={handleOffload}
        handleLoadAll={handleLoadAll}
        isConnected={isConnected}
        isEvmActive={isEvmActive}
        isPreloading={isPreloading}
        setSettingsOpen={setSettingsOpen}
        setModelHubOpen={setModelHubOpen}
        setBenchmarkOpen={setBenchmarkOpen}
      />

      <div className="main">
        {!sidebarOpen && (
          <button className="floating-open-btn" onClick={() => setSidebarOpen(true)}>
            <Menu size={18} />
          </button>
        )}

        <div className="chat-area">
          <MessageList
            history={history}
            isGenerating={isGenerating}
            currentLogs={currentLogs}
            currentStream={currentStream}
            displayText={displayText}
            onStarterClick={(text) => {
              setPrompt(text);
              setTimeout(() => handleSend(text, null, setPrompt, setAttachedImage, textareaRef), 50);
            }}
          />
        </div>

        <InputArea
          prompt={prompt}
          setPrompt={setPrompt}
          isGenerating={isGenerating}
          attachedImage={attachedImage}
          setAttachedImage={setAttachedImage}
          menuOpen={menuOpen}
          setMenuOpen={setMenuOpen}
          searchMode={searchMode}
          setSearchMode={setSearchMode}
          handleSend={onSendPrompt}
          handleStop={handleStop}
          setSettingsOpen={setSettingsOpen}
          textareaRef={textareaRef}
        />
      </div>

      <SettingsModal
        settingsOpen={settingsOpen}
        setSettingsOpen={setSettingsOpen}
        serverUrl={serverUrl}
        setServerUrl={setServerUrl}
        deviceMode={deviceMode}
        setDeviceMode={setDeviceMode}
        routingMode={routingMode}
        setRoutingMode={setRoutingMode}
        contextLength={contextLength}
        maxTokens={maxTokens}
        temperature={temperature}
        searchMode={searchMode}
      />

      <ModelHubModal
        open={modelHubOpen}
        setOpen={setModelHubOpen}
        serverUrl={serverUrl}
      />

      <BenchmarkModal
        open={benchmarkOpen}
        setOpen={setBenchmarkOpen}
        serverUrl={serverUrl}
      />
    </div>
  );
}
