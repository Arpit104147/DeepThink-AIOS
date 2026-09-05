import React from "react";
import { Menu, Plus, Trash2, Box, BarChart3, Loader, Zap, MessageSquare, X, Settings } from "lucide-react";

/**
 * @component Sidebar
 * Session management sidebar with chat history, memory controls,
 * connection status indicator, and user profile footer.
 */
const Sidebar = ({
  sidebarOpen,
  setSidebarOpen,
  sessions,
  currentSessionId,
  createNewChat,
  loadSession,
  deleteSession,
  handleOffload,
  handleLoadAll,
  isConnected,
  isEvmActive,
  isPreloading,
  setSettingsOpen,
  setModelHubOpen,
  setBenchmarkOpen,
}) => {
  return (
    <div className={`sidebar ${!sidebarOpen ? "closed" : ""}`}>
      <div className="sidebar-top">
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(false)}>
          <Menu size={18} />
        </button>
        {/* Connection Status Badge */}
        <div className={`connection-badge ${isConnected ? "connected" : "disconnected"}`}>
          <span className="connection-dot"></span>
          <span className="connection-text">{isConnected ? "Online" : "Offline"}</span>
        </div>
      </div>

      <button className="new-chat-btn" onClick={createNewChat}>
        <Plus size={16} /> New chat
      </button>

      <div className="sidebar-nav">
        <button className="nav-item" onClick={handleOffload}>
          <span className="nav-icon"><Trash2 size={15} /></span> Offload Memory
        </button>
        <button className="nav-item" onClick={() => setModelHubOpen(true)}>
          <span className="nav-icon"><Box size={15} /></span> Model Hub
        </button>
        <button className="nav-item" onClick={() => setBenchmarkOpen(true)}>
          <span className="nav-icon"><BarChart3 size={15} /></span> Benchmark Studio
        </button>
        <button
          className="nav-item"
          onClick={handleLoadAll}
          disabled={!isConnected || isPreloading}
          title={
            !isConnected
              ? "Backend disconnected"
              : isPreloading
                ? "Loading models into System RAM..."
                : "Pre-load all downloaded models into System RAM"
          }
        >
          <span className="nav-icon">{isPreloading ? <Loader size={15} className="spin" /> : <Zap size={15} />}</span>
          {isPreloading ? "Loading Swarm..." : "Load All Models"}
        </button>
      </div>

      <div className="sidebar-section-title">Recents</div>
      <div className="history-list">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`history-item ${s.id === currentSessionId ? "active" : ""}`}
            onClick={() => loadSession(s.id)}
          >
            <span className="history-item-title"><MessageSquare size={13} /> {s.title}</span>
            <button className="delete-btn" onClick={(e) => deleteSession(s.id, e)}>
              <X size={14} />
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <div className="history-empty">No recent chats.</div>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="user-row">
          <div className="user-avatar">A</div>
          <span className="user-name">ARPIT BEHERA</span>
          <button
            className="sidebar-settings-btn"
            onClick={(e) => { e.stopPropagation(); setSettingsOpen(true); }}
            title="Settings"
          >
            <Settings size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
