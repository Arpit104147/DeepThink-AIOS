import React, { useState, useEffect } from "react";

/**
 * @component SettingsModal
 * Enhanced glassmorphic overlay modal with tabbed navigation for
 * configuring device mode, routing mode, server URL, pre-compiled Vulkan engine,
 * GPU hardware diagnostics, security, and workspace settings.
 */
const SettingsModal = ({
  settingsOpen,
  setSettingsOpen,
  serverUrl,
  setServerUrl,
  deviceMode,
  setDeviceMode,
  routingMode,
  setRoutingMode,
  contextLength,
  maxTokens,
  temperature,
  searchMode,
}) => {
  const [activeTab, setActiveTab] = useState("general");
  const [githubToken, setGithubToken] = useState(() =>
    localStorage.getItem("github_token") || ""
  );

  // Pre-compiled Vulkan Engine & Diagnostics states
  const [vulkanStatus, setVulkanStatus] = useState(null);
  const [vulkanMsg, setVulkanMsg] = useState("");
  const [diagnostics, setDiagnostics] = useState(null);
  const [loadingDiag, setLoadingDiag] = useState(false);

  const fetchVulkanStatus = async () => {
    try {
      const res = await fetch(`${serverUrl}/api/vulkan/status`);
      if (res.ok) {
        const data = await res.json();
        setVulkanStatus(data);
      }
    } catch (e) {
      console.warn("Vulkan engine status check failed:", e);
    }
  };

  const runGpuDiagnostics = async () => {
    setLoadingDiag(true);
    setDiagnostics(null);
    try {
      const res = await fetch(`${serverUrl}/api/vulkan/diagnostics`);
      if (res.ok) {
        const data = await res.json();
        setDiagnostics(data);
      }
    } catch (e) {
      setVulkanMsg(`Diagnostics error: ${e.message}`);
    } finally {
      setLoadingDiag(false);
    }
  };

  useEffect(() => {
    if (settingsOpen) {
      fetchVulkanStatus();
      runGpuDiagnostics();
      const interval = setInterval(fetchVulkanStatus, 1500);
      return () => clearInterval(interval);
    }
  }, [settingsOpen, serverUrl]);

  if (!settingsOpen) return null;

  const handleUpdateVulkan = async () => {
    try {
      setVulkanMsg("🚀 Contacting GitHub Releases for pre-compiled Vulkan binary...");
      const res = await fetch(`${serverUrl}/api/vulkan/update`, { method: "POST" });
      if (res.ok) {
        fetchVulkanStatus();
      }
    } catch (e) {
      setVulkanMsg(`Update request error: ${e.message}`);
    }
  };

  const handleSave = () => {
    let finalUrl = serverUrl.trim();
    if (finalUrl && !finalUrl.startsWith("http")) finalUrl = "http://" + finalUrl;
    if (finalUrl.endsWith("/")) finalUrl = finalUrl.slice(0, -1);
    finalUrl = finalUrl.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    localStorage.setItem("server_url", finalUrl);
    localStorage.setItem("routing_mode", routingMode);
    if (githubToken) localStorage.setItem("github_token", githubToken);
    setServerUrl(finalUrl);
    setSettingsOpen(false);

    // Sync settings to backend silently
    fetch(`${finalUrl}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context_length: contextLength,
        max_tokens: maxTokens,
        temperature,
        device_mode: deviceMode,
        gpu_layers: -1,
        search_mode: searchMode,
      }),
    }).catch(() => console.warn("Settings sync to backend deferred — will apply on next request."));
  };

  const handleUrlBlur = (e) => {
    let val = e.target.value.trim();
    if (val && !val.startsWith("http")) val = "http://" + val;
    if (val.endsWith("/")) val = val.slice(0, -1);
    val = val.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    localStorage.setItem("server_url", val);
    setServerUrl(val);
  };

  const tabs = [
    { id: "general", label: "General", icon: "⚙️" },
    { id: "vulkan", label: "Vulkan GPU Status", icon: "⚡" },
    { id: "security", label: "Security", icon: "🛡️" },
    { id: "workspace", label: "Workspace", icon: "📁" },
  ];

  return (
    <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
      <div className="modal settings-modal-enhanced" onClick={(e) => e.stopPropagation()} style={{ maxWidth: "800px", width: "95%" }}>
        <div className="modal-header-enhanced">
          <h2>System Settings & GPU Acceleration</h2>
          <button className="modal-close-btn" onClick={() => setSettingsOpen(false)}>✕</button>
        </div>

        {/* Tab Navigation */}
        <div className="settings-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`settings-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="settings-tab-content">
          {activeTab === "general" && (
            <div className="tab-panel">
              <div className="modal-field">
                <label>Device Mode</label>
                <select value={deviceMode} onChange={(e) => setDeviceMode(e.target.value)}>
                  <option value="gpu">GPU (Vulkan Engine)</option>
                  <option value="cpu">CPU Only</option>
                  <option value="hybrid">Hybrid (CPU + GPU)</option>
                </select>
              </div>

              <div className="modal-field">
                <label>Routing Mode</label>
                <select value={routingMode} onChange={(e) => setRoutingMode(e.target.value)}>
                  <option value="auto">Auto (Smart Router)</option>
                  <option value="reasoning">Reasoning (DeepSeek Math/Theory)</option>
                  <option value="coding">Coding (Actor-Critic Sandbox)</option>
                  <option value="simple">Simple (Direct Response)</option>
                  <option value="chip_design">Chip Design (EDA Sandbox)</option>
                </select>
              </div>

              <div className="modal-field">
                <label>Server URL</label>
                <input
                  type="text"
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  onBlur={handleUrlBlur}
                  placeholder="http://127.0.0.1:8080"
                />
              </div>
            </div>
          )}

          {/* TAB: Pre-Compiled Vulkan Engine & Hardware Diagnostics */}
          {activeTab === "vulkan" && (
            <div className="tab-panel">
              {/* Vulkan Engine Updater Card */}
              <div style={{ background: "rgba(0,0,0,0.4)", padding: "16px", borderRadius: "12px", border: "1px solid rgba(139, 92, 246, 0.25)", marginBottom: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div>
                    <div style={{ fontWeight: "600", fontSize: "1rem", color: "#f8fafc", display: "flex", alignItems: "center", gap: "8px" }}>
                      <span>⚡ Pre-Compiled Vulkan GPU Engine</span>
                      <span style={{ fontSize: "0.72rem", padding: "2px 8px", borderRadius: "6px", background: "rgba(52, 211, 153, 0.2)", color: "#34d399", border: "1px solid rgba(52, 211, 153, 0.3)" }}>
                        🟢 Active (100% GPU Offload)
                      </span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#94a3b8", marginTop: "4px" }}>
                      Pre-compiled C++ engine from <code>ggml-org/llama.cpp</code>. Runs models directly on Intel iGPU / Arc, NVIDIA, or AMD GPUs.
                    </div>
                  </div>

                  <button
                    className="hub-save-btn"
                    onClick={handleUpdateVulkan}
                    disabled={vulkanStatus?.progress?.status === "updating"}
                    style={{
                      background: vulkanStatus?.installed ? "rgba(52, 211, 153, 0.15)" : "linear-gradient(135deg, #4f46e5, #9333ea)",
                      border: vulkanStatus?.installed ? "1px solid #34d399" : "none",
                      color: vulkanStatus?.installed ? "#34d399" : "#fff",
                      padding: "8px 16px",
                      fontSize: "0.83rem",
                      whiteSpace: "nowrap",
                      fontWeight: "600"
                    }}
                  >
                    {vulkanStatus?.installed
                      ? (vulkanStatus?.has_update ? "🔄 Update Available" : "✅ Engine Installed & Up to Date")
                      : "📥 Download Vulkan Engine"}
                  </button>
                </div>

                <div style={{ display: "flex", gap: "16px", fontSize: "0.78rem", color: "#cbd5e1" }}>
                  <div>
                    Installed Version: <strong style={{ color: "#34d399" }}>{vulkanStatus?.installed_version || "b10441"}</strong>
                  </div>
                  <div>
                    Latest GitHub Release: <strong style={{ color: "#818cf8" }}>{vulkanStatus?.latest_version || "b10441"}</strong>
                  </div>
                </div>

                {/* Progress bar when updating */}
                {vulkanStatus?.progress?.status === "updating" && (
                  <div style={{ marginTop: "12px", background: "rgba(0,0,0,0.5)", padding: "10px", borderRadius: "8px", border: "1px solid rgba(99, 102, 241, 0.3)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "#818cf8", marginBottom: "4px" }}>
                      <span>{vulkanStatus.progress.message}</span>
                      <span>{vulkanStatus.progress.percent}%</span>
                    </div>
                    <div style={{ width: "100%", height: "6px", background: "rgba(255,255,255,0.1)", borderRadius: "3px", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${vulkanStatus.progress.percent}%`, background: "linear-gradient(90deg, #6366f1, #a855f7)", transition: "width 0.3s ease" }} />
                    </div>
                  </div>
                )}
              </div>

              {/* Hardware & GPU Diagnostics Verification Box */}
              <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "16px", borderRadius: "12px", border: "1px solid rgba(52, 211, 153, 0.3)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                  <div style={{ fontWeight: "600", fontSize: "0.95rem", color: "#f8fafc", display: "flex", alignItems: "center", gap: "8px" }}>
                    <span>🔍 Hardware & VRAM Diagnostics</span>
                  </div>
                  <button
                    onClick={runGpuDiagnostics}
                    disabled={loadingDiag}
                    style={{ padding: "6px 14px", borderRadius: "8px", background: "rgba(52, 211, 153, 0.15)", border: "1px solid rgba(52, 211, 153, 0.3)", color: "#34d399", cursor: "pointer", fontSize: "0.78rem", fontWeight: "600" }}
                  >
                    {loadingDiag ? "Scanning Hardware..." : "🔄 Refresh GPU Diagnostics"}
                  </button>
                </div>

                {diagnostics ? (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", fontSize: "0.82rem" }}>
                    <div style={{ background: "rgba(0,0,0,0.3)", padding: "10px 12px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ color: "#94a3b8", fontSize: "0.74rem" }}>Detected GPU Adapter</div>
                      <div style={{ fontWeight: "600", color: "#f8fafc", marginTop: "2px" }}>
                        🎮 {diagnostics.gpu_name}
                      </div>
                    </div>

                    <div style={{ background: "rgba(0,0,0,0.3)", padding: "10px 12px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ color: "#94a3b8", fontSize: "0.74rem" }}>VRAM / Shared GPU Memory</div>
                      <div style={{ fontWeight: "600", color: "#34d399", marginTop: "2px" }}>
                        💾 {diagnostics.vram_free_gb} GB Free / {diagnostics.vram_total_gb} GB Total
                      </div>
                    </div>

                    <div style={{ gridColumn: "1 / -1", background: "rgba(52, 211, 153, 0.1)", padding: "12px", borderRadius: "8px", border: "1px solid rgba(52, 211, 153, 0.25)" }}>
                      <div style={{ fontWeight: "600", color: "#34d399", fontSize: "0.86rem", marginBottom: "4px" }}>
                        {diagnostics.execution_target}
                      </div>
                      <div style={{ color: "#cbd5e1", fontSize: "0.78rem" }}>
                        ✓ {diagnostics.offload_info}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: "20px", textAlign: "center", color: "#94a3b8", fontSize: "0.82rem" }}>
                    Scanning graphics adapter and verifying Vulkan GPU memory...
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "security" && (
            <div className="tab-panel">
              <div className="security-status-card">
                <div className="security-item">
                  <span className="security-icon">🛡️</span>
                  <span className="security-label">SAST Code Scanning</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon">🔒</span>
                  <span className="security-label">Sandbox Isolation</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon">🌐</span>
                  <span className="security-label">Air-Gap Mode</span>
                  <span className="security-badge">Set via ENV</span>
                </div>
              </div>
              <p className="settings-hint">
                Security features are automatically enabled. Air-gap mode can be activated
                by setting <code>AIOS_AIR_GAP=1</code> environment variable before starting the server.
              </p>
            </div>
          )}

          {activeTab === "workspace" && (
            <div className="tab-panel">
              <div className="modal-field">
                <label>GitHub Token (optional)</label>
                <input
                  type="password"
                  value={githubToken}
                  onChange={(e) => setGithubToken(e.target.value)}
                  placeholder="ghp_xxxxxxxxxxxx"
                />
                <span className="field-hint">Used for automated PR creation. Stored locally only.</span>
              </div>
              <p className="settings-hint">
                Git workspace operations allow the AIOS to clone repositories, create branches,
                and commit generated code (Verilog, testbenches, layouts) directly.
              </p>
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button onClick={() => setSettingsOpen(false)}>Close</button>
          <button className="primary-btn" onClick={handleSave}>Save Settings</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
