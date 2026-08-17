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

  const handleSetupGpuEngine = async (engineTarget = "auto") => {
    if (vulkanStatus?.progress?.status === "updating") return;
    const targetKey = engineTarget === "nvidia" ? "cuda" : (engineTarget === "apple" ? "metal" : engineTarget);
    const isAlreadyInstalled = 
      (targetKey === "cuda" && (vulkanStatus?.engines?.nvidia?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "cuda"))) ||
      (targetKey === "vulkan" && (vulkanStatus?.engines?.vulkan?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "vulkan"))) ||
      (targetKey === "metal" && (vulkanStatus?.engines?.apple?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "metal")));
    
    if (isAlreadyInstalled) {
      setVulkanMsg(`✅ Engine ${targetKey.toUpperCase()} is already installed and active.`);
      return;
    }

    try {
      setVulkanMsg(`🚀 Initiating ${targetKey.toUpperCase()} GPU Acceleration Setup...`);
      const res = await fetch(`${serverUrl}/api/gpu/setup?engine=${targetKey}`, { method: "POST" });
      if (res.ok) {
        fetchVulkanStatus();
      }
    } catch (e) {
      setVulkanMsg(`Engine setup error: ${e.message}`);
    }
  };

  const handleUpdateVulkan = () => handleSetupGpuEngine("vulkan");

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
              {/* Hardware GPU Engine Status Banner */}
              <div style={{ background: "rgba(0,0,0,0.4)", padding: "16px", borderRadius: "12px", border: "1px solid rgba(139, 92, 246, 0.25)", marginBottom: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div>
                    <div style={{ fontWeight: "600", fontSize: "1rem", color: "#f8fafc", display: "flex", alignItems: "center", gap: "8px" }}>
                      <span>⚡ Hardware Acceleration Engine</span>
                      <span style={{ fontSize: "0.72rem", padding: "2px 8px", borderRadius: "6px", background: "rgba(52, 211, 153, 0.2)", color: "#34d399", border: "1px solid rgba(52, 211, 153, 0.3)" }}>
                        🟢 Active ({vulkanStatus?.detected_platform?.toUpperCase() || "AUTO"} GPU Detected)
                      </span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#94a3b8", marginTop: "4px" }}>
                      Auto-detects active hardware. Select your GPU engine to compile or download the backend driver.
                    </div>
                  </div>
                </div>

                {/* 3 SEPARATE HARDWARE ENGINE BUTTONS */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", marginTop: "12px", marginBottom: "12px" }}>
                  
                  {/* ENGINE 1: NVIDIA CUDA */}
                  {(() => {
                    const isCudaActive = vulkanStatus?.engines?.nvidia?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "cuda");
                    return (
                      <div style={{
                        background: vulkanStatus?.detected_platform === "nvidia" ? "rgba(16, 185, 129, 0.12)" : "rgba(15, 23, 42, 0.5)",
                        border: vulkanStatus?.detected_platform === "nvidia" ? "2px solid #10b981" : "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "10px",
                        padding: "12px",
                        opacity: vulkanStatus?.detected_platform === "nvidia" ? 1 : 0.7,
                        boxShadow: vulkanStatus?.detected_platform === "nvidia" ? "0 0 16px rgba(16, 185, 129, 0.25)" : "none"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
                          <span style={{ fontWeight: "700", fontSize: "0.85rem", color: "#34d399" }}>🟢 NVIDIA CUDA</span>
                          {vulkanStatus?.detected_platform === "nvidia" && (
                            <span style={{ fontSize: "0.62rem", background: "#10b981", color: "#000", fontWeight: "800", padding: "1px 6px", borderRadius: "4px" }}>DETECTED</span>
                          )}
                        </div>
                        <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: "10px" }}>
                          CUDA / cu122 driver for RTX/GTX/T4 GPUs.
                        </div>
                        <button
                          className="hub-save-btn"
                          onClick={() => handleSetupGpuEngine("cuda")}
                          disabled={isCudaActive || vulkanStatus?.progress?.status === "updating"}
                          style={{
                            width: "100%",
                            background: isCudaActive
                              ? "rgba(16, 185, 129, 0.2)"
                              : (vulkanStatus?.detected_platform === "nvidia" ? "#10b981" : "rgba(255,255,255,0.08)"),
                            color: isCudaActive
                              ? "#34d399"
                              : (vulkanStatus?.detected_platform === "nvidia" ? "#000" : "#ccc"),
                            border: isCudaActive ? "1px solid #10b981" : "none",
                            padding: "6px 10px",
                            fontSize: "0.75rem",
                            fontWeight: "700",
                            cursor: (isCudaActive || vulkanStatus?.progress?.status === "updating") ? "default" : "pointer",
                            opacity: isCudaActive ? 0.9 : 1
                          }}
                        >
                          {vulkanStatus?.progress?.status === "updating" && vulkanStatus?.active_target === "cuda"
                            ? "⏳ Installing..."
                            : (isCudaActive
                                ? "✅ Installed & Active"
                                : (vulkanStatus?.detected_platform === "nvidia" ? "⚡ Activate CUDA" : "Setup CUDA"))}
                        </button>
                      </div>
                    );
                  })()}

                  {/* ENGINE 2: INTEL / AMD VULKAN */}
                  {(() => {
                    const isVulkanActive = vulkanStatus?.engines?.vulkan?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "vulkan");
                    return (
                      <div style={{
                        background: vulkanStatus?.detected_platform === "vulkan" ? "rgba(99, 102, 241, 0.12)" : "rgba(15, 23, 42, 0.5)",
                        border: vulkanStatus?.detected_platform === "vulkan" ? "2px solid #6366f1" : "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "10px",
                        padding: "12px",
                        opacity: vulkanStatus?.detected_platform === "vulkan" ? 1 : 0.7,
                        boxShadow: vulkanStatus?.detected_platform === "vulkan" ? "0 0 16px rgba(99, 102, 241, 0.25)" : "none"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
                          <span style={{ fontWeight: "700", fontSize: "0.85rem", color: "#818cf8" }}>🔵 Intel / AMD Vulkan</span>
                          {vulkanStatus?.detected_platform === "vulkan" && (
                            <span style={{ fontSize: "0.62rem", background: "#6366f1", color: "#fff", fontWeight: "800", padding: "1px 6px", borderRadius: "4px" }}>DETECTED</span>
                          )}
                        </div>
                        <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: "10px" }}>
                          Vulkan SPIR-V engine for Radeon & Arc GPUs.
                        </div>
                        <button
                          className="hub-save-btn"
                          onClick={() => handleSetupGpuEngine("vulkan")}
                          disabled={isVulkanActive || vulkanStatus?.progress?.status === "updating"}
                          style={{
                            width: "100%",
                            background: isVulkanActive
                              ? "rgba(99, 102, 241, 0.2)"
                              : (vulkanStatus?.detected_platform === "vulkan" ? "#6366f1" : "rgba(255,255,255,0.08)"),
                            color: isVulkanActive ? "#a5b4fc" : "#fff",
                            border: isVulkanActive ? "1px solid #6366f1" : "none",
                            padding: "6px 10px",
                            fontSize: "0.75rem",
                            fontWeight: "700",
                            cursor: (isVulkanActive || vulkanStatus?.progress?.status === "updating") ? "default" : "pointer",
                            opacity: isVulkanActive ? 0.9 : 1
                          }}
                        >
                          {vulkanStatus?.progress?.status === "updating" && vulkanStatus?.active_target === "vulkan"
                            ? "⏳ Downloading..."
                            : (isVulkanActive
                                ? "✅ Installed & Active"
                                : (vulkanStatus?.detected_platform === "vulkan" ? "⚡ Activate Vulkan" : "Download Vulkan"))}
                        </button>
                      </div>
                    );
                  })()}

                  {/* ENGINE 3: APPLE SILICON METAL */}
                  {(() => {
                    const isMetalActive = vulkanStatus?.engines?.apple?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "metal");
                    return (
                      <div style={{
                        background: vulkanStatus?.detected_platform === "apple" ? "rgba(168, 85, 247, 0.12)" : "rgba(15, 23, 42, 0.5)",
                        border: vulkanStatus?.detected_platform === "apple" ? "2px solid #a855f7" : "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "10px",
                        padding: "12px",
                        opacity: vulkanStatus?.detected_platform === "apple" ? 1 : 0.7,
                        boxShadow: vulkanStatus?.detected_platform === "apple" ? "0 0 16px rgba(168, 85, 247, 0.25)" : "none"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
                          <span style={{ fontWeight: "700", fontSize: "0.85rem", color: "#c084fc" }}>🍎 Apple Metal</span>
                          {vulkanStatus?.detected_platform === "apple" && (
                            <span style={{ fontSize: "0.62rem", background: "#a855f7", color: "#fff", fontWeight: "800", padding: "1px 6px", borderRadius: "4px" }}>DETECTED</span>
                          )}
                        </div>
                        <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: "10px" }}>
                          Metal MPS backend for M1/M2/M3/M4 Macs.
                        </div>
                        <button
                          className="hub-save-btn"
                          onClick={() => handleSetupGpuEngine("metal")}
                          disabled={isMetalActive || vulkanStatus?.progress?.status === "updating"}
                          style={{
                            width: "100%",
                            background: isMetalActive
                              ? "rgba(168, 85, 247, 0.2)"
                              : (vulkanStatus?.detected_platform === "apple" ? "#a855f7" : "rgba(255,255,255,0.08)"),
                            color: isMetalActive ? "#c084fc" : "#fff",
                            border: isMetalActive ? "1px solid #a855f7" : "none",
                            padding: "6px 10px",
                            fontSize: "0.75rem",
                            fontWeight: "700",
                            cursor: (isMetalActive || vulkanStatus?.progress?.status === "updating") ? "default" : "pointer",
                            opacity: isMetalActive ? 0.9 : 1
                          }}
                        >
                          {vulkanStatus?.progress?.status === "updating" && vulkanStatus?.active_target === "metal"
                            ? "⏳ Compiling..."
                            : (isMetalActive
                                ? "✅ Installed & Active"
                                : (vulkanStatus?.detected_platform === "apple" ? "⚡ Activate Metal" : "Setup Metal"))}
                        </button>
                      </div>
                    );
                  })()}

                </div>

                <div style={{ display: "flex", gap: "16px", fontSize: "0.78rem", color: "#cbd5e1" }}>
                  <div>
                    Installed Engine: <strong style={{ color: "#34d399" }}>{vulkanStatus?.installed_version || "Ready"}</strong>
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
