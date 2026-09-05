import { useState, useEffect } from "react";
import { Settings, Zap, Shield, FolderOpen, RefreshCw, Gpu, Monitor, HardDrive, Check, Loader, Lock, Globe } from "lucide-react";
import Modal from "../common/Modal";

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
      console.warn(`Diagnostics error: ${e.message}`);
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

  const activeEngineTarget = (() => {
    if (vulkanStatus?.engines?.nvidia?.installed) return "cuda";
    if (vulkanStatus?.engines?.vulkan?.installed) return "vulkan";
    if (vulkanStatus?.engines?.apple?.installed) return "metal";
    const detected = vulkanStatus?.detected_platform;
    if (detected === "nvidia") return "cuda";
    if (detected === "apple") return "metal";
    if (detected === "vulkan") return "vulkan";
    return "auto";
  })();

  const activeEngineTitle = (() => {
    if (activeEngineTarget === "cuda") return "NVIDIA CUDA";
    if (activeEngineTarget === "metal") return "Apple Metal";
    if (activeEngineTarget === "vulkan") return "Intel / AMD Vulkan";
    return "GPU Acceleration";
  })();

  const handleSetupGpuEngine = async (engineTarget = "auto", forceUpdate = false) => {
    if (vulkanStatus?.progress?.status === "updating") return;
    const targetKey = engineTarget === "nvidia" ? "cuda" : (engineTarget === "apple" ? "metal" : engineTarget);
    const isAlreadyInstalled = 
      (targetKey === "cuda" && (vulkanStatus?.engines?.nvidia?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "cuda"))) ||
      (targetKey === "vulkan" && (vulkanStatus?.engines?.vulkan?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "vulkan"))) ||
      (targetKey === "metal" && (vulkanStatus?.engines?.apple?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "metal")));
    
    if (isAlreadyInstalled && !forceUpdate) {
      console.log(`✅ Engine ${targetKey.toUpperCase()} is already installed and active.`);
      return;
    }

    try {
      console.log(`🚀 Initiating ${targetKey.toUpperCase()} GPU Acceleration ${forceUpdate ? "Update" : "Setup"}...`);
      const res = await fetch(`${serverUrl}/api/gpu/setup?engine=${targetKey}`, { method: "POST" });
      if (res.ok) {
        fetchVulkanStatus();
      }
    } catch (e) {
      console.warn(`Engine setup error: ${e.message}`);
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
    { id: "general", label: "General", icon: <Settings size={15} /> },
    { id: "vulkan", label: "GPU Acceleration", icon: <Zap size={15} /> },
    { id: "security", label: "Security", icon: <Shield size={15} /> },
    { id: "workspace", label: "Workspace", icon: <FolderOpen size={15} /> },
  ];

  return (
    <Modal open={settingsOpen} onClose={() => setSettingsOpen(false)} title="System Settings & GPU Acceleration" maxWidth="800px">
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
              <div className="gpu-status-banner" style={{ background: "var(--dt-surface-1)", padding: "16px", borderRadius: "var(--dt-radius-lg)", border: "1px solid rgba(139, 92, 246, 0.25)", marginBottom: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div>
                    <div style={{ fontWeight: "600", fontSize: "1rem", color: "var(--dt-text)", display: "flex", alignItems: "center", gap: "8px" }}>
                      <Zap size={16} /> Hardware Acceleration Engine
                      <span style={{ fontSize: "0.72rem", padding: "2px 8px", borderRadius: "6px", background: "var(--dt-success-bg)", color: "var(--dt-success)", border: "1px solid var(--dt-success-border)" }}>
                        Active ({vulkanStatus?.detected_platform?.toUpperCase() || "AUTO"} GPU Detected)
                      </span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--dt-text-muted)", marginTop: "4px" }}>
                      Auto-detects active hardware. Select your GPU engine to compile or download the backend driver.
                    </div>
                  </div>
                </div>

                {/* GPU Engine Cards */}
                <div className="gpu-engine-grid">
                  {[{
                    key: "nvidia", platform: "nvidia", target: "cuda",
                    title: "NVIDIA CUDA", desc: "CUDA / cu122 driver for RTX/GTX/T4 GPUs.",
                    engineCheck: vulkanStatus?.engines?.nvidia?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "cuda"),
                    labels: { installing: "Installing...", installed: "Installed & Active", activate: "Activate CUDA", setup: "Setup CUDA" },
                  }, {
                    key: "vulkan", platform: "vulkan", target: "vulkan",
                    title: "Intel / AMD Vulkan", desc: "Vulkan SPIR-V engine for Radeon & Arc GPUs.",
                    engineCheck: vulkanStatus?.engines?.vulkan?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "vulkan"),
                    labels: { installing: "Downloading...", installed: "Installed & Active", activate: "Activate Vulkan", setup: "Download Vulkan" },
                  }, {
                    key: "metal", platform: "apple", target: "metal",
                    title: "Apple Metal", desc: "Metal MPS backend for M1/M2/M3/M4 Macs.",
                    engineCheck: vulkanStatus?.engines?.apple?.installed || (vulkanStatus?.progress?.status === "completed" && vulkanStatus?.active_target === "metal"),
                    labels: { installing: "Compiling...", installed: "Installed & Active", activate: "Activate Metal", setup: "Setup Metal" },
                  }].map((engine) => {
                    const isDetected = vulkanStatus?.detected_platform === engine.platform;
                    const isInstalled = engine.engineCheck;
                    const isUpdating = vulkanStatus?.progress?.status === "updating";
                    return (
                      <div key={engine.key} className={`gpu-engine-card ${engine.key} ${isDetected ? "detected" : ""}`}>
                        <div className="gpu-engine-card-header">
                          <span className={`gpu-engine-card-title ${engine.key}`}>
                            <Monitor size={14} /> {engine.title}
                          </span>
                          {isDetected && (
                            <span className={`gpu-detected-badge ${engine.key}`}>DETECTED</span>
                          )}
                        </div>
                        <div className="gpu-engine-card-desc">{engine.desc}</div>
                        {isInstalled ? (
                          <div style={{ display: "flex", gap: "6px" }}>
                            <div
                              className={`gpu-engine-btn installed ${engine.key}`}
                              style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "4px" }}
                            >
                              <Check size={12} /> Active
                            </div>
                            <button
                              className={`gpu-engine-btn update ${engine.key}`}
                              style={{ flex: 1 }}
                              onClick={() => handleSetupGpuEngine(engine.target, true)}
                              disabled={isUpdating}
                              title={`Update ${engine.title} to latest release`}
                            >
                              {isUpdating && vulkanStatus?.active_target === engine.target ? (
                                <><Loader size={12} className="spin" /> Updating</>
                              ) : (
                                <><RefreshCw size={12} /> Update</>
                              )}
                            </button>
                          </div>
                        ) : (
                          <button
                            className={`gpu-engine-btn ${engine.key} ${!isDetected ? "inactive" : ""}`}
                            onClick={() => handleSetupGpuEngine(engine.target)}
                            disabled={isUpdating}
                          >
                            {isUpdating && vulkanStatus?.active_target === engine.target
                              ? (<><Loader size={12} className="spin" /> {engine.labels.installing}</>)
                              : isDetected
                                ? (<><Zap size={12} /> {engine.labels.activate}</>)
                                : engine.labels.setup}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "14px", padding: "10px 14px", background: "rgba(255, 255, 255, 0.03)", borderRadius: "var(--dt-radius-md)", border: "1px solid var(--dt-border-subtle)", flexWrap: "wrap", gap: "10px" }}>
                  <div style={{ display: "flex", gap: "16px", fontSize: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
                    <div>
                      Installed Engine: <strong style={{ color: "var(--dt-success)" }}>{vulkanStatus?.installed_version || "Ready"}</strong>
                    </div>
                    <div>
                      Latest GitHub Release: <strong style={{ color: "#818cf8" }}>{vulkanStatus?.latest_version || "Latest"}</strong>
                    </div>
                    {vulkanStatus?.has_update && (
                      <span style={{ fontSize: "0.7rem", padding: "2px 8px", borderRadius: "12px", background: "var(--dt-warning-bg)", color: "var(--dt-warning)", border: "1px solid var(--dt-warning-border)", fontWeight: "600" }}>
                        Update Available
                      </span>
                    )}
                  </div>

                  <button
                    className="gpu-update-btn"
                    onClick={() => handleSetupGpuEngine(activeEngineTarget, true)}
                    disabled={vulkanStatus?.progress?.status === "updating"}
                    title={`Update installed ${activeEngineTitle} (${activeEngineTarget.toUpperCase()})`}
                  >
                    {vulkanStatus?.progress?.status === "updating" ? (
                      <><Loader size={13} className="spin" /> Updating...</>
                    ) : (
                      <><RefreshCw size={13} /> Update {activeEngineTarget.toUpperCase()} Engine</>
                    )}
                  </button>
                </div>

                {/* Progress bar when updating */}
                {vulkanStatus?.progress?.status === "updating" && (
                  <div style={{ marginTop: "12px", background: "rgba(0,0,0,0.5)", padding: "10px", borderRadius: "var(--dt-radius-md)", border: "1px solid rgba(99, 102, 241, 0.3)" }}>
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

              {/* Hardware Diagnostics */}
              <div className="diagnostics-panel">
                <div className="diagnostics-panel-header">
                  <div className="diagnostics-panel-title">
                    <HardDrive size={16} /> Hardware & VRAM Diagnostics
                  </div>
                  <button
                    onClick={runGpuDiagnostics}
                    disabled={loadingDiag}
                    className="diagnostics-refresh-btn"
                  >
                    {loadingDiag ? (<><Loader size={13} className="spin" /> Scanning...</>) : (<><RefreshCw size={13} /> Refresh GPU Diagnostics</>)}
                  </button>
                </div>

                {diagnostics ? (
                  <div className="diagnostics-grid">
                    <div className="diagnostics-card">
                      <div className="diagnostics-card-label">Detected GPU Adapter</div>
                      <div className="diagnostics-card-value">
                        <Gpu size={14} /> {diagnostics.gpu_name}
                      </div>
                    </div>
                    <div className="diagnostics-card">
                      <div className="diagnostics-card-label">VRAM / Shared GPU Memory</div>
                      <div className="diagnostics-card-value" style={{ color: "var(--dt-success)" }}>
                        <HardDrive size={14} /> {diagnostics.vram_free_gb} GB Free / {diagnostics.vram_total_gb} GB Total
                      </div>
                    </div>
                    <div className="diagnostics-summary">
                      <div className="diagnostics-summary-title">
                        {diagnostics.execution_target}
                      </div>
                      <div className="diagnostics-summary-detail">
                        <Check size={13} /> {diagnostics.offload_info}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="diagnostics-loading">
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
                  <span className="security-icon"><Shield size={16} /></span>
                  <span className="security-label">SAST Code Scanning</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon"><Lock size={16} /></span>
                  <span className="security-label">Sandbox Isolation</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon"><Globe size={16} /></span>
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
    </Modal>
  );
};

export default SettingsModal;
