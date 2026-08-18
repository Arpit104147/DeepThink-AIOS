import React, { useState, useEffect } from "react";

const ModelHubModal = ({ open, setOpen, serverUrl }) => {
  const [activeTab, setActiveTab] = useState("explorer");
  const [modelsStatus, setModelsStatus] = useState({});
  const [roles, setRoles] = useState({
    router: "router",
    coding: "ornith",
    reasoning: "deepseek_r1",
    linter: "vibethinker",
    vision: "qwen_vl"
  });

  // LM Studio Explorer States
  const [searchQuery, setSearchQuery] = useState("glm 4.7 flash");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [repoDetails, setRepoDetails] = useState(null);
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [activeDownloadKey, setActiveDownloadKey] = useState(null);
  const [downloadBannerMsg, setDownloadBannerMsg] = useState("");
  const [systemInfo, setSystemInfo] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${serverUrl}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setModelsStatus(data.models || {});
        if (data.system) {
          setSystemInfo(data.system);
        }
      }
    } catch (err) {
      console.error("Failed to fetch model status:", err);
    }
  };

  const calculateVramFit = (fileSizeGb) => {
    if (!fileSizeGb || isNaN(fileSizeGb)) {
      return {
        status: "unknown",
        badgeText: "Size Unknown",
        badgeStyle: { background: "rgba(148, 163, 184, 0.15)", border: "1px solid rgba(148, 163, 184, 0.4)", color: "#94a3b8", padding: "6px 14px", borderRadius: "6px", fontWeight: "600", fontSize: "0.82rem" },
        description: "Model size data unavailable."
      };
    }

    const size = parseFloat(fileSizeGb);
    const overhead = 1.5; // Context & KV cache overhead
    const totalRequired = size + overhead;

    const vramTotal = systemInfo?.vram_total_gb ? parseFloat(systemInfo.vram_total_gb) : 0;
    const ramTotal = systemInfo?.ram_total_gb ? parseFloat(systemInfo.ram_total_gb) : (systemInfo?.ram_total ? parseFloat(systemInfo.ram_total) : 16);

    if (vramTotal > 0 && totalRequired <= vramTotal) {
      return {
        status: "vram_safe",
        badgeText: "🟢 Likely Fit (VRAM Safe)",
        badgeStyle: {
          background: "rgba(52, 211, 153, 0.15)",
          border: "1px solid rgba(52, 211, 153, 0.5)",
          color: "#34d399",
          padding: "6px 14px",
          borderRadius: "6px",
          fontWeight: "600",
          fontSize: "0.82rem",
          display: "inline-flex",
          alignItems: "center"
        },
        description: `Fits entirely in GPU VRAM (${vramTotal.toFixed(1)} GB available). Full GPU acceleration.`
      };
    }

    const totalMem = vramTotal + ramTotal;
    if (totalRequired <= totalMem) {
      return {
        status: "ram_fit",
        badgeText: "🟡 Partial Offload (System RAM Fit)",
        badgeStyle: {
          background: "rgba(251, 191, 36, 0.15)",
          border: "1px solid rgba(251, 191, 36, 0.5)",
          color: "#fbbf24",
          padding: "6px 14px",
          borderRadius: "6px",
          fontWeight: "600",
          fontSize: "0.82rem",
          display: "inline-flex",
          alignItems: "center"
        },
        description: vramTotal > 0 
          ? `Exceeds VRAM (${vramTotal.toFixed(1)} GB), but fits in System RAM (${ramTotal.toFixed(1)} GB). CPU offloading active.`
          : `Fits in System RAM (${ramTotal.toFixed(1)} GB). CPU inference.`
      };
    }

    return {
      status: "oom_risk",
      badgeText: "🔴 Exceeds Memory (OOM Risk)",
      badgeStyle: {
        background: "rgba(248, 113, 113, 0.15)",
        border: "1px solid rgba(248, 113, 113, 0.5)",
        color: "#f87171",
        padding: "6px 14px",
        borderRadius: "6px",
        fontWeight: "600",
        fontSize: "0.82rem",
        display: "inline-flex",
        alignItems: "center"
      },
      description: `Requires ~${totalRequired.toFixed(1)} GB (model + context overhead), exceeding system capacity (${totalMem.toFixed(1)} GB). Risk of OOM.`
    };
  };

  const fetchRoles = async () => {
    try {
      const res = await fetch(`${serverUrl}/api/models/roles`);
      if (res.ok) {
        const data = await res.json();
        if (data.roles) setRoles(data.roles);
      }
    } catch (err) {
      console.error("Failed to fetch model roles:", err);
    }
  };

  const handleSearch = async (queryToSearch) => {
    const q = queryToSearch !== undefined ? queryToSearch : searchQuery;
    setLoadingSearch(true);
    setActionMessage("");
    try {
      const res = await fetch(`${serverUrl}/api/models/search?q=${encodeURIComponent(q)}&limit=20`);
      if (res.ok) {
        const data = await res.json();
        const models = data.models || [];
        setSearchResults(models);
        if (models.length > 0) {
          handleSelectRepo(models[0]);
        } else {
          setSelectedRepo(null);
          setRepoDetails(null);
        }
      }
    } catch (e) {
      setActionMessage(`Search failed: ${e.message}`);
    } finally {
      setLoadingSearch(false);
    }
  };

  const handleSelectRepo = async (repo) => {
    if (!repo) return;
    const targetRepoId = repo.id || repo.model_id || repo.repo_id;
    if (!targetRepoId) return;

    setSelectedRepo(repo);
    setLoadingDetails(true);
    setSelectedFileIdx(0);
    setActionMessage("");
    try {
      const res = await fetch(`${serverUrl}/api/models/repo_details?repo_id=${encodeURIComponent(targetRepoId)}`);
      if (res.ok) {
        const data = await res.json();
        setRepoDetails(data);
      }
    } catch (e) {
      setActionMessage(`Failed to fetch repo details: ${e.message}`);
    } finally {
      setLoadingDetails(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchStatus();
      fetchRoles();
      if (searchResults.length === 0) {
        handleSearch("glm 4.7 flash");
      }
      const interval = setInterval(fetchStatus, 3000);
      return () => clearInterval(interval);
    }
  }, [open, serverUrl]);

  // Auto-clear download banner after 4 seconds
  useEffect(() => {
    if (downloadBannerMsg) {
      const timer = setTimeout(() => setDownloadBannerMsg(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [downloadBannerMsg]);

  if (!open) return null;

  const handleRoleChange = (roleKey, modelKey) => {
    setRoles((prev) => ({ ...prev, [roleKey]: modelKey }));
  };

  const handleSaveRoles = async () => {
    try {
      const res = await fetch(`${serverUrl}/api/models/roles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(roles)
      });
      if (res.ok) {
        setActionMessage("✅ Swarm role mappings saved successfully!");
        fetchStatus();
      }
    } catch (e) {
      setActionMessage(`Role save error: ${e.message}`);
    }
  };

  const filesList = repoDetails?.files || repoDetails?.gguf_files || [];
  const currentFile = filesList[selectedFileIdx];
  const currentKeyClean = currentFile ? currentFile.filename.replace(".gguf", "").replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase() : null;
  const currentModelStatus = currentKeyClean ? modelsStatus[currentKeyClean] : null;

  const handleDownloadSelectedQuant = async () => {
    if (!currentFile || !selectedRepo) return;
    const targetRepoId = selectedRepo.id || selectedRepo.model_id || selectedRepo.repo_id;

    setActionMessage(`🚀 Starting download for ${currentFile.filename}...`);
    setDownloadBannerMsg(`🚀 Started downloading ${currentFile.filename} (${currentFile.size_gb} GB)...`);

    try {
      const res = await fetch(`${serverUrl}/api/models/download_hf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_id: targetRepoId,
          filename: currentFile.filename,
          name: selectedRepo.model_name || targetRepoId.split("/").pop()
        })
      });
      if (res.ok) {
        const data = await res.json();
        setActiveDownloadKey(data.key);
        fetchStatus();
      } else {
        const errData = await res.json();
        setActionMessage(`Download failed: ${errData.detail || "Server error"}`);
      }
    } catch (e) {
      setActionMessage(`Download request error: ${e.message}`);
    }
  };

  const handleCancelDownload = async (modelKey) => {
    try {
      const res = await fetch(`${serverUrl}/api/models/cancel_download/${modelKey}`, {
        method: "POST"
      });
      if (res.ok) {
        setActionMessage(`⏹️ Cancelled download for ${modelKey}`);
        setDownloadBannerMsg("");
        fetchStatus();
      }
    } catch (e) {
      setActionMessage(`Cancel error: ${e.message}`);
    }
  };

  const handleDeleteModelFile = async (modelKey) => {
    if (!window.confirm(`Are you sure you want to delete the downloaded GGUF file(s) for "${modelKey}" from disk?`)) {
      return;
    }
    try {
      const res = await fetch(`${serverUrl}/api/models/delete/${modelKey}`, {
        method: "DELETE"
      });
      if (res.ok) {
        const data = await res.json();
        setActionMessage(`🗑️ Deleted model files for ${modelKey} (freed ${data.freed_mb || 0} MB)`);
        fetchStatus();
      } else {
        const d = await res.json();
        setActionMessage(`Cannot delete: ${d.detail || "Error"}`);
      }
    } catch (e) {
      setActionMessage(`Delete error: ${e.message}`);
    }
  };

  const handleDeleteCustomModel = async (modelKey) => {
    try {
      const res = await fetch(`${serverUrl}/api/models/custom/${modelKey}`, {
        method: "DELETE"
      });
      if (res.ok) {
        setActionMessage(`🗑️ Removed custom model ${modelKey}`);
        fetchStatus();
      } else {
        const d = await res.json();
        setActionMessage(`Cannot delete: ${d.detail || "Error"}`);
      }
    } catch (e) {
      setActionMessage(`Delete error: ${e.message}`);
    }
  };

  const currentRepoId = selectedRepo ? (selectedRepo.id || selectedRepo.model_id || selectedRepo.repo_id) : "";

  return (
    <div className="modal-overlay" onClick={() => setOpen(false)}>
      <div className="modal hub-modal-wrapper" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="hub-header">
          <div className="hub-title-group">
            <span className="hub-title-icon">🤖</span>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <h2 className="hub-title">Model Hub & Discovery</h2>
                <span className="hub-subtitle-badge">AIOS Engine</span>
              </div>
              <p className="hub-description">Search HuggingFace GGUF repository cards, choose quantization variants, and monitor live download progress.</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={() => setOpen(false)}>✕</button>
        </div>

        {/* Global Download Banner Notification */}
        {downloadBannerMsg && (
          <div style={{ background: "rgba(99, 102, 241, 0.2)", border: "1px solid rgba(99, 102, 241, 0.4)", borderRadius: "8px", padding: "8px 12px", marginBottom: "10px", fontSize: "0.8rem", color: "#a5b4fc", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{downloadBannerMsg}</span>
            <button onClick={() => setDownloadBannerMsg("")} style={{ background: "none", border: "none", color: "#a5b4fc", cursor: "pointer" }}>✕</button>
          </div>
        )}

        {/* Navigation Tab Bar */}
        <div className="hub-nav-tabs">
          <button className={`hub-tab-btn ${activeTab === "explorer" ? "active" : ""}`} onClick={() => setActiveTab("explorer")}>
            🔍 HuggingFace Model Explorer
          </button>
          <button className={`hub-tab-btn ${activeTab === "swarm" ? "active" : ""}`} onClick={() => setActiveTab("swarm")}>
            🎯 Swarm Role Assignment
          </button>
          <button className={`hub-tab-btn ${activeTab === "models" ? "active" : ""}`} onClick={() => setActiveTab("models")}>
            🍱 Installed Library
          </button>
        </div>

        {/* Tab Content Container */}
        <div className="hub-tab-content" style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
          {/* TAB 1: HuggingFace Explorer */}
          {activeTab === "explorer" && (
            <div className="lmstudio-explorer">
              {/* Sidebar Search List */}
              <div className="lmstudio-sidebar">
                <form onSubmit={(e) => { e.preventDefault(); handleSearch(); }} className="lmstudio-search-bar">
                  <span className="search-icon">🔍</span>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search HuggingFace models (e.g. qwen 2.5 vl, deepseek, phi)..."
                    className="lmstudio-search-input"
                  />
                  {searchQuery && (
                    <button type="button" onClick={() => { setSearchQuery(""); handleSearch(""); }} className="clear-search-btn">✕</button>
                  )}
                </form>

                {/* Quick Search Preset Tags */}
                <div style={{ display: "flex", gap: "6px", overflowX: "auto", padding: "4px 0 8px 0" }}>
                  {[
                    { label: "👁️ Vision (Qwen-VL)", query: "qwen 2.5 vl gguf" },
                    { label: "🧠 Reasoning (R1)", query: "deepseek r1 gguf" },
                    { label: "💻 Coding (Ornith)", query: "ornith 1.0 gguf" },
                    { label: "⚡ Router (Phi-3.5)", query: "phi-3.5 gguf" },
                  ].map((chip) => (
                    <button
                      key={chip.label}
                      type="button"
                      onClick={() => { setSearchQuery(chip.query); handleSearch(chip.query); }}
                      style={{
                        padding: "3px 8px",
                        fontSize: "0.72rem",
                        borderRadius: "12px",
                        background: searchQuery.includes(chip.query.split(" ")[0]) ? "rgba(168, 85, 247, 0.3)" : "rgba(255, 255, 255, 0.06)",
                        border: "1px solid rgba(255, 255, 255, 0.12)",
                        color: "#f8fafc",
                        cursor: "pointer",
                        whiteSpace: "nowrap"
                      }}
                    >
                      {chip.label}
                    </button>
                  ))}
                </div>

                <div className="lmstudio-repo-list">
                  {loadingSearch ? (
                    <div style={{ padding: "20px", textAlign: "center", color: "#94a3b8", fontSize: "0.85rem" }}>
                      Searching HuggingFace Hub...
                    </div>
                  ) : searchResults.length === 0 ? (
                    <div style={{ padding: "20px", textAlign: "center", color: "#94a3b8", fontSize: "0.85rem" }}>
                      No GGUF model repositories found.
                    </div>
                  ) : (
                    searchResults.map((m) => {
                      const itemRepoId = m.id || m.model_id || m.repo_id;
                      const isSelected = currentRepoId && itemRepoId && currentRepoId === itemRepoId;
                      const isVisionModel = itemRepoId.toLowerCase().includes("vl") || itemRepoId.toLowerCase().includes("vision") || itemRepoId.toLowerCase().includes("llava") || itemRepoId.toLowerCase().includes("smolvlm");
                      return (
                        <div
                          key={itemRepoId}
                          className={`lmstudio-repo-card ${isSelected ? "active" : ""}`}
                          onClick={() => handleSelectRepo(m)}
                        >
                          <div className="card-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                              <span>🤗</span> <span>{m.model_name || itemRepoId.split("/").pop()}</span>
                            </div>
                            {isVisionModel && (
                              <span style={{ fontSize: "0.65rem", background: "rgba(168,85,247,0.25)", border: "1px solid rgba(168,85,247,0.4)", color: "#c084fc", padding: "1px 5px", borderRadius: "4px" }}>
                                👁️ VL Model
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: "0.75rem", opacity: 0.75, marginTop: "2px" }}>{m.author}</div>
                          <div className="card-meta">
                            <span>❤️ {m.likes}</span>
                            <span>⬇️ {m.downloads?.toLocaleString() || 0}</span>
                            <span>recently</span>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Main Model Inspector View */}
              <div className="lmstudio-main-panel">
                {selectedRepo ? (
                  <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: "12px" }}>
                    {/* Header Details */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <h3 style={{ fontSize: "1.1rem", fontWeight: "700", color: "#f8fafc" }}>{currentRepoId}</h3>
                        <div style={{ display: "flex", gap: "12px", fontSize: "0.78rem", color: "#94a3b8", marginTop: "4px" }}>
                          <span>⬇️ {selectedRepo.downloads?.toLocaleString() || repoDetails?.downloads?.toLocaleString() || 0} downloads</span>
                          <span>⭐ {selectedRepo.likes || repoDetails?.likes || 0} likes</span>
                          <span style={{ color: "#a855f7" }}>FORMAT: GGUF</span>
                        </div>
                        {currentKeyClean === "qwen_vl" && (
                          <div style={{ marginTop: "6px", fontSize: "0.75rem", padding: "4px 10px", borderRadius: "6px", background: "rgba(168, 85, 247, 0.15)", border: "1px solid rgba(168, 85, 247, 0.35)", color: "#c084fc", display: "inline-flex", alignItems: "center", gap: "6px" }}>
                            <span>👁️ Multimodal Vision Model</span>
                            <span style={{ color: "#e9d5ff" }}>— Automatically downloads 2 files (Main GGUF + mmproj-BF16.gguf Projector)</span>
                          </div>
                        )}
                      </div>
                      <a
                        href={`https://huggingface.co/${currentRepoId}`}
                        target="_blank"
                        rel="noreferrer"
                        className="hub-save-btn"
                        style={{ padding: "6px 12px", fontSize: "0.78rem", background: "rgba(255,255,255,0.06)", textDecoration: "none" }}
                      >
                        🤗 Model Card ↗
                      </a>
                    </div>

                    {/* Quantization Selector & Download CTA */}
                    <div style={{ background: "rgba(0,0,0,0.3)", padding: "14px", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ fontWeight: "600", fontSize: "0.88rem", color: "#cbd5e1", marginBottom: "8px" }}>
                        Download Options
                      </div>

                      {loadingDetails ? (
                        <div style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Loading available GGUF quantizations...</div>
                      ) : filesList.length > 0 ? (
                        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                          <select
                            value={selectedFileIdx}
                            onChange={(e) => setSelectedFileIdx(Number(e.target.value))}
                            style={{
                              background: "rgba(0,0,0,0.5)",
                              border: "1px solid rgba(255,255,255,0.12)",
                              color: "#f8fafc",
                              padding: "10px",
                              borderRadius: "8px",
                              fontSize: "0.85rem"
                            }}
                          >
                            {filesList.map((f, idx) => {
                              const fFit = calculateVramFit(f.size_gb);
                              const icon = fFit.status === "vram_safe" ? "🟢" : fFit.status === "ram_fit" ? "🟡" : "🔴";
                              const label = fFit.status === "vram_safe" ? "VRAM Safe" : fFit.status === "ram_fit" ? "RAM Fit" : "OOM Risk";
                              return (
                                <option key={f.filename} value={idx}>
                                  {icon} GGUF {selectedRepo.model_name || currentRepoId.split("/").pop()} ({f.quant}) — {f.size_gb} GB ({label})
                                </option>
                              );
                            })}
                          </select>

                          {/* Live Download Progress Bar or Dynamic Fit Evaluation CTA */}
                          {!currentModelStatus?.downloaded && currentModelStatus?.progress && currentModelStatus.progress.status === "downloading" ? (
                            <div style={{ background: "rgba(0,0,0,0.4)", padding: "14px", borderRadius: "12px", border: "1px solid rgba(99, 102, 241, 0.3)" }}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                                <div style={{ fontSize: "0.85rem", fontWeight: "600", color: "#818cf8" }}>
                                  🚀 Downloading {currentKeyClean === "qwen_vl" ? "Qwen 2.5-VL + mmproj Vision Projector" : currentFile?.quant}... {currentModelStatus.progress.percent}%
                                </div>
                                <button
                                  onClick={() => handleCancelDownload(currentKeyClean)}
                                  style={{
                                    padding: "4px 10px",
                                    borderRadius: "6px",
                                    background: "rgba(239, 68, 68, 0.2)",
                                    border: "1px solid rgba(239, 68, 68, 0.4)",
                                    color: "#ef4444",
                                    cursor: "pointer",
                                    fontSize: "0.78rem",
                                    fontWeight: "600"
                                  }}
                                >
                                  ⏹️ Cancel Download
                                </button>
                              </div>

                              <div style={{ width: "100%", height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden", marginBottom: "6px" }}>
                                <div
                                  style={{
                                    height: "100%",
                                    width: `${currentModelStatus.progress.percent}%`,
                                    background: "linear-gradient(90deg, #6366f1, #a855f7)",
                                    transition: "width 0.3s ease"
                                  }}
                                />
                              </div>

                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.76rem", color: "#94a3b8" }}>
                                <span>Downloaded: {currentModelStatus.progress.downloaded_gb} GB</span>
                                <span>Total: {currentFile?.size_gb || currentModelStatus.progress.total_gb || "?"} GB</span>
                              </div>
                            </div>
                          ) : (() => {
                            const currentFit = calculateVramFit(currentFile?.size_gb);
                            return (
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
                                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                                  <span style={currentFit.badgeStyle}>
                                    {currentFit.badgeText}
                                  </span>
                                  <span style={{ fontSize: "0.74rem", color: "#94a3b8", maxWidth: "340px" }}>
                                    {currentFit.description}
                                  </span>
                                </div>

                                {currentModelStatus?.downloaded ? (
                                  <div style={{ background: "rgba(52, 211, 153, 0.15)", border: "1px solid #34d399", color: "#34d399", padding: "8px 18px", borderRadius: "8px", fontWeight: "600", fontSize: "0.88rem" }}>
                                    ✅ Downloaded & Ready in Library ({currentFile?.size_gb || currentModelStatus.size} GB)
                                  </div>
                                ) : (
                                  <button
                                    className="hub-save-btn"
                                    onClick={handleDownloadSelectedQuant}
                                    style={{
                                      background: currentFit.status === "oom_risk" ? "linear-gradient(135deg, #b91c1c, #dc2626)" : "linear-gradient(135deg, #4f46e5, #6366f1)",
                                      padding: "10px 24px",
                                      fontSize: "0.92rem"
                                    }}
                                  >
                                    {currentFit.status === "oom_risk" ? "⚠️ Download Anyway" : "📥 Download"} {currentFile ? `${currentFile.size_gb} GB` : ""}
                                  </button>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                      ) : (
                        <div style={{ color: "#fbbf24", fontSize: "0.85rem" }}>
                          No .gguf files found in this repository.
                        </div>
                      )}
                    </div>

                    {/* README Preview */}
                    <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
                      <div style={{ fontWeight: "600", fontSize: "0.88rem", color: "#cbd5e1", marginBottom: "8px" }}>
                        README / Model Documentation
                      </div>
                      <div style={{ flex: 1, background: "rgba(0,0,0,0.5)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", padding: "12px", overflowY: "auto", fontFamily: "monospace", fontSize: "0.78rem", color: "#94a3b8" }}>
                        {repoDetails?.readme || "No README available for this model."}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#94a3b8" }}>
                    Select a model from the search list to inspect quants and download.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: Swarm Intelligence Role Mapping */}
          {activeTab === "swarm" && (
            <div className="tab-panel">
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "14px" }}>
                Assign which local LLM handles each swarm intelligence capability role:
              </p>

              <div style={{ display: "grid", gap: "12px", maxHeight: "380px", overflowY: "auto" }}>
                {[
                  { key: "router", label: "Smart Router LLM", icon: "⚡", desc: "Intent classification & role dispatcher" },
                  { key: "coding", label: "Primary Coding LLM", icon: "💻", desc: "Full code generation & refactoring" },
                  { key: "reasoning", label: "Reasoning Engine LLM", icon: "🧠", desc: "Deep chain-of-thought logic & math synthesis" },
                  { key: "linter", label: "Syntax Linter / Patch LLM", icon: "🧬", desc: "Fast AST-aware code search/replace patcher" },
                  { key: "vision", label: "Vision / OCR Parsing Model", icon: "👁️", desc: "Transcribes uploaded image content & code screenshots" },
                ].map((role) => (
                  <div key={role.key} style={{ padding: "14px", background: "rgba(255,255,255,0.03)", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span>{role.icon}</span>
                        <div>
                          <div style={{ fontWeight: "600", fontSize: "0.9rem", color: "#f8fafc" }}>{role.label}</div>
                          <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>{role.desc}</div>
                        </div>
                      </div>
                      <span style={{ fontSize: "0.75rem", padding: "2px 8px", borderRadius: "4px", background: modelsStatus[roles[role.key]]?.downloaded ? "rgba(52, 211, 153, 0.2)" : "rgba(251, 191, 36, 0.2)", color: modelsStatus[roles[role.key]]?.downloaded ? "#34d399" : "#fbbf24" }}>
                        {modelsStatus[roles[role.key]]?.downloaded ? "● Ready" : "● Not Downloaded"}
                      </span>
                    </div>

                    <select
                      value={roles[role.key] || ""}
                      onChange={(e) => handleRoleChange(role.key, e.target.value)}
                      style={{ width: "100%", padding: "8px 12px", background: "rgba(0,0,0,0.4)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "6px", color: "#f8fafc", fontSize: "0.82rem" }}
                    >
                      {Object.entries(modelsStatus).map(([mKey, mInfo]) => (
                        <option key={mKey} value={mKey}>
                          {mInfo.name} — {mInfo.downloaded ? "✅ (Downloaded)" : "⌛ (Not Downloaded)"}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: "16px", display: "flex", justifyContent: "flex-end" }}>
                <button className="hub-save-btn" onClick={handleSaveRoles}>
                  💾 Save Swarm Role Mapping
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: Installed Local Models */}
          {activeTab === "models" && (
            <div className="tab-panel">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                <span style={{ fontSize: "0.82rem", color: "#94a3b8" }}>
                  Installed GGUF model library on local system ({Object.values(modelsStatus).filter(m => m.downloaded).length} ready).
                </span>
                <button onClick={fetchStatus} style={{ padding: "6px 12px", borderRadius: "8px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", color: "#fff", cursor: "pointer", fontSize: "0.78rem" }}>🔄 Refresh</button>
              </div>

              {Object.keys(modelsStatus).length === 0 ? (
                <div style={{ padding: "40px", textAlign: "center", color: "#94a3b8", fontSize: "0.88rem" }}>
                  Searching for local GGUF models... Click <strong>Refresh</strong> if models are loaded in Kaggle/disk.
                </div>
              ) : (
                <div style={{ display: "grid", gap: "10px", maxHeight: "380px", overflowY: "auto" }}>
                  {Object.entries(modelsStatus).map(([key, info]) => (
                    <div key={key} style={{ padding: "14px", background: "rgba(255, 255, 255, 0.025)", borderRadius: "12px", border: "1px solid rgba(255, 255, 255, 0.07)", display: "flex", flexDirection: "column", gap: "8px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div>
                          <div style={{ fontWeight: "600", fontSize: "0.95rem", color: "#f8fafc", display: "flex", alignItems: "center", gap: "8px" }}>
                            {info.name || key}
                            {info.is_custom && (
                              <span style={{ fontSize: "0.68rem", background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.4)", color: "#818cf8", padding: "2px 6px", borderRadius: "4px" }}>
                                Custom
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: "0.78rem", color: "#94a3b8", marginTop: "3px" }}>
                            Repo: <code style={{ background: "rgba(0,0,0,0.4)", padding: "2px 6px", borderRadius: "4px" }}>{info.repo_id}</code> | File: <code style={{ background: "rgba(0,0,0,0.4)", padding: "2px 6px", borderRadius: "4px" }}>{info.filename}</code>
                          </div>
                        </div>

                        <div style={{ fontSize: "0.78rem", display: "flex", alignItems: "center", gap: "8px" }}>
                          {info.downloaded ? (
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <span style={{ color: "#34d399", fontWeight: "600" }}>✅ Downloaded {info.size ? `(${info.size})` : ""}</span>
                              <button
                                onClick={() => handleDeleteModelFile(key)}
                                style={{ padding: "4px 9px", borderRadius: "5px", background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.35)", color: "#f87171", cursor: "pointer", fontSize: "0.74rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "4px", transition: "background 0.2s" }}
                                title="Delete downloaded GGUF model files from disk to free storage"
                              >
                                🗑️ Delete File
                              </button>
                            </div>
                          ) : info.progress && info.progress.status === "downloading" ? (
                            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                              <span style={{ color: "#818cf8", fontWeight: "600" }}>🚀 Downloading {info.progress.percent}%</span>
                              <button
                                onClick={() => handleCancelDownload(key)}
                                style={{ padding: "3px 8px", borderRadius: "4px", background: "rgba(239,68,68,0.2)", border: "1px solid rgba(239,68,68,0.4)", color: "#ef4444", cursor: "pointer", fontSize: "0.72rem" }}
                              >
                                ⏹️ Cancel
                              </button>
                            </div>
                          ) : (
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <span style={{ color: "#fbbf24", fontWeight: "600" }}>⏳ Not Downloaded</span>
                              {info.is_custom && (
                                <button
                                  onClick={() => handleDeleteCustomModel(key)}
                                  style={{ padding: "3px 8px", borderRadius: "4px", background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171", cursor: "pointer", fontSize: "0.72rem" }}
                                  title="Remove custom model definition"
                                >
                                  🗑️ Remove
                                </button>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Vision Projector mmproj Status Banner */}
                      {info.is_vision && (
                        <div style={{ fontSize: "0.75rem", padding: "6px 10px", borderRadius: "6px", background: info.has_mmproj ? "rgba(52, 211, 153, 0.08)" : "rgba(251, 191, 36, 0.1)", border: info.has_mmproj ? "1px solid rgba(52, 211, 153, 0.25)" : "1px solid rgba(251, 191, 36, 0.3)", color: info.has_mmproj ? "#34d399" : "#fbbf24", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                          <span>👁️ Vision Projector (mmproj): <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 5px", borderRadius: "4px" }}>{info.mmproj_filename || "mmproj-BF16.gguf"}</code></span>
                          <span>{info.has_mmproj ? "✅ Projector Ready" : "⚠️ Projector Missing (Vision requires mmproj)"}</span>
                        </div>
                      )}

                      {/* Progress Bar in Local Models list (ONLY when downloading) */}
                      {!info.downloaded && info.progress && info.progress.status === "downloading" && (
                        <div style={{ width: "100%", background: "rgba(0,0,0,0.3)", padding: "8px 12px", borderRadius: "8px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#818cf8", marginBottom: "4px" }}>
                            <span>Downloading: {info.progress.downloaded_gb} GB</span>
                            <span>{info.progress.percent}%</span>
                          </div>
                          <div style={{ width: "100%", height: "6px", background: "rgba(255,255,255,0.1)", borderRadius: "3px", overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${info.progress.percent}%`, background: "linear-gradient(90deg, #6366f1, #a855f7)", transition: "width 0.3s ease" }} />
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer Actions */}
        <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px solid rgba(255,255,255,0.08)", display: "flex", justifyContent: "flex-end", flexShrink: 0 }}>
          <button onClick={() => setOpen(false)} style={{ padding: "8px 22px", borderRadius: "8px", background: "rgba(255,255,255,0.08)", color: "#cbd5e1", border: "none", cursor: "pointer", fontWeight: "600" }}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelHubModal;
