import React, { useState, useEffect, useRef, useMemo } from "react";

/**
 * @component BenchmarkModal
 * State-of-the-art Human-Engineered AI OS Benchmark Studio & Performance Monitor.
 * Provides real-time parallel worker telemetry, accuracy scoring against published
 * AI model baselines (GPT-4o, Claude 3.5 Sonnet, Llama 3 70B), category pills, and live streaming logs.
 */
export default function BenchmarkModal({ open, setOpen, serverUrl }) {
  const [activeCategory, setActiveCategory] = useState("HumanEval");
  const [logFilter, setLogFilter] = useState("all"); // 'all', 'passed', 'failed', 'info'
  const [copiedLogs, setCopiedLogs] = useState(false);
  const [status, setStatus] = useState({
    active: false,
    category: null,
    progress: 0,
    total: 0,
    passed: 0,
    failed: 0,
    accuracy: 0.0,
    tokens_per_sec: 0.0,
    avg_latency: 0.0,
    elapsed_seconds: 0.0,
    workers: [],
    logs: [],
    history: {},
    comparison_baselines: {}
  });
  const [loading, setLoading] = useState(false);
  const consoleRef = useRef(null);

  // Poll benchmark status when modal is open
  useEffect(() => {
    if (!open) return;

    const fetchStatus = async () => {
      try {
        const res = await fetch(`${serverUrl}/api/benchmark/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch (err) {
        console.error("Failed to fetch benchmark status:", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [serverUrl, open]);

  // Auto-scroll log console internally without scrolling the parent modal window
  useEffect(() => {
    if (open && consoleRef.current && status.active) {
      const el = consoleRef.current;
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
      if (isNearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [status.logs, open, status.active, logFilter]);

  const categories = [
    { id: "HumanEval", label: "HumanEval", icon: "🐍", desc: "Python Function Synthesis" },
    { id: "MBPP", label: "MBPP", icon: "⚙️", desc: "Basic Python Programming" },
    { id: "GSM8K", label: "GSM8K", icon: "🧮", desc: "Grade Math Multi-Step" },
    { id: "MATH", label: "MATH", icon: "📐", desc: "Competition Mathematics" },
    { id: "GPQA (PhD Science)", label: "GPQA", icon: "🔬", desc: "PhD-Level Science" },
    { id: "AIME (Olympiad Logic)", label: "AIME", icon: "🏆", desc: "Olympiad Reasoning" },
    { id: "MuSR (PhD Logic)", label: "MuSR", icon: "🧠", desc: "Murder Mystery Logic" },
    { id: "MMLU-Pro (Prof STEM)", label: "MMLU-Pro", icon: "🏛️", desc: "Professional STEM" },
    { id: "SWE-bench Lite", label: "SWE-Lite", icon: "🛠️", desc: "Software Issue Fixes" },
    { id: "SWE-bench Pro", label: "SWE-Pro", icon: "⚡", desc: "Complex Architecture" },
    { id: "SearchQA / HotpotQA", label: "SearchQA", icon: "🔍", desc: "Multi-Hop Web Search" }
  ];

  const handleStart = async (cat = activeCategory) => {
    setLoading(true);
    try {
      await fetch(`${serverUrl}/api/benchmark/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: cat })
      });
    } catch (err) {
      console.error("Failed to start benchmark:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAll = async () => {
    setLoading(true);
    try {
      await fetch(`${serverUrl}/api/benchmark/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: "ALL" })
      });
    } catch (err) {
      console.error("Failed to start all benchmarks:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${serverUrl}/api/benchmark/stop`, { method: "POST" });
    } catch (err) {
      console.error("Failed to stop benchmark:", err);
    }
  };

  const handleCopyLogs = () => {
    if (!status.logs || status.logs.length === 0) return;
    navigator.clipboard.writeText(status.logs.join("\n"));
    setCopiedLogs(true);
    setTimeout(() => setCopiedLogs(false), 2000);
  };

  const handleExportReport = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      active_category: status.category || activeCategory,
      accuracy: status.accuracy,
      passed: status.passed,
      failed: status.failed,
      total: status.total,
      avg_latency_s: status.avg_latency,
      tokens_per_sec: status.tokens_per_sec,
      elapsed_seconds: status.elapsed_seconds,
      history: status.history || {},
      logs: status.logs || []
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deepthink_aios_benchmark_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredLogs = useMemo(() => {
    if (!status.logs) return [];
    if (logFilter === "passed") return status.logs.filter((l) => l.includes("✅") || l.includes("PASSED"));
    if (logFilter === "failed") return status.logs.filter((l) => l.includes("❌") || l.includes("failed") || l.includes("FAILED") || l.includes("Error"));
    if (logFilter === "info") return status.logs.filter((l) => !l.includes("✅") && !l.includes("❌"));
    return status.logs;
  }, [status.logs, logFilter]);

  const currentBaseline = (status?.comparison_baselines && status.comparison_baselines[status?.category || activeCategory]) || {
    gpt4: 90.0,
    claude35_sonnet: 92.0,
    llama3_70b: 86.0,
    deepthink_aios: 91.5
  };

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={() => setOpen(false)}>
      <div className="modal-content benchmark-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="benchmark-header">
          <div className="benchmark-header-title">
            <span className="benchmark-icon">⚡</span>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <h2>DeepThink AIOS Benchmark Studio</h2>
                {status.active && (
                  <span
                    style={{
                      background: "rgba(239, 68, 68, 0.15)",
                      border: "1px solid rgba(239, 68, 68, 0.4)",
                      color: "#f87171",
                      padding: "2px 8px",
                      borderRadius: "12px",
                      fontSize: "0.72rem",
                      fontWeight: "700",
                      display: "flex",
                      alignItems: "center",
                      gap: "5px"
                    }}
                  >
                    <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#ef4444", display: "inline-block", animation: "pulse 1.5s infinite" }}></span>
                    LIVE EVALUATION
                  </span>
                )}
              </div>
              <p className="benchmark-subtitle">
                Parallel Hardware Telemetry, Real-time Code Execution & Published Baselines
              </p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={() => setOpen(false)}>
            ✕
          </button>
        </div>

        {/* Category Selector Pills */}
        <div style={{ marginBottom: "18px" }}>
          <div style={{ fontSize: "0.8rem", color: "#94a3b8", fontWeight: "600", textTransform: "uppercase", marginBottom: "8px", letterSpacing: "0.04em" }}>
            Select Benchmark Suite:
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {categories.map((c) => {
              const isSelected = (status.category || activeCategory) === c.id;
              const hasHistory = status.history && status.history[c.id];
              return (
                <button
                  key={c.id}
                  onClick={() => {
                    if (!status.active) setActiveCategory(c.id);
                  }}
                  disabled={status.active}
                  title={c.desc}
                  style={{
                    background: isSelected
                      ? "linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(168, 85, 247, 0.25))"
                      : "rgba(255, 255, 255, 0.03)",
                    border: isSelected ? "1px solid #818cf8" : "1px solid rgba(255, 255, 255, 0.08)",
                    color: isSelected ? "#ffffff" : "#94a3b8",
                    padding: "7px 14px",
                    borderRadius: "20px",
                    fontSize: "0.82rem",
                    fontWeight: isSelected ? "600" : "500",
                    cursor: status.active ? "not-allowed" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    transition: "all 0.2s ease"
                  }}
                >
                  <span>{c.icon}</span>
                  <span>{c.label}</span>
                  {hasHistory && (
                    <span style={{ fontSize: "0.72rem", color: hasHistory.accuracy >= 75 ? "#34d399" : "#fbbf24", fontWeight: "700" }}>
                      ({hasHistory.accuracy}%)
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Action Controls */}
        <div className="benchmark-controls" style={{ marginTop: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "0.88rem", color: "#cbd5e1" }}>
              Active Suite: <strong style={{ color: "#818cf8" }}>{status.category || activeCategory}</strong>
            </span>
          </div>

          <div className="action-buttons" style={{ display: "flex", gap: "10px" }}>
            {!status.active ? (
              <>
                <button
                  className="btn-run-benchmark"
                  onClick={() => handleStart(activeCategory)}
                  disabled={loading}
                >
                  {loading ? "Starting..." : `▶ Run ${activeCategory.split(" ")[0]}`}
                </button>
                <button
                  className="btn-run-all-benchmark"
                  onClick={handleRunAll}
                  disabled={loading}
                  style={{
                    background: "linear-gradient(135deg, #a855f7, #ec4899)",
                    color: "#ffffff",
                    border: "none",
                    padding: "10px 18px",
                    borderRadius: "8px",
                    fontWeight: "600",
                    fontSize: "0.88rem",
                    cursor: "pointer",
                    boxShadow: "0 4px 12px rgba(168, 85, 247, 0.3)",
                    transition: "transform 0.2s ease, box-shadow 0.2s ease"
                  }}
                >
                  ⚡ Run All 11 Suites
                </button>
              </>
            ) : (
              <button className="btn-stop-benchmark" onClick={handleStop}>
                ⏹ Cancel Evaluation
              </button>
            )}

            <button
              onClick={handleExportReport}
              style={{
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                color: "#cbd5e1",
                padding: "8px 14px",
                borderRadius: "8px",
                fontSize: "0.82rem",
                cursor: "pointer"
              }}
              title="Export complete benchmark history & logs as JSON"
            >
              📥 Export Report
            </button>
          </div>
        </div>

        {/* KPI Metrics Dashboard */}
        <div className="benchmark-kpi-grid">
          <div className="kpi-card highlight">
            <div className="kpi-label">Pass Rate (Accuracy)</div>
            <div className="kpi-value">{status.accuracy}%</div>
            <div className="kpi-progress-bar">
              <div
                className="kpi-progress-fill"
                style={{ width: `${Math.min(100, status.accuracy)}%` }}
              ></div>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Passed / Total</div>
            <div className="kpi-value">
              {status.passed} <span className="kpi-sub">/ {status.total}</span>
            </div>
            <div className="kpi-subtext">Failed: {status.failed}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Avg Latency</div>
            <div className="kpi-value">{status.avg_latency}s</div>
            <div className="kpi-subtext">Elapsed: {status.elapsed_seconds}s</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Throughput</div>
            <div className="kpi-value">{status.tokens_per_sec}</div>
            <div className="kpi-subtext">tokens / sec</div>
          </div>
        </div>

        {/* Comparative Model Baselines */}
        <div className="benchmark-section">
          <h3>Published Model Comparison ({status.category || activeCategory})</h3>
          <div className="baseline-bars-grid">
            <div className="baseline-item">
              <div className="baseline-header">
                <span>GPT-4o</span>
                <span>{currentBaseline.gpt4}%</span>
              </div>
              <div className="baseline-track">
                <div
                  className="baseline-fill gpt4"
                  style={{ width: `${currentBaseline.gpt4}%` }}
                ></div>
              </div>
            </div>

            <div className="baseline-item">
              <div className="baseline-header">
                <span>Claude 3.5 Sonnet</span>
                <span>{currentBaseline.claude35_sonnet}%</span>
              </div>
              <div className="baseline-track">
                <div
                  className="baseline-fill claude"
                  style={{ width: `${currentBaseline.claude35_sonnet}%` }}
                ></div>
              </div>
            </div>

            <div className="baseline-item">
              <div className="baseline-header">
                <span>Llama 3 70B</span>
                <span>{currentBaseline.llama3_70b}%</span>
              </div>
              <div className="baseline-track">
                <div
                  className="baseline-fill llama"
                  style={{ width: `${currentBaseline.llama3_70b}%` }}
                ></div>
              </div>
            </div>

            <div className="baseline-item active-system">
              <div className="baseline-header">
                <span>⚡ Current AIOS Engine</span>
                <span style={{ color: "#34d399", fontWeight: "700" }}>{status.accuracy}%</span>
              </div>
              <div className="baseline-track">
                <div
                  className="baseline-fill aios"
                  style={{ width: `${Math.min(100, status.accuracy)}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Worker Telemetry Grid */}
        <div className="benchmark-section">
          <h3>Parallel Worker Nodes ({status.workers?.length || 0} active workers)</h3>
          <div className="worker-nodes-grid">
            {status.workers?.map((w) => (
              <div
                key={w.id}
                className={`worker-card ${w.status !== "Idle" ? "working" : ""}`}
              >
                <div className="worker-card-header">
                  <span className="worker-id">Worker #{w.id + 1}</span>
                  <span className="worker-status-tag">{w.status}</span>
                </div>
                <div className="worker-task-title">{w.task || "Idle"}</div>
                <div className="worker-progress-bar">
                  <div
                    className="worker-progress-fill"
                    style={{ width: `${w.progress}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Stored Benchmark History Table */}
        {status?.history && Object.keys(status.history).length > 0 && (
          <div className="benchmark-section">
            <h3>🏆 Benchmark Suite History ({Object.keys(status.history).length} Completed)</h3>
            <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "10px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)", marginBottom: "16px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", color: "#e2e8f0" }}>
                <thead>
                  <tr style={{ background: "rgba(255,255,255,0.05)", textTransform: "uppercase", fontSize: "0.74rem", color: "#94a3b8" }}>
                    <th style={{ padding: "10px 14px", textAlign: "left" }}>Category / Suite</th>
                    <th style={{ padding: "10px 14px", textAlign: "center" }}>Accuracy</th>
                    <th style={{ padding: "10px 14px", textAlign: "center" }}>Passed / Total</th>
                    <th style={{ padding: "10px 14px", textAlign: "center" }}>Duration</th>
                    <th style={{ padding: "10px 14px", textAlign: "right" }}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(status.history).map(([cat, res]) => (
                    <tr key={cat} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                      <td style={{ padding: "10px 14px", fontWeight: "600", color: "#cbd5e1" }}>{cat}</td>
                      <td style={{ padding: "10px 14px", textAlign: "center", color: res.accuracy >= 75 ? "#34d399" : res.accuracy >= 50 ? "#fbbf24" : "#f87171", fontWeight: "700" }}>
                        {res.accuracy}%
                      </td>
                      <td style={{ padding: "10px 14px", textAlign: "center" }}>{res.passed} / {res.total}</td>
                      <td style={{ padding: "10px 14px", textAlign: "center" }}>{res.elapsed_seconds}s</td>
                      <td style={{ padding: "10px 14px", textAlign: "right", color: "#94a3b8", fontSize: "0.76rem" }}>{res.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Live Execution Console Log with Filter Bar */}
        <div className="benchmark-section">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h3>Live Telemetry & Execution Log</h3>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ display: "flex", background: "rgba(255,255,255,0.04)", borderRadius: "6px", padding: "2px", border: "1px solid rgba(255,255,255,0.08)" }}>
                {["all", "passed", "failed", "info"].map((f) => (
                  <button
                    key={f}
                    onClick={() => setLogFilter(f)}
                    style={{
                      background: logFilter === f ? "#6366f1" : "transparent",
                      color: logFilter === f ? "#fff" : "#94a3b8",
                      border: "none",
                      padding: "3px 8px",
                      borderRadius: "4px",
                      fontSize: "0.74rem",
                      cursor: "pointer",
                      textTransform: "capitalize",
                      fontWeight: logFilter === f ? "600" : "400"
                    }}
                  >
                    {f}
                  </button>
                ))}
              </div>

              <button
                onClick={handleCopyLogs}
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: "#cbd5e1",
                  padding: "4px 8px",
                  borderRadius: "6px",
                  fontSize: "0.74rem",
                  cursor: "pointer"
                }}
              >
                {copiedLogs ? "✅ Copied" : "📋 Copy Logs"}
              </button>
            </div>
          </div>

          <div className="benchmark-console" ref={consoleRef}>
            {filteredLogs.length === 0 ? (
              <div className="console-empty">
                Console ready. Select a benchmark suite above and click "Run Suite" or "Run All 11 Suites".
              </div>
            ) : (
              filteredLogs.map((log, index) => {
                const isSuccess = log.includes("✅") || log.includes("PASSED");
                const isFail = log.includes("❌") || log.includes("failed") || log.includes("FAILED");
                return (
                  <div
                    key={index}
                    className="console-line"
                    style={{
                      color: isSuccess ? "#34d399" : isFail ? "#f87171" : "#38bdf8"
                    }}
                  >
                    {log}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

