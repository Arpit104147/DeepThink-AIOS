import React, { useState, useEffect, useRef } from "react";

/**
 * @component BenchmarkModal
 * State-of-the-art Human-Engineered AI OS Benchmark Studio & Performance Monitor.
 * Provides real-time parallel worker telemetry, accuracy scoring against published
 * AI model baselines (GPT-4o, Claude 3.5 Sonnet, Llama 3 70B), and live streaming logs.
 */
export default function BenchmarkModal({ open, setOpen, serverUrl }) {
  const [activeCategory, setActiveCategory] = useState("HumanEval");
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
      // Only auto-scroll if user is near bottom or when benchmark is active
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
      if (isNearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [status.logs, open, status.active]);

  if (!open) return null;

  const categories = [
    { id: "HumanEval", label: "HumanEval (Python Code)" },
    { id: "MBPP", label: "MBPP (Basic Python)" },
    { id: "GSM8K", label: "GSM8K (Grade Math)" },
    { id: "MATH", label: "MATH (Competition Math)" },
    { id: "GPQA (PhD Science)", label: "GPQA (PhD Science)" },
    { id: "AIME (Olympiad Logic)", label: "AIME (Olympiad Math)" },
    { id: "MuSR (PhD Logic)", label: "MuSR (Logical Reasoning)" },
    { id: "MMLU-Pro (Prof STEM)", label: "MMLU-Pro (STEM)" },
    { id: "SWE-bench Lite", label: "SWE-bench Lite (Git Fixes)" },
    { id: "SWE-bench Pro", label: "SWE-bench Pro (Complex Git)" },
    { id: "SearchQA / HotpotQA", label: "SearchQA (RAG Search)" }
  ];

  const handleStart = async () => {
    setLoading(true);
    try {
      await fetch(`${serverUrl}/api/benchmark/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: activeCategory })
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

  const currentBaseline = (status?.comparison_baselines && status.comparison_baselines[status?.category || activeCategory]) || {
    gpt4: 90.0,
    claude35_sonnet: 92.0,
    llama3_70b: 86.0,
    deepthink_aios: 91.5
  };

  return (
    <div className="modal-backdrop" onClick={() => setOpen(false)}>
      <div className="modal-content benchmark-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="benchmark-header">
          <div className="benchmark-header-title">
            <span className="benchmark-icon">📊</span>
            <div>
              <h2>AIOS Benchmark Studio</h2>
              <p className="benchmark-subtitle">
                Parallel Execution Telemetry & Performance Baselines
              </p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={() => setOpen(false)}>
            ✕
          </button>
        </div>

        {/* Control Bar */}
        <div className="benchmark-controls">
          <div className="category-select-wrapper">
            <label>Select Suite:</label>
            <select
              value={activeCategory}
              onChange={(e) => setActiveCategory(e.target.value)}
              disabled={status.active}
            >
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>

          <div className="action-buttons" style={{ display: "flex", gap: "8px" }}>
            {!status.active ? (
              <>
                <button
                  className="btn-run-benchmark"
                  onClick={handleStart}
                  disabled={loading}
                >
                  {loading ? "Starting..." : "▶ Run Suite"}
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
                  ⚡ Run All Benchmarks
                </button>
              </>
            ) : (
              <button className="btn-stop-benchmark" onClick={handleStop}>
                ⏹ Cancel Evaluation
              </button>
            )}
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
            <div className="kpi-label">Avg Task Latency</div>
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
                <span>{status.accuracy}%</span>
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
            <h3>🏆 Stored Benchmark Results ({Object.keys(status.history).length} Suites Evaluated)</h3>
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

        {/* Live Execution Console Log */}
        <div className="benchmark-section">
          <h3>Live Telemetry & Execution Log</h3>
          <div className="benchmark-console" ref={consoleRef}>
            {status.logs?.length === 0 ? (
              <div className="console-empty">
                Console ready. Select a benchmark suite above and click "Run Suite" or "Run All Benchmarks".
              </div>
            ) : (
              status.logs.map((log, index) => (
                <div key={index} className="console-line">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
