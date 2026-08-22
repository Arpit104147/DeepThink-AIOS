import React, { useState, useEffect } from "react";
import useKatexReady from "../../hooks/useKatexReady";
import { splitSpecialSegments, parseAndRenderSegment } from "../../utils/markdownParser";
import PredictiveMetricsCard from "../visualizations/PredictiveMetrics";
import PlotlyChart from "../visualizations/PlotlyChart";
import ArtifactSandbox from "../visualizations/ArtifactSandbox";

/**
 * @component CodeBlock
 */
const CodeBlock = ({ lang, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const displayLang = (lang || "code").toUpperCase();

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-lang-badge">{displayLang}</span>
        <button className={`copy-btn ${copied ? "copied" : ""}`} onClick={handleCopy}>
          {copied ? "Copied! ✓" : "Copy"}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
};

/**
 * @component MessageRenderer
 * Renders AI response text with support for:
 * - Markdown (headings, lists, bold, code blocks)
 * - LaTeX math (block and inline via KaTeX)
 * - Embedded Plotly charts
 * - Embedded HTML artifact sandboxes
 * - Predictive metrics cards
 * - Multi-volume textbook quick navigation
 * - Quick copy full response
 */
const MessageRenderer = ({ text, animate = false }) => {
  useKatexReady();
  const [displayedText, setDisplayedText] = useState(animate ? "" : text);
  const [copiedFull, setCopiedFull] = useState(false);

  useEffect(() => {
    if (!animate) {
      setDisplayedText(text);
      return;
    }

    let currentLength = 0;
    const step = 8;
    const intervalTime = 12;

    const interval = setInterval(() => {
      currentLength += step;
      if (currentLength >= text.length) {
        setDisplayedText(text);
        clearInterval(interval);
      } else {
        let targetLength = currentLength;
        const sub = text.substring(0, targetLength);

        // Prevent partial comment/tag slicing
        const openComments = (sub.match(/<!--/g) || []).length;
        const closeComments = (sub.match(/-->/g) || []).length;
        if (openComments > closeComments) {
          const nextClose = text.indexOf("-->", targetLength);
          if (nextClose !== -1) {
            targetLength = nextClose + 3;
          }
        }

        setDisplayedText(text.substring(0, targetLength));
      }
    }, intervalTime);

    return () => clearInterval(interval);
  }, [text, animate]);

  if (!displayedText) return null;

  const isMultiVolume = displayedText.includes("Volume I:") && displayedText.includes("Volume II:");

  const handleCopyFull = () => {
    navigator.clipboard.writeText(displayedText);
    setCopiedFull(true);
    setTimeout(() => setCopiedFull(false), 2000);
  };

  const segments = splitSpecialSegments(displayedText);

  return (
    <div className="message-renderer">
      {/* Response Action Bar for long academic/computational outputs */}
      {displayedText.length > 500 && (
        <div style={{
          display: "flex",
          justifyContent: isMultiVolume ? "space-between" : "flex-end",
          alignItems: "center",
          marginBottom: "12px",
          paddingBottom: "8px",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          flexWrap: "wrap",
          gap: "8px"
        }}>
          {isMultiVolume && (
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted, #888)", alignSelf: "center", marginRight: "4px" }}>
                Jump to:
              </span>
              <button
                className="study-nav-btn"
                style={{
                  padding: "3px 8px",
                  borderRadius: "6px",
                  fontSize: "11px",
                  background: "rgba(59, 130, 246, 0.15)",
                  color: "#60a5fa",
                  border: "1px solid rgba(59, 130, 246, 0.3)",
                  cursor: "pointer"
                }}
                onClick={() => {
                  const el = document.querySelector(".message-renderer h2, .message-renderer h1");
                  if (el) el.scrollIntoView({ behavior: "smooth" });
                }}
              >
                📚 Volume I
              </button>
              <button
                className="study-nav-btn"
                style={{
                  padding: "3px 8px",
                  borderRadius: "6px",
                  fontSize: "11px",
                  background: "rgba(16, 185, 129, 0.15)",
                  color: "#34d399",
                  border: "1px solid rgba(16, 185, 129, 0.3)",
                  cursor: "pointer"
                }}
                onClick={() => {
                  const el = document.querySelectorAll(".message-renderer h2");
                  if (el && el[1]) el[1].scrollIntoView({ behavior: "smooth" });
                }}
              >
                📝 Volume II (Exam)
              </button>
            </div>
          )}
          <button
            className={`copy-full-btn ${copiedFull ? "copied" : ""}`}
            style={{
              padding: "4px 10px",
              borderRadius: "6px",
              fontSize: "11px",
              background: copiedFull ? "rgba(16, 185, 129, 0.2)" : "rgba(255,255,255,0.06)",
              color: copiedFull ? "#34d399" : "var(--color-text-muted, #aaa)",
              border: "1px solid rgba(255,255,255,0.1)",
              cursor: "pointer",
              transition: "all 0.2s ease"
            }}
            onClick={handleCopyFull}
          >
            {copiedFull ? "Copied Markdown ✓" : "📋 Copy Full Output"}
          </button>
        </div>
      )}

      {segments.map((segment, si) => {
        if (segment.type === "metrics") {
          return <PredictiveMetricsCard key={`predictive-${si}`} jsonStr={segment.content.trim()} />;
        }

        if (segment.type === "plotly") {
          if (!segment.closed) {
            return (
              <div key={`plotly-${si}`} className="plotly-chart-container plotly-loading">
                <div className="plotly-loading-dot" />
                <span className="plotly-loading-text">Generating interactive 3D Plotly chart...</span>
              </div>
            );
          }
          return <PlotlyChart key={`plotly-${si}`} jsonStr={segment.content.trim()} />;
        }

        if (segment.type === "html") {
          if (!segment.closed) {
            return (
              <div key={`artifact-${si}`} className="plotly-chart-container artifact-loading">
                <div className="artifact-loading-dot" />
                <span className="artifact-loading-text">Building 3D simulation sandbox...</span>
              </div>
            );
          }
          return <ArtifactSandbox key={`artifact-${si}`} htmlCode={segment.content.trim()} />;
        }

        // Standard markdown text with code block support
        const parts = segment.content.split(/(```[\s\S]*?```)/g);
        return (
          <React.Fragment key={`seg-${si}`}>
            {parts.map((part, i) => {
              if (part.startsWith("```") && part.endsWith("```")) {
                const lines = part.slice(3, -3).split("\n");
                const lang = lines[0].trim().split(" ")[0];
                const code = lines.slice(1).join("\n");
                return <CodeBlock key={i} lang={lang} code={code} />;
              }
              return (
                <div key={i} className="md-content">
                  {parseAndRenderSegment(part)}
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default MessageRenderer;
