import React, { useState, useEffect } from "react";
import useKatexReady from "../../hooks/useKatexReady";
import { splitSpecialSegments, parseAndRenderSegment } from "../../utils/markdownParser";
import PredictiveMetricsCard from "../visualizations/PredictiveMetrics";
import PlotlyChart from "../visualizations/PlotlyChart";
import ArtifactSandbox from "../visualizations/ArtifactSandbox";

/**
 * @component MessageRenderer
 * Renders AI response text with support for:
 * - Markdown (headings, lists, bold, code blocks)
 * - LaTeX math (block and inline via KaTeX)
 * - Embedded Plotly charts
 * - Embedded HTML artifact sandboxes
 * - Predictive metrics cards
 * - Optional typewriter animation
 */
const MessageRenderer = ({ text, animate = false }) => {
  // Subscribe to KaTeX load event for re-render on library availability
  useKatexReady();
  const [displayedText, setDisplayedText] = useState(animate ? "" : text);

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

  const segments = splitSpecialSegments(displayedText);

  return (
    <div className="message-renderer">
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
                return (
                  <div key={i} className="code-block-wrapper">
                    <div className="code-block-header">
                      <span>{lang || "code"}</span>
                      <button className="copy-btn" onClick={() => navigator.clipboard.writeText(code)}>
                        Copy
                      </button>
                    </div>
                    <pre><code>{code}</code></pre>
                  </div>
                );
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
