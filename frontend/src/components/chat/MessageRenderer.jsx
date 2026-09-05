import React, { useState, useEffect } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Clipboard, ClipboardCheck, ChevronDown } from "lucide-react";
import useKatexReady from "../../hooks/useKatexReady";
import { splitSpecialSegments, parseAndRenderSegment } from "../../utils/markdownParser";
import PredictiveMetricsCard from "../visualizations/PredictiveMetrics";
import PlotlyChart from "../visualizations/PlotlyChart";
import ArtifactSandbox from "../visualizations/ArtifactSandbox";

const syntaxThemeOverrides = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: "#1a1a2e",
    borderRadius: "0 0 8px 8px",
    margin: 0,
    fontSize: "13px",
    lineHeight: "1.6",
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: "none",
    fontSize: "13px",
  },
};

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
  const highlightLang = (lang || "text").toLowerCase();

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-lang-badge">{displayLang}</span>
        <button className={`copy-btn ${copied ? "copied" : ""}`} onClick={handleCopy}>
          {copied ? "Copied! ✓" : "Copy"}
        </button>
      </div>
      <SyntaxHighlighter
        language={highlightLang}
        style={syntaxThemeOverrides}
        showLineNumbers={code.split("\n").length > 5}
        wrapLongLines={true}
        customStyle={{ margin: 0 }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

/**
 * @component MessageRenderer
 * Renders AI response text with support for:
 * - Markdown (headings, lists, bold, code blocks)
 * - LaTeX math (block and inline via KaTeX)
 * - Syntax-highlighted code blocks
 * - Embedded Plotly charts
 * - Embedded HTML artifact sandboxes
 * - Predictive metrics cards
 * - Generic heading-based section navigation
 * - Quick copy full response
 */
const MessageRenderer = ({ text, animate = false }) => {
  useKatexReady();
  const [displayedText, setDisplayedText] = useState(animate ? "" : text);
  const [copiedFull, setCopiedFull] = useState(false);
  const [navOpen, setNavOpen] = useState(false);

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

  // Extract ## and ### headings for generic section navigation
  const headingMatches = displayedText.match(/^#{2,3}\s+.+$/gm) || [];
  const hasNavigation = headingMatches.length >= 3 && displayedText.length > 500;

  const handleCopyFull = () => {
    navigator.clipboard.writeText(displayedText);
    setCopiedFull(true);
    setTimeout(() => setCopiedFull(false), 2000);
  };

  const scrollToHeading = (headingText) => {
    const cleanText = headingText.replace(/^#{2,3}\s+/, "").trim();
    const headings = document.querySelectorAll(".message-renderer .md-content h2, .message-renderer .md-content h3");
    for (const el of headings) {
      if (el.textContent.trim() === cleanText) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        setNavOpen(false);
        break;
      }
    }
  };

  const segments = splitSpecialSegments(displayedText);

  return (
    <div className="message-renderer">
      {/* Response Action Bar */}
      {displayedText.length > 500 && (
        <div className="response-action-bar">
          {hasNavigation && (
            <div className="section-nav-wrap">
              <button
                className="section-nav-toggle"
                onClick={() => setNavOpen(!navOpen)}
              >
                <ChevronDown size={14} style={{ transform: navOpen ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
                Jump to Section
              </button>
              {navOpen && (
                <div className="section-nav-dropdown">
                  {headingMatches.map((h, i) => (
                    <button
                      key={i}
                      className="section-nav-item"
                      onClick={() => scrollToHeading(h)}
                    >
                      {h.replace(/^#{2,3}\s+/, "")}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          <button
            className={`copy-full-btn ${copiedFull ? "copied" : ""}`}
            onClick={handleCopyFull}
          >
            {copiedFull ? (
              <><ClipboardCheck size={13} /> Copied Markdown</>
            ) : (
              <><Clipboard size={13} /> Copy Full Output</>
            )}
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
