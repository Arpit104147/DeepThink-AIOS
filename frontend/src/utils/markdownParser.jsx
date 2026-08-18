import React from "react";

/**
 * Render a TeX string using KaTeX. Falls back to raw text if KaTeX isn't loaded.
 * @param {string} tex - Raw LaTeX string
 * @param {boolean} isBlock - Whether to render as display math
 */
export const renderMath = (tex, isBlock) => {
  if (window.katex) {
    try {
      return (
        <span
          dangerouslySetInnerHTML={{
            __html: window.katex.renderToString(tex, {
              displayMode: isBlock,
              throwOnError: false,
            }),
          }}
        />
      );
    } catch (e) {
      console.error(e);
    }
  }
  return isBlock ? (
    <div className="math-block-fallback">{tex}</div>
  ) : (
    <span className="math-inline-fallback">{tex}</span>
  );
};

/**
 * Parse a line of text and render inline elements: math, bold, inline code.
 * Handles both \\( ... \\) and $ ... $ math delimiters.
 */
export const renderInlineElements = (text) => {
  // Split on inline math (\( ... \) or $...$), bold (**...**), or inline code (`...`).
  // Previous pattern was over-escaped and silently matched nothing.
  const inlineParts = text.split(/(\\\([\s\S]*?\\\)|\$[^$\n]+\$|\*\*[^*\n]+\*\*|`[^`\n]+`)/g);
  return inlineParts.map((chunk, index) => {
    if (chunk == null || chunk === "") return null;
    if (chunk.startsWith("\\(") && chunk.endsWith("\\)")) {
      return <React.Fragment key={index}>{renderMath(chunk.slice(2, -2).trim(), false)}</React.Fragment>;
    }
    if (chunk.startsWith("$") && chunk.endsWith("$") && chunk.length > 2) {
      const content = chunk.slice(1, -1).trim();
      // Only treat as math if it contains at least one math-y token, and is not plain currency/prose.
      const mathTokens = /[=+\-*/^_{}\\]|\\frac|\\sqrt|\\int|\\sum|\\alpha|\\beta|\\gamma|\\theta|\\pi|\\lambda|\\mu|\\sigma/;
      const isCurrency = /^\d+(\.\d{1,2})?$/;
      if (isCurrency.test(content) || !mathTokens.test(content)) {
        return chunk;
      }
      return <React.Fragment key={index}>{renderMath(content, false)}</React.Fragment>;
    }
    if (chunk.startsWith("**") && chunk.endsWith("**")) {
      return <strong key={index}>{chunk.slice(2, -2)}</strong>;
    }
    if (chunk.startsWith("`") && chunk.endsWith("`")) {
      return <code key={index}>{chunk.slice(1, -1)}</code>;
    }
    return chunk;
  });
};

/**
 * Render a Markdown Table block into an HTML table with header, body, and alignment.
 */
export const renderMarkdownTable = (tableLines, tableKey) => {
  if (!tableLines || tableLines.length === 0) return null;

  const cleanedLines = tableLines
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && l.startsWith("|"));

  if (cleanedLines.length === 0) return null;

  const parseRow = (line) => {
    const trimmed = line.replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").map((cell) => cell.trim());
  };

  const headerCells = parseRow(cleanedLines[0]);
  let hasSeparator = false;

  if (cleanedLines.length > 1) {
    const sepCandidate = parseRow(cleanedLines[1]);
    hasSeparator = sepCandidate.every((cell) => /^:?-+:?$/.test(cell.replace(/\s/g, "")));
  }

  let alignments = headerCells.map(() => "left");
  if (hasSeparator) {
    const sepCells = parseRow(cleanedLines[1]);
    alignments = sepCells.map((cell) => {
      const c = cell.trim();
      if (c.startsWith(":") && c.endsWith(":")) return "center";
      if (c.endsWith(":")) return "right";
      return "left";
    });
  }

  const dataRows = hasSeparator ? cleanedLines.slice(2) : cleanedLines.slice(1);

  return (
    <div key={tableKey} className="md-table-wrapper">
      <table className="md-table">
        <thead>
          <tr>
            {headerCells.map((h, i) => (
              <th key={i} style={{ textAlign: alignments[i] || "left" }}>
                {renderInlineElements(h)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((rowLine, ri) => {
            const cells = parseRow(rowLine);
            return (
              <tr key={ri}>
                {headerCells.map((_, ci) => (
                  <td key={ci} style={{ textAlign: alignments[ci] || "left" }}>
                    {renderInlineElements(cells[ci] || "")}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

/**
 * Parse a text segment and render block-level markdown elements:
 * block math, headings, tables, lists, horizontal rules, paragraphs.
 */
export const parseAndRenderSegment = (segment) => {
  // Split on block math \[ ... \] or $$ ... $$. Previous pattern was over-escaped.
  const parts = segment.split(/(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/g);
  return parts.map((part, index) => {
    if (part.startsWith("\\[") && part.endsWith("\\]")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }
    if (part.startsWith("$$") && part.endsWith("$$")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }

    const lines = part.split("\n");
    const elements = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      // Detect Table block (lines starting with |)
      if (trimmed.startsWith("|") && (trimmed.endsWith("|") || trimmed.includes("|"))) {
        const tableLines = [];
        while (i < lines.length) {
          const curTrim = lines[i].trim();
          if (curTrim.startsWith("|")) {
            tableLines.push(curTrim);
            i++;
          } else if (curTrim === "" && i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) {
            // Skip empty newline between table rows
            i++;
          } else {
            break;
          }
        }
        elements.push(renderMarkdownTable(tableLines, `table-${index}-${i}`));
        continue;
      }

      if (trimmed === "") {
        elements.push(<div key={`empty-${i}`} style={{ height: "6px" }} />);
        i++;
        continue;
      }

      if (trimmed === "---") {
        elements.push(<hr key={`hr-${i}`} className="md-hr" />);
        i++;
        continue;
      }

      if (line.startsWith("##### ")) {
        elements.push(<h5 key={`h5-${i}`} className="md-h5">{renderInlineElements(line.slice(6))}</h5>);
        i++;
        continue;
      }
      if (line.startsWith("#### ")) {
        elements.push(<h4 key={`h4-${i}`} className="md-h4">{renderInlineElements(line.slice(5))}</h4>);
        i++;
        continue;
      }
      if (line.startsWith("### ")) {
        elements.push(<h3 key={`h3-${i}`} className="md-h3">{renderInlineElements(line.slice(4))}</h3>);
        i++;
        continue;
      }
      if (line.startsWith("## ")) {
        elements.push(<h2 key={`h2-${i}`} className="md-h2">{renderInlineElements(line.slice(3))}</h2>);
        i++;
        continue;
      }
      if (line.startsWith("# ")) {
        elements.push(<h1 key={`h1-${i}`} className="md-h1">{renderInlineElements(line.slice(2))}</h1>);
        i++;
        continue;
      }

      const listMatch = trimmed.match(/^([-*]|\d+\.)\s+(.*)/);
      if (listMatch) {
        const indent = line.length - line.trimStart().length;
        const marker = listMatch[1];
        const content = listMatch[2];
        const isNumbered = /^\d+\.$/.test(marker);
        elements.push(
          <div
            key={`list-${i}`}
            className={`md-list-item ${isNumbered ? "numbered" : "bullet"}`}
            style={{ paddingLeft: `${indent * 8 + 12}px` }}
          >
            {isNumbered ? (
              <span className="num-prefix">{marker}</span>
            ) : (
              <span className="bullet-dot">•</span>
            )}
            <span className="bullet-content">{renderInlineElements(content)}</span>
          </div>
        );
        i++;
        continue;
      }

      elements.push(
        <p key={`p-${i}`} className="md-p">{renderInlineElements(line)}</p>
      );
      i++;
    }

    return <React.Fragment key={index}>{elements}</React.Fragment>;
  });
};

/**
 * Split raw streamed text into typed segments: text, plotly, html, metrics.
 * Each segment carries a type and content for specialized rendering.
 */
export const splitSpecialSegments = (text) => {
  const segments = [];
  let currentPos = 0;

  while (currentPos < text.length) {
    const plotlyIdx = text.indexOf("<!--PLOTLY_JSON-->", currentPos);
    const htmlIdx = text.indexOf("<!--ARTIFACT_HTML-->", currentPos);
    const metricsIdx = text.indexOf("=== PREDICTIVE_METRICS ===", currentPos);

    const candidates = [
      { idx: plotlyIdx, type: "plotly", open: "<!--PLOTLY_JSON-->", close: "<!--/PLOTLY_JSON-->" },
      { idx: htmlIdx, type: "html", open: "<!--ARTIFACT_HTML-->", close: "<!--/ARTIFACT_HTML-->" },
      { idx: metricsIdx, type: "metrics", open: "=== PREDICTIVE_METRICS ===", close: "=== /PREDICTIVE_METRICS ===" },
    ].filter((c) => c.idx !== -1);

    if (candidates.length === 0) {
      segments.push({ type: "text", content: text.substring(currentPos) });
      break;
    }

    candidates.sort((a, b) => a.idx - b.idx);
    const { idx: earliestIdx, type: tagType, open: openTag, close: closeTag } = candidates[0];

    if (earliestIdx > currentPos) {
      segments.push({ type: "text", content: text.substring(currentPos, earliestIdx) });
    }

    const startOfData = earliestIdx + openTag.length;
    const closeIdx = text.indexOf(closeTag, startOfData);

    if (closeIdx !== -1) {
      segments.push({ type: tagType, content: text.substring(startOfData, closeIdx), closed: true });
      currentPos = closeIdx + closeTag.length;
    } else {
      segments.push({ type: tagType, content: text.substring(startOfData), closed: false });
      currentPos = text.length;
    }
  }

  return segments;
};
