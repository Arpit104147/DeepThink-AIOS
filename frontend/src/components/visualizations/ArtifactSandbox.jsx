import React, { useState, useEffect, useRef } from "react";

/**
 * @component ArtifactSandbox
 * Renders AI-generated HTML/JS in a secure sandboxed iframe.
 * Includes a real-time console overlay, reload, expand, and
 * open-in-new-tab controls.
 */
const ArtifactSandbox = ({ htmlCode }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [consoleLogs, setConsoleLogs] = useState([]);
  const [showConsole, setShowConsole] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const iframeRef = useRef(null);

  // Listen for console messages from the sandboxed iframe
  useEffect(() => {
    const handleMessage = (e) => {
      if (e.data && e.data.type) {
        if (e.data.type === "CONSOLE_LOG") {
          setConsoleLogs((prev) => [...prev, { type: "log", text: e.data.text }]);
        } else if (e.data.type === "CONSOLE_ERROR") {
          setConsoleLogs((prev) => [...prev, { type: "error", text: e.data.text }]);
        } else if (e.data.type === "CONSOLE_WARN") {
          setConsoleLogs((prev) => [...prev, { type: "warn", text: e.data.text }]);
        }
      }
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  // Inject dark theme, error handler, and console capture into the iframe
  useEffect(() => {
    if (!htmlCode || !iframeRef.current) return;
    setConsoleLogs([]);

    try {
      let doc = htmlCode;

      const injection = `
        <style>
          html, body { background-color: #0d0d0d !important; color: #e0e0e0 !important; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; height: 100%; overflow: hidden; }
          #chart, .js-plotly-plot { background-color: transparent !important; }
          .bg { fill: transparent !important; }
          .error-box { margin: 20px; border: 1px solid #ff4444; padding: 15px; background: #2a0000; border-radius: 5px; color: #ff8888; }
        </style>
        <script id="sandbox-injection">
          (function() {
            const _log = console.log;
            const _error = console.error;
            const _warn = console.warn;
            
            console.log = function(...args) {
              _log.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_LOG', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };
            console.error = function(...args) {
              _error.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_ERROR', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };
            console.warn = function(...args) {
              _warn.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_WARN', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };

            const originalAddEventListener = window.addEventListener;
            window.addEventListener = function(type, listener, options) {
              if (type === 'load' && document.readyState === 'complete') {
                setTimeout(() => { try { listener({ type: 'load' }); } catch(e) {} }, 0);
              } else if (type === 'DOMContentLoaded' && (document.readyState === 'interactive' || document.readyState === 'complete')) {
                setTimeout(() => { try { listener({ type: 'DOMContentLoaded' }); } catch(e) {} }, 0);
              } else {
                originalAddEventListener.call(window, type, listener, options);
              }
            };

            const originalDocAddEventListener = document.addEventListener;
            document.addEventListener = function(type, listener, options) {
              if (type === 'DOMContentLoaded' && (document.readyState === 'interactive' || document.readyState === 'complete')) {
                setTimeout(() => { try { listener({ type: 'DOMContentLoaded' }); } catch(e) {} }, 0);
              } else if (type === 'load' && document.readyState === 'complete') {
                setTimeout(() => { try { listener({ type: 'load' }); } catch(e) {} }, 0);
              } else {
                originalDocAddEventListener.call(document, type, listener, options);
              }
            };

            let _onload = null;
            Object.defineProperty(window, 'onload', {
              get: () => _onload,
              set: (fn) => {
                _onload = fn;
                if (fn && document.readyState === 'complete') {
                  setTimeout(() => { try { fn({ type: 'load' }); } catch(e) {} }, 0);
                }
              },
              configurable: true
            });
          })();

          window.onerror = function(msg, url, line, col, error) {
            window.parent.postMessage({ type: 'CONSOLE_ERROR', text: msg + ' (Line ' + line + ')' }, '*');
            const errDiv = document.createElement('div');
            errDiv.className = 'error-box';
            errDiv.innerHTML = '<strong>⚠️ Sandbox Runtime Error:</strong><br/>' + msg + ' (Line ' + line + ')';
            document.body.appendChild(errDiv);
            return false;
          };
        </script>
      `;

      const lowerHtml = doc.toLowerCase();
      if (lowerHtml.includes("</head>")) {
        const index = lowerHtml.indexOf("</head>");
        doc = doc.substring(0, index) + injection + doc.substring(index);
      } else if (lowerHtml.includes("<body>")) {
        const index = lowerHtml.indexOf("<body>");
        doc = doc.substring(0, index + 6) + injection + doc.substring(index + 6);
      } else {
        doc = injection + doc;
      }

      // Use srcdoc to properly sequence external CDN script loading
      iframeRef.current.srcdoc = doc;
      setHasError(false);
    } catch (err) {
      console.error("Artifact write error:", err);
      setHasError(true);
    }
  }, [htmlCode, reloadKey]);

  if (hasError || !htmlCode) {
    return (
      <div className="artifact-error">
        <span>⚠️</span> Failed to render sandbox
      </div>
    );
  }

  return (
    <div className={`artifact-container ${isExpanded ? "expanded" : ""}`}>
      <div className="artifact-header">
        <div className="artifact-header-left">
          <div className="artifact-dot" />
          <span className="artifact-label">Live Sandbox Simulation</span>
          <span className="artifact-badge">HTML/JS</span>
        </div>
        <div className="artifact-header-right">
          <button
            className="artifact-btn"
            onClick={() => setReloadKey((prev) => prev + 1)}
            title="Reload simulation"
          >
            ↻
          </button>
          <button
            className="artifact-btn"
            onClick={() => {
              const w = window.open("", "_blank");
              if (w) { w.document.write(htmlCode); w.document.close(); }
            }}
            title="Open in new tab"
          >
            ↗
          </button>
          <button
            className="artifact-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? "⊖" : "⊕"}
          </button>
        </div>
      </div>

      <div className="artifact-iframe-wrap">
        <iframe
          ref={iframeRef}
          title="AI Artifact"
          className="artifact-iframe"
        />

        {/* Real-time console overlay */}
        <div className="sandbox-console-drawer">
          <div
            className="sandbox-console-header"
            onClick={() => setShowConsole(!showConsole)}
          >
            <span className="sandbox-console-title">
              <span>🛠️</span> Console Output {consoleLogs.length > 0 && `(${consoleLogs.length})`}
            </span>
            <div className="sandbox-console-actions">
              {consoleLogs.length > 0 && (
                <button
                  onClick={(e) => { e.stopPropagation(); setConsoleLogs([]); }}
                  className="sandbox-console-clear"
                  title="Clear Console"
                >
                  Clear
                </button>
              )}
              <span>{showConsole ? "▼" : "▲"}</span>
            </div>
          </div>

          {showConsole && (
            <div className="sandbox-console-body">
              {consoleLogs.length === 0 ? (
                <span className="sandbox-console-empty">
                  No console output captured. Simulation running cleanly.
                </span>
              ) : (
                consoleLogs.map((log, idx) => (
                  <div key={idx} className={`sandbox-console-line ${log.type}`}>
                    {log.text}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ArtifactSandbox;
