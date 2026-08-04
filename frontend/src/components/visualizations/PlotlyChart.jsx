import React, { useState, useEffect, useRef } from "react";

/**
 * @component PlotlyChart
 * Renders interactive Plotly.js charts from raw JSON data.
 * Automatically applies dark theme styling and responsive sizing.
 */
const PlotlyChart = ({ jsonStr }) => {
  const chartRef = useRef(null);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    if (!chartRef.current) return;
    setErrorMsg(null);

    try {
      const fig = JSON.parse(jsonStr);

      // Clear fixed dimensions from backend; let CSS handle sizing
      if (fig.layout) {
        delete fig.layout.width;
        delete fig.layout.height;
      }

      const layout = {
        autosize: true,
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e0e0e0" },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        ...fig.layout,
      };

      const data = Array.isArray(fig.data) ? fig.data : [fig.data];

      if (!window.Plotly) {
        throw new Error("Plotly.js library failed to load from CDN.");
      }

      window.Plotly.react(chartRef.current, data, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }).catch((err) => {
        setErrorMsg(`Plotly drawing error: ${err.message}`);
      });
    } catch (err) {
      console.error("Plotly render error:", err);
      setErrorMsg(err.message || String(err));
    }
  }, [jsonStr]);

  if (errorMsg) {
    return (
      <div className="plotly-chart-container plotly-error">
        <span className="plotly-error-icon">⚠️</span>
        <h3 className="plotly-error-title">Visualization Render Error</h3>
        <p className="plotly-error-message">{errorMsg}</p>
        <details className="plotly-error-details">
          <summary>Show Raw JSON Data</summary>
          <pre className="plotly-error-json">{jsonStr}</pre>
        </details>
      </div>
    );
  }

  return (
    <div
      ref={chartRef}
      className="plotly-chart-container"
    />
  );
};

export default PlotlyChart;
