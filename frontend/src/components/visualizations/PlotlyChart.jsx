import { useState, useEffect, useRef, useMemo } from "react";

/**
 * @component PlotlyChart
 * Renders interactive Plotly.js charts from raw JSON data.
 * Automatically applies dark theme styling and responsive sizing.
 */
const PlotlyChart = ({ jsonStr }) => {
  const chartRef = useRef(null);
  const [drawError, setDrawError] = useState(null);

  const { fig, parseError } = useMemo(() => {
    try {
      const parsed = JSON.parse(jsonStr);
      return { fig: parsed, parseError: null };
    } catch (err) {
      return { fig: null, parseError: err.message || String(err) };
    }
  }, [jsonStr]);

  useEffect(() => {
    const currentElem = chartRef.current;
    if (!currentElem || !fig || !window.Plotly) return;

    try {
      const layout = {
        autosize: true,
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e0e0e0" },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        ...fig.layout,
      };
      delete layout.width;
      delete layout.height;

      const data = Array.isArray(fig) ? fig : (Array.isArray(fig?.data) ? fig.data : (fig?.data ? [fig.data] : []));

      window.Plotly.react(currentElem, data, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }).then(() => {
        setDrawError(null);
      }).catch((err) => {
        setDrawError(`Plotly drawing error: ${err.message}`);
      });
    } catch (err) {
      setDrawError(err.message || String(err));
    }
    
    return () => {
      if (currentElem && window.Plotly) {
        window.Plotly.purge(currentElem);
      }
    };
  }, [fig]);

  const plotlyMissing = fig && !window.Plotly ? "Plotly.js library failed to load from CDN." : null;
  const errorMsg = parseError || plotlyMissing || drawError;

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
