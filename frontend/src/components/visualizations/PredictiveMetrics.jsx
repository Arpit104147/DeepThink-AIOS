import { useMemo } from "react";

/**
 * @component PredictiveMetricsCard
 * Renders a glassmorphic card displaying ML forecasting results
 * including metric scores and tabular future predictions.
 */
const PredictiveMetricsCard = ({ jsonStr }) => {
  const { data, errorMsg } = useMemo(() => {
    try {
      return { data: JSON.parse(jsonStr), errorMsg: null };
    } catch (err) {
      console.error("Error parsing predictive metrics JSON:", err);
      return { data: null, errorMsg: "Failed to parse predictive metrics." };
    }
  }, [jsonStr]);

  if (errorMsg) {
    return (
      <div className="predictive-error-card">{errorMsg}</div>
    );
  }

  if (!data) return null;

  const { metric_name, metric_value, forecast, dates } = data;

  return (
    <div className="predictive-metrics-card">
      <div className="predictive-header">
        <h4 className="predictive-title">🔮 Forecasting & Prediction Report</h4>
        <div className="predictive-badge">API Verified</div>
      </div>

      <div className="predictive-grid">
        <div className="predictive-stat-box">
          <div className="predictive-stat-label">Target Metric</div>
          <div className="predictive-stat-value">{metric_name || "Accuracy"}</div>
        </div>
        <div className="predictive-stat-box">
          <div className="predictive-stat-label">Model Score</div>
          <div className={`predictive-stat-value score ${Number.isFinite(metric_value) && metric_value > 0.8 ? "high" : "low"}`}>
            {Number.isFinite(metric_value) ? metric_value.toFixed(4) : metric_value || "N/A"}
          </div>
        </div>
      </div>

      {Array.isArray(forecast) && forecast.length > 0 && (
        <div className="predictive-forecast">
          <div className="predictive-forecast-label">Future Predictions:</div>
          <div className="predictive-table-wrap">
            <table className="predictive-table">
              <thead>
                <tr>
                  <th>Time/Step</th>
                  <th className="text-right">Forecast Value</th>
                </tr>
              </thead>
              <tbody>
                {forecast.map((val, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? "even" : ""}>
                    <td>{dates && dates[idx] ? dates[idx] : `Step +${idx + 1}`}</td>
                    <td className="text-right forecast-value">
                      {Number.isFinite(val)
                        ? val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })
                        : val}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictiveMetricsCard;
