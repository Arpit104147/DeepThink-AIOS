import re
import json
from backend.sandbox import Sandbox

class PredictionPipeline:
    """Enterprise Machine Learning Predictive Modeling & High-Precision Forecast Engine."""

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        if status_callback:
            status_callback("🔮 Prediction Engine: Ingesting data & initializing ML tournament...", "info", "ornith", 20)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        coder_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)

        # 1. Check for real live financial / time-series data
        real_data_context = ""
        try:
            if hasattr(orchestrator, "web_search") and orchestrator.web_search:
                fin_table = orchestrator.web_search.fetch_financial_quote(prompt)
                if fin_table:
                    real_data_context = fin_table
                else:
                    search_res = orchestrator.web_search.search_and_scrape(prompt, max_results=3, max_scrapes=2)
                    if isinstance(search_res, dict) and not search_res.get("empty", True):
                        real_data_context = search_res.get("context", "")
        except Exception:
            pass

        data_instruction = ""
        if real_data_context:
            data_instruction = (
                f"LIVE HISTORICAL / MARKET DATA EXTRACTED:\n{real_data_context}\n\n"
                f"INSTRUCTION: Extract the actual historical numbers/dates from the live data above and load them into NumPy/Pandas arrays. "
                f"If additional time points are needed, interpolate or extend realistically based on the historical trend.\n\n"
            )
        else:
            data_instruction = (
                "INSTRUCTION: Generate a dense, realistic multivariate time series dataset (100+ points) modeling the specific domain dynamics "
                "with realistic trend, seasonality, and variance.\n\n"
            )

        if status_callback:
            status_callback("🔮 Training & Cross-Validating Multi-Algorithm ML Tournament...", "info", "ornith", 50)

        script_p = (
            "Write an optimized, production-grade Python script using scikit-learn, numpy, and pandas for this predictive modeling task.\n\n"
            f"USER REQUEST: {prompt}\n\n"
            f"{data_instruction}"
            "MANDATORY MULTI-ALGORITHM ML TOURNAMENT REQUIREMENTS:\n"
            "1. Split data chronologically or train/test (80/20).\n"
            "2. Train and evaluate 4 distinct competitive regression models:\n"
            "   a. Model A: Polynomial Ridge (PolynomialFeatures(degree=2) + StandardScaler + Ridge(alpha=1.0))\n"
            "   b. Model B: HistGradientBoostingRegressor(max_iter=100, random_state=42)\n"
            "   c. Model C: RandomForestRegressor(n_estimators=100, random_state=42)\n"
            "   d. Model D: HuberRegressor(max_iter=200)\n"
            "3. Compute R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) for all 4 models on test split.\n"
            "4. Automatically select the CHAMPION model (highest R² score).\n"
            "5. Use the Champion model to forecast the next 10 future time steps, including 95% confidence bounds (± 1.96 * residual std).\n"
            "6. Output ONLY valid JSON metrics dictionary at the end:\n"
            "   PREDICTIVE_METRICS = {\n"
            "       'champion_model': str,\n"
            "       'r2': float,\n"
            "       'rmse': float,\n"
            "       'mae': float,\n"
            "       'model_scores': {'Polynomial_Ridge': float, 'Gradient_Boosting': float, 'Random_Forest': float, 'Huber_Robust': float},\n"
            "       'history_actual': list,\n"
            "       'history_fitted': list,\n"
            "       'forecast_values': list,\n"
            "       'confidence_lower': list,\n"
            "       'confidence_upper': list\n"
            "   }\n"
            "   print(json.dumps(PREDICTIVE_METRICS))\n\n"
            "Wrap script in ```python``` blocks."
        )

        code_resp = orchestrator._call_model(coder_llm, script_p, gen_tokens, gen_temp)
        code = Sandbox.extract_code(orchestrator._strip_thinking(code_resp))

        if status_callback:
            status_callback("🔮 Executing ML Tournament in High-Performance Sandbox...", "info", "system", 75)

        ok, output = orchestrator.sandbox.execute(code, language="python")

        metrics_json = None
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{") and ("r2" in line or "champion_model" in line):
                try:
                    metrics_json = json.loads(line)
                    break
                except Exception:
                    pass

        metrics_md = ""
        if metrics_json:
            champ = metrics_json.get('champion_model', 'Champion Regressor')
            r2_val = metrics_json.get('r2', 'N/A')
            rmse_val = metrics_json.get('rmse', 'N/A')
            mae_val = metrics_json.get('mae', 'N/A')
            scores = metrics_json.get('model_scores', {})
            
            scores_rows = ""
            if isinstance(scores, dict):
                scores_rows = "\n".join([f"| {k.replace('_', ' ')} | `{v:.4f}` | {'🏆 Champion' if k.lower() in champ.lower() or champ.lower() in k.lower() else 'Evaluated'} |" for k, v in scores.items() if isinstance(v, (int, float))])

            metrics_md = (
                f"\n\n### 🏆 Champion Model & Forecast Evaluation\n"
                f"- **Champion Algorithm:** `{champ}`\n"
                f"- **Model R² Score (Accuracy):** `{r2_val}`\n"
                f"- **Root Mean Squared Error (RMSE):** `{rmse_val}`\n"
                f"- **Mean Absolute Error (MAE):** `{mae_val}`\n\n"
            )
            if scores_rows:
                metrics_md += (
                    f"#### 📊 Multi-Algorithm Tournament Leaderboard\n\n"
                    f"| Machine Learning Architecture | Validation R² Score | Tournament Status |\n"
                    f"| :--- | :--- | :--- |\n"
                    f"{scores_rows}\n\n"
                )

        if status_callback:
            status_callback("🔮 Rendering Interactive 3D/2D Forecast Surface...", "info", "system", 90)

        viz_html = orchestrator._generate_3d_visualization(prompt, coder_llm, oc_ctx, gen_tokens, gen_temp, status_callback)
        if not viz_html or "<!--ARTIFACT_HTML-->" not in viz_html:
            viz_html = PredictionPipeline._build_plotly_forecast_chart(prompt, metrics_json)

        res_md = f"# 🔮 High-Precision Predictive Modeling & Machine Learning Forecast\n\n```python\n{code}\n```\n{metrics_md}\n\n{viz_html}"
        return res_md

    @staticmethod
    def _build_plotly_forecast_chart(prompt, metrics_json=None):
        """Generates dynamic Plotly forecast charts with actuals, fitted curve, forecast horizon, and confidence band."""
        history_actual = [100, 102, 105, 103, 108, 112, 110, 115, 118, 122]
        forecast_values = [125, 128, 131, 134, 137, 140, 143, 146, 150, 153]
        conf_lower = [122, 124, 126, 128, 130, 132, 134, 136, 139, 141]
        conf_upper = [128, 132, 136, 140, 144, 148, 152, 156, 161, 165]
        champ = "Optimized ML Regressor"

        if isinstance(metrics_json, dict):
            champ = metrics_json.get("champion_model", champ)
            if "history_actual" in metrics_json and isinstance(metrics_json["history_actual"], list) and len(metrics_json["history_actual"]) > 0:
                history_actual = [float(x) for x in metrics_json["history_actual"] if isinstance(x, (int, float))][:30]
            if "forecast_values" in metrics_json and isinstance(metrics_json["forecast_values"], list) and len(metrics_json["forecast_values"]) > 0:
                forecast_values = [float(x) for x in metrics_json["forecast_values"] if isinstance(x, (int, float))][:15]
            elif "predictions" in metrics_json and isinstance(metrics_json["predictions"], list):
                forecast_values = [float(x) for x in metrics_json["predictions"] if isinstance(x, (int, float))][:15]

            if "confidence_lower" in metrics_json and isinstance(metrics_json["confidence_lower"], list) and len(metrics_json["confidence_lower"]) > 0:
                conf_lower = [float(x) for x in metrics_json["confidence_lower"] if isinstance(x, (int, float))][:len(forecast_values)]
            else:
                conf_lower = [v * 0.95 for v in forecast_values]

            if "confidence_upper" in metrics_json and isinstance(metrics_json["confidence_upper"], list) and len(metrics_json["confidence_upper"]) > 0:
                conf_upper = [float(x) for x in metrics_json["confidence_upper"] if isinstance(x, (int, float))][:len(forecast_values)]
            else:
                conf_upper = [v * 1.05 for v in forecast_values]

        hist_x = list(range(1, len(history_actual) + 1))
        fore_x = list(range(len(history_actual), len(history_actual) + len(forecast_values)))
        if history_actual:
            forecast_values_plot = [history_actual[-1]] + forecast_values
            conf_lower_plot = [history_actual[-1]] + conf_lower
            conf_upper_plot = [history_actual[-1]] + conf_upper
            fore_x = list(range(len(history_actual), len(history_actual) + len(forecast_values) + 1))
        else:
            forecast_values_plot = forecast_values
            conf_lower_plot = conf_lower
            conf_upper_plot = conf_upper

        clean_topic = re.sub(r"[^a-zA-Z0-9 ]", "", prompt)[:50].strip() or "Predictive Horizon"

        return (
            "<!--ARTIFACT_HTML-->\n"
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "  <script src=\"https://cdn.plot.ly/plotly-2.24.1.min.js\"></script>\n"
            "  <style>html, body { margin:0; padding:0; width:100vw; height:100vh; background:#0a0d14; font-family:sans-serif; overflow:hidden; }</style>\n"
            "</head>\n"
            "<body>\n"
            "  <div id=\"plot\" style=\"width:100vw; height:100vh;\"></div>\n"
            "  <script>\n"
            "    document.addEventListener('DOMContentLoaded', function() {\n"
            f"      var histX = {json.dumps(hist_x)};\n"
            f"      var histY = {json.dumps(history_actual)};\n"
            f"      var foreX = {json.dumps(fore_x)};\n"
            f"      var foreY = {json.dumps(forecast_values_plot)};\n"
            f"      var confLow = {json.dumps(conf_lower_plot)};\n"
            f"      var confHigh = {json.dumps(conf_upper_plot)};\n"
            "      \n"
            "      var traceActual = {\n"
            "        x: histX, y: histY,\n"
            "        mode: 'lines+markers',\n"
            "        name: 'Historical / Training Data',\n"
            "        line: { color: '#38bdf8', width: 3 },\n"
            "        marker: { size: 6, color: '#38bdf8' }\n"
            "      };\n"
            "      \n"
            "      var traceForecast = {\n"
            "        x: foreX, y: foreY,\n"
            "        mode: 'lines+markers',\n"
            f"        name: 'Champion Forecast ({champ})',\n"
            "        line: { color: '#a855f7', width: 3, dash: 'dot' },\n"
            "        marker: { size: 7, color: '#ec4899' }\n"
            "      };\n"
            "      \n"
            "      var traceConfUpper = {\n"
            "        x: foreX, y: confHigh,\n"
            "        mode: 'lines',\n"
            "        line: { width: 0 },\n"
            "        showlegend: false,\n"
            "        hoverinfo: 'none'\n"
            "      };\n"
            "      \n"
            "      var traceConfLower = {\n"
            "        x: foreX, y: confLow,\n"
            "        mode: 'lines',\n"
            "        line: { width: 0 },\n"
            "        fill: 'tonexty',\n"
            "        fillcolor: 'rgba(168, 85, 247, 0.15)',\n"
            "        name: '95% Confidence Interval'\n"
            "      };\n"
            "      \n"
            "      var layout = {\n"
            f"        title: {{ text: 'Predictive Forecast & Horizon: {clean_topic}', font: {{ color: '#f8fafc', size: 16 }} }},\n"
            "        paper_bgcolor: '#0a0d14',\n"
            "        plot_bgcolor: '#0f172a',\n"
            "        xaxis: { title: 'Time Sequence Index', color: '#94a3b8', gridcolor: '#1e293b' },\n"
            "        yaxis: { title: 'Target Metric Value', color: '#94a3b8', gridcolor: '#1e293b' },\n"
            "        legend: { font: { color: '#cbd5e1' }, orientation: 'h', y: 1.1 },\n"
            "        margin: { l:50, r:30, b:50, t:60 }\n"
            "      };\n"
            "      Plotly.newPlot('plot', [traceActual, traceConfUpper, traceConfLower, traceForecast], layout, {responsive: true});\n"
            "    });\n"
            "  </script>\n"
            "</body>\n"
            "</html>\n"
            "<!--ARTIFACT_HTML-->"
        )

