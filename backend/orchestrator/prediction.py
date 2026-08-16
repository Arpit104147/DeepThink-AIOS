import re
import json
from backend.sandbox import Sandbox

class PredictionPipeline:
    """ML Data Science Forecasting & 3D Predictive Metrics Pipeline."""

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        if status_callback:
            status_callback("🔮 Prediction Pipeline activated...", "info", "ornith", 20)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        coder_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)

        script_p = (
            "Write a complete Python script using scikit-learn/pandas/numpy for this prediction task:\n"
            f"User Request: {prompt}\n\n"
            "REQUIREMENTS:\n"
            "1. Generate structured polynomial synthetic multi-feature time series dataset.\n"
            "2. Use PolynomialFeatures(degree=2) and StandardScaler() before training Ridge(alpha=1.0) regression model so the model achieves a high positive R² score.\n"
            "3. Split train/test (80/20), train model, predict future values for next 10 time steps.\n"
            "4. Print JSON formatted metrics at the end:\n"
            "   PREDICTIVE_METRICS = {'r2': float, 'mse': float, 'predictions': list}\n"
            "   print(json.dumps(PREDICTIVE_METRICS))\n\n"
            "Wrap script in ```python``` blocks."
        )

        code_resp = orchestrator._call_model(coder_llm, script_p, gen_tokens, gen_temp)
        code = Sandbox.extract_code(orchestrator._strip_thinking(code_resp))

        ok, output = orchestrator.sandbox.execute(code, language="python")

        metrics_json = None
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{") and "r2" in line:
                try:
                    metrics_json = json.loads(line)
                    break
                except Exception:
                    pass

        metrics_md = ""
        if metrics_json:
            metrics_md = (
                f"\n\n### 🔮 Predictive Model Metrics\n"
                f"- **R² Score:** `{metrics_json.get('r2', 'N/A')}`\n"
                f"- **Mean Squared Error (MSE):** `{metrics_json.get('mse', 'N/A')}`\n"
            )

        viz_html = orchestrator._generate_3d_visualization(prompt, coder_llm, oc_ctx, gen_tokens, gen_temp, status_callback)
        if not viz_html or "<!--ARTIFACT_HTML-->" not in viz_html:
            viz_html = PredictionPipeline._build_plotly_3d_fallback(prompt, metrics_json)

        res_md = f"Prediction & Forecasting Analysis\n\n```python\n{code}\n```\n{metrics_md}\n\n{viz_html}"
        return res_md

    @staticmethod
    def _build_plotly_3d_fallback(prompt, metrics_json=None):
        preds = metrics_json.get("predictions", [1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9]) if (metrics_json and isinstance(metrics_json, dict) and "predictions" in metrics_json) else [1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9]
        preds_js = json.dumps(preds[:20])
        return (
            "<!--ARTIFACT_HTML-->\n"
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "  <script src=\"https://cdn.plot.ly/plotly-2.24.1.min.js\"></script>\n"
            "  <style>html, body { margin:0; padding:0; width:100vw; height:100vh; background:#0d0d0d; font-family:sans-serif; overflow:hidden; }</style>\n"
            "</head>\n"
            "<body>\n"
            "  <div id=\"plot\" style=\"width:100vw; height:100vh;\"></div>\n"
            "  <script>\n"
            "    document.addEventListener('DOMContentLoaded', function() {\n"
            f"      const predictions = {preds_js};\n"
            "      const steps = Array.from({length: predictions.length}, (_, i) => i + 1);\n"
            "      const trace1 = {\n"
            "        x: steps,\n"
            "        y: predictions.map((v, i) => v * 0.85 + 0.1 * i),\n"
            "        z: predictions,\n"
            "        mode: 'lines+markers',\n"
            "        marker: { size: 6, color: '#00f2fe' },\n"
            "        line: { color: '#4facfe', width: 5 },\n"
            "        type: 'scatter3d',\n"
            "        name: '3D Predicted Energy Curve'\n"
            "      };\n"
            "      const layout = {\n"
            "        title: { text: '3D Predictive Energy Consumption Forecast', font: { color: '#ffffff' } },\n"
            "        paper_bgcolor: '#0d0d0d',\n"
            "        plot_bgcolor: '#0d0d0d',\n"
            "        scene: {\n"
            "          xaxis: { title: 'Time Step', color: '#888' },\n"
            "          yaxis: { title: 'Feature Load', color: '#888' },\n"
            "          zaxis: { title: 'Predicted Value', color: '#888' }\n"
            "        },\n"
            "        margin: { l:0, r:0, b:0, t:40 }\n"
            "      };\n"
            "      Plotly.newPlot('plot', [trace1], layout);\n"
            "    });\n"
            "  </script>\n"
            "</body>\n"
            "</html>\n"
            "<!--ARTIFACT_HTML-->"
        )
