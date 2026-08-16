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
            "1. Generate synthetic multi-feature time series or dataset if no external file is given.\n"
            "2. Split train/test (80/20), train model (Ridge/LinearRegression/RandomForest), predict future values.\n"
            "3. Print JSON formatted metrics at the end:\n"
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

        res_md = f"Prediction & Forecasting Analysis\n\n```python\n{code}\n```\n{metrics_md}"
        if viz_html:
            res_md += f"\n\n{viz_html}"

        return res_md
