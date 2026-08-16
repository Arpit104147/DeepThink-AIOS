import re
import os
import shutil
from backend.sandbox import Sandbox

class ChipDesignPipeline:
    """Verilog HDL & SPICE EDA Simulation Pipeline with 3D Semiconductor Layout Rendering."""

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        if status_callback:
            status_callback("🔬 Chip Design Pipeline activated...", "info", "ornith", 15)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        
        # Check available EDA tools
        eda_tools = {
            'iverilog': shutil.which('iverilog') is not None,
            'yosys': shutil.which('yosys') is not None,
            'ngspice': shutil.which('ngspice') is not None,
            'gdstk': True
        }

        tool_status = " | ".join([f"{k} {'✅' if v else '❌'}" for k, v in eda_tools.items()])
        if status_callback:
            status_callback(f"EDA Tools: {tool_status}", "info", "system", 20)

        is_spice = any(kw in prompt.lower() for kw in ['spice', 'ngspice', 'netlist', '.subckt', 'opamp', 'transistor'])
        req_lang = "spice" if is_spice else "verilog"

        # Stage 1: Architecture Decomposition
        if status_callback:
            status_callback("Stage 1: Architecture Decomposition...", "info", "deepseek_r1", 25)

        ds_llm = orchestrator._get_model("deepseek_r1", required_ctx=ds_ctx)
        arch_prompt = (
            f"You are a principal semiconductor architect.\n"
            f"Decompose the following hardware request into a detailed block specification:\n"
            f"{prompt}\n\n"
            f"Include: Sub-module breakdown, Input/Output port list with bit widths, Clocking & reset strategy, "
            f"Interconnection topology, and Testbench test vector scenarios."
        )
        arch_plan = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, arch_prompt, gen_tokens, 0.4))

        # Stage 2: HDL / SPICE Generation
        if status_callback:
            status_callback("Stage 2: Generating HDL Code...", "info", "ornith", 45)

        coder_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)
        if is_spice:
            hdl_prompt = (
                f"Write a complete, syntactically correct SPICE netlist for this specification:\n{arch_plan[:2000]}\n\n"
                f"Wrap code in ```spice``` blocks. Include .subckt, transistor parameters, voltage sources, and .tran analysis."
            )
        else:
            hdl_prompt = (
                f"Write complete, production-grade Verilog-2001 code for this specification:\n{arch_plan[:2000]}\n\n"
                f"REQUIREMENTS:\n"
                f"1. Output TWO separate code blocks:\n"
                f"   - Block 1: Design module in ```verilog```\n"
                f"   - Block 2: Complete self-contained testbench with $dumpfile/$dumpvars in ```verilog```\n"
                f"2. Use valid Verilog syntax (`always @(posedge clock) begin ... end`). Do NOT use C-style syntax."
            )

        hdl_resp = orchestrator._strip_thinking(orchestrator._call_model(coder_llm, hdl_prompt, gen_tokens, gen_temp))
        hdl_clean = Sandbox.extract_code(hdl_resp)

        # Stage 3: 3D Semiconductor Layout Rendering
        if status_callback:
            status_callback("Stage 3: 3D Chip Visualization...", "info", "ornith", 75)

        viz_prompt = (
            "Write a COMPLETE HTML page rendering an interactive 3D semiconductor chip layout.\n\n"
            "RULES:\n"
            "1. Three.js r128 CDN: https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js\n"
            "2. OrbitControls CDN: https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\n"
            "3. Dark background #0d0d0d. MeshPhongMaterial with distinct colors per layer.\n"
            "4. Show stacked semiconductor layers with gaps:\n"
            "   - Silicon Substrate (gray, y=0)\n"
            "   - N-Well/P-Well (blue/red, y=0.5)\n"
            "   - Polysilicon gates (green, y=1.0)\n"
            "   - Metal 1 (cyan, y=1.5)\n"
            "   - Via (yellow spheres, y=2.0)\n"
            "   - Metal 2 (orange, y=2.5)\n"
            "5. Glassmorphic info panel (top-right) with design name and layer legend\n"
            "6. Auto-rotation with OrbitControls\n"
            "7. Ambient + directional lighting\n"
            "8. No ES6 imports. Use global THREE and THREE.OrbitControls.\n"
            "9. VARIABLE SCOPE SAFETY: Use 'let' or 'var' for geometry/material variables inside loops. NEVER redeclare 'const geometry' or 'const material' inside loops.\n\n"
            f"Design Context:\n{arch_plan[:1000]}\n\n"
            "Output ONLY complete HTML in ```html``` blocks."
        )

        viz_resp = orchestrator._call_model(coder_llm, viz_prompt, max_tokens=2048, temperature=0.2)
        html_extract = Sandbox.extract_code(orchestrator._strip_thinking(viz_resp))

        viz_html = ""
        if html_extract and "THREE" in html_extract:
            viz_html = f"<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"

        output_parts = [
            f"🏗️ Stage 1: Architecture Decomposition\n\n{arch_plan}\n\n",
            f"⚡ Stage 2: HDL Design\n\n```verilog\n{hdl_clean}\n```\n\n"
        ]

        if viz_html:
            output_parts.append(f"🔬 Stage 3: 3D Chip Architecture Visualization\n\n{viz_html}")

        if not eda_tools['iverilog']:
            output_parts.append("\n\n📦 Missing EDA Tools\n```bash\nsudo apt-get install -y iverilog yosys ngspice\n```")

        if status_callback:
            status_callback("✅ Chip Design Pipeline complete!", "success", "system", 100)

        return "".join(output_parts)
