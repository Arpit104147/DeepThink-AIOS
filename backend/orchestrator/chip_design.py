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
                f"Write complete, production-grade IEEE 1364-2001 Verilog code for this request:\n{prompt}\n\n"
                f"Architecture Plan:\n{arch_plan[:1500]}\n\n"
                f"STRICT VERILOG RULES:\n"
                f"1. Write standard IEEE 1364-2001 Verilog syntax:\n"
                f"```verilog\n"
                f"module bcd_counter (\n"
                f"    input wire clk,\n"
                f"    input wire reset,\n"
                f"    input wire load,\n"
                f"    input wire up_down,\n"
                f"    input wire [3:0] data_in,\n"
                f"    output reg [3:0] count,\n"
                f"    output reg carry_out\n"
                f");\n"
                f"    always @(posedge clk or posedge reset) begin\n"
                f"        if (reset) count <= 4'b0; ...\n"
                f"    end\n"
                f"endmodule\n"
                f"```\n"
                f"2. NEVER output pseudo-code or non-Verilog keywords like 'begins with', 'sub module', 'always on clock', or C-style braces '{{}}'. Use standard Verilog 'module', 'always @(posedge clk)', 'begin ... end', 'if ... else'.\n"
                f"3. Output TWO separate code blocks:\n"
                f"   - Block 1: Design module in ```verilog```\n"
                f"   - Block 2: Complete self-contained testbench with $dumpfile/$dumpvars in ```verilog```"
            )

        hdl_resp = orchestrator._strip_thinking(orchestrator._call_model(coder_llm, hdl_prompt, gen_tokens, gen_temp))
        hdl_clean = Sandbox.extract_code(hdl_resp)
        if not hdl_clean:
            hdl_clean = hdl_resp

        # Stage 3: 3D Semiconductor Layout Rendering
        if status_callback:
            status_callback("Stage 3: 3D Chip Visualization...", "info", "ornith", 75)

        viz_prompt = (
            "Write a COMPLETE, FULLY WORKING HTML page rendering an interactive 3D semiconductor chip architecture layout using Three.js.\n\n"
            "MANDATORY HTML STRUCTURE & RULES:\n"
            "1. Include CDN script tags in <head>:\n"
            "   <script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>\n"
            "   <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>\n"
            "2. CSS: html, body { margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background: #0d0d0d; font-family: sans-serif; }\n"
            "3. 3D SCENE SETUP:\n"
            "   - Declare 'var scene, camera, renderer, controls;' ONCE at the top of the <script> block.\n"
            "   - Initialize scene, camera, renderer. Append renderer.domElement to document.body.\n"
            "   - Set renderer size to window.innerWidth, window.innerHeight.\n"
            "   - Add AmbientLight and DirectionalLight.\n"
            "   - Add OrbitControls(camera, renderer.domElement) with enableDamping=true.\n"
            "4. 3D CHIP LAYERS (Stack with MeshPhongMaterial):\n"
            "   - Silicon Substrate: Gray box (size: 10x0.5x10, y: 0)\n"
            "   - Diffusion / Well: Blue & Red boxes (size: 2x0.2x2, y: 0.5)\n"
            "   - Polysilicon Gates: Green strips (size: 8x0.1x0.5, y: 0.8)\n"
            "   - Metal 1 Traces: Cyan tracks (size: 0.4x0.15x9, y: 1.2)\n"
            "   - Via Interconnects: Yellow cylinders (radius: 0.15, height: 0.4, y: 1.5)\n"
            "   - Metal 2 Traces: Orange tracks (size: 9x0.15x0.4, y: 1.8)\n"
            "5. GLASSMORPHIC HUD PANEL (top-right absolute overlay div): Show design title '4-Bit BCD Counter Chip Layout' and color legend for layers.\n"
            "6. RENDER LOOP: Implement animate() calling controls.update(), scene rotation, and renderer.render(scene, camera). RequestAnimationFrame(animate).\n"
            "7. RESIZE LISTENER: Add window.onresize handler updating camera aspect and renderer size.\n"
            "8. NO ES6 IMPORTS. Use global THREE and THREE.OrbitControls.\n"
            "9. VARIABLE SCOPE SAFETY: Use 'var' at top of script. NEVER re-declare 'let camera' or 'const camera' inside event listeners or functions to prevent SyntaxError.\n\n"
            f"Topic: {prompt}\n\n"
            "Output ONLY valid, complete HTML inside ```html``` blocks."
        )

        viz_resp = orchestrator._call_model(coder_llm, viz_prompt, max_tokens=2048, temperature=0.2)
        html_extract = Sandbox.extract_code(orchestrator._strip_thinking(viz_resp))
        if not html_extract and ("<!DOCTYPE" in viz_resp or "<html" in viz_resp):
            html_extract = viz_resp

        viz_html = ""
        if html_extract:
            viz_html = f"<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"

        output_parts = [
            f"### 🏗️ Stage 1: Architecture Decomposition\n\n{arch_plan}\n\n",
            f"### ⚡ Stage 2: HDL Design\n\n```verilog\n{hdl_clean}\n```\n\n"
        ]

        if viz_html:
            output_parts.append(f"### 🔬 Stage 3: 3D Chip Architecture Visualization\n\n{viz_html}")

        if not eda_tools['iverilog']:
            output_parts.append("\n\n### 📦 Missing EDA Tools\n```bash\nsudo apt-get install -y iverilog yosys ngspice\n```")

        if status_callback:
            status_callback("✅ Chip Design Pipeline complete!", "success", "system", 100)

        return "".join(output_parts)
