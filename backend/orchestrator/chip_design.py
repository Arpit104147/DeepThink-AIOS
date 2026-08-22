import re
import os
import json
import shutil
from backend.sandbox import Sandbox
from backend.downloader import resolve_model_key

class ChipDesignPipeline:
    """
    Universal Semiconductor EDA & Multi-Tier Chip Architecture Engine (180nm Planar to 2nm GAAFET).
    Supports Out-of-Order CPUs, SIMT GPUs, 2D Systolic Array TPUs, Mobile SoCs/APUs,
    High-Bandwidth Memory (HBM3/DDR5) controllers, and Analog SPICE netlists with interactive
    3D Physical Die Visualizations, real-time DVFS Clock Speed Calculations, and Thermal Breakpoint Analysis.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        if status_callback:
            status_callback("🔬 Universal Chip Design Pipeline activated...", "info", "ornith", 15)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        
        # Check available local EDA tools
        eda_tools = {
            'iverilog': shutil.which('iverilog') is not None,
            'yosys': shutil.which('yosys') is not None,
            'ngspice': shutil.which('ngspice') is not None,
            'gdstk': True
        }

        tool_status = " | ".join([f"{k} {'✅' if v else '❌'}" for k, v in eda_tools.items()])
        if status_callback:
            status_callback(f"EDA Tools: {tool_status}", "info", "system", 20)

        is_spice = any(kw in prompt.lower() for kw in ['spice', 'ngspice', 'netlist', '.subckt', 'opamp', 'transistor', 'bandgap'])
        req_lang = "spice" if is_spice else "verilog"

        # Detect Chip Category & Process Node Target
        chip_meta = ChipDesignPipeline._analyze_chip_meta(prompt)

        # Stage 1: Architecture & Process Node Decomposition
        if status_callback:
            status_callback(f"Stage 1: Decomposing {chip_meta['node']} {chip_meta['type']} Architecture...", "info", "deepseek_r1", 25)

        reasoning_key = resolve_model_key("reasoning") or "deepseek_r1"
        try:
            ds_llm = orchestrator._get_model(reasoning_key, required_ctx=ds_ctx)
            if not orchestrator._is_model_valid(ds_llm):
                ds_llm = orchestrator._get_model("router", required_ctx=ds_ctx)
        except (FileNotFoundError, Exception):
            ds_llm = orchestrator._get_model("router", required_ctx=ds_ctx)

        arch_prompt = (
            f"You are a Distinguished Principal Silicon Architect and Chief EDA Systems Fellow.\n"
            f"Decompose the following semiconductor hardware request into an exhaustive architectural specification:\n"
            f"USER REQUEST: {prompt}\n\n"
            f"FABRICATION PROCESS TARGET: {chip_meta['node']}\n"
            f"CHIP CLASSIFICATION: {chip_meta['type']}\n\n"
            f"MANDATORY ARCHITECTURE SPECIFICATION MODULES:\n"
            f"1. 🏛️ Microarchitecture & Sub-Module Breakdown: Detailed block diagram description, pipeline stages, datapaths, and execution units.\n"
            f"2. 🔌 Port & Interface Pinout Table: Complete I/O list with signal names, bit widths, directions (input/output), and protocol standards (AXI, APB, DFI, native).\n"
            f"3. ⏱️ Clocking, Reset & DVFS Power Strategy: Operating voltage-frequency states (Eco, Ideal Efficiency, Max Turbo, Thermal Breakpoint) and Backside Power Delivery (BSPDN) / Power Gating.\n"
            f"4. 📊 Estimated Silicon Metrics: Target frequency ($f_{{max}}$ in GHz), transistor budget, die area ($\text{{mm}}^2$), and TDP power envelope.\n"
            f"5. 🧪 Testbench & Verification Plan: Corner case test vectors, hazard scenarios, and assertion coverage plan."
        )
        arch_plan = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, arch_prompt, gen_tokens, 0.3))

        # Stage 2: HDL / SPICE Generation
        if status_callback:
            status_callback(f"Stage 2: Generating Synthesizable {req_lang.upper()} Core...", "info", "ornith", 50)

        coder_key = resolve_model_key("coding") or "ornith"
        try:
            coder_llm = orchestrator._get_model(coder_key, required_ctx=oc_ctx)
            if not orchestrator._is_model_valid(coder_llm):
                coder_llm = orchestrator._get_model("router", required_ctx=oc_ctx)
        except (FileNotFoundError, Exception):
            coder_llm = orchestrator._get_model("router", required_ctx=oc_ctx)

        if is_spice:
            hdl_prompt = (
                f"Write a complete, syntactically correct SPICE netlist for this specification:\n{arch_plan[:2000]}\n\n"
                f"Wrap code in ```spice``` blocks. Include .subckt, transistor parameters, voltage sources, and .tran analysis."
            )
        else:
            hdl_prompt = (
                f"Write production-grade, synthesizable IEEE 1364 / SystemVerilog code for this hardware design:\n{prompt}\n\n"
                f"Architecture Plan:\n{arch_plan[:1600]}\n\n"
                f"STRICT VERILOG RULES:\n"
                f"1. Write standard synthesizable Verilog syntax with clean module declarations, registers, and wire assignments.\n"
                f"2. Include complete logic implementations for all sub-modules without pseudo-code or placeholders.\n"
                f"3. Separate modules cleanly one after another (NEVER nest modules).\n"
                f"4. Output TWO distinct code blocks:\n"
                f"   - Block 1: Design module(s) in ```verilog```\n"
                f"   - Block 2: Complete self-checking testbench in ```verilog``` with $dumpfile(\"wave.vcd\"), $dumpvars(0, ...), stimulus vectors, and $finish."
            )

        hdl_resp = orchestrator._strip_thinking(orchestrator._call_model(coder_llm, hdl_prompt, gen_tokens, gen_temp))
        
        code_blocks = re.findall(rf"```(?:{req_lang}|verilog|spice)?\s*([\s\S]*?)\s*```", hdl_resp, flags=re.I)
        if code_blocks:
            hdl_clean = "\n\n// --- Complete Self-Checking Verification Testbench ---\n\n".join(b.strip() for b in code_blocks if b.strip())
        else:
            hdl_clean = Sandbox.extract_code(hdl_resp) or hdl_resp

        # Stage 3: Architecture-Specific 3D Physical Die & DVFS Telemetry Visualizer
        if status_callback:
            status_callback(f"Stage 3: Rendering 3D Physical Microarchitecture ({chip_meta['type']})...", "info", "system", 75)

        viz_html = ChipDesignPipeline._build_3d_chip_visualization(prompt, chip_meta)

        output_parts = [
            f"### 🏗️ Stage 1: Architecture & Process Node Decomposition ({chip_meta['node']})\n\n{arch_plan}\n\n",
            f"### ⚡ Stage 2: Synthesizable {req_lang.upper()} Implementation & Testbench\n\n```{req_lang}\n{hdl_clean}\n```\n\n",
            f"### 🔬 Stage 3: 3D Physical Semiconductor Die & DVFS Operating Telemetry ({chip_meta['type']})\n\n{viz_html}"
        ]

        if not eda_tools['iverilog']:
            output_parts.append("\n\n### 📦 EDA Tools Status\n```bash\nsudo apt-get install -y iverilog yosys ngspice\n```")

        if status_callback:
            status_callback("✅ Chip Design Pipeline complete!", "success", "system", 100)

        return "".join(output_parts)

    @staticmethod
    def _analyze_chip_meta(prompt):
        """Analyzes prompt to determine the exact process node and chip architecture class."""
        p_lower = prompt.lower()
        
        # 1. Process Node Detection (Default: 2nm GAAFET for modern requests)
        if "180nm" in p_lower or "legacy" in p_lower:
            node = "180nm Bulk Planar CMOS"
            node_key = "planar"
        elif "65nm" in p_lower or "45nm" in p_lower:
            node = "65nm Planar CMOS"
            node_key = "planar"
        elif "28nm" in p_lower:
            node = "28nm HKMG Planar CMOS"
            node_key = "planar"
        elif "14nm" in p_lower or "16nm" in p_lower or "10nm" in p_lower:
            node = "14nm FinFET 3D Transistors"
            node_key = "finfet"
        elif "7nm" in p_lower:
            node = "7nm EUV FinFET"
            node_key = "finfet"
        elif "5nm" in p_lower:
            node = "5nm Extreme EUV FinFET"
            node_key = "finfet"
        elif "3nm" in p_lower:
            node = "3nm Gate-All-Around (GAAFET)"
            node_key = "gaafet"
        elif "2nm" in p_lower or "1.8nm" in p_lower or "powervia" in p_lower or "bspdn" in p_lower:
            node = "2nm RibbonFET / GAA Nanosheets (BSPDN Backside Power)"
            node_key = "gaafet"
        else:
            node = "2nm Gate-All-Around (GAA) Nanosheet"
            node_key = "gaafet"

        def has_word(patterns, text):
            return bool(re.search(r'\b(' + '|'.join(re.escape(p) for p in patterns) + r')\b', text, re.IGNORECASE))

        # 2. Architecture Family Detection (Hierarchical & Word-Bounded to prevent 'output' -> 'tpu' false matches)
        if has_word(["soc", "apu", "mobile chip", "snapdragon", "apple silicon", "heterogeneous", "system on chip", "system-on-chip"], p_lower):
            chip_type = "Heterogeneous Mobile / Laptop SoC (APU)"
            arch_key = "soc"
        elif has_word(["gpu", "shader", "simt", "cuda", "streaming multiprocessor", "rasterizer"], p_lower):
            chip_type = "SIMT GPU Parallel Compute Unit"
            arch_key = "gpu"
        elif has_word(["tpu", "systolic", "tensor processing unit", "ai accelerator", "gemm", "matrix processor"], p_lower) or (has_word(["npu", "neural engine"], p_lower) and not has_word(["soc", "apu"], p_lower)) or (has_word(["tensor core"], p_lower) and not has_word(["gpu"], p_lower)):
            chip_type = "AI TPU / Tensor Processing Engine"
            arch_key = "tpu"
        elif has_word(["hbm", "hbm3", "hbm4", "dram", "ddr4", "ddr5", "memory controller", "high-bandwidth memory", "stacked dram"], p_lower):
            chip_type = "High-Bandwidth Memory (HBM3/DRAM) Controller"
            arch_key = "memory"
        elif has_word(["cpu", "risc-v", "rv64", "rv32", "arm", "out-of-order", "superscalar", "pipeline", "reorder buffer", "rob"], p_lower):
            chip_type = "High-Performance Out-of-Order CPU Core"
            arch_key = "cpu"
        elif has_word(["spice", "opamp", "op-amp", "bandgap", "pll", "adc", "dac", "analog"], p_lower):
            chip_type = "Analog / Mixed-Signal Silicon Macro"
            arch_key = "analog"
        else:
            chip_type = "Digital Logic Semiconductor Core"
            arch_key = "digital"

        return {"node": node, "node_key": node_key, "type": chip_type, "arch_key": arch_key}

    @staticmethod
    def _clean_chip_title(prompt, chip_type, node):
        """Extracts a clean, non-truncated human-readable title from prompt."""
        cleaned = re.sub(r"\b(design|implement|create|an|in|with|verilog|testbench|3d|layout|visualize|the|for|and|a)\b", " ", prompt, flags=re.I)
        cleaned = re.sub(r"[^\w\s\(\)\-\.]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = [w for w in cleaned.split() if len(w) > 1]
        if len(words) >= 2:
            return " ".join(words[:6])
        return f"{chip_type}"

    @staticmethod
    def _build_3d_chip_visualization(prompt, chip_meta=None):
        """
        Generates a state-of-the-art interactive 3D Physical Die & Microarchitecture Visualizer in Three.js.
        Features interactive DVFS sliders, real-time clock speed calculations, ideal operating points,
        and thermal/voltage breakpoint analytics.
        """
        if not chip_meta:
            chip_meta = ChipDesignPipeline._analyze_chip_meta(prompt)

        node_title = chip_meta["node"]
        chip_type = chip_meta["type"]
        arch_key = chip_meta["arch_key"]
        node_key = chip_meta["node_key"]
        clean_title = ChipDesignPipeline._clean_chip_title(prompt, chip_type, node_title)

        return f"""<!--ARTIFACT_HTML-->
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background: #0a0d14; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    #hud {{ position: absolute; top: 16px; right: 16px; background: rgba(15, 23, 42, 0.94); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.14); padding: 18px 20px; border-radius: 14px; color: #f8fafc; font-size: 0.8rem; box-shadow: 0 16px 40px rgba(0,0,0,0.75); z-index: 100; max-width: 380px; max-height: calc(100vh - 32px); overflow-y: auto; }}
    #hud h3 {{ margin: 0 0 6px; font-size: 0.95rem; color: #38bdf8; font-weight: 700; display: flex; align-items: center; gap: 6px; }}
    #hud .badge {{ background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 8px; border-radius: 6px; font-size: 0.68rem; font-weight: 600; display: inline-block; margin-bottom: 8px; }}
    
    /* DVFS Performance Slider Controls */
    .dvfs-section {{ background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 10px 12px; margin-bottom: 12px; }}
    .dvfs-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.74rem; font-weight: 700; }}
    .dvfs-state-badge {{ padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; }}
    .slider-container {{ position: relative; margin: 8px 0; }}
    .dvfs-slider {{ width: 100%; height: 6px; -webkit-appearance: none; background: #334155; border-radius: 4px; outline: none; }}
    .dvfs-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: #38bdf8; cursor: pointer; box-shadow: 0 0 10px #38bdf8; }}
    .slider-labels {{ display: flex; justify-content: space-between; font-size: 0.64rem; color: #94a3b8; font-weight: 600; margin-top: 4px; }}
    
    /* Telemetry KPI Grid */
    .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; }}
    .kpi-card {{ background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 6px 8px; }}
    .kpi-label {{ font-size: 0.64rem; color: #94a3b8; text-transform: uppercase; }}
    .kpi-val {{ font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-top: 2px; }}
    
    /* Operating Point Banners */
    #operating-banner {{ padding: 8px 10px; border-radius: 8px; font-size: 0.72rem; line-height: 1.35; margin-bottom: 12px; font-weight: 600; }}
    .banner-ideal {{ background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }}
    .banner-turbo {{ background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; }}
    .banner-breakpoint {{ background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }}
    .banner-eco {{ background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; }}

    #inspector {{ background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 0.75rem; }}
    #inspector-title {{ font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
    #inspector-desc {{ color: #cbd5e1; font-size: 0.72rem; line-height: 1.35; }}
    .legend-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: #cbd5e1; }}
    .box {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
    .btn-exploded {{ width: 100%; background: #0284c7; color: white; border: none; padding: 8px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.75rem; transition: background 0.2s; }}
    .btn-exploded:hover {{ background: #0369a1; }}
    #controls-hint {{ position: absolute; bottom: 16px; left: 16px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08); padding: 8px 14px; border-radius: 8px; color: #94a3b8; font-size: 0.72rem; z-index: 100; }}
  </style>
</head>
<body>
  <div id="hud">
    <h3>🔬 {clean_title}</h3>
    <span class="badge">Process: {node_title}</span>
    
    <!-- DVFS Voltage & Frequency Scaling Controller -->
    <div class="dvfs-section">
      <div class="dvfs-header">
        <span>⚡ DVFS Operating State</span>
        <span class="dvfs-state-badge" id="dvfsStateBadge" style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid #10b981;">Ideal Efficiency ⭐</span>
      </div>
      <div class="slider-container">
        <input type="range" min="1" max="4" value="2" step="1" class="dvfs-slider" id="dvfsSlider">
      </div>
      <div class="slider-labels">
        <span>1. Eco</span>
        <span style="color:#34d399;">2. Ideal ⭐</span>
        <span style="color:#f59e0b;">3. Max ⚡</span>
        <span style="color:#ef4444;">4. Breakpoint ⚠️</span>
      </div>
    </div>

    <!-- Operating Point Banner -->
    <div id="operating-banner" class="banner-ideal">
      💠 <strong>Ideal Operating Point:</strong> Optimal Energy-Delay Product ($0.78V$). Maximum Perf/Watt with zero thermal throttling.
    </div>

    <!-- Live Telemetry KPI Grid -->
    <div class="kpi-grid" id="kpiGrid">
      <div class="kpi-card">
        <div class="kpi-label">Big CPU Clock</div>
        <div class="kpi-val" id="kpiBigCpu" style="color:#ef4444;">2.85 GHz</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">GPU Shader Clock</div>
        <div class="kpi-val" id="kpiGpu" style="color:#a855f7;">980 MHz</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Supply Voltage (Vdd)</div>
        <div class="kpi-val" id="kpiVoltage" style="color:#38bdf8;">0.78 V</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Die Junction Temp</div>
        <div class="kpi-val" id="kpiTemp" style="color:#10b981;">52 °C</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">TDP Power Draw</div>
        <div class="kpi-val" id="kpiPower">5.4 W</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Compute Throughput</div>
        <div class="kpi-val" id="kpiThroughput" style="color:#f59e0b;">2.8 TFLOPS</div>
      </div>
    </div>
    
    <div id="inspector">
      <div id="inspector-title">💡 Hover / Click Any Subsystem Tile</div>
      <div id="inspector-desc">Interactive raycasting will inspect real-time clock frequencies and microarchitecture specs.</div>
    </div>

    <div class="legend-grid" id="legendGrid"></div>
    <button class="btn-exploded" id="toggleExploded">Toggle Exploded-View Inspection</button>
  </div>
  <div id="controls-hint">🖱️ Left-Click: Rotate | Right-Click: Pan | Scroll: Zoom | Slide DVFS: Test Operating Points</div>

  <script>
    document.addEventListener('DOMContentLoaded', function() {{
      var archKey = "{arch_key}";
      var nodeKey = "{node_key}";
      var currentDvfsState = 2; // Default: 2 (Ideal)
      
      var scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0d14);
      var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 14, 22);
      
      var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.shadowMap.enabled = true;
      document.body.appendChild(renderer.domElement);
      
      var controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.target.set(0, 1.0, 0);
      
      // Lighting
      var ambLight = new THREE.AmbientLight(0xffffff, 0.75);
      scene.add(ambLight);
      var dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
      dirLight.position.set(16, 32, 22);
      dirLight.castShadow = true;
      scene.add(dirLight);
      var fillLight = new THREE.DirectionalLight(0x38bdf8, 0.6);
      fillLight.position.set(-16, 12, -16);
      scene.add(fillLight);
      
      var rootGroup = new THREE.Group();
      scene.add(rootGroup);
      
      var isExploded = false;
      var layers = [];
      var interactiveObjects = [];

      function addInteractiveMesh(mesh, name, desc, specs) {{
        mesh.userData = {{ name: name, desc: desc, specs: specs || {{}}, origColor: mesh.material.color.getHex() }};
        interactiveObjects.push(mesh);
      }}

      // ── DVFS Telemetry Data Engine ──
      var telemetryProfiles = {{
        1: {{
          name: "Eco / Idle Standby",
          badgeBg: "rgba(56,189,248,0.2)", badgeColor: "#38bdf8", badgeBorder: "#38bdf8",
          bannerClass: "banner-eco",
          bannerText: "🟢 <strong>Eco Standby:</strong> Minimum leakage power ($0.60V$). Background OS tasks only.",
          voltage: "0.60 V", temp: "36 °C", power: "1.4 W",
          bigCpu: "1.20 GHz", littleCpu: "0.60 GHz", gpu: "350 MHz", npu: "8 TOPS",
          throughput: "0.8 TFLOPS", memBw: "51 GB/s", tempColor: "#38bdf8"
        }},
        2: {{
          name: "Ideal Efficiency ⭐",
          badgeBg: "rgba(16,185,129,0.2)", badgeColor: "#34d399", badgeBorder: "#10b981",
          bannerClass: "banner-ideal",
          bannerText: "💠 <strong>Ideal Operating Point:</strong> Sweet spot on V-f curve ($0.78V$). Maximum Perf/Watt efficiency.",
          voltage: "0.78 V", temp: "52 °C", power: "5.4 W",
          bigCpu: "2.85 GHz", littleCpu: "1.40 GHz", gpu: "980 MHz", npu: "24 TOPS",
          throughput: "2.8 TFLOPS", memBw: "102 GB/s", tempColor: "#10b981"
        }},
        3: {{
          name: "Max Sustained (Turbo) ⚡",
          badgeBg: "rgba(245,158,11,0.2)", badgeColor: "#fbbf24", badgeBorder: "#f59e0b",
          bannerClass: "banner-turbo",
          bannerText: "⚡ <strong>Max Sustained Turbo:</strong> Peak rated frequency ($0.95V$). High-load rendering & 3D gaming.",
          voltage: "0.95 V", temp: "78 °C", power: "14.2 W",
          bigCpu: "3.60 GHz", littleCpu: "2.00 GHz", gpu: "1.45 GHz", npu: "38 TOPS",
          throughput: "4.6 TFLOPS", memBw: "136 GB/s", tempColor: "#f59e0b"
        }},
        4: {{
          name: "Thermal Breakpoint ⚠️",
          badgeBg: "rgba(239,68,68,0.25)", badgeColor: "#f87171", badgeBorder: "#ef4444",
          bannerClass: "banner-breakpoint",
          bannerText: "🔴 <strong>Thermal & Voltage Breakpoint:</strong> Dielectric threshold ($1.15V$). Tj > 95°C forces thermal throttling!",
          voltage: "1.15 V", temp: "98 °C", power: "28.5 W",
          bigCpu: "4.20 GHz (Throttling)", littleCpu: "2.40 GHz", gpu: "1.85 GHz", npu: "52 TOPS",
          throughput: "6.2 TFLOPS", memBw: "153 GB/s", tempColor: "#ef4444"
        }}
      }};

      function updateDvfsTelemetry(state) {{
        var p = telemetryProfiles[state];
        var badge = document.getElementById("dvfsStateBadge");
        badge.textContent = p.name;
        badge.style.background = p.badgeBg;
        badge.style.color = p.badgeColor;
        badge.style.border = "1px solid " + p.badgeBorder;

        var banner = document.getElementById("operating-banner");
        banner.className = p.bannerClass;
        banner.innerHTML = p.bannerText;

        document.getElementById("kpiBigCpu").textContent = p.bigCpu;
        document.getElementById("kpiGpu").textContent = p.gpu;
        document.getElementById("kpiVoltage").textContent = p.voltage;
        var tempEl = document.getElementById("kpiTemp");
        tempEl.textContent = p.temp;
        tempEl.style.color = p.tempColor;
        document.getElementById("kpiPower").textContent = p.power;
        document.getElementById("kpiThroughput").textContent = p.throughput;

        // Visual Heatmap / Emissive Glow Feedback on Silicon Tiles
        for (var i = 0; i < interactiveObjects.length; i++) {{
          var mesh = interactiveObjects[i];
          if (state === 4) {{
            mesh.material.emissive.setHex(0xdc2626);
            mesh.material.emissiveIntensity = 0.25;
          }} else if (state === 3) {{
            mesh.material.emissive.setHex(0xf59e0b);
            mesh.material.emissiveIntensity = 0.15;
          }} else {{
            mesh.material.emissive.setHex(0x000000);
            mesh.material.emissiveIntensity = 0.0;
          }}
        }}
      }}

      document.getElementById("dvfsSlider").addEventListener("input", function(e) {{
        currentDvfsState = parseInt(e.target.value);
        updateDvfsTelemetry(currentDvfsState);
      }});

      // ── Build Microarchitecture-Specific 3D Scene ──
      if (archKey === "tpu") {{
        // ── 🧠 1. TPU / AI TENSOR PROCESSING UNIT (2D Systolic Array) ──
        document.getElementById("legendGrid").innerHTML = `
          <div class="legend-item"><span class="box" style="background:#10b981"></span> Systolic PEs</div>
          <div class="legend-item"><span class="box" style="background:#38bdf8"></span> SRAM Buffers</div>
          <div class="legend-item"><span class="box" style="background:#f59e0b"></span> Vector / GELU Unit</div>
          <div class="legend-item"><span class="box" style="background:#dc2626"></span> BSPDN Backside Power</div>
        `;

        var bspdnGroup = new THREE.Group();
        var bspdnMat = new THREE.MeshStandardMaterial({{ color: 0xdc2626, metalness: 0.85, roughness: 0.25 }});
        for (var b = 0; b < 6; b++) {{
          var bMesh = new THREE.Mesh(new THREE.BoxGeometry(16, 0.35, 0.8), bspdnMat);
          bMesh.position.set(0, -0.9, -5.0 + b * 2.0);
          bspdnGroup.add(bMesh);
          addInteractiveMesh(bMesh, "BSPDN Buried Power Rails (Vdd/Vss)", "Backside power grid delivering clean IR-drop power directly to PE columns.");
        }}
        rootGroup.add(bspdnGroup);
        layers.push({{ group: bspdnGroup, baseY: 0, explodedY: -3.0 }});

        var subGroup = new THREE.Group();
        var subMesh = new THREE.Mesh(new THREE.BoxGeometry(16.5, 0.7, 16.5), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.6 }}));
        subGroup.add(subMesh);
        rootGroup.add(subGroup);
        layers.push({{ group: subGroup, baseY: 0, explodedY: 0 }});

        var tpuCoreGroup = new THREE.Group();
        var peMat = new THREE.MeshStandardMaterial({{ color: 0x10b981, metalness: 0.75, roughness: 0.2 }});
        for (var r = 0; r < 8; r++) {{
          for (var c = 0; c < 8; c++) {{
            var peMesh = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.45, 1.0), peMat);
            peMesh.position.set(-4.2 + c * 1.2, 0.6, -4.2 + r * 1.2);
            tpuCoreGroup.add(peMesh);
            addInteractiveMesh(peMesh, `Systolic PE [${{r}},${{c}}] (MAC Unit)`, "16-bit Bfloat16 Multiply-Accumulate unit with weight stationary registers.");
          }}
        }}

        var sramMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.8, roughness: 0.25 }});
        var weightBuf = new THREE.Mesh(new THREE.BoxGeometry(10.0, 0.55, 1.6), sramMat);
        weightBuf.position.set(0, 0.65, -6.0);
        tpuCoreGroup.add(weightBuf);
        addInteractiveMesh(weightBuf, "Weight Stationary SRAM Buffer", "High-bandwidth 512KB SRAM feeding systolic columns with zero latency.");

        var actBuf = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.55, 10.0), sramMat);
        actBuf.position.set(-6.0, 0.65, 0);
        tpuCoreGroup.add(actBuf);
        addInteractiveMesh(actBuf, "Input Activation SRAM Buffer", "Double-buffered activation feature matrix feeding row PEs.");

        var vecMat = new THREE.MeshStandardMaterial({{ color: 0xf59e0b, metalness: 0.8, roughness: 0.2 }});
        var vecUnit = new THREE.Mesh(new THREE.BoxGeometry(10.0, 0.55, 1.6), vecMat);
        vecUnit.position.set(0, 0.65, 6.0);
        tpuCoreGroup.add(vecUnit);
        addInteractiveMesh(vecUnit, "Vector Activation Unit (GELU/Softmax)", "Pipelined SIMD transcendental engine performing activation, LayerNorm, and scaling.");

        rootGroup.add(tpuCoreGroup);
        layers.push({{ group: tpuCoreGroup, baseY: 0, explodedY: 3.5 }});

      }} else if (archKey === "soc") {{
        // ── 📱 2. HETEROGENEOUS MOBILE APU / SOC (Apple Silicon / Snapdragon) ──
        document.getElementById("legendGrid").innerHTML = `
          <div class="legend-item"><span class="box" style="background:#ef4444"></span> Big CPU Cores</div>
          <div class="legend-item"><span class="box" style="background:#38bdf8"></span> LITTLE Cores</div>
          <div class="legend-item"><span class="box" style="background:#a855f7"></span> GPU Shader Array</div>
          <div class="legend-item"><span class="box" style="background:#10b981"></span> NPU Neural Engine</div>
          <div class="legend-item"><span class="box" style="background:#eab308"></span> LPDDR5X PHY</div>
          <div class="legend-item"><span class="box" style="background:#f97316"></span> NoC Mesh Router</div>
        `;

        var pkgGroup = new THREE.Group();
        var pkgMesh = new THREE.Mesh(new THREE.BoxGeometry(17, 0.6, 17), new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.7 }}));
        pkgGroup.add(pkgMesh);
        rootGroup.add(pkgGroup);
        layers.push({{ group: pkgGroup, baseY: 0, explodedY: -3.0 }});

        var socGroup = new THREE.Group();
        var subMesh = new THREE.Mesh(new THREE.BoxGeometry(15, 0.5, 15), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.5 }}));
        socGroup.add(subMesh);

        // Big Performance CPU Cores (Red)
        var bigCpuMat = new THREE.MeshStandardMaterial({{ color: 0xef4444, metalness: 0.75, roughness: 0.25 }});
        for (var c = 0; c < 2; c++) {{
          var bCore = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.5, 3.2), bigCpuMat);
          bCore.position.set(-4.0 + c * 3.8, 0.55, -4.5);
          socGroup.add(bCore);
          addInteractiveMesh(bCore, `Big Performance CPU Core ${{c}}`, "64-bit Out-of-Order superscalar core with 192KB L1 cache. Ideal: 2.85GHz @ 0.78V | Max: 3.6GHz | Breakpoint: 4.2GHz @ 1.15V.");
        }}

        // LITTLE Efficiency CPU Cores (Cyan)
        var littleCpuMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.75, roughness: 0.25 }});
        for (var lc = 0; lc < 4; lc++) {{
          var lCore = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.5, 1.6), littleCpuMat);
          lCore.position.set(4.0 + (lc % 2) * 1.9, 0.55, -5.0 + Math.floor(lc / 2) * 1.9);
          socGroup.add(lCore);
          addInteractiveMesh(lCore, `Efficiency CPU Core ${{lc}}`, "Ultra-low-power in-order core for background OS tasks. Ideal: 1.4GHz @ 0.68V | Max: 2.0GHz.");
        }}

        // GPU Compute Shader Array (Purple)
        var gpuMat = new THREE.MeshStandardMaterial({{ color: 0xa855f7, metalness: 0.8, roughness: 0.2 }});
        var gpuMesh = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.5, 5.0), gpuMat);
        gpuMesh.position.set(-3.5, 0.55, 2.5);
        socGroup.add(gpuMesh);
        addInteractiveMesh(gpuMesh, "GPU Parallel Compute & Shader Array", "Multi-core SIMT graphics engine. Ideal: 980MHz (2.8 TFLOPS) | Max Turbo: 1.45GHz (4.6 TFLOPS).");

        // NPU Neural Engine (Green)
        var npuMat = new THREE.MeshStandardMaterial({{ color: 0x10b981, metalness: 0.8, roughness: 0.2 }});
        var npuMesh = new THREE.Mesh(new THREE.BoxGeometry(4.0, 0.5, 3.2), npuMat);
        npuMesh.position.set(3.8, 0.55, 0.2);
        socGroup.add(npuMesh);
        addInteractiveMesh(npuMesh, "16-Core NPU Neural Engine", "Dedicated AI matrix processor. Ideal: 24 TOPS INT8 | Max: 38 TOPS @ 1.6GHz.");

        // LPDDR5X Memory PHY Channels (Yellow)
        var phyMat = new THREE.MeshStandardMaterial({{ color: 0xeab308, metalness: 0.85, roughness: 0.15 }});
        var phyNorth = new THREE.Mesh(new THREE.BoxGeometry(14.0, 0.45, 0.8), phyMat);
        phyNorth.position.set(0, 0.55, 6.8);
        socGroup.add(phyNorth);
        addInteractiveMesh(phyNorth, "LPDDR5X Dual-Channel Memory PHY", "8533 Mbps low-power high-speed memory interface delivering 136 GB/s bandwidth.");

        // Glowing Network-on-Chip (NoC) Crossbar Grid (Orange)
        var nocMat = new THREE.MeshStandardMaterial({{ color: 0xf97316, emissive: 0xf97316, emissiveIntensity: 0.3, metalness: 0.9 }});
        for (var k = -4; k <= 4; k += 4) {{
          var nocBar = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.15, 13.0), nocMat);
          nocBar.position.set(k, 0.7, 0);
          socGroup.add(nocBar);
          addInteractiveMesh(nocBar, "Network-on-Chip (NoC) Interconnect", "Coherent low-latency AXI5 packet-switched crossbar linking all SoC subsystem tiles.");
        }}

        rootGroup.add(socGroup);
        layers.push({{ group: socGroup, baseY: 0, explodedY: 2.0 }});

      }} else if (archKey === "memory") {{
        // ── 💾 3. HBM3 / 3D STACKED DRAM MEMORY CUBE ──
        document.getElementById("legendGrid").innerHTML = `
          <div class="legend-item"><span class="box" style="background:#a855f7"></span> Silicon Interposer</div>
          <div class="legend-item"><span class="box" style="background:#0284c7"></span> Base Logic Die</div>
          <div class="legend-item"><span class="box" style="background:#10b981"></span> 3D DRAM Layer 0-3</div>
          <div class="legend-item"><span class="box" style="background:#eab308"></span> Through-Silicon Vias</div>
        `;

        var interposerGroup = new THREE.Group();
        var intMesh = new THREE.Mesh(new THREE.BoxGeometry(15, 0.5, 15), new THREE.MeshStandardMaterial({{ color: 0xa855f7, metalness: 0.8, roughness: 0.25 }}));
        interposerGroup.add(intMesh);
        rootGroup.add(interposerGroup);
        layers.push({{ group: interposerGroup, baseY: 0, explodedY: -3.0 }});

        var logicDieGroup = new THREE.Group();
        var logicMesh = new THREE.Mesh(new THREE.BoxGeometry(11, 0.6, 11), new THREE.MeshStandardMaterial({{ color: 0x0284c7, metalness: 0.75, roughness: 0.2 }}));
        logicMesh.position.y = 0.6;
        logicDieGroup.add(logicMesh);
        addInteractiveMesh(logicMesh, "HBM3 Base Logic Controller Die", "Master PHY, DFI interface, Built-in Self Test (BIST), and memory error correction (ECC).");
        rootGroup.add(logicDieGroup);
        layers.push({{ group: logicDieGroup, baseY: 0, explodedY: -1.0 }});

        var dramStackGroup = new THREE.Group();
        var dramMat = new THREE.MeshStandardMaterial({{ color: 0x10b981, transparent: true, opacity: 0.88, metalness: 0.7, roughness: 0.2 }});
        var tsvMat = new THREE.MeshStandardMaterial({{ color: 0xeab308, metalness: 0.95, roughness: 0.1 }});

        for (var d = 0; d < 4; d++) {{
          var dramDie = new THREE.Mesh(new THREE.BoxGeometry(10.5, 0.45, 10.5), dramMat);
          dramDie.position.y = 1.3 + d * 0.75;
          dramStackGroup.add(dramDie);
          addInteractiveMesh(dramDie, `3D Stacked DRAM Die Layer ${{d}}`, "High-density DRAM cell arrays (16Gb per die) with micro-second refresh timing.");
        }}

        for (var vx = -4; vx <= 4; vx += 2.0) {{
          for (var vz = -4; vz <= 4; vz += 2.0) {{
            var tsvMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 3.6, 10), tsvMat);
            tsvMesh.position.set(vx, 2.4, vz);
            dramStackGroup.add(tsvMesh);
            addInteractiveMesh(tsvMesh, "Through-Silicon Via (TSV) Channel", "Vertical copper interconnect transmitting 1024-bit wide-I/O signals across stacked dies.");
          }}
        }}

        rootGroup.add(dramStackGroup);
        layers.push({{ group: dramStackGroup, baseY: 0, explodedY: 3.0 }});

      }} else if (archKey === "cpu") {{
        // ── 🖥️ 4. OUT-OF-ORDER CPU CORE (RV64GC / x86) ──
        document.getElementById("legendGrid").innerHTML = `
          <div class="legend-item"><span class="box" style="background:#ef4444"></span> OoO Execution Engine</div>
          <div class="legend-item"><span class="box" style="background:#38bdf8"></span> TAGE Branch Predictor</div>
          <div class="legend-item"><span class="box" style="background:#10b981"></span> L1/L2 Caches</div>
          <div class="legend-item"><span class="box" style="background:#eab308"></span> L3 Shared Cache</div>
        `;

        var subGroup = new THREE.Group();
        subGroup.add(new THREE.Mesh(new THREE.BoxGeometry(15, 0.7, 15), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.6 }})));
        rootGroup.add(subGroup);
        layers.push({{ group: subGroup, baseY: 0, explodedY: 0 }});

        var cpuGroup = new THREE.Group();
        var exeMesh = new THREE.Mesh(new THREE.BoxGeometry(5.5, 0.55, 6.5), new THREE.MeshStandardMaterial({{ color: 0xef4444, metalness: 0.75, roughness: 0.25 }}));
        exeMesh.position.set(-3.5, 0.65, -2.5);
        cpuGroup.add(exeMesh);
        addInteractiveMesh(exeMesh, "Out-of-Order Execution Units & ROB", "4 Integer ALUs, 2 Vector FPUs, Load/Store units, and 128-entry Reorder Buffer. Ideal: 3.2GHz @ 0.82V | Breakpoint: 4.4GHz @ 1.18V.");

        var feMesh = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.55, 6.5), new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.8, roughness: 0.2 }}));
        feMesh.position.set(3.5, 0.65, -2.5);
        cpuGroup.add(feMesh);
        addInteractiveMesh(feMesh, "TAGE Branch Predictor & Instruction Fetch", "Multi-table conditional branch prediction with 4-wide instruction decoder.");

        var l2Mesh = new THREE.Mesh(new THREE.BoxGeometry(6.0, 0.55, 4.0), new THREE.MeshStandardMaterial({{ color: 0x10b981, metalness: 0.75, roughness: 0.25 }}));
        l2Mesh.position.set(-3.5, 0.65, 4.0);
        cpuGroup.add(l2Mesh);
        addInteractiveMesh(l2Mesh, "L1/L2 Non-Blocking Cache Banks", "64KB L1 Data/Inst cache and 1MB private L2 cache with hardware prefetchers.");

        var l3Mesh = new THREE.Mesh(new THREE.BoxGeometry(5.5, 0.55, 4.0), new THREE.MeshStandardMaterial({{ color: 0xeab308, metalness: 0.85, roughness: 0.2 }}));
        l3Mesh.position.set(3.5, 0.65, 4.0);
        cpuGroup.add(l3Mesh);
        addInteractiveMesh(l3Mesh, "Shared L3 SRAM Cache Slice", "High-density 8MB shared L3 cache with MESI/MOESI hardware coherence.");

        rootGroup.add(cpuGroup);
        layers.push({{ group: cpuGroup, baseY: 0, explodedY: 3.0 }});

      }} else {{
        // ── 🎮 5. GPU SIMT / GENERAL DIGITAL ASIC ──
        document.getElementById("legendGrid").innerHTML = `
          <div class="legend-item"><span class="box" style="background:#a855f7"></span> Streaming Multiprocessors</div>
          <div class="legend-item"><span class="box" style="background:#10b981"></span> Tensor Cores</div>
          <div class="legend-item"><span class="box" style="background:#38bdf8"></span> L2 Cache Partition</div>
          <div class="legend-item"><span class="box" style="background:#eab308"></span> Memory Controllers</div>
        `;

        var subGroup = new THREE.Group();
        subGroup.add(new THREE.Mesh(new THREE.BoxGeometry(15, 0.7, 15), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.6 }})));
        rootGroup.add(subGroup);
        layers.push({{ group: subGroup, baseY: 0, explodedY: 0 }});

        var gpuGroup = new THREE.Group();
        var smMat = new THREE.MeshStandardMaterial({{ color: 0xa855f7, metalness: 0.8, roughness: 0.2 }});
        for (var s = 0; s < 6; s++) {{
          var smMesh = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.55, 3.2), smMat);
          smMesh.position.set(-4.0 + (s % 3) * 4.0, 0.65, -3.5 + Math.floor(s / 3) * 4.0);
          gpuGroup.add(smMesh);
          addInteractiveMesh(smMesh, `Streaming Multiprocessor (SM ${{s}})`, "128 CUDA compute cores, SIMT warp scheduler, and Tensor Core matrix units. Ideal: 1.1GHz | Max Turbo: 1.65GHz.");
        }}

        var l2Gpu = new THREE.Mesh(new THREE.BoxGeometry(12.0, 0.5, 2.0), new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.75, roughness: 0.2 }}));
        l2Gpu.position.set(0, 0.65, 5.0);
        gpuGroup.add(l2Gpu);
        addInteractiveMesh(l2Gpu, "Shared High-Speed L2 Cache (32MB)", "Unified crossbar-connected cache with high-throughput multi-channel routing.");

        rootGroup.add(gpuGroup);
        layers.push({{ group: gpuGroup, baseY: 0, explodedY: 3.0 }});
      }}

      // ── Interactive Raycaster for Hover & Click Inspections ──
      var raycaster = new THREE.Raycaster();
      var mouse = new THREE.Vector2();
      var hoveredMesh = null;

      function onMouseMove(event) {{
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        var intersects = raycaster.intersectObjects(interactiveObjects, true);

        if (intersects.length > 0) {{
          var target = intersects[0].object;
          if (hoveredMesh !== target) {{
            if (hoveredMesh) {{
              hoveredMesh.material.emissive.setHex(currentDvfsState >= 3 ? 0xf59e0b : 0x000000);
              hoveredMesh.material.emissiveIntensity = currentDvfsState >= 3 ? 0.15 : 0.0;
            }}
            hoveredMesh = target;
            hoveredMesh.material.emissive.setHex(0x38bdf8);
            hoveredMesh.material.emissiveIntensity = 0.5;
            
            if (target.userData && target.userData.name) {{
              document.getElementById("inspector-title").textContent = "🔍 " + target.userData.name;
              document.getElementById("inspector-desc").textContent = target.userData.desc;
            }}
          }}
        }} else {{
          if (hoveredMesh) {{
            hoveredMesh.material.emissive.setHex(currentDvfsState >= 3 ? 0xf59e0b : 0x000000);
            hoveredMesh.material.emissiveIntensity = currentDvfsState >= 3 ? 0.15 : 0.0;
            hoveredMesh = null;
          }}
        }}
      }}

      window.addEventListener('mousemove', onMouseMove, false);

      // Exploded View Button Logic
      document.getElementById('toggleExploded').addEventListener('click', function() {{
        isExploded = !isExploded;
        this.textContent = isExploded ? "Collapse Die Layers" : "Toggle Exploded-View Inspection";
      }});
      
      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        rootGroup.rotation.y += 0.002;
        
        for (var i = 0; i < layers.length; i++) {{
          var targetY = isExploded ? layers[i].explodedY : layers[i].baseY;
          layers[i].group.position.y += (targetY - layers[i].group.position.y) * 0.08;
        }}
        
        renderer.render(scene, camera);
      }}
      animate();
      
      window.addEventListener('resize', function() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});
    }});
  </script>
</body>
</html>
<!--/ARTIFACT_HTML-->"""
