<div align="center">

# 🧠 DeepThink AIOS

### Fully Local Multi-Agent AI Operating System, Semiconductor EDA Studio & Autonomous Research Fleet

*An orchestrated fleet of specialized SLMs/LLMs running on consumer hardware — zero cloud dependencies.*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React 18](https://img.shields.io/badge/React-18.0%2B-blue)
![Vite 8](https://img.shields.io/badge/Vite-8.0%2B-646CFF)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Architecture Multi-Agent](https://img.shields.io/badge/Architecture-Multi--Agent-purple)
![Semiconductor 2nm](https://img.shields.io/badge/Semiconductor-2nm%20GAAFET-orange)
![Benchmarks 11 Suites](https://img.shields.io/badge/Benchmarks-11%20Suites-emerald)
![Security Hardened](https://img.shields.io/badge/Security-SSRF%20%7C%20PathTraversal%20Patched-success)
![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen)

</div>

---

DeepThink AIOS is an **enterprise-grade, fully local multi-agent AI Operating System** that routes user queries across specialized neural pipelines for software engineering, theoretical mathematical reasoning, machine learning forecasting tournaments, 2-volume university textbook authoring, 3D physical semiconductor layout synthesis (from 180nm planar to 2nm GAAFET), and hardware-accelerated benchmarking — all running locally with dynamic hardware scaling from Intel iGPUs to NVIDIA H100s.

---

### 💻 System Requirements

| Specification | Minimum (Lightweight LLMs) | Recommended (Full Swarm Fleet) |
|---|---|---|
| **System RAM** | **8 GB RAM** *(using 1.5B–3B quants)* | **16 GB – 32 GB RAM** *(for 7B–9B quants)* |
| **GPU VRAM** | Integrated iGPU / 2–4 GB VRAM | 8 GB – 16 GB+ VRAM *(Vulkan / CUDA / Metal)* |
| **Storage** | 10 GB free disk space | 30 GB SSD space for full local GGUF fleet |
| **OS** | Linux (Ubuntu/Debian), macOS (Apple Silicon), Windows 10/11 | Linux / Kaggle Cloud VM / macOS |

---

### ✨ Key Features & Specialized Pipelines

- **🎨 Human-Centered Modernized UI/UX** — Production-grade glassmorphic interface engineered with **Lucide SVG icons** across all navigation and modal controls, a **30+ Design Token CSS architecture**, a **unified accessible `<Modal>` primitive**, code blocks with **syntax highlighting (`vscDarkPlus`) & line numbers**, an **interactive empty-state with 1-click starter prompt cards**, dynamic **"Jump to Section" heading navigation**, and responsive mobile breakpoints (768px/900px).
- **🎓 Master 2-Volume Study Engine (`🎓 Study`)** — Pedagogical textbook synthesis powered by **DeepSeek-R1**. Authors 15–20 page master reference books with centered display KaTeX formulas (`$$ ... $$`), embedded **Mermaid architectural flowcharts**, pedagogical alert callouts (`> [!TIP]`, `> [!IMPORTANT]`), a **1-Page High-Yield Formula Cheat-Sheet**, multi-dimensional comparison tables, a **10-Problem Solved Question Bank**, and a **Standardized University Mock Exam**. Supports direct PDF/Slide ingestion.
- **🔮 Maximum Power Prediction Engine (`🔮 Predict`)** — High-precision time-series forecasting across Financial Markets, Climate/Weather, Energy & Battery State-of-Health (SOH) degradation over 500 charge-discharge cycles, and Cloud Telemetry. Features a **14-Signal Alpha Feature Space**, an **8-Algorithm Tournament** with **Bayesian Softmax Inverse-Loss Stacking** ($\beta = 3.5$), and **Conformal Prediction Probabilistic Uncertainty Bands** ($80\%$ & $95\%$ corridors in Plotly).
- **🔬 Scientific Semiconductor EDA & 3D Physical Die Visualizer** — Synthesizes synthesizable Verilog/SystemVerilog HDL and SPICE netlists across all process nodes (180nm Planar to 2nm RibbonFET / GAA Nanosheets with Backside Power Delivery). Supports 6 full silicon architectures with procedural WebGL/Three.js silicon die rendering and a live clock stepping toolbar.
- **⚡ Zero-Hallucination Category-Aware Mathematical Reasoning (`⚡ Reason`)** — Solves complex theoretical derivations across General Relativity, Real/Complex Analysis, Quantum Mechanics, and General Mathematics. Features **dynamic category prompt routing**, publication-grade centered KaTeX display formulas (`$$ ... $$`), automatic line-break sanitization, and verified closed-form fallbacks (exact 9 Schwarzschild Christoffel symbols, Kretschmann scalar $K = \frac{48G^2M^2}{c^4 r^6}$, and the Bose-Einstein Riemann Zeta integral $\int_0^\infty \frac{x^3}{e^x - 1} dx = \frac{\pi^4}{15}$).
- **💻 Production-Grade Autonomous Coding Pipeline** — Multi-phase software engineering with Big-O complexity optimization ($O(N)$ / $O(N \log N)$), automated **AST Static Analysis Linting (SAST)**, strict type annotations, Google-style docstrings, C++17 shared mutex concurrency (`std::shared_mutex`, lock-free SPSC queues), and multi-language execution sandboxes.
- **🛡️ Enterprise Security & Air-Gap Hardening** — Full sandbox protection with **SSRF prevention** (blocking private, loopback, and link-local CIDR ranges), **Path Traversal guards** in git workspace commits, **Supply-Chain Auto-pip Allowlisting** (restricted to ~35 vetted scientific packages), **TarSlip/ZipSlip guards** in archive extraction, and `shell=False` process safety.
- **📊 Benchmark Studio & Telemetry Dashboard** — Parallel evaluation across **11 standard suites** (HumanEval, MBPP, GSM8K, MATH, GPQA, AIME, MuSR, MMLU-Pro, SWE-bench Lite, SWE-bench Pro, SearchQA) with real-time scoring vs GPT-4o and Claude 3.5 Sonnet baselines, live throughput ($\text{tok/s}$), and JSON report exports.
- **🌐 100% Keyless Multi-Tier Web Search (`🌐 Search` & `🔬 Extreme`)** — Scrapes live financial quotes, real-time weather, and multi-source academic publications with deep synthesis without API keys.
- **⚡ Elastic VRAM Management (EVM) & DMA** — Zero-cost dynamic model hot-swapping between System RAM and GPU VRAM with thread-safe progress locks and bounded SQLite vector memory recall (`LIMIT 500`).

---

### 🌟 Flagship Golden Prompts Showcase

| Pipeline | Example Prompt to Try in the UI | Key Output Artifacts |
|---|---|---|
| 🔬 **Chip Design** | `Design a 2nm GAAFET TPU with an 8x8 Systolic Array of Bfloat16 PEs, Backside Power Delivery (BSPDN), synthesizable Verilog, and 3D silicon layout.` | Synthesizable RTL, self-checking testbench, SPICE deck, and interactive 3D WebGL silicon die with live clock stepping |
| 🔮 **Prediction** | `Predict lithium-ion battery State-of-Health (SOH) degradation over 500 charge-discharge cycles under high ambient temperature stress.` | Multi-feature polynomial & Bayesian stacking forecast, SOH capacity decay curve, and Plotly 80%/95% confidence corridor |
| ⚡ **Reasoning (Math)**| `Compute the exact definite integral of x^3 / (e^x - 1) from x=0 to infinity, showing the Riemann zeta function connection and full series expansion.` | Publication-grade KaTeX derivation ($$ ... $$) demonstrating $\zeta(4) = \frac{\pi^4}{90}$ and final value $\frac{\pi^4}{15}$ |
| ⚡ **Reasoning (GR)**| `Derive the Schwarzschild metric from Einstein's field equations R_uv = 0, computing all Christoffel symbols, Newtonian limit, and Kretschmann invariant.` | Full step-by-step tensor derivation, exact 9 non-zero Christoffel symbols, and Kretschmann scalar invariant proof |
| 💻 **Coding** | `Implement a lock-free SPSC ring buffer queue in Rust with atomic operations, memory ordering, doc tests, and cache-line padding.` | Production-grade Rust module, AST verified, memory-order annotated, passing unit test harness |
| 🎓 **Study** | `Teach me Transformer Attention Mechanism (Self-Attention, Multi-Head, KV-Cache) from first principles as an exhaustive graduate textbook.` | 2-Volume Master Treatise with Mermaid architecture diagram, display math, 1-page formula cheat-sheet, 10 solved problems & mock exam |

---

## 🤖 System Model Fleet

| System Role | Model Name | HuggingFace Repo ID | GGUF Filename & Projector | Quants |
|---|---|---|---|---|
| **Master Router** | **Phi-3.5-Mini** / **Llama-3.2** | `bartowski/Phi-3.5-mini-instruct-GGUF` | `Phi-3.5-mini-instruct-Q6_K.gguf` | Q6_K / Q4_K |
| **Agentic Coder** | **Qwen2.5-Coder / Ornith** | `deepreinforce-ai/Ornith-1.0-9B-GGUF` | `ornith-1.0-9b-Q6_K.gguf` | Q6_K / Q4_K |
| **Reasoning Engine** | **DeepSeek-R1 Distill** | `unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF` | `DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf` | Q6_K / Q4_K |
| **Syntax Linter** | **VibeThinker 3B** | `prithivMLmods/VibeThinker-3B-GGUF` | `VibeThinker-3B.Q6_K.gguf` | Q6_K / Q4_K |
| **Vision & OCR** | **Qwen-2.5-VL / Qwen3-VL** | `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` | `Qwen2.5-VL-7B-Instruct-UD-Q6_K_XL.gguf` + `mmproj-BF16.gguf` | Q6_K / Q4_K / Q8_0 |

---

## 🔀 Pipeline Architecture

```mermaid
flowchart TD
    %% ── TOP-LEVEL INGESTION ──
    USER([User Prompt / Image / PDF]) --> MODE_CHECK{"Pipeline Mode Selected?"}
    
    MODE_CHECK -->|🎓 Study| STUDY_PIPE["Study Pipeline: 2-Volume Master Curriculum + Mermaid + 10 Problems + Exam"]
    MODE_CHECK -->|🔮 Predict| PREDICT_PIPE["Predict Pipeline: 14-Signal Alpha Features + 8-Model Bayesian Tournament"]
    MODE_CHECK -->|🔬 Extreme| EXTREME_PIPE["Extreme WebSearch: Multi-Source Academic Survey"]
    MODE_CHECK -->|🌐 Search| SEARCH_PIPE["Simple Search: Live Real-Time Web Data"]
    MODE_CHECK -->|📊 Benchmark| BENCH_PIPE["Benchmark Studio: 11-Suite Parallel Worker Evaluation"]
    MODE_CHECK -->|Auto / Prompt| ROUTER["Fast-Path & Router Intent Classifier"]

    %% ── Intent Classification Branches ──
    ROUTER --> PATH_CODING["1. CODING (AST Linter + O(N) Complexity + Type Hints)"]
    ROUTER --> PATH_REASONING["2. REASONING (SymPy CAS Grounding & Kretschmann Scalar)"]
    ROUTER --> PATH_CHIP["3. CHIP DESIGN (2nm GAAFET, BSPDN & 3D Live Clock)"]
    ROUTER --> PATH_VISION["4. VISION & OCR"]
    ROUTER --> PATH_SIMPLE["5. DIRECT / CONVERSATIONAL"]

    %% ── Execution Pathways ──
    STUDY_PIPE --> STUDY_OUT["Master Textbook + 1-Page Cheat Sheet + 10 Solved Problems + Mock Exam"]
    PREDICT_PIPE --> PREDICT_OUT["8-Model Bayesian Stacking + 80%/95% Conformal Fan Chart"]
    BENCH_PIPE --> BENCH_OUT["Real-Time Throughput / Accuracy Telemetry vs Baselines"]
    PATH_CODING --> CODE_SB{"Execution Sandbox"} --> CODE_PASS["Verified Working Polyglot Code"]
    PATH_REASONING --> PAL_SB{"SymPy / Math Sandbox"} --> PAL_PASS["Verified KaTeX Proof ($$ ... $$)"]
    PATH_CHIP --> EDA_SB{"Icarus / Yosys / SPICE"} --> CHIP_OUT["Verilog Module + Interactive 3D Die Visualizer"]
```

---

## ⚡ Quick Start

### 1. Local System Startup

```bash
git clone https://github.com/Arpit104147/DeepThink-AIOS.git
cd DeepThink-AIOS

# Launch servers (Backend on :8000, Web UI on :5173)
./start.sh
```

Open **`http://localhost:5173`** in your browser.

---

### 2. Kaggle / Remote Cloud GPU Setup (Continuous Runner)

Paste and run this complete Python script inside a single Kaggle Notebook cell:

```python
# =========================================================================
# 🚀 DEEPTHINK-AIOS: KAGGLE BACKEND + CLOUDFLARE TUNNEL (CONTINUOUS RUNNER)
# =========================================================================

import os, subprocess, time, re, sys

# 1. Clone or Auto-Update Repository
if os.path.exists("/kaggle/working/DeepThink-AIOS"):
    os.chdir("/kaggle/working/DeepThink-AIOS")
    subprocess.run(["git", "pull", "origin", "main"], check=True)
else:
    os.chdir("/kaggle/working")
    subprocess.run(["git", "clone", "https://github.com/Arpit104147/DeepThink-AIOS.git"], check=True)
    os.chdir("/kaggle/working/DeepThink-AIOS")

# 1.5 Install System EDA Chip Design Tools (Icarus Verilog, Yosys, NGSPICE, KLayout)
print("🔬 Installing System EDA Chip Design Tools (iverilog, yosys, ngspice, klayout)...", flush=True)
subprocess.run(["apt-get", "update", "-y", "-q"], check=False)
subprocess.run(["apt-get", "install", "-y", "-q", "iverilog", "yosys", "ngspice", "klayout"], check=False)

# 2. Install dependencies & Pre-compile CUDA llama-cpp-python for Kaggle GPU
print("⚡ Installing requirements & pre-compiling CUDA llama-cpp-python for Kaggle GPU...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)

try:
    import torch
    if torch.cuda.is_available():
        print("🔥 Pre-installing CUDA-accelerated llama-cpp-python for Kaggle GPU...", flush=True)
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env["FORCE_CMAKE"] = "1"
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python", "--force-reinstall", "--no-cache-dir", "-q"
        ], env=env, check=False)
except Exception as e:
    print(f"⚠️ CUDA setup note: {e}")

# 3. Download Cloudflare Tunnel binary
subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/tmp/cloudflared"], check=True)
subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=True)

# 4. Launch FastAPI Backend
print("⏳ Launching FastAPI Backend on Port 8000...", flush=True)
backend_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"])

# 5. Create Cloudflare Tunnel
print("🌐 Creating Secure Cloudflare Tunnel...", flush=True)
tunnel_proc = subprocess.Popen(
    ["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(4)

# 6. Extract & Print Public URL
public_url = None
for line in tunnel_proc.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        print("\n" + "="*72, flush=True)
        print("🎉 YOUR KAGGLE BACKEND PUBLIC URL:", flush=True)
        print(f"👉 {public_url}", flush=True)
        print("="*72, flush=True)
        print("📌 COPY the URL above and paste it into your local Laptop Frontend!", flush=True)
        print("="*72 + "\n", flush=True)
        break

# 7. Continuous Heartbeat to keep Kaggle alive overnight
start_time = time.time()
print("⚡ Backend is ACTIVE & serving requests continuously...", flush=True)

try:
    while True:
        time.sleep(120)
        elapsed_min = int((time.time() - start_time) // 60)
        print(f"💓 [HEARTBEAT - {elapsed_min}m elapsed] DeepThink-AIOS Backend Running | URL: {public_url}", flush=True)
except KeyboardInterrupt:
    print("Stopping server...", flush=True)
    backend_proc.terminate()
    tunnel_proc.terminate()
```

Copy the printed `https://xxxx.trycloudflare.com` URL, open **`http://localhost:5173`** in your local browser, click **Settings (`⚙️`)**, and paste the URL into **Server URL**.

---

## 🧪 Comprehensive Automated Verification Suites

The system includes automated pre-flight audit suites that test every Python file, frontend build, mathematical post-processor, and router fast-path:

```bash
# 1. Run 6-Stage Deep Line-by-Line System Audit
python3 backend/test_deep_line_by_line_audit.py

# 2. Run 7-Subsystem End-to-End Project Audit
python3 backend/test_full_project_audit.py

# 3. Test React 18 / Vite Production Bundle
npm run build --prefix frontend
```

---

## 💻 Tech Stack

- **Backend:** FastAPI, Uvicorn, Python 3.10+, PyTorch, Vulkan SDK, `llama-cpp-python`, ChromaDB, PyPDF, Scikit-Learn, Icarus Verilog, Yosys, NGSPICE, SymPy, NumPy, Pandas
- **Frontend:** React 18, Vite 8, Lucide React (`lucide-react`), React Syntax Highlighter (`react-syntax-highlighter`), KaTeX Typography, Plotly.js, Three.js / WebGL, CSS Design Tokens
- **Hardware Acceleration:** Vulkan Compute (NVIDIA, AMD, Intel iGPU/dGPU), NVIDIA CUDA, Apple Metal MPS, Multi-Core CPU Fallback
- **Security:** Safe SAST Execution Sandbox, SSRF Defense, Path Traversal Defense, Supply-Chain Package Allowlisting, TarSlip/ZipSlip Guards

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
