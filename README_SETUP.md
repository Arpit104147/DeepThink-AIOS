# DeepThink AIOS Setup & Hardware Acceleration Guide

Because DeepThink AIOS runs fully local 7B reasoning agents, it requires **hardware acceleration (GPU)**. 

The Python code is OS-agnostic, but PyTorch and `llama.cpp` require different installation commands depending on if you are using an Apple MacBook, a Windows NVIDIA PC, or a Linux Intel machine.

Please follow the instructions for your specific operating system to ensure the models run on your GPU rather than your CPU.

---

## Step 1: System Prerequisites (For Polyglot Sandbox)
DeepThink AIOS can execute code in multiple languages. For this to work, ensure your system has the necessary native compilers installed.

*   **Mac:** Run `xcode-select --install` in terminal (installs `gcc`/`g++`).
*   **Linux:** Run `sudo apt install build-essential openjdk-17-jdk nodejs` (installs C/C++, Java, and Node.js).
*   **Windows:** Install [MinGW](https://www.mingw-w64.org/) for C/C++ and Node.js for Javascript.

*(Note: If a compiler is missing, the AI engine will not crash; it will gracefully fall back to Python execution).*

### 🔧 EDA Toolchain (Optional — For Chip Design Pipeline)
If you plan to use the Chip Design EDA Sandbox for Verilog/SPICE hardware simulation:

```bash
# 🐧 Linux / Kaggle Notebooks:
!apt-get update -y && !apt-get install -y iverilog yosys ngspice klayout

# 🍎 Mac (Homebrew):
brew install icarus-verilog yosys ngspice

# 🪟 Windows (Chocolatey / WSL2):
choco install icarus-verilog ngspice
```

*(These tools are optional. If missing, DeepThink AIOS will gracefully output the code and 3D chip architecture layout while skipping hardware binary simulation).*

### 🔒 Kernel Isolation Prerequisites (Optional — Enhanced Security)
The sandbox automatically uses Linux kernel namespaces (`unshare`) for network/process isolation when available. No extra setup is needed on most modern Linux systems.

*   **Root users:** Full namespace isolation (`--net --pid --ipc`) is automatically enabled.
*   **Non-root users:** User namespaces are probed automatically. If not available, the existing 3-layer sandbox (process isolation + builtins stripping + resource limits) remains fully active.

---

## ⚡ Kaggle / Remote Cloud GPU Setup (Continuous Runner)

For zero-setup Kaggle deployment, paste and run this complete Python script inside a single Kaggle Notebook cell:

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

---

## Step 2: Install Base Python Dependencies
First, install the core dependencies that work on all operating systems:

```bash
# Clone the repository and enter it
git clone https://github.com/Arpit104147/DeepThink-AIOS.git
cd DeepThink-AIOS

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all package requirements
pip install -r requirements.txt
```

---

## Step 3: Install Hardware Acceleration

Run the specific block below that matches your computer's hardware to enable GPU execution.

### 🍎 Mac (Apple Silicon M1/M2/M3)
Apple uses the "Metal" framework for GPU acceleration.

```bash
# 1. Install Mac-optimized PyTorch
pip install torch torchvision torchaudio

# 2. Install Mac-optimized Llama.cpp (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

### 🪟 Windows / Linux (NVIDIA GPUs)
NVIDIA uses "CUDA" for GPU acceleration. Ensure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed (v12.1+ recommended).

#### 1. Setup CUDA Environment Variables (Linux only - e.g. Kaggle/L4/L40S)
If the installer cannot find your CUDA compiler (`nvcc`), make sure it is in your path:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. Install NVIDIA-optimized PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### ⚙️ Hardware Acceleration Engines (3-Choice Auto-Detected Setup)

DeepThink AIOS includes **3 Dedicated GPU Hardware Engines** built directly into the UI. The system automatically detects your hardware architecture and highlights the optimal GPU driver:

#### 1-Click Setup via Web UI (Recommended):
1. Start the system: `python backend/app.py` and `npm run dev` in frontend.
2. Open **Settings (⚙️)** in the sidebar and click the **⚡ Hardware GPU Engine** tab.
3. The UI automatically **detects and highlights your active hardware platform**:
   - 🟢 **NVIDIA CUDA Engine** (Highlighted on NVIDIA GPUs — installs CUDA wheel / `cu122`).
   - 🔵 **Intel / AMD Vulkan Engine** (Highlighted on Intel UHD/Iris Xe/Arc & AMD Radeon — downloads pre-compiled Vulkan binary).
   - 🍎 **Apple Silicon Metal Engine** (Highlighted on Mac M1/M2/M3/M4 — compiles Metal Performance Shaders backend).
4. Click your desired GPU setup button to compile or download the backend driver for 100% GPU offloading instantly!

---

#### Alternative Manual CLI Download (Optional):
If you prefer downloading pre-compiled binaries via CLI:
```bash
# On Linux / macOS:
python -c "from backend.vulkan_engine import _download_and_extract_vulkan; _download_and_extract_vulkan()"
```

> [!NOTE]
> **Universal Hardware Acceleration:** By using Vulkan (`-DGGML_VULKAN=on`), `llama-cpp-python` offloads compute layers directly to any Vulkan 1.2+ compliant GPU (Intel UHD/Iris Xe/Arc, AMD, or NVIDIA) without requiring proprietary vendor drivers.

---

## Step 4: Download Models & Start the System

Once all dependencies are installed, head to **[STARTUP.md](./STARTUP.md)** for:
- How to download the AI model weights (~18 GB)
- How to start the backend and frontend on Ubuntu, Mac, and Windows
- How to configure enterprise security features (JWT auth, air-gap mode)
- Troubleshooting common errors

---

## 🔐 Enterprise Configuration (Optional)

These environment variables enable enterprise-grade features. They are **entirely optional** for local development:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIOS_AUTH_ENABLED` | `0` | Set to `1` to require JWT tokens on all API endpoints |
| `AIOS_JWT_SECRET` | auto-generated | Custom secret key for JWT token signing |
| `AIOS_ADMIN_PASSWORD` | `admin` | Password for the `/api/auth/login` endpoint |
| `AIOS_AIR_GAP` | `0` | Set to `1` to disable all outbound network features |
| `GITHUB_TOKEN` | empty | GitHub Personal Access Token for automated PR creation |

Example (air-gapped deployment with auth):
```bash
export AIOS_AUTH_ENABLED=1
export AIOS_JWT_SECRET="your-secret-key-here"
export AIOS_ADMIN_PASSWORD="strong-password"
export AIOS_AIR_GAP=1
python backend/app.py
```
