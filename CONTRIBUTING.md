# Contributing to DeepThink AIOS

We welcome contributions from fellow students, researchers, and developers! Whether you are fixing a bug, adding support for a new programming language in the sandbox, or refining the semiconductor EDA synthesis pipeline, here is how you can get started.

---

## 🛠️ Local Development Setup

To set up a local environment for coding and verification:

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Team_Trenches.git
   cd Team_Trenches
   ```

2. **Set up the Virtual Environment & Dependencies:**
   Follow the detailed guide in [README_SETUP.md](./README_SETUP.md) to install Python dependencies, configure hardware acceleration (Metal/CUDA/XPU), and optionally install EDA toolchains (`iverilog`, `yosys`, `ngspice`, `klayout`).

3. **Verify Setup:**
   Run the quick verification command:
   ```bash
   python3 -m py_compile backend/app.py backend/sandbox.py backend/security.py
   ```

---

## 🛡️ Guidelines for Contributions

### Code Style & Structure
- Keep Python code type-annotated and document new helper functions.
- Keep CSS variables centered in `frontend/src/index.css` to maintain theme consistency (glassmorphism / dark mode).
- Do not introduce external dependencies to `requirements.txt` unless absolutely necessary.

### Security (SAST & Sandbox Isolation)
- All sandbox execution features must maintain the 3-layer security system (Builtins stripping, Resource constraints, and Linux Namespace wrapping).
- If your contribution edits `backend/sandbox.py` or executes guest code, ensure it does not bypass the pre-execution SAST scan in `backend/security.py`.

### Verilog & Chip Design EDA
- When updating hardware verification components, ensure compatibility with Yosys mapping rules and KLayout DRC reports.

---

## 🚀 Submitting a Pull Request

When you are ready to submit your changes:

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```
2. Commit your changes with clear, structured messages:
   ```bash
   git commit -m "feat: add support for XYZ verification sandbox"
   ```
3. Push to your fork and open a **Pull Request**.
4. Fill out the provided Pull Request template completely so we can review and test your changes quickly!
