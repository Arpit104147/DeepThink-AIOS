import re

class TaskRouter:
    """Helper module for intent classification, prompt routing heuristics,
    and prediction guards."""

    @staticmethod
    def classify_task(orchestrator, router_llm, prompt):
        """Classify user query into task types (CODING, REASONING, CHIP_DESIGN, SIMPLE)."""
        p_lower = prompt.lower()

        # 1. Deterministic Fast-Path for Hardware & EDA Chip Design
        chip_keywords = [
            "verilog", "systemverilog", "vhdl", "ngspice", "spice netlist",
            "chip design", "semiconductor layout", "eda tools", "iverilog", "yosys",
            "gdstk", "cla adder", "carry-lookahead", "binary counter", "up/down counter",
            "fsm in verilog", "flip-flop", "alu in verilog", "dumpfile", "dumpvars",
            "3d semiconductor", "asic", "fpga"
        ]
        if any(kw in p_lower for kw in chip_keywords):
            return "CHIP_DESIGN"

        # 2. Deterministic Fast-Path for Pure Math / Physics Proofs
        reasoning_keywords = [
            "derive", "proof of", "prove that", "eigenvalues and eigenfunctions",
            "lorentz transformations", "hamiltonian", "schrodinger", "dirac ladder",
            "commutation relation", "time dilation", "special relativity"
        ]
        if any(kw in p_lower for kw in reasoning_keywords) and not any(k in p_lower for k in ["python", "c++", "cpp", "javascript", "script"]):
            return "REASONING"

        p = (
            "Classify this user request into ONE of the following 4 categories:\n"
            "1. CODING: Software programming (Python, C, C++, Rust, Go, Java, JavaScript, Bash, HTML/CSS, Web Apps).\n"
            "2. REASONING: Step-by-step logic proofs, math derivations, physics proofs, or theoretical deductions.\n"
            "3. CHIP_DESIGN: Verilog / SystemVerilog HDL hardware design, FPGA/ASIC digital circuits, SPICE analog simulation, or 3D semiconductor layout.\n"
            "4. SIMPLE: General knowledge, weather, facts, conversational QA, summaries, or direct questions.\n\n"
            "Reply ONLY with ONE category word (CODING, REASONING, CHIP_DESIGN, or SIMPLE).\n\n"
            f"User Request: {prompt[:500]}"
        )
        try:
            raw = orchestrator._call_model(router_llm, p, max_tokens=64, temperature=0.1).strip()
            res = orchestrator._strip_thinking(raw).strip().upper()
            # Match the LAST occurrence of a category keyword to avoid
            # false matches in reasoning text like "This is not CODING..."
            last_match = None
            for cat in ["CODING", "REASONING", "CHIP_DESIGN", "SIMPLE"]:
                idx = res.rfind(cat)
                if idx != -1:
                    if last_match is None or idx > last_match[1]:
                        last_match = (cat, idx)
            if last_match:
                return last_match[0]
        except Exception:
            pass

        return "SIMPLE"

    @staticmethod
    def is_playground_applicable(orchestrator, router_llm, prompt):
        """Check if reasoning can be verified via Python sandbox."""
        if not orchestrator._is_model_valid(router_llm):
            router_llm = orchestrator._get_model("router", required_ctx=1024)
        auto_keywords = [
            "solve_ivp", "scipy", "sympy", "z3-solver", "networkx", "astropy", "biopython", "rdkit",
            "verification script", "run playground", "sandbox verification"
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in auto_keywords):
            return True

        p = (
            "Determine if this request can be numerically verified or proven using a short Python validation script.\n"
            "Return YES if: It involves computing a specific value, simulating a differential equation (like ODEs/trajectory with numbers), "
            "solving constraints (SAT/Z3), verifying encryption/decryption roundtrips, or testing a specific algorithm's logic.\n"
            "Return NO if: It is a request for general explanations, derivations (like Euler-Lagrange, mathematical proofs), "
            "conceptual descriptions, or open-ended theoretical physics/math questions without concrete inputs/assertions.\n\n"
            "Reply ONLY 'YES' or 'NO'.\n\n"
            f"Query: {prompt[:500]}"
        )
        result = orchestrator._call_model(router_llm, p, max_tokens=64, temperature=0.1)
        cleaned = orchestrator._strip_thinking(str(result)).strip().upper()
        # Use word boundary match to avoid false positive from "NO: ... YES ..."
        if re.search(r'\bYES\b', cleaned):
            # Make sure YES appears after any NO (last answer wins)
            yes_pos = cleaned.rfind("YES")
            no_pos = cleaned.rfind("NO")
            return yes_pos > no_pos
        return False

    @staticmethod
    def looks_numeric_problem(prompt):
        """Heuristic to decide if prompt is a math/arithmetic problem."""
        prompt_lower = prompt.lower()
        math_keywords = ["solve", "calculate", "find the", "compute", "integral", "derivative", "probability", "equation"]
        return any(kw in prompt_lower for kw in math_keywords)
