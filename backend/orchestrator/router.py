import re

class TaskRouter:
    """Helper module for intent classification, prompt routing heuristics,
    and prediction guards."""

    @staticmethod
    def classify_task(orchestrator, router_llm, prompt):
        """Classify user query into task types (CODING, REASONING, CHIP_DESIGN, SIMPLE)."""
        p = (
            "Classify this user request into ONE of the following 4 categories:\n"
            "1. CODING: Request for writing, modifying, debugging, or optimizing code (Python, C, C++, Verilog, HTML/JS, etc.).\n"
            "2. REASONING: Step-by-step logic proofs, math derivations, physics proofs, or complex multi-step deductions.\n"
            "3. CHIP_DESIGN: Verilog HDL hardware design, SPICE analog simulation, or circuit netlists.\n"
            "4. SIMPLE: General knowledge, weather, facts, conversational QA, summaries, or direct questions.\n\n"
            "Reply ONLY with ONE category word (CODING, REASONING, CHIP_DESIGN, or SIMPLE).\n\n"
            f"User Request: {prompt[:500]}"
        )
        try:
            res = orchestrator._call_model(router_llm, p, max_tokens=15, temperature=0.1).strip().upper()
            res = orchestrator._strip_thinking(res)
            for cat in ["CODING", "REASONING", "CHIP_DESIGN", "SIMPLE"]:
                if cat in res:
                    return cat
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
        result = orchestrator._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        return "YES" in str(result).upper()

    @staticmethod
    def looks_numeric_problem(prompt):
        """Heuristic to decide if prompt is a math/arithmetic problem."""
        prompt_lower = prompt.lower()
        math_keywords = ["solve", "calculate", "find the", "compute", "integral", "derivative", "probability", "equation"]
        return any(kw in prompt_lower for kw in math_keywords)
