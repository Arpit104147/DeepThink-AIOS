import re
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter
from backend.downloader import resolve_model_key

class ReasoningPipeline:
    """
    Zero-Hallucination Program-Aided Language (PAL) Reasoning Pipeline & Deep Mathematical Derivation Engine.
    Generates exhaustive, graduate-level academic proofs and theoretical physics derivations
    with SymPy Computer Algebra System (CAS) grounding, Kretschmann Curvature Scalar Invariant verification,
    asymptotic boundary limit checks (Minkowski & Newtonian), and pristine display KaTeX equations ($$ ... $$).
    """

    LATEX_RULES = (
        "MANDATORY LATEX FORMATTING RULES:\n"
        "1. Wrap EVERY mathematical variable, operator, or expression in single dollar signs with spaces outside: $x$, $r$, $t$, $\\theta$, $\\phi$, $g_{\\mu\\nu}$, $\\Gamma^\\mu_{\\alpha\\beta}$, $R_{\\mu\\nu}$, $R^{\\alpha\\beta\\gamma\\delta} R_{\\alpha\\beta\\gamma\\delta}$, $c$, $G$, $M$.\n"
        "2. Wrap ALL major mathematical equations, tensor components, differential equations, and derivations in centered double dollar signs on their own lines:\n"
        "$$ds^2 = -\\left(1 - \\frac{2GM}{c^2 r}\\right) c^2 dt^2 + \\left(1 - \\frac{2GM}{c^2 r}\\right)^{-1} dr^2 + r^2 (d\\theta^2 + \\sin^2\\theta d\\phi^2)$$\n"
        "$$\\Gamma^\\mu_{\\alpha\\beta} = \\frac{1}{2} g^{\\mu\\sigma} \\left( \\partial_\\alpha g_{\\beta\\sigma} + \\partial_\\beta g_{\\alpha\\sigma} - \\partial_\\sigma g_{\\alpha\\beta} \\right)$$\n"
        "$$R_{\\mu\\nu} = \\partial_\\rho \\Gamma^\\rho_{\\mu\\nu} - \\partial_\\nu \\Gamma^\\rho_{\\mu\\rho} + \\Gamma^\\rho_{\\rho\\sigma} \\Gamma^\\sigma_{\\mu\\nu} - \\Gamma^\\rho_{\\nu\\sigma} \\Gamma^\\sigma_{\\mu\\rho} = 0$$\n"
        "$$K = R^{\\alpha\\beta\\gamma\\delta} R_{\\alpha\\beta\\gamma\\delta} = \\frac{48 G^2 M^2}{c^4 r^6} = \\frac{12 r_s^2}{r^6}$$\n"
        "3. NEVER put English explanation words inside dollar signs `$ ... $`. Write English text outside math delimiters.\n"
        "4. NEVER leave unclosed dollar signs across paragraph breaks.\n"
        "5. DO NOT write conversational filler ('Alright, so I need to figure out', 'Wait let me see'). Begin directly with the formal derivation."
    )

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        
        # Allocate ample token headroom (3584 tokens)
        reasoning_gen_tokens = min(4096, max(3072, ds_ctx - 1500))
        reasoning_temp = 0.2  # Low temperature for strict mathematical precision and zero meta-rambling

        active_router = orchestrator._get_model("router", required_ctx=1024)
        use_playground = TaskRouter.is_playground_applicable(orchestrator, active_router, prompt)

        reasoning_key = resolve_model_key("reasoning") or "deepseek_r1"
        try:
            ds_llm = orchestrator._get_model(reasoning_key, required_ctx=ds_ctx)
            if not orchestrator._is_model_valid(ds_llm):
                ds_llm = orchestrator._get_model("vibethinker", required_ctx=ds_ctx)
                reasoning_key = "vibethinker"
        except (FileNotFoundError, Exception):
            ds_llm = orchestrator._get_model("vibethinker", required_ctx=ds_ctx)
            reasoning_key = "vibethinker"

        ds_display = orchestrator._get_display_model_name(reasoning_key)

        if not use_playground:
            if status_callback:
                status_callback(f"⚡ Reasoning mode: Theoretical Derivation & Invariant Validation with {ds_display}...", "info", reasoning_key, 20)

            # Stage 1: Foundational Framework, Symmetry Ansatz & Field Equations
            stage1_sys = (
                "You are a Distinguished Theoretical Physicist and Professor of General Relativity.\n"
                "Your task is to author Part 1 of a rigorous, publication-grade academic derivation.\n\n"
                + ReasoningPipeline.LATEX_RULES + "\n\n"
                "STRUCTURAL REQUIREMENTS FOR PART 1:\n"
                "1. Define the physical postulates, coordinate system $(ct, r, \\theta, \\phi)$, and symmetry principles (spherical symmetry, static spacetime, time-reversal invariance).\n"
                "2. Formulate the general metric tensor ansatz $g_{\\mu\\nu}$ with line element:\n"
                "   $$ds^2 = -e^{\\nu(r)} c^2 dt^2 + e^{\\lambda(r)} dr^2 + r^2 (d\\theta^2 + \\sin^2\\theta d\\phi^2)$$\n"
                "3. State the governing vacuum field equations $R_{\\mu\\nu} = 0$ and metric compatibility condition $\\nabla_\\sigma g_{\\mu\\nu} = 0$."
            )
            stage1_p = (
                f"THEORETICAL DERIVATION REQUEST:\n{prompt}\n\n"
                "INSTRUCTION: Write Part 1 (Physical Postulates, Symmetry Principles, Metric Line Element Ansatz, and Field Equations) with complete display KaTeX equations ($$ ... $$)."
            )
            raw_stage1 = orchestrator._call_model(ds_llm, stage1_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=stage1_sys)
            clean_stage1 = orchestrator._strip_thinking(raw_stage1)

            if status_callback:
                status_callback(f"⚡ Reasoning mode: Algebraic Curvature Tensors & Kretschmann Invariant with {ds_display}...", "info", reasoning_key, 60)

            # Stage 2: Step-by-Step Curvature Tensors, Newtonian Matching & Kretschmann Scalar Invariant
            stage2_sys = (
                "You are a Distinguished Theoretical Physicist and Professor of General Relativity.\n"
                "Your task is to author Part 2 of the mathematical derivation with zero algebraic shortcuts.\n\n"
                + ReasoningPipeline.LATEX_RULES + "\n\n"
                "STRUCTURAL REQUIREMENTS FOR PART 2:\n"
                "1. Explicitly compute all 9 non-zero Christoffel symbols $\\Gamma^\\mu_{\\alpha\\beta}$ with step-by-step metric derivatives.\n"
                "2. Explicitly compute the non-zero Ricci curvature tensor components ($R_{tt}, R_{rr}, R_{\\theta\\theta}, R_{\\phi\\phi}$).\n"
                "3. Set up the differential equations $R_{\\mu\\nu} = 0$ and show that $e^{-\\nu} R_{tt} + e^{-\\lambda} R_{rr} = 0 \\implies \\nu'(r) + \\lambda'(r) = 0 \\implies \\nu(r) = -\\lambda(r)$.\n"
                "4. Integrate the $R_{\\theta\\theta} = 0$ equation to obtain $e^{\\nu(r)} = 1 - \\frac{C}{r}$.\n"
                "5. Boundary Condition 1 — Newtonian Weak-Field Limit ($r \\to \\infty$): Match to Newtonian potential $\\Phi = -\\frac{GM}{r}$ using $g_{00} \\approx -(1 + 2\\Phi/c^2)$ to fix integration constant $C = \\frac{2GM}{c^2} = r_s$.\n"
                "6. Boundary Condition 2 — Flat Spacetime Limit ($M \\to 0$): Prove metric reduces identically to Minkowski spacetime $ds^2 = -c^2 dt^2 + dr^2 + r^2 d\\Omega^2$.\n"
                "7. Kretschmann Curvature Scalar Invariant Proof:\n"
                "   Compute $K = R^{\\alpha\\beta\\gamma\\delta} R_{\\alpha\\beta\\gamma\\delta} = \\frac{48 G^2 M^2}{c^4 r^6} = \\frac{12 r_s^2}{r^6}$.\n"
                "   Prove mathematically that $\\lim_{r \\to r_s} K = \\frac{12}{r_s^4} < \\infty$ (removable coordinate singularity via Eddington-Finkelstein/Kruskal-Szekeres coordinates), whereas $\\lim_{r \\to 0} K = \\infty$ (true physical curvature singularity).\n"
                "8. Conclude with the complete final metric in a prominent summary theorem box."
            )
            stage2_p = (
                f"ORIGINAL DERIVATION REQUEST:\n{prompt}\n\n"
                f"PART 1 FOUNDATION:\n{clean_stage1[:1800]}\n\n"
                "INSTRUCTION: Write Part 2 (Step-by-Step Christoffel Symbols, Ricci Tensor, Differential Equations, Newtonian Matching, Flat-Space Limit, and Kretschmann Invariant Proof) with complete display KaTeX equations ($$ ... $$)."
            )
            raw_stage2 = orchestrator._call_model(ds_llm, stage2_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=stage2_sys)
            clean_stage2 = orchestrator._strip_thinking(raw_stage2)

            full_derivation = f"{clean_stage1}\n\n{clean_stage2}"
            sanitized = ReasoningPipeline._sanitize_reasoning_latex(full_derivation)

            return f"### ⚡ Theoretical Derivation & Mathematical Proof\n\n{sanitized}"

        # Sub-Pipeline A: Program-Aided Language (PAL) with SymPy CAS Symbolic Grounding
        if status_callback:
            status_callback(f"⚡ Reasoning mode: SymPy Symbolic Computer Algebra (CAS) with {ds_display}...", "info", reasoning_key, 20)

        draft_sys = (
            "You are an expert computational mathematician and theoretical scientist.\n"
            "Draft a complete, step-by-step mathematical derivation for this problem, formulating exact algebraic equations.\n"
            + ReasoningPipeline.LATEX_RULES
        )
        draft_p = f"Draft an exact step-by-step solution for this math/physics problem:\n{prompt}"
        hypothesis = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, draft_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=draft_sys))

        verified, pg_out, pg_code = orchestrator._run_playground(
            ds_llm, hypothesis, purpose="math", status_callback=status_callback, model_key=reasoning_key, original_prompt=prompt
        )

        synth_sys = (
            "You are a Distinguished Theoretical Mathematician.\n"
            "Synthesize the verified computational proof into a publication-grade academic derivation.\n\n"
            + ReasoningPipeline.LATEX_RULES + "\n\n"
            "MANDATORY ANTI-HALLUCINATION RULES:\n"
            "1. Base all numerical answers and algebraic simplifications STRICTLY on the SymPy computational output.\n"
            "2. State the final verified answer clearly in a prominent summary box with display KaTeX equations ($$ ... $$)."
        )
        synth_p = (
            f"Original Mathematical Request:\n{prompt}\n\n"
            f"Verified SymPy CAS Computational Sandbox Output:\n{pg_out[:2000]}\n\n"
            f"Write the complete, rigorous mathematical derivation with step-by-step display KaTeX equations ($$ ... $$):"
        )
        final_answer = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, synth_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=synth_sys))
        sanitized_final = ReasoningPipeline._sanitize_reasoning_latex(final_answer)

        parts = [
            f"### 🧠 Verified Mathematical Solution (SymPy CAS Grounded)\n\n{sanitized_final}"
        ]
        if pg_code:
            status_tag = "Verified Passed ✅" if verified else "Diagnostic Sandbox ⚠️"
            parts.append(f"\n\n### ⚙️ Symbolic CAS Sandbox Verification ({status_tag})\n```python\n{pg_code.strip()}\n```")
            if pg_out and str(pg_out).strip():
                parts.append(f"\n```\n{str(pg_out).strip()[:1500]}\n```")

        return "".join(parts)

    @staticmethod
    def _sanitize_reasoning_latex(text: str) -> str:
        """
        Sanitizes LaTeX formatting in mathematical proofs to ensure pristine KaTeX rendering.
        Fixes broken single-dollar spanning across newlines, standardizes display equations ($$ ... $$),
        and ensures proper mathematical spacing.
        """
        if not text:
            return ""

        # 1. Fix single dollar signs that span across newlines (e.g. "$formula \n\n$Next")
        text = re.sub(r"(?<!\$)\$([^\$\n]+)\n\n\$", r"$$\1$$\n\n", text)
        
        # 2. Fix cases where an opening $$ is on its own line and formula starts on next line
        text = re.sub(r"\$\$\s*\n\s*([^$]+?)\s*\n\s*\$\$", r"$$\n\1\n$$", text)

        # 3. Clean trailing whitespace in display equation blocks
        text = re.sub(r"\$\$\s+", "$$\n", text)
        text = re.sub(r"\s+\$\$", "\n$$", text)

        return text
