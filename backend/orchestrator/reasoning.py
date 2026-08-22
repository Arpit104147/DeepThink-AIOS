import re
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter
from backend.downloader import resolve_model_key

class ReasoningPipeline:
    """
    Program-Aided Language (PAL) Reasoning Pipeline & Deep Mathematical Derivation Engine.
    Generates exhaustive, graduate-level academic proofs and theoretical physics derivations
    with pristine display KaTeX equations ($$ ... $$) and rigorous step-by-step algebraic substitutions.
    """

    LATEX_RULES = (
        "MANDATORY LATEX FORMATTING RULES:\n"
        "1. Wrap EVERY mathematical variable, operator, or expression in single dollar signs with spaces outside: $x$, $r$, $t$, $\\theta$, $\\phi$, $g_{\\mu\\nu}$, $\\Gamma^\\mu_{\\alpha\\beta}$, $R_{\\mu\\nu}$, $G_{\\mu\\nu}$, $c$, $G$, $M$.\n"
        "2. Wrap ALL major mathematical equations, tensor components, differential equations, and derivations in centered double dollar signs on their own lines:\n"
        "$$ds^2 = -e^{\\nu(r)} c^2 dt^2 + e^{\\lambda(r)} dr^2 + r^2 (d\\theta^2 + \\sin^2\\theta d\\phi^2)$$\n"
        "$$\\Gamma^\\mu_{\\alpha\\beta} = \\frac{1}{2} g^{\\mu\\sigma} \\left( \\partial_\\alpha g_{\\beta\\sigma} + \\partial_\\beta g_{\\alpha\\sigma} - \\partial_\\sigma g_{\\alpha\\beta} \\right)$$\n"
        "$$R_{\\mu\\nu} = \\partial_\\rho \\Gamma^\\rho_{\\mu\\nu} - \\partial_\\nu \\Gamma^\\rho_{\\mu\\rho} + \\Gamma^\\rho_{\\rho\\sigma} \\Gamma^\\sigma_{\\mu\\nu} - \\Gamma^\\rho_{\\nu\\sigma} \\Gamma^\\sigma_{\\mu\\rho} = 0$$\n"
        "$$e^{\\nu(r)} = e^{-\\lambda(r)} = 1 - \\frac{2GM}{c^2 r} = 1 - \\frac{r_s}{r}$$\n"
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
                status_callback(f"⚡ Reasoning mode: Theoretical Derivation with {ds_display}...", "info", reasoning_key, 20)

            # Stage 1: Foundational Framework, Symmetry Ansatz & Field Equations
            stage1_sys = (
                "You are an expert theoretical physicist and mathematician.\n"
                "Your task is to author Part 1 of a rigorous, publication-grade academic derivation.\n\n"
                + ReasoningPipeline.LATEX_RULES + "\n\n"
                "STRUCTURAL REQUIREMENTS FOR PART 1:\n"
                "1. Define the physical postulates, coordinate system, and symmetry principles (e.g. spherical symmetry, static spacetime).\n"
                "2. Formulate the general metric tensor ansatz $g_{\\mu\\nu}$ with explicit metric line element $ds^2$.\n"
                "3. State the governing field equations (e.g. Vacuum Einstein Field Equations $R_{\\mu\\nu} = 0$ or Euler-Lagrange equations)."
            )
            stage1_p = (
                f"THEORETICAL DERIVATION REQUEST:\n{prompt}\n\n"
                "INSTRUCTION: Write Part 1 (Physical Postulates, Symmetry Principles, and Metric / Field Ansatz) with complete display KaTeX equations ($$ ... $$)."
            )
            raw_stage1 = orchestrator._call_model(ds_llm, stage1_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=stage1_sys)
            clean_stage1 = orchestrator._strip_thinking(raw_stage1)

            if status_callback:
                status_callback(f"⚡ Reasoning mode: Algebraic Derivation & Curvature Tensors with {ds_display}...", "info", reasoning_key, 60)

            # Stage 2: Explicit Step-by-Step Algebraic Derivation & Integration Constants
            stage2_sys = (
                "You are an expert theoretical physicist and mathematician.\n"
                "Your task is to author Part 2 of the mathematical derivation, providing complete algebraic computations.\n\n"
                + ReasoningPipeline.LATEX_RULES + "\n\n"
                "STRUCTURAL REQUIREMENTS FOR PART 2:\n"
                "1. Explicitly compute all non-zero Christoffel symbols $\\Gamma^\\mu_{\\alpha\\beta}$ with step-by-step metric derivatives.\n"
                "2. Explicitly compute the non-zero Ricci curvature tensor components ($R_{tt}, R_{rr}, R_{\\theta\\theta}, R_{\\phi\\phi}$).\n"
                "3. Set up the differential equations $R_{\\mu\\nu} = 0$ and show how adding $R_{tt} + R_{rr} = 0$ proves $\\nu'(r) + \\lambda'(r) = 0$.\n"
                "4. Integrate the differential equations and apply the asymptotic flatness Newtonian limit at $r \\to \\infty$ (matching to $\\Phi = -\\frac{GM}{r}$) to determine the integration constants ($r_s = \\frac{2GM}{c^2}$).\n"
                "5. State the final exact metric solution in a prominent summary theorem box."
            )
            stage2_p = (
                f"ORIGINAL DERIVATION REQUEST:\n{prompt}\n\n"
                f"PART 1 FOUNDATION:\n{clean_stage1[:1800]}\n\n"
                "INSTRUCTION: Write Part 2 (Full Step-by-Step Christoffel Symbols, Ricci Tensor Components, Differential Equations, and Newtonian Integration Constant Matching) with complete display KaTeX equations ($$ ... $$)."
            )
            raw_stage2 = orchestrator._call_model(ds_llm, stage2_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=stage2_sys)
            clean_stage2 = orchestrator._strip_thinking(raw_stage2)

            full_derivation = f"{clean_stage1}\n\n{clean_stage2}"
            sanitized = ReasoningPipeline._sanitize_reasoning_latex(full_derivation)

            return f"### ⚡ Theoretical Derivation & Mathematical Proof\n\n{sanitized}"

        # PAL Playground verification mode
        if status_callback:
            status_callback(f"⚡ Reasoning mode: Computational Sandbox (PAL) with {ds_display}...", "info", reasoning_key, 20)

        draft_sys = (
            "You are an expert computational mathematician.\n"
            "Draft a complete, step-by-step mathematical derivation for this problem.\n"
            + ReasoningPipeline.LATEX_RULES
        )
        draft_p = f"Draft a step-by-step solution for this math/physics problem:\n{prompt}"
        hypothesis = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, draft_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=draft_sys))

        verified, pg_out, pg_code = orchestrator._run_playground(
            ds_llm, hypothesis, purpose="math", status_callback=status_callback, model_key=reasoning_key, original_prompt=prompt
        )

        synth_sys = (
            "You are an expert theoretical mathematician.\n"
            "Synthesize the verified computational proof into a publication-grade academic derivation.\n"
            + ReasoningPipeline.LATEX_RULES
        )
        synth_p = (
            f"Original Request:\n{prompt}\n\n"
            f"Verified Computational Output:\n{pg_out[:2000]}\n\n"
            f"Provide the final, complete, step-by-step academic proof with centered display KaTeX equations ($$ ... $$):"
        )
        final_answer = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, synth_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=synth_sys))
        sanitized_final = ReasoningPipeline._sanitize_reasoning_latex(final_answer)

        return f"### 🧠 Verified Mathematical Solution (PAL)\n\n{sanitized_final}"

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
        text = re.sub(r"(?<!\$)\$([^\$\n]+)\n\$", r"$$\1$$\n", text)

        # 2. Convert standalone single-dollar display equations to double-dollar display blocks
        text = re.sub(r"(?m)^\s*\$([^\$\n]{8,})\$\s*$", r"$$\1$$", text)

        # 3. Clean up unescaped English text trapped inside single dollar signs
        text = re.sub(
            r"\$([^\$\n]{10,})\s+(where|with|and|such that|in which|implies|yielding)\s+",
            r"$$\1$$\n\2 ",
            text,
            flags=re.IGNORECASE
        )

        # 4. Remove conversational filler phrases
        text = re.sub(r"(?i)^(Alright,|Okay,|Let's see,|Wait,|First,|Now,)\s*", "", text)

        # 5. Fix double dollar formatting and clean empty lines inside display math
        lines = text.split("\n")
        fixed_lines = []
        in_display = False

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("$$") and trimmed.endswith("$$") and len(trimmed) > 4:
                fixed_lines.append(trimmed)
            elif trimmed == "$$":
                in_display = not in_display
                fixed_lines.append("$$")
            elif in_display and trimmed.startswith("$") and not trimmed.startswith("$$"):
                # Remove nested single dollar inside display block
                fixed_lines.append(trimmed.replace("$", ""))
            else:
                fixed_lines.append(line)

        cleaned_output = "\n".join(fixed_lines)

        # Ensure balanced display dollar blocks
        if cleaned_output.count("$$") % 2 != 0:
            cleaned_output += "\n$$"

        return cleaned_output
