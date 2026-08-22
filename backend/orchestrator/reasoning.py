import re
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter
from backend.downloader import resolve_model_key

class ReasoningPipeline:
    """Program-Aided Language (PAL) Reasoning Pipeline & Deep Mathematical Derivation Engine."""

    LATEX_RULES = (
        "MANDATORY LATEX FORMATTING RULES:\n"
        "1. Wrap EVERY mathematical variable, operator, or expression in single dollar signs. "
        "Examples: $x$, $p$, $a$, $a^\\dagger$, $\\hat{H}$, $\\hbar$, $\\omega$, $E_n$, $\\psi_n(x)$.\n"
        "2. Wrap ALL major equations, commutation proofs, integrals, and derivations in centered double dollar signs on their own line:\n"
        "$$[a, a^\\dagger] = aa^\\dagger - a^\\dagger a = 1$$\n"
        "$$E_n = \\hbar\\omega\\left(n + \\frac{1}{2}\\right)$$\n"
        "$$\\psi_0(x) = \\left(\\frac{m\\omega}{\\pi\\hbar}\\right)^{1/4} \\exp\\left(-\\frac{m\\omega x^2}{2\\hbar}\\right)$$\n"
        "3. NEVER output unformatted raw math text like 'a a†', 'hbar omega', or 'E0=1/2hbar w'. Always use standard KaTeX.\n"
        "4. DO NOT output conversational rambling (e.g. 'First let me recall', 'Wait maybe I should'). Proceed directly to the rigorous proof."
    )

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        
        # Allocate ample token headroom (3584 tokens) so long theoretical proofs never get cut off mid-derivation
        reasoning_gen_tokens = min(4096, max(3072, ds_ctx - 1500))
        reasoning_temp = 0.2  # Low temperature for strict mathematical precision and zero rambling

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

            theory_sys = (
                "You are an expert theoretical mathematician and mathematical physicist.\n"
                "Your task is to provide a complete, rigorous, publication-grade academic derivation and proof.\n\n"
                + ReasoningPipeline.LATEX_RULES + "\n\n"
                "STRUCTURAL REQUIREMENTS:\n"
                "1. Divide the proof into clear numbered academic sections (e.g. 1. Operator Definitions, 2. Commutation Proof, 3. Energy Spectrum & Zero-Point Energy, 4. Explicit Wavefunctions).\n"
                "2. Provide step-by-step algebraic substitutions showing how each equation follows from the previous one.\n"
                "3. Conclude with a clear summary box or theorem statement highlighting the final results."
            )
            theory_p = (
                f"REQUEST / THEOREM TO PROVE:\n{prompt}\n\n"
                f"Write the complete, rigorous theoretical derivation with full display KaTeX equations:"
            )
            raw = orchestrator._call_model(ds_llm, theory_p, max_tokens=reasoning_gen_tokens, temperature=reasoning_temp, system_prompt=theory_sys)
            cleaned = orchestrator._strip_thinking(raw)
            return f"### ⚡ Theoretical Derivation & Mathematical Proof\n\n{cleaned}"

        # PAL Playground verification mode
        if status_callback:
            status_callback(f"⚡ Reasoning mode: Playground-Verified (PAL) with {ds_display}...", "info", reasoning_key, 20)

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

        return f"### 🧠 Verified Mathematical Solution (PAL)\n\n{final_answer}"
