import re
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter

class ReasoningPipeline:
    """Program-Aided Language (PAL) Reasoning Pipeline & SymPy Sandbox Verifier."""

    LATEX_RULES = (
        "MANDATORY LATEX FORMATTING RULES:\n"
        "1. Wrap EVERY mathematical variable, function, or expression in single dollar signs. "
        "Examples: $x$, $n$, $f(x)$, $\\ln n$, $n \\ge 2$, $p^2 - 1$.\n"
        "2. Wrap ALL standalone equations, integrals, series, and limits in display double dollar signs on their own line. "
        "Example:\n$$\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}$$\n"
        "3. NEVER output raw unformatted math text like 'f(x)f(x)', 'lnn', or 'n>=2'. "
        "Always use proper LaTeX: $f(x)$, $\\ln n$, $n \\ge 2$.\n"
        "4. For fractions use $\\frac{a}{b}$, for square roots use $\\sqrt{x}$, "
        "for summations use $\\sum$, for integrals use $\\int$.\n"
    )

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        active_router = orchestrator._get_model("router", required_ctx=1024)
        use_playground = TaskRouter.is_playground_applicable(orchestrator, active_router, prompt)

        if not use_playground:
            if status_callback:
                status_callback("⚡ Reasoning mode: Theoretical Derivation", "info", "deepseek_r1", 20)
            ds_llm = orchestrator._get_model("deepseek_r1", required_ctx=ds_ctx)
            theory_p = (
                "You are an expert theoretical mathematician and physicist.\n"
                "Provide a rigorous, step-by-step academic derivation.\n\n"
                + ReasoningPipeline.LATEX_RULES +
                f"\nRequest: {prompt}"
            )
            raw = orchestrator._call_model(ds_llm, theory_p, gen_tokens, gen_temp)
            cleaned = orchestrator._strip_thinking(raw)
            return f"### ⚡ Theoretical Derivation & Proof\n\n{cleaned}"

        # PAL Playground verification mode
        if status_callback:
            status_callback("⚡ Reasoning mode: Playground-Verified (PAL)", "info", "deepseek_r1", 20)

        ds_llm = orchestrator._get_model("deepseek_r1", required_ctx=ds_ctx)
        draft_p = f"Draft a step-by-step solution for this math/physics problem:\n{prompt}"
        hypothesis = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, draft_p, gen_tokens, gen_temp))

        verified, pg_out, pg_code = orchestrator._run_playground(
            ds_llm, hypothesis, purpose="math", status_callback=status_callback, model_key="deepseek_r1", original_prompt=prompt
        )

        synth_p = (
            f"Original Request:\n{prompt}\n\n"
            f"Verified Solution Output:\n{pg_out[:2000]}\n\n"
            "Provide a final, clear, step-by-step academic explanation.\n\n"
            + ReasoningPipeline.LATEX_RULES
        )
        final_answer = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, synth_p, gen_tokens, gen_temp))

        return f"### 🧠 Verified Mathematical Solution (PAL)\n\n{final_answer}"
