import re
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter

class ReasoningPipeline:
    """Program-Aided Language (PAL) Reasoning Pipeline & SymPy Sandbox Verifier."""

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
                "Provide a rigorous, step-by-step academic derivation with complete LaTeX equations.\n\n"
                "MANDATORY LATEX FORMATTING RULES:\n"
                "1. Wrap EVERY single mathematical variable, function, or expression in standard single dollar signs ($x$, $n \\ge 2$, $\\ln n$, $f(x) = \\frac{1}{x (\\ln x)^2}$).\n"
                "2. Wrap ALL major standalone equations, integrals, series, and limits in centered display double dollar signs ($$ ... $$).\n"
                "3. NEVER output raw concatenated text (like 'lnnlnn' or 'f(x)f(x)'). Always use proper LaTeX notation ($f(x)$, $\\ln n$).\n\n"
                f"Request: {prompt}"
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
            "Provide a final, clear, step-by-step academic explanation.\n"
            "MANDATORY LATEX RULES:\n"
            "1. Wrap ALL inline variables and terms in single dollar signs ($x$, $n$, $\\ln n$).\n"
            "2. Wrap ALL standalone formulas and integrals in centered display double dollar signs ($$ ... $$).\n"
            "3. Format all math terms cleanly in LaTeX so KaTeX renders them perfectly."
        )
        final_answer = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, synth_p, gen_tokens, gen_temp))

        return f"### 🧠 Verified Mathematical Solution (PAL)\n\n{final_answer}"
