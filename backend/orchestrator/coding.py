import os
import re
import gc
import uuid
import re
import shutil
from backend.sandbox import Sandbox
from backend.orchestrator.router import TaskRouter

class CodingPipeline:
    """7-Phase Agentic Actor-Critic Coding Pipeline & AST Linter Surgical Patcher."""

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None):
        req_lang = CodingPipeline._detect_target_language(prompt)
        lang_name = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "html": "HTML/Web App",
            "cpp": "C++",
            "c": "C",
            "bash": "Bash",
            "java": "Java",
            "go": "Go",
            "rust": "Rust"
        }.get(req_lang, req_lang)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        
        is_web_design = ("html" in prompt.lower() or "website" in prompt.lower() or "landing page" in prompt.lower())
        if is_web_design or req_lang == "html":
            gen_tokens = min(12288, int(gen_tokens * 1.5))
            min_ctx = min(ds_ctx, oc_ctx)
            if min_ctx - gen_tokens < 1500:
                gen_tokens = max(2048, min_ctx - 1500)

        logic_temp = 0.6
        crunch_budget = max(1024, ds_ctx - gen_tokens - 1000)
        ds_safe = orchestrator._crunch_prompt(prompt, "deepseek_r1", crunch_budget, status_callback)

        is_benchmark = (mode == "BENCHMARK_CODING" or mode == "BENCHMARK")
        max_resets = 1 if is_benchmark else 2
        lessons = ""
        initial_failed_code = ""
        initial_failed_error = ""
        c_header_rule = ""

        planner_sys = (
            "You are a world-class software architect and algorithm planner.\n"
            "Your task is to draft a clean, step-by-step logic plan for the user's request.\n\n"
            "MANDATORY ACCURACY RULES:\n"
            "1. ADHERE TO USER SPECIFICATIONS: Pay strict attention to the target programming language, data structures, algorithms, and libraries requested by the user.\n"
            "2. ALGORITHMIC & ARCHITECTURAL RIGOR: Outline the exact data structures, time/space complexity, function signatures, and step-by-step implementation logic.\n"
            "3. LANGUAGE ACCURACY: Use standard idioms, built-in types, and standard library headers for the target language (e.g. <stdio.h>, <stdlib.h>, <pthread.h>, <stdatomic.h> for C).\n"
            "4. OUTPUT FORMAT: Write a numbered list of steps explaining the logic, data structures, and edge cases.\n"
            "5. THINKING CONSTRAINT: Keep your reasoning focused and proceed directly to the implementation plan."
        )

        coder_sys = (
            "You are an expert computational programmer and software engineer.\n"
            "Your job is to translate the logic plan into a complete, clean, and immediately runnable Python script.\n\n"
            "STRICT RULES:\n"
            "1. Implement equations EXACTLY as described in the plan.\n"
            "2. Do NOT write placeholders, mock functions, or abbreviated loop bodies.\n"
            "3. Handle edge cases: division by zero, array bounds, negative sqrt.\n"
            "4. AUTOMATED SELF-TESTING MANDATE: Append a test suite at the bottom of the script using strict assertions."
        )

        benchmark_coder_sys = (
            "You are an expert computational programmer and algorithm engineer.\n"
            "Your task is to write a complete, efficient, and bug-free Python solution implementing the requested function.\n"
            "STRICT RULES:\n"
            "1. Output ONLY valid Python code inside ```python``` blocks.\n"
            "2. Complete the requested function directly with pure Python logic.\n"
            "3. Do NOT add artificial type checking or raise unexpected ValueErrors.\n"
            "4. Do NOT generate example print statements, dummy calls, or top-level assertion tests."
        )

        for reset in range(max_resets):
            max_rounds = 1 if (is_benchmark or reset > 0) else 2
            for rnd in range(max_rounds):
                # Phase 1: Logic Plan
                orchestrator._check_cancelled("code:draft_logic")
                is_nuclear = (reset > 0)
                model_key = "deepseek_r1" if is_nuclear else "vibethinker"
                model_name = orchestrator._get_display_model_name(model_key)

                if status_callback:
                    lbl = f"🧠 {model_name} drafting logic..." if is_benchmark else f"🧠 {model_name} drafting logic (Attempt {rnd+1}/{max_rounds})..."
                    status_callback(lbl, "info", model_key, 20 + rnd*10)

                ds_llm = orchestrator._get_model(model_key, required_ctx=ds_ctx)
                ds_display = orchestrator._get_display_model_name(model_key)
                plan_p = f"Create a step-by-step logic plan:\n{ds_safe}"
                if lessons:
                    plan_p += f"\n\nLESSONS FROM PREVIOUS FAILURES:\n{lessons[:800]}"
                ds_draft = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, plan_p, gen_tokens, logic_temp, system_prompt=planner_sys))

                # Phase 2: Reasoning Sandbox (Only for Python interactive queries, skip for benchmarks)
                orchestrator._check_cancelled("code:verify_logic")
                active_router = orchestrator._get_model("router", required_ctx=1024)
                use_logic_playground = (not is_benchmark) and (req_lang == "python") and TaskRouter.is_playground_applicable(orchestrator, active_router, prompt)
                verified = True
                pg_out = ""
                if use_logic_playground:
                    if status_callback:
                        status_callback(f"Reasoning Sandbox: Verifying logic with {ds_display}...", "info", model_key, 30 + rnd*10)
                    verified, pg_out, _ = orchestrator._run_playground(ds_llm, ds_draft, "logic", status_callback=status_callback, model_key=model_key, original_prompt=prompt)

                if not verified:
                    ds_llm = orchestrator._get_model("deepseek_r1", required_ctx=ds_ctx)
                    ds_display = orchestrator._get_display_model_name("deepseek_r1")
                    fix_p = f"ORIGINAL REQUEST:\n{prompt}\n\nLogic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\nError:\n{pg_out[:1000]}\nRewrite a corrected logic plan."
                    ds_draft = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, fix_p, gen_tokens, logic_temp, system_prompt=planner_sys))

                compiled_plan = ds_draft

                # Phase 3: Write Code
                orchestrator._check_cancelled("code:write_code")
                oc_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)
                coder_display = orchestrator._get_display_model_name("ornith")
                if status_callback:
                    status_callback(f"💻 {coder_display} writing code...", "info", "ornith", 50 + rnd*10)

                if is_benchmark:
                    code_p = f"Implement a complete, optimal Python solution for this task and plan:\n\nTask:\n{prompt}\n\nPlan:\n{compiled_plan}\n\nOutput ONLY python code in ```python```."
                    sys_prompt = benchmark_coder_sys
                elif req_lang == "python":
                    code_p = f"Write a complete Python script for this plan:\n{compiled_plan}\n\nWrap in ```python```."
                    sys_prompt = coder_sys
                elif req_lang == "html":
                    code_p = f"Implement a complete web application for this plan:\n{compiled_plan}\n\nWrap each file in <file path=\"...\">...</file> blocks."
                    sys_prompt = "You are an expert full-stack web developer."
                else:
                    c_header_rule = ""
                    if req_lang == "c" or req_lang == "cpp":
                        c_header_rule = (
                            "STRICT C/C++ SYNTAX & ENTRYPOINT RULES:\n"
                            "1. MANDATORY int main(): You MUST write a complete 'int main(void)' function with unit tests at the end of the code. Without main(), linking WILL FAIL with 'undefined reference to main'.\n"
                            "2. HEADERS: Use standard C11 <stdatomic.h> and <pthread.h>. NEVER include <atomic.h>.\n"
                            "3. ARRAY POINTER SYNTAX: queue->queue is 'int*'. Access elements as 'queue->queue[i] = val'. NEVER write '*queue->queue[i] = val' or '*queue->queue = (int*)0'.\n"
                            "4. FUNCTION PROTOTYPES: Declare all function prototypes (or place function definitions) BEFORE main() and caller functions to prevent implicit declaration errors.\n"
                            "5. LOCK-FREE ATOMICS: Use standard C11 atomic_compare_exchange_weak_explicit(&var, &expected, desired, memory_order_relaxed, memory_order_relaxed) or atomic_compare_exchange_strong.\n\n"
                        )
                    code_p = (
                        f"Write a complete, fully working, self-contained {lang_name} program for this request:\n{prompt}\n\n"
                        f"{c_header_rule}"
                        f"Plan:\n{compiled_plan}\n\n"
                        f"Write ONLY valid {lang_name} code inside ```{req_lang}``` blocks. Include main() with unit tests."
                    )
                    sys_prompt = f"You are an expert {lang_name} systems engineer. Output ONLY code in ```{req_lang}``` blocks. {c_header_rule}"

                raw_model_output = orchestrator._strip_thinking(orchestrator._call_model(oc_llm, code_p, gen_tokens, gen_temp, system_prompt=sys_prompt))
                
                files_dict = None
                if req_lang == "html":
                    files_dict = Sandbox.parse_multi_file_manifest(raw_model_output)
                    code = raw_model_output if files_dict else Sandbox.extract_code(raw_model_output)
                else:
                    code = Sandbox.extract_code(raw_model_output)

                # For automated benchmark evaluation, return code immediately for official harness testing
                if is_benchmark:
                    return f"```python\n{code}\n```"

                # Phase 4: Execution Sandbox
                orchestrator._check_cancelled("code:execute_sandbox")
                if status_callback:
                    status_callback(f"Executing in Sandbox (Attempt {rnd+1}/{max_rounds})...", "info", "sandbox", 60 + rnd*10)

                if files_dict:
                    ok, output_log, temp_dir = orchestrator.sandbox.execute_workspace(files_dict)
                    if ok:
                        return f"### 📂 Multi-File Workspace Generated successfully!\n\nBelow is the live simulation:\n\n{output_log}"
                else:
                    ok, output = orchestrator.sandbox.execute(code, language=req_lang)
                    if ok:
                        orchestrator.memory.save(prompt, code)
                        return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

                if not initial_failed_code:
                    initial_failed_code = code
                    initial_failed_error = output

                # Agent IDE surgical patch / rewrite
                oc_fix = orchestrator._get_model("ornith", required_ctx=oc_ctx)
                fix_p = f"Fix the following {lang_name} code:\n\n{c_header_rule}CODE:\n{code[:2000]}\n\nERROR:\n{output[:800]}\n\nOutput complete script in ```{req_lang}```."
                code = Sandbox.extract_code(orchestrator._strip_thinking(orchestrator._call_model(oc_fix, fix_p, gen_tokens, gen_temp)))
                ok, output = orchestrator.sandbox.execute(code, language=req_lang)
                if ok:
                    orchestrator.memory.save(prompt, code)
                    return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

        # Fallback output if all retries exhausted
        return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

    @staticmethod
    def _detect_target_language(prompt):
        p_lower = prompt.lower()
        if "verilog" in p_lower or "systemverilog" in p_lower:
            return "verilog"
        elif "spice" in p_lower or "ngspice" in p_lower:
            return "spice"
        elif "html" in p_lower or "css" in p_lower or "web app" in p_lower or "website" in p_lower:
            return "html"
        elif " c " in p_lower or "in c language" in p_lower or "in c," in p_lower or p_lower.endswith("in c"):
            return "c"
        elif "c++" in p_lower or "cpp" in p_lower:
            return "cpp"
        elif "java" in p_lower:
            return "java"
        elif "rust" in p_lower:
            return "rust"
        elif "go " in p_lower or "golang" in p_lower:
            return "go"
        elif "bash" in p_lower or "shell script" in p_lower:
            return "bash"
        return "python"
