import os
import re
import gc
import uuid
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

        logic_temp = 0.5
        crunch_budget = max(1024, ds_ctx - gen_tokens - 1000)
        ds_safe = orchestrator._crunch_prompt(prompt, "deepseek_r1", crunch_budget, status_callback)

        is_benchmark = (mode == "BENCHMARK_CODING" or mode == "BENCHMARK")
        max_resets = 1 if is_benchmark else 2
        lessons = ""
        initial_failed_code = ""
        initial_failed_error = ""

        planner_sys = (
            f"You are a distinguished principal software architect.\n"
            f"Your task is to draft a rigorous, step-by-step logic and algorithmic architecture plan for {lang_name}.\n\n"
            f"MANDATORY ARCHITECTURAL RULES:\n"
            f"1. ALGORITHMIC SPECIFICATION: Detail the exact data structures, concurrency primitives, memory management, time/space complexity, and edge cases.\n"
            f"2. PROSE ONLY: Write clear structured English explanation steps. DO NOT write raw pseudo-code or syntactically incomplete code blocks.\n"
            f"3. THOROUGHNESS: Clearly define all state transitions, lock-free pointer swaps, invariants, and unit test scenarios."
        )

        coder_sys = (
            f"You are an expert computational programmer and software engineer.\n"
            f"Your job is to translate the logic plan into a complete, clean, and immediately runnable {lang_name} program.\n\n"
            f"STRICT RULES:\n"
            f"1. Implement all structures and algorithms EXACTLY as requested.\n"
            f"2. Do NOT write placeholders, mock functions, or abbreviated loop bodies.\n"
            f"3. Handle edge cases: bounds checking, thread safety, memory deallocation.\n"
            f"4. AUTOMATED SELF-TESTING MANDATE: Append a complete self-testing test suite at the bottom with assertions.\n"
            f"5. Output ONLY valid {lang_name} code inside ```{req_lang}``` blocks."
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

        # Track the last code and output for fallback
        code = ""
        output = ""
        compiled_plan = ""

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
                plan_p = f"TARGET PROGRAMMING LANGUAGE: {lang_name}\n\nUSER REQUEST: {ds_safe}\n\nCreate a step-by-step logic and algorithmic architecture plan strictly for {lang_name}:"
                
                # Recall past verified experiences from persistent memory
                try:
                    if hasattr(orchestrator, "memory") and orchestrator.memory and not is_benchmark:
                        past_mem = orchestrator.memory.recall(prompt, n_results=1)
                        if past_mem:
                            plan_p += f"\n\n{past_mem}"
                except Exception:
                    pass

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
                    fix_p = f"ORIGINAL REQUEST ({lang_name}):\n{prompt}\n\nLogic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\nError:\n{pg_out[:1000]}\nRewrite a corrected logic plan strictly for {lang_name}."
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
                    code_p = f"USER REQUEST: {prompt}\n\nPLAN:\n{compiled_plan}\n\nWrite a complete, robust Python script implementing this plan with self-testing assertions. Wrap in ```python```."
                    sys_prompt = coder_sys
                elif req_lang == "html":
                    code_p = f"Implement a complete web application for this plan:\n{compiled_plan}\n\nWrap each file in <file path=\"...\">...</file> blocks."
                    sys_prompt = "You are an expert full-stack web developer."
                else:
                    c_header_rule = ""
                    if req_lang == "cpp":
                        c_header_rule = (
                            "STRICT C++ SYNTAX & STRUCT RULES:\n"
                            "1. HEADERS: Include all necessary standard headers: <iostream>, <atomic>, <memory>, <vector>, <thread>, <mutex>, <cassert>, <chrono>.\n"
                            "2. STRUCT ORDERING: Define all node structs (e.g., `struct Node { ... };`) completely BEFORE any class or method references them. NEVER reference undeclared types.\n"
                            "3. SIGNATURES: Use standard modern C++ syntax (e.g., `Node* push_back(const std::pair<int, std::string>& key)`). Never write invalid parameter casts.\n"
                            "4. ATOMICS: Use `std::atomic<Node*> next;` with explicit memory orderings (`std::memory_order_acquire`, `std::memory_order_release`, `std::memory_order_relaxed`).\n"
                            "5. ENTRYPOINT & UNIT TESTS: Write a complete `int main() { ... }` executing comprehensive unit test assertions.\n\n"
                        )
                    elif req_lang == "c":
                        c_header_rule = (
                            "STRICT C11 SYNTAX & ENTRYPOINT RULES:\n"
                            "1. HEADERS: Use standard C11 <stdatomic.h>, <pthread.h>, <stdio.h>, <stdlib.h>, <assert.h>. NEVER include <atomic.h>.\n"
                            "2. STRUCTS: Define `typedef struct Node { ... } Node;` at the top before usage.\n"
                            "3. ATOMICS: Use standard C11 `atomic_compare_exchange_weak_explicit(&var, &expected, desired, memory_order_relaxed, memory_order_relaxed)`.\n"
                            "4. ENTRYPOINT: Write a complete `int main(void)` with working unit test assertions.\n\n"
                        )

                    code_p = (
                        f"Write a complete, fully working, self-contained {lang_name} program for this request:\n{prompt}\n\n"
                        f"{c_header_rule}"
                        f"Plan:\n{compiled_plan}\n\n"
                        f"Write ONLY valid {lang_name} code inside ```{req_lang}``` blocks. Include main() with unit tests."
                    )
                    sys_prompt = f"You are an expert {lang_name} systems engineer. Output ONLY code in ```{req_lang}``` blocks.\n{c_header_rule}"

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
                    output = output_log
                else:
                    ok, output = orchestrator.sandbox.execute(code, language=req_lang)
                    if ok:
                        orchestrator.memory.save(prompt, code)
                        return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang, execution_passed=True)

                if not initial_failed_code:
                    initial_failed_code = code
                    initial_failed_error = output

                # Agent IDE surgical patch / rewrite
                oc_fix = orchestrator._get_model("ornith", required_ctx=oc_ctx)
                fix_p = f"Fix the following {lang_name} code to resolve the compilation/runtime errors:\n\nCODE:\n{code[:2000]}\n\nCOMPILER ERROR:\n{output[:800]}\n\nOutput complete fixed program in ```{req_lang}```."
                code = Sandbox.extract_code(orchestrator._strip_thinking(orchestrator._call_model(oc_fix, fix_p, gen_tokens, gen_temp)))
                ok, output = orchestrator.sandbox.execute(code, language=req_lang)
                if ok:
                    orchestrator.memory.save(prompt, code)
                    return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang, execution_passed=True)

        # Fallback output if all retries exhausted
        return orchestrator._synthesize_coding_response(prompt, compiled_plan, code, output, ds_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang, execution_passed=False)

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
