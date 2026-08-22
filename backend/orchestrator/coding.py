import os
import re
import gc
import ast
import uuid
import shutil
from backend.sandbox import Sandbox
from backend.downloader import resolve_model_key
from backend.orchestrator.router import TaskRouter

class CodingPipeline:
    """
    Production-Grade Autonomous Coding Pipeline & Algorithmic Complexity Optimizer.
    Features 6-Phase Dual-Agent Logic Planning, Strict Type-Safe Code Generation,
    AST Static Analysis Linting, Cross-Model Devil's Advocate Critic Auditing,
    and Real-Time Sandboxed Automated Verification.
    """

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

        logic_temp = 0.3
        crunch_budget = max(1024, ds_ctx - gen_tokens - 1000)
        ds_safe = orchestrator._crunch_prompt(prompt, "deepseek_r1", crunch_budget, status_callback)

        is_benchmark = (mode == "BENCHMARK_CODING" or mode == "BENCHMARK")
        max_resets = 1 if is_benchmark else 2
        lessons = ""
        initial_failed_code = ""
        initial_failed_error = ""

        planner_sys = (
            f"You are a Distinguished Principal Software Architect and Chief Systems Fellow.\n"
            f"Your task is to draft an exhaustive, optimal algorithmic architecture plan strictly for {lang_name}.\n\n"
            f"MANDATORY ARCHITECTURAL & COMPLEXITY RULES:\n"
            f"1. ALGORITHMIC OPTIMIZATION (Big-O): Enforce optimal time and space complexity bounds (prefer O(1), O(N), or O(N log N) using hash-maps, two-pointers, sliding windows, prefix arrays, or binary search). Avoid naive O(N^2) brute force.\n"
            f"2. MEMORY & DATA STRUCTURES: Specify cache-friendly contiguous data layouts, cacheline padding (64-byte alignment), and atomic lock-free orderings (acquire/release) where concurrency is needed.\n"
            f"3. PROSE ONLY — NO CODE BLOCKS: Write numbered English specification steps. DO NOT write code blocks or triple backticks (```) in the logic plan.\n"
            f"4. DEFENSIVE BOUNDARIES: Detail exact edge cases: null/None inputs, empty collections, single-item collections, boundary integers, and thread-safety invariants."
        )

        coder_sys = (
            f"You are an Expert Principal Computational Software Engineer.\n"
            f"Your task is to implement a complete, optimized, production-grade {lang_name} program based on the logic plan.\n\n"
            f"MANDATORY PRODUCTION CODE QUALITY STANDARDS:\n"
            f"1. STRICT TYPE ANNOTATIONS: Use explicit type hints across all function signatures, parameters, and return types (e.g. from typing import List, Dict, Optional, Tuple, Union, Any in Python; const correctness and templates in C++).\n"
            f"2. GOOGLE-STYLE DOCSTRINGS: Write clean docstrings for every class and public function explaining Args, Returns, Raises, and practical usage Examples.\n"
            f"3. ALGORITHMIC EFFICIENCY: Implement optimal algorithms with minimal memory overhead and zero redundant allocations.\n"
            f"4. DEFENSIVE ERROR HANDLING: Handle edge cases gracefully with explicit exception types and input validation (never bare except:).\n"
            f"5. AUTOMATED SELF-TESTING HARNESS IN main(): Append a complete, executable main() test suite containing assertions covering nominal operations and edge cases (empty inputs, single items, boundary limits).\n"
            f"6. NO LAZINESS OR PLACEHOLDERS: Output ONLY complete, runnable code inside ```{req_lang}``` blocks. Never write '// TODO', mock functions, or abbreviated loop bodies."
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

        # Fast path for automated benchmark evaluations (HumanEval, MBPP, SWE-bench)
        if is_benchmark:
            orchestrator._check_cancelled("code:benchmark_solve")
            oc_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)
            coder_display = orchestrator._get_display_model_name("ornith")
            if status_callback:
                status_callback(f"💻 {coder_display} solving benchmark function...", "info", "ornith", 50)
            code_p = (
                f"Complete this Python function implementation for the automated test harness.\n\n"
                f"PROBLEM:\n{prompt}\n\n"
                f"Output ONLY the complete Python code in ```python``` blocks."
            )
            raw_model_output = orchestrator._strip_thinking(
                orchestrator._call_model(oc_llm, code_p, gen_tokens=1024, temperature=0.1, system_prompt=benchmark_coder_sys)
            )
            code = Sandbox.extract_code(raw_model_output)
            if not code or len(code.strip()) < 5:
                code = raw_model_output
            return f"```python\n{code}\n```"

        for reset in range(max_resets):
            max_rounds = 1 if reset > 0 else 2
            for rnd in range(max_rounds):
                # Phase 1: Logic & Algorithmic Architecture Plan
                orchestrator._check_cancelled("code:draft_logic")
                is_nuclear = (reset > 0)
                model_key = "deepseek_r1" if is_nuclear else "vibethinker"
                model_name = orchestrator._get_display_model_name(model_key)

                if status_callback:
                    lbl = f"🧠 {model_name} drafting algorithmic architecture (Attempt {rnd+1}/{max_rounds})..."
                    status_callback(lbl, "info", model_key, 20 + rnd*10)

                ds_llm = orchestrator._get_model(model_key, required_ctx=ds_ctx)
                ds_display = orchestrator._get_display_model_name(model_key)
                plan_p = f"TARGET PROGRAMMING LANGUAGE: {lang_name}\n\nUSER REQUEST: {ds_safe}\n\nCreate an optimal algorithmic architecture and data structure plan strictly for {lang_name}:"
                
                # Recall past verified experiences from persistent vector memory
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

                # Phase 2: Reasoning Sandbox Logic Verification
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
                    fix_p = f"ORIGINAL REQUEST ({lang_name}):\n{prompt}\n\nLogic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\nError:\n{pg_out[:1000]}\nRewrite an optimized, corrected logic plan strictly for {lang_name}."
                    ds_draft = orchestrator._strip_thinking(orchestrator._call_model(ds_llm, fix_p, gen_tokens, logic_temp, system_prompt=planner_sys))

                compiled_plan = ds_draft

                # Phase 3: Production Code Synthesis
                orchestrator._check_cancelled("code:write_code")
                oc_llm = orchestrator._get_model("ornith", required_ctx=oc_ctx)
                coder_display = orchestrator._get_display_model_name("ornith")
                if status_callback:
                    status_callback(f"💻 {coder_display} synthesizing production {lang_name} code...", "info", "ornith", 50 + rnd*10)

                if req_lang == "python":
                    code_p = (
                        f"USER REQUEST: {prompt}\n\n"
                        f"ARCHITECTURAL PLAN:\n{compiled_plan}\n\n"
                        f"Write a complete, highly optimized, type-annotated, production-grade Python script with Google docstrings and self-testing assertions in main(). Wrap in ```python```."
                    )
                    sys_prompt = coder_sys
                elif req_lang == "html":
                    is_3d_viz = any(k in prompt.lower() for k in [
                        "3d", "three", "webgl", "model", "visual", "diagram", "dna",
                        "crispr", "protein", "membrane", "simulation", "molecule", "cell"
                    ])
                    if is_3d_viz:
                        sys_prompt = (
                            "You are a master WebGL and 3D scientific visualization engineer.\n"
                            "Create a complete, single-file, interactive 3D HTML page using Three.js (r128).\n\n"
                            "STRICT RULES:\n"
                            "1. Use CDN scripts in <head>:\n"
                            "   <script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>\n"
                            "   <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>\n"
                            "2. Visual Design: Dark theme background (#0a0d14), studio lighting (AmbientLight + DirectionalLight), and 60 FPS OrbitControls.\n"
                            "3. Accurate Scientific Geometry: Implement custom Three.js meshes and parametric curves accurately representing the requested biological, chemical, or physical structure.\n"
                            "4. Interactive HUD: Include a clean glassmorphic HUD overlay in the top-right corner with title, color legend, and component descriptions.\n"
                            "5. Output ONLY valid, runnable single-file HTML inside ```html``` blocks."
                        )
                        code_p = f"USER REQUEST: {prompt}\n\nPLAN:\n{compiled_plan}\n\nWrite the complete, interactive 3D Three.js HTML page wrapped in ```html```."
                    else:
                        code_p = f"Implement a complete, beautiful web application for this request:\n{prompt}\n\nPlan:\n{compiled_plan}\n\nWrap each file in <file path=\"...\">...</file> blocks or output single-file HTML in ```html```."
                        sys_prompt = "You are an expert full-stack web developer. Output complete, production-grade web code."
                else:
                    c_header_rule = ""
                    if req_lang == "cpp":
                        c_header_rule = (
                            "STRICT MODERN C++17 CONCURRENCY & DATA STRUCTURE RULES:\n"
                            "1. HEADERS: Use standard C++17 headers: <iostream>, <vector>, <unordered_map>, <mutex>, <shared_mutex>, <thread>, <cassert>, <atomic>, <memory>, <chrono>.\n"
                            "   NEVER write '#include <lock_guard>' (not a header file; include <mutex> or <shared_mutex> instead).\n"
                            "2. MUTEX & LOCK SCOPE: Store `std::mutex` or `mutable std::shared_mutex` as class member variables.\n"
                            "   NEVER declare `std::lock_guard` or `std::unique_lock` as a struct/class field member. Instantiate `std::unique_lock<std::shared_mutex> lock(mutex_);` or `std::shared_lock<std::shared_mutex> lock(mutex_);` ONLY inside function bodies.\n"
                            "3. LRU CACHE ARCHITECTURE: If building an LRU cache, implement a clean `LRUCache` class with `std::unordered_map` + custom doubly-linked list (`Node* head_, tail_`).\n"
                            "   - `get(key, value)`: Finds node, moves it to head, returns true/false.\n"
                            "   - `put(key, value)`: Updates node or inserts at head, evicting tail if size > capacity.\n"
                            "4. SPSC LOCK-FREE QUEUE: If building a lock-free queue, use `alignas(64) std::atomic<size_t> head_{0}; alignas(64) std::atomic<size_t> tail_{0};` with `std::memory_order_acquire` and `std::memory_order_release`.\n"
                            "5. MANDATORY UNIT TESTS IN main(): Write a complete `int main()` that spawns multiple threads to test concurrent operations and verifies state using `assert(...)`.\n\n"
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
                        f"Write a complete, fully working, self-contained, high-performance {lang_name} program for this request:\n{prompt}\n\n"
                        f"{c_header_rule}"
                        f"Plan:\n{compiled_plan}\n\n"
                        f"Write ONLY valid {lang_name} code inside ```{req_lang}``` blocks. Include main() with complete unit tests."
                    )
                    sys_prompt = f"You are an expert {lang_name} systems engineer. Output ONLY complete, runnable code in ```{req_lang}``` blocks.\n{c_header_rule}"

                raw_model_output = orchestrator._strip_thinking(orchestrator._call_model(oc_llm, code_p, gen_tokens, gen_temp, system_prompt=sys_prompt))
                
                files_dict = None
                if req_lang == "html":
                    files_dict = Sandbox.parse_multi_file_manifest(raw_model_output)
                    code = raw_model_output if files_dict else Sandbox.extract_code(raw_model_output)
                else:
                    code = Sandbox.extract_code(raw_model_output)

                # Phase 4: AST Static Analysis & Linting Pass (Python)
                if req_lang == "python" and code and not files_dict:
                    ast_ok, ast_err = CodingPipeline._lint_python_ast(code)
                    if not ast_ok:
                        if status_callback:
                            status_callback("🛡️ AST Linter detected syntax defect, repairing...", "warning", "ornith", 55 + rnd*10)
                        fix_ast_p = f"Fix Python syntax error in the following code:\n\nERROR:\n{ast_err}\n\nCODE:\n{code}\n\nOutput complete fixed Python code in ```python```."
                        code = Sandbox.extract_code(orchestrator._strip_thinking(orchestrator._call_model(oc_llm, fix_ast_p, gen_tokens, gen_temp, system_prompt=coder_sys)))

                # Phase 5: Execution Sandbox & Automated Assertion Testing
                orchestrator._check_cancelled("code:execute_sandbox")
                if status_callback:
                    status_callback(f"Executing in Sandbox (Attempt {rnd+1}/{max_rounds})...", "info", "sandbox", 65 + rnd*10)

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

                # Phase 6: Agentic Compiler Error Auto-Patching Loop
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
    def _lint_python_ast(code):
        """Validates Python code via Abstract Syntax Tree (AST) parsing before sandbox execution."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}"
        except Exception as ex:
            return False, str(ex)

    @staticmethod
    def _detect_target_language(prompt):
        p_lower = prompt.lower()
        if "verilog" in p_lower or "systemverilog" in p_lower:
            return "verilog"
        elif "spice" in p_lower or "ngspice" in p_lower:
            return "spice"
        elif any(k in p_lower for k in [
            "html", "css", "web app", "website", "landing page",
            "3d model", "3d visualization", "3d diagram", "three.js", "threejs",
            "webgl", "interactive 3d", "render 3d", "interactive diagram", "canvas",
            "double helix", "crispr", "dna diagram", "protein 3d", "cell membrane",
            "molecular model", "3d simulation"
        ]):
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
