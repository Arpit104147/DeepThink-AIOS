"""
Evaluation and Verification Engines for AIOS Benchmarks.
Includes sandbox unit test verification and regex math answer parsing.
"""
import re
import asyncio
from typing import Dict, Any, Tuple

async def evaluate_problem_solution(
    orchestrator: Any,
    problem: Dict[str, Any],
    response: str,
    worker_id: int,
    add_log_fn=None
) -> Tuple[bool, int]:
    """
    Evaluates a model output against dataset test cases or expected answers.
    Returns (success_bool, token_count).
    """
    success = False
    
    if "test" in problem and problem["test"]:
        # Programmatic Python execution (HumanEval / MBPP / SWE-bench)
        from backend.sandbox import Sandbox
        extracted_code = Sandbox.extract_code(response)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            raw_lines = [l for l in response.split('\n') if l.strip().startswith(('def ', 'import ', 'from ', 'return '))]
            if raw_lines:
                extracted_code = "\n".join(raw_lines)
                
        if extracted_code and len(extracted_code.strip()) >= 5:
            entry_point = problem.get("entry_point")
            prompt_clean = problem.get("prompt", "").strip()
            
            # Ensure function signature is present exactly once
            if entry_point and f"def {entry_point}" not in extracted_code:
                matches = re.findall(rf"(def {entry_point}\b[\s\S]*?)(?=\ndef |\Z)", response)
                if matches:
                    extracted_code = "\n\n".join(matches)
                else:
                    extracted_code = prompt_clean + "\n" + extracted_code

            typing_imports = (
                "from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable\n"
                "import math\n"
                "import numpy as np\n\n"
            )
            
            test_code = typing_imports + extracted_code + "\n\n" + problem["test"]
            if entry_point and f"check({entry_point})" not in problem["test"]:
                test_code += f"\n\ncheck({entry_point})"
                
            is_success, output = await asyncio.to_thread(orchestrator.sandbox.execute, test_code, "python", timeout=5.0)
            success = is_success
            
            if not success and add_log_fn:
                error_lines = [l for l in str(output).strip().split('\n') if l.strip()]
                fail_reason = error_lines[-1] if error_lines else "Unknown"
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']} failed: {fail_reason[:120]}")
        else:
            if add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']}: No Python code found in response")
    else:
        # Fallback for theoretical/math datasets without programmatic test blocks (GPQA, GSM8K, MATH)
        if "answer" in problem and problem["answer"]:
            expected_answer = str(problem["answer"]).strip().lower()
            if "####" in expected_answer:
                expected_answer = expected_answer.split("####")[-1].strip()
                
            resp_lower = response.lower()
            expected_clean = expected_answer.replace(",", "").replace("$", "").strip()
            
            if expected_clean and re.fullmatch(r"-?\d+(?:\.\d+)?", expected_clean):
                resp_clean = re.sub(r"[,$]", "", resp_lower)
                success = re.search(rf"(?<!\d){re.escape(expected_clean)}(?!\d)", resp_clean) is not None
            else:
                success = expected_answer in resp_lower
        else:
            lower_resp = response.lower()
            has_error = any(err in lower_resp for err in [
                "traceback (most recent call last)",
                "timeouterror:",
                "syntaxerror:",
                "assertionerror:"
            ])
            success = not has_error
            
    generated_tokens = len(response) // 4
    return success, generated_tokens
