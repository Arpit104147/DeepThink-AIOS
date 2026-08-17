"""
Evaluation and Verification Engines for AIOS Benchmarks.
Includes sandbox unit test verification and regex math answer parsing.
"""
import re
import asyncio
import time
from typing import Dict, Any, Tuple


def _extract_last_python_block(text: str) -> str:
    """
    Extract the LAST Python code block from a markdown response.
    
    The AIOS _coding_pipeline returns a rich markdown response with multiple code blocks:
      1. Planner's pseudo-code/logic sketch (FIRST block — NOT the real code)
      2. Sandbox execution output (text block)  
      3. Verified Working Code (LAST python block — THIS is the real code)
    
    Using re.search (which grabs the first match) was grabbing the planner's incomplete
    pseudo-code instead of the final verified code. This function grabs the LAST match.
    """
    # Find ALL python code blocks
    pattern = r"```\s*(?:python|py)\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the LAST python code block (the verified working code)
        code = matches[-1].strip()
        # Remove hallucinated pip install commands
        code = re.sub(r'^[!%]\s*pip\s+install.*$', '', code, flags=re.MULTILINE).strip()
        return code
    
    # Fallback: try generic code blocks
    generic_matches = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_matches:
        code = generic_matches[-1].strip()
        # Skip if it looks like a text/output block
        first_line = code.split('\n')[0].strip().lower()
        known_tags = ['text', 'output', 'bash', 'sh', 'json', 'xml', 'html', 'css']
        if first_line in known_tags:
            # Try the second-to-last block
            if len(generic_matches) >= 2:
                code = generic_matches[-2].strip()
            else:
                return ""
        # Strip language tag from first line if present
        lang_tags = ['python', 'py', 'javascript', 'js', 'c', 'cpp', 'java']
        if first_line in lang_tags:
            code = '\n'.join(code.split('\n')[1:])
        return re.sub(r'^[!%]\s*pip\s+install.*$', '', code, flags=re.MULTILINE).strip()
    
    return ""


def _clean_pipeline_artifacts(code: str, entry_point: str = "") -> str:
    """
    Cleanly strips pipeline self-test artifacts, example calls, and standalone assertions
    from extracted code to ensure pristine integration with official benchmark harnesses.
    """
    lines = code.split('\n')
    cleaned = []
    skip_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip markdown code fence leaks
        if stripped.startswith('```'):
            continue
            
        # Skip if __name__ == '__main__' block and its indented children
        if re.match(r"if\s+__name__\s*==\s*['\"]__main__['\"]", stripped):
            skip_block = True
            continue
        if skip_block:
            if line.startswith(('    ', '\t')) or stripped == '':
                continue
            else:
                skip_block = False
        
        # Skip top-level standalone assertions outside function definitions
        if stripped.startswith('assert ') and not line.startswith(('    ', '\t')):
            continue
        
        # Skip top-level print calls
        if stripped.startswith('print(') and not line.startswith(('    ', '\t')):
            continue
            
        # Skip top-level sample function calls (e.g. is_prime(5))
        if entry_point and re.match(rf"^{re.escape(entry_point)}\s*\(", stripped) and not line.startswith(('    ', '\t')):
            continue
        
        # Skip comment headers for test suites
        if (stripped.startswith('# Test') or stripped.startswith('# Example') or stripped.startswith('# Self-test')) and not line.startswith(('    ', '\t')):
            continue
            
        cleaned.append(line)
    
    return '\n'.join(cleaned).rstrip()


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
        entry_point = problem.get("entry_point", "")
        prompt_clean = problem.get("prompt", "").strip()
        
        # Extract the LAST python code block
        extracted_code = _extract_last_python_block(response)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            from backend.sandbox import Sandbox
            extracted_code = Sandbox.extract_code(response)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            raw_lines = [l for l in response.split('\n') 
                        if l.strip().startswith(('def ', 'import ', 'from ', 'return ', '    ', 'class '))]
            if raw_lines:
                extracted_code = "\n".join(raw_lines)
                
        if extracted_code and len(extracted_code.strip()) >= 5:
            extracted_code = _clean_pipeline_artifacts(extracted_code, entry_point)
            
            # Ensure function/class signature is present
            has_def = entry_point and (
                re.search(rf"\bdef\s+{re.escape(entry_point)}\b", extracted_code) is not None or
                re.search(rf"\bclass\s+{re.escape(entry_point)}\b", extracted_code) is not None
            )

            if entry_point and not has_def:
                matches = re.findall(rf"((?:def|class)\s+{re.escape(entry_point)}\b[\s\S]*?)(?=\n(?:def|class)\s+|\Z)", response)
                if matches:
                    extracted_code = _clean_pipeline_artifacts(matches[-1], entry_point)
                elif prompt_clean and prompt_clean not in extracted_code:
                    extracted_code = prompt_clean + "\n" + extracted_code

            typing_imports = (
                "from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable, Iterable\n"
                "import math\n"
                "import re as _re\n"
                "import hashlib\n"
                "import copy\n"
                "from itertools import *\n"
                "from collections import *\n\n"
            )
            
            # Assemble: imports + solution code + official test harness
            test_code = typing_imports + extracted_code + "\n\n" + problem["test"]
            if entry_point:
                if f"check({entry_point})" not in problem["test"] and "def check(" in problem["test"]:
                    test_code += f"\n\ncheck({entry_point})"
                
            try:
                is_success, output = await asyncio.to_thread(
                    orchestrator.sandbox.execute, test_code, "python", timeout=15.0
                )
                success = is_success
            except Exception as eval_err:
                success = False
                output = f"Execution Error: {eval_err}"
            
            if not success and add_log_fn:
                error_lines = [l for l in str(output).strip().split('\n') if l.strip()]
                fail_reason = error_lines[-1] if error_lines else "Assertion failed"
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']} failed: {fail_reason[:120]}")
            elif success and add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ✅ {problem['id']} passed!")
        else:
            if add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']}: No Python code found in response")
    else:
        # Theoretical / Math datasets (GSM8K, MATH, GPQA, AIME, MuSR, MMLU)
        if "answer" in problem and problem["answer"]:
            expected_answer = str(problem["answer"]).strip()
            if "####" in expected_answer:
                expected_answer = expected_answer.split("####")[-1].strip()
                
            resp_lower = response.lower()
            expected_clean = expected_answer.lower().replace(",", "").replace("$", "").strip()
            
            # Extract boxed answers if present \boxed{...}
            boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", response)
            if boxed_matches:
                boxed_clean = boxed_matches[-1].strip().lower().replace(",", "").replace("$", "")
                if boxed_clean == expected_clean:
                    success = True
            
            if not success:
                if expected_clean and re.fullmatch(r"-?\d+(?:\.\d+)?", expected_clean):
                    resp_clean = re.sub(r"[,$]", "", resp_lower)
                    success = re.search(rf"(?<!\d){re.escape(expected_clean)}(?!\d)", resp_clean) is not None
                else:
                    success = expected_clean in resp_lower
                    
            if success and add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ✅ {problem['id']} passed! (Answer: {expected_answer[:30]})")
            elif not success and add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']} failed. Expected: {expected_answer[:40]}")
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
