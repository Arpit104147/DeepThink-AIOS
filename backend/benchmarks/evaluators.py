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
    Remove pipeline self-test artifacts from extracted code that would
    interfere with HumanEval's official check() test harness.
    
    Strips:
    - if __name__ == '__main__' blocks
    - Standalone assert statements (not inside functions)
    - Standalone print("ALL TESTS PASSED") and similar
    - Direct function call invocations at module level
    """
    lines = code.split('\n')
    cleaned = []
    skip_block = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip if __name__ == '__main__' block and everything after
        if re.match(r"if\s+__name__\s*==\s*['\"]__main__['\"]", stripped):
            skip_block = True
            continue
        if skip_block:
            # Skip indented lines that are part of the __main__ block
            if line.startswith(('    ', '\t')) or stripped == '':
                continue
            else:
                skip_block = False
        
        # Skip standalone test assertions (not inside functions — no leading indent)
        if stripped.startswith('assert ') and not line.startswith(('    ', '\t')):
            continue
        
        # Skip standalone print calls at module level
        if stripped.startswith('print(') and not line.startswith(('    ', '\t')):
            continue
            
        # Skip standalone function call tests at module level
        if entry_point and stripped.startswith(f'{entry_point}(') and not line.startswith(('    ', '\t')):
            continue
        
        # Skip "# Test" / "# Self-test" comment headers at module level  
        if stripped.startswith('# Test') and not line.startswith(('    ', '\t')):
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
        
        # Extract the LAST python code block (the verified working code from _coding_pipeline)
        extracted_code = _extract_last_python_block(response)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            # Fallback: try the old extract_code method
            from backend.sandbox import Sandbox
            extracted_code = Sandbox.extract_code(response)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            # Last resort: grab raw function-like lines
            raw_lines = [l for l in response.split('\n') 
                        if l.strip().startswith(('def ', 'import ', 'from ', 'return ', '    '))]
            if raw_lines:
                extracted_code = "\n".join(raw_lines)
                
        if extracted_code and len(extracted_code.strip()) >= 5:
            # Clean pipeline artifacts (self-tests, print statements, etc.)
            extracted_code = _clean_pipeline_artifacts(extracted_code, entry_point)
            
            # Ensure function signature is present exactly once
            if entry_point and f"def {entry_point}" not in extracted_code:
                # Try to find the function in the full response
                matches = re.findall(rf"(def {re.escape(entry_point)}\b[\s\S]*?)(?=\ndef |\Z)", response)
                if matches:
                    # Use the LAST match (most likely the verified code)
                    extracted_code = _clean_pipeline_artifacts(matches[-1], entry_point)
                else:
                    # Prepend the original prompt stub
                    extracted_code = prompt_clean + "\n" + extracted_code

            typing_imports = (
                "from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable\n"
                "import math\n"
                "import re as _re\n"
                "import hashlib\n"
                "from itertools import *\n"
                "from collections import *\n\n"
            )
            
            # Assemble: imports + function code + official test harness
            test_code = typing_imports + extracted_code + "\n\n" + problem["test"]
            if entry_point and f"check({entry_point})" not in problem["test"]:
                test_code += f"\n\ncheck({entry_point})"
                
            is_success, output = await asyncio.to_thread(
                orchestrator.sandbox.execute, test_code, "python", timeout=10.0
            )
            success = is_success
            
            if not success and add_log_fn:
                error_lines = [l for l in str(output).strip().split('\n') if l.strip()]
                fail_reason = error_lines[-1] if error_lines else "Unknown"
                add_log_fn(f"[Worker {worker_id}] ❌ {problem['id']} failed: {fail_reason[:120]}")
            elif success and add_log_fn:
                add_log_fn(f"[Worker {worker_id}] ✅ {problem['id']} passed!")
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
