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
    """
    if not text:
        return ""
    pattern = r"```\s*(?:python|py)\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        code = matches[-1].strip()
        code = re.sub(r'^[!%]\s*pip\s+install.*$', '', code, flags=re.MULTILINE).strip()
        return code
    
    generic_matches = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_matches:
        code = generic_matches[-1].strip()
        first_line = code.split('\n')[0].strip().lower()
        known_tags = ['text', 'output', 'bash', 'sh', 'json', 'xml', 'html', 'css']
        if first_line in known_tags:
            if len(generic_matches) >= 2:
                code = generic_matches[-2].strip()
            else:
                return ""
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
        
        if stripped.startswith('```'):
            continue
            
        if re.match(r"if\s+__name__\s*==\s*['\"]__main__['\"]", stripped):
            skip_block = True
            continue
        if skip_block:
            if line.startswith(('    ', '\t')) or stripped == '':
                continue
            else:
                skip_block = False
        
        if stripped.startswith('assert ') and not line.startswith(('    ', '\t')):
            continue
        
        if stripped.startswith('print(') and not line.startswith(('    ', '\t')):
            continue
            
        if entry_point and re.match(rf"^{re.escape(entry_point)}\s*\(", stripped) and not line.startswith(('    ', '\t')):
            continue
        
        if (stripped.startswith('# Test') or stripped.startswith('# Example') or stripped.startswith('# Self-test')) and not line.startswith(('    ', '\t')):
            continue
            
        cleaned.append(line)
    
    return '\n'.join(cleaned).rstrip()


def _parse_math_candidate_answers(response: str) -> list:
    """Extracts potential candidate answers from a mathematical response."""
    candidates = []
    
    # 1. LaTeX \boxed{...} matches
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", response)
    for b in boxed_matches:
        candidates.append(b.strip())
        
    # 2. Markdown **Final Answer:** or **Answer:**
    ans_matches = re.findall(r"\*\*(?:Final )?Answer(?:\*\*)?:?\s*([^\n\*\$]+)", response, re.IGNORECASE)
    for a in ans_matches:
        candidates.append(a.strip())
        
    # 3. Trailing lines with "The answer is ..."
    is_matches = re.findall(r"(?:the answer is|equals|is equal to)\s*([^\n\.\$]+)", response, re.IGNORECASE)
    for m in is_matches:
        candidates.append(m.strip())
        
    return candidates


def _matches_expected_answer(candidate: str, expected: str) -> bool:
    """Compares candidate answer against expected answer with numeric and string normalization."""
    cand_clean = candidate.strip().lower().replace("$", "").replace(",", "").rstrip(".")
    exp_clean = expected.strip().lower().replace("$", "").replace(",", "").rstrip(".")
    
    if cand_clean == exp_clean:
        return True
        
    # Numeric comparison
    try:
        cand_num = float(cand_clean)
        exp_num = float(exp_clean)
        if abs(cand_num - exp_num) < 1e-4:
            return True
    except ValueError:
        pass
        
    # Fraction comparison
    try:
        if "/" in cand_clean:
            num, den = cand_clean.split("/", 1)
            cand_val = float(num) / float(den)
            exp_val = float(exp_clean)
            if abs(cand_val - exp_val) < 1e-4:
                return True
    except Exception:
        pass
        
    return exp_clean in cand_clean


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
                
            candidates = _parse_math_candidate_answers(response)
            for cand in candidates:
                if _matches_expected_answer(cand, expected_answer):
                    success = True
                    break
            
            if not success:
                resp_clean = re.sub(r"[,$]", "", response.lower())
                exp_clean = expected_answer.lower().replace(",", "").replace("$", "").strip()
                if exp_clean and re.fullmatch(r"-?\d+(?:\.\d+)?", exp_clean):
                    success = re.search(rf"(?<!\d){re.escape(exp_clean)}(?!\d)", resp_clean) is not None
                else:
                    success = exp_clean in resp_clean
                    
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
