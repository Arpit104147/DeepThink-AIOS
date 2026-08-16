"""
Evaluation and Verification Engines for AIOS Benchmarks.
Includes sandbox unit test verification, direct model completion for HumanEval/MBPP,
and regex math answer parsing.
"""
import re
import asyncio
import time
from typing import Dict, Any, Tuple


# ── HumanEval / MBPP: Direct Model Completion ────────────────────────────────
# Standard benchmark methodology: ONE model call with the function stub as prompt.
# The model completes the function body. No planning, no reflexion, no sandbox self-test.
# This matches how OpenAI, Google, Meta, and every published leaderboard evaluates models.

HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given Python function.\n"
    "RULES:\n"
    "1. Output ONLY the function body (the implementation), nothing else.\n"
    "2. Do NOT repeat the function signature or docstring.\n"
    "3. Do NOT write test cases, assertions, print statements, or if __name__ blocks.\n"
    "4. Do NOT add any explanation, markdown, or comments outside the code.\n"
    "5. Do NOT wrap your answer in ```python``` code fences.\n"
    "6. Your output will be directly appended after the function signature and docstring.\n"
    "7. Use proper indentation (4 spaces) as this is inside a function.\n"
)


async def direct_model_completion(orchestrator, problem: Dict[str, Any], worker_id: int, add_log_fn=None) -> str:
    """
    Generate a completion for a HumanEval/MBPP problem using a single direct LLM call.
    Bypasses the full 7-phase pipeline which is designed for complex coding tasks,
    not standardized function-completion benchmarks.
    
    Returns the raw model response string.
    """
    prompt = problem["prompt"]
    entry_point = problem.get("entry_point", "")
    
    # Build a focused completion prompt
    user_prompt = (
        f"Complete the following Python function. Output ONLY the function body "
        f"(the lines of code that go inside the function). Do not repeat the signature or docstring.\n\n"
        f"{prompt}"
    )
    
    def _do_inference():
        """Run inference in a thread — uses the orchestrator's model and lock safely."""
        # Get the router model (smallest/fastest available model)
        llm = orchestrator._get_model("router")
        
        # Use a modest token budget — HumanEval solutions are typically short
        response = orchestrator._call_model(
            llm, 
            user_prompt, 
            max_tokens=512, 
            temperature=0.1,
            system_prompt=HUMANEVAL_SYSTEM_PROMPT
        )
        return response
    
    response = await asyncio.to_thread(_do_inference)
    return response


def _extract_function_body(response: str, prompt: str, entry_point: str) -> str:
    """
    Extract the function implementation from a model's response for HumanEval evaluation.
    
    Strategy:
    1. If the response contains the full function (signature + body), extract just that function.
    2. If the response is just indented code (the body), prepend the original prompt stub.
    3. Strip any markdown fences, test code, print statements, and if __name__ blocks.
    """
    text = response.strip()
    
    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    
    # Remove any if __name__ block and everything after it
    main_match = re.search(r'\nif\s+__name__\s*==\s*["\']__main__["\']', text)
    if main_match:
        text = text[:main_match.start()]
    
    # Remove standalone test/assert/print lines at the end (pipeline artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone assertions and print calls that aren't inside functions
        if stripped.startswith(('assert ', 'print(')) and not line.startswith(('    ', '\t')):
            continue
        # Skip lines that call the entry point directly (test invocations)
        if entry_point and stripped.startswith(f'{entry_point}('):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines).rstrip()
    
    # Check if the response includes the full function definition
    if entry_point and f"def {entry_point}" in text:
        # Extract just this function (including its body)
        func_match = re.search(
            rf"(def {re.escape(entry_point)}\b.*?)(?=\ndef |\Z)", 
            text, re.DOTALL
        )
        if func_match:
            return func_match.group(1).rstrip()
    
    # Response is just the body — prepend the original prompt stub
    # Ensure proper indentation
    if text and not text.startswith(('def ', 'class ')):
        # Add indentation if missing
        body_lines = text.split('\n')
        needs_indent = any(line and not line.startswith(('    ', '\t')) for line in body_lines if line.strip())
        if needs_indent:
            text = '\n'.join('    ' + line if line.strip() else line for line in body_lines)
        return prompt.rstrip() + '\n' + text
    
    return text


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
        
        # Extract function implementation from the model response
        extracted_code = _extract_function_body(response, problem.get("prompt", ""), entry_point)
        
        if not extracted_code or len(extracted_code.strip()) < 10:
            # Fallback: try to find any function-like lines
            raw_lines = [l for l in response.split('\n') if l.strip().startswith(('def ', 'import ', 'from ', 'return ', '    '))]
            if raw_lines:
                extracted_code = problem.get("prompt", "").rstrip() + "\n" + "\n".join(raw_lines)

        if extracted_code and len(extracted_code.strip()) >= 5:
            # Ensure the entry point function exists in the code
            if entry_point and f"def {entry_point}" not in extracted_code:
                # Prepend the original prompt stub
                extracted_code = problem.get("prompt", "").rstrip() + "\n" + extracted_code

            typing_imports = (
                "from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable\n"
                "import math\n"
                "import re\n"
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
