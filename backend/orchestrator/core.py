import os
import gc
import re
import time
import shutil
import uuid
import datetime
import threading
from backend.downloader import get_model_path, is_model_downloaded, resolve_model_key, MODEL_DEFINITIONS
from backend.sandbox import Sandbox
from backend.memory import Memory
from backend.search import WebSearch

from backend.orchestrator.dma import TransformerWrapper, psutil
from backend.orchestrator.router import TaskRouter
from backend.orchestrator.vision import VisionEngine
from backend.orchestrator.coding import CodingPipeline
from backend.orchestrator.reasoning import ReasoningPipeline
from backend.orchestrator.prediction import PredictionPipeline
from backend.orchestrator.chip_design import ChipDesignPipeline
from backend.orchestrator.study import StudyPipeline

try:
    import torch
except ImportError:
    torch = None


class AgentOrchestrator:
    """Production-grade multi-agent coordinator with Dynamic Memory Allocation (DMA),
    VRAM/RAM hardware tracking, and 7-way intelligent pipeline routing."""

    def __init__(self, cancel_event=None):
        self.cancel_event = cancel_event
        self.loaded_models = {}
        self.model_access_order = []
        self.model_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.memory = Memory()
        self.sandbox = Sandbox()
        self.web_search = WebSearch()

        # Dynamic Memory Allocator Safety Thresholds
        self.ram_safety_gb = 1.5
        self.vram_safety_gb = 1.0
        self.device_mode = "gpu"
        self.gpu_layers = -1
        self.search_mode = "off"
        self.context_length = 0
        self.max_tokens = 2048
        self.temperature = 0.7
        self.dual_gpu_pipeline = False
        self.memory_mode = "normal"
        self.max_auto_ctx = 16384

    @property
    def router(self):
        return TaskRouter

    def update_settings(self, context_length=0, max_tokens=2048, temperature=0.7, device_mode="gpu", gpu_layers=-1, search_mode="off"):
        """Update runtime orchestrator configuration settings."""
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device_mode = device_mode
        self.gpu_layers = gpu_layers
        self.search_mode = search_mode

    # ── DMA Memory Management Helpers ────────────────────────────────────
    def _get_ram_free_gb(self):
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return 8.0

    def _get_vram_free_gb(self, gpu_idx=0):
        if torch and torch.cuda.is_available():
            try:
                free_b, _ = torch.cuda.mem_get_info(gpu_idx)
                return free_b / (1024 ** 3)
            except Exception:
                pass
        return None

    def _is_model_valid(self, model_obj):
        if model_obj is None:
            return False
        if hasattr(model_obj, 'close') and callable(getattr(model_obj, 'close')):
            if hasattr(model_obj, '_stack') and getattr(model_obj, '_stack') is None:
                return False
        return True

    def _is_gpu_resident(self, model_obj):
        if model_obj is None:
            return False
        if hasattr(model_obj, '_n_gpu_layers'):
            return getattr(model_obj, '_n_gpu_layers', 0) != 0
        if hasattr(model_obj, 'device'):
            return str(getattr(model_obj, 'device', 'cpu')).lower() in ['cuda', 'xpu']
        return False

    def _touch_model(self, model_key):
        if model_key in self.model_access_order:
            self.model_access_order.remove(model_key)
        self.model_access_order.append(model_key)

    def _empty_gpu_caches(self):
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _close_model(self, model_obj, name=None):
        if model_obj is None:
            return
        chat_handler = getattr(model_obj, "chat_handler", None)
        if chat_handler:
            exit_stack = getattr(chat_handler, "_exit_stack", None)
            if exit_stack:
                try:
                    exit_stack.close()
                except Exception:
                    pass
            model_obj.chat_handler = None
        try:
            setattr(model_obj, '__del__', lambda *args, **kwargs: None)
        except Exception:
            pass
        if hasattr(model_obj, 'close'):
            try:
                close_fn = getattr(model_obj, 'close', None)
                if callable(close_fn):
                    model_obj.close = lambda *args, **kwargs: None
                    close_fn()
            except Exception:
                pass
        gc.collect()
        self._empty_gpu_caches()

    def unload_all_models(self):
        with self.model_lock:
            with self.inference_lock:
                for key in list(self.loaded_models.keys()):
                    model_obj = self.loaded_models[key]
                    self._close_model(model_obj, key)
                    del self.loaded_models[key]
                self.loaded_models.clear()
                self.model_access_order.clear()
        gc.collect()
        self._empty_gpu_caches()

    def _evict_lru_gpu_model(self, incoming_key):
        """Evict the least-recently-used GPU-resident model to free VRAM for a new load."""
        # Find GPU-resident models that aren't the one we're about to load
        gpu_models = [
            k for k in self.model_access_order
            if k != incoming_key and k in self.loaded_models and self._is_gpu_resident(self.loaded_models[k])
        ]
        if not gpu_models:
            return
        # Check if we have enough free VRAM (need ~2GB minimum for a new model)
        free_vram = self._get_vram_free_gb()
        if free_vram is not None and free_vram > 2.0:
            return  # Enough VRAM, no eviction needed
        # Evict the oldest (least recently used) GPU model
        evict_key = gpu_models[0]
        print(f"♻️ DMA: Evicting '{evict_key}' from GPU to free VRAM for '{incoming_key}'...")
        model_obj = self.loaded_models.pop(evict_key, None)
        if evict_key in self.model_access_order:
            self.model_access_order.remove(evict_key)
        self._close_model(model_obj, evict_key)

    def _get_model(self, model_key, required_ctx=None, force_cpu=False):
        model_key = resolve_model_key(model_key)
        if required_ctx is None:
            required_ctx = self.context_length if self.context_length > 0 else 8192

        is_cpu = (self.device_mode == "cpu" or force_cpu)

        with self.model_lock:
            if model_key in self.loaded_models:
                model_obj = self.loaded_models[model_key]
                if not self._is_model_valid(model_obj):
                    self.loaded_models.pop(model_key, None)
                    if model_key in self.model_access_order:
                        self.model_access_order.remove(model_key)
                elif not is_cpu and not self._is_gpu_resident(model_obj):
                    # Model was loaded on CPU (e.g. preloaded in RAM), but GPU is requested!
                    # Evict from CPU cache and reload properly into GPU VRAM
                    print(f"🔄 Promoting '{model_key}' from CPU RAM to GPU VRAM...")
                    self.loaded_models.pop(model_key, None)
                    if model_key in self.model_access_order:
                        self.model_access_order.remove(model_key)
                    self._close_model(model_obj, model_key)
                else:
                    self._touch_model(model_key)
                    return model_obj

            return self._load_model_synchronized(model_key, required_ctx, force_cpu)

    def _load_model_synchronized(self, model_key, required_ctx=None, force_cpu=False):
        if required_ctx is None:
            required_ctx = self.context_length if self.context_length > 0 else 8192

        model_path = get_model_path(model_key)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for '{model_key}' not found on disk.")

        if model_path.endswith('.gguf'):
            from llama_cpp import Llama

            # Determine GPU layers: respect user setting, default to all layers (-1)
            if self.device_mode == "cpu" or force_cpu:
                gpu_layers = 0
            elif self.gpu_layers is not None and self.gpu_layers >= 0:
                gpu_layers = self.gpu_layers
            else:
                gpu_layers = -1

            # VRAM-aware eviction: free stale GPU models before loading new one
            if gpu_layers != 0:
                self._evict_lru_gpu_model(model_key)

            kwargs = {
                "model_path": model_path,
                "n_ctx": required_ctx,
                "n_gpu_layers": gpu_layers,
                "verbose": False
            }

            # Check mmproj for vision models
            if model_key == "qwen_vl" or "vl" in model_path.lower():
                search_dirs = [os.path.dirname(model_path), "models"]
                mmproj_path = None
                for s_dir in search_dirs:
                    if os.path.exists(s_dir):
                        for f in os.listdir(s_dir):
                            if "mmproj" in f.lower() and f.endswith(".gguf"):
                                mmproj_path = os.path.join(s_dir, f)
                                break
                    if mmproj_path:
                        break
                if mmproj_path:
                    try:
                        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                        kwargs["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                        kwargs["clip_model_path"] = mmproj_path
                    except Exception:
                        pass

            llm = Llama(**kwargs)
            llm._n_gpu_layers = kwargs["n_gpu_layers"]
            self.loaded_models[model_key] = llm
            self._touch_model(model_key)
            return llm
        elif model_path.endswith('.safetensors') or os.path.isdir(model_path):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_dir = os.path.dirname(model_path) if model_path.endswith('.safetensors') else model_path
            device = "cuda" if (self.device_mode != "cpu" and torch and torch.cuda.is_available()) else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
                trust_remote_code=True
            )
            wrapper = TransformerWrapper(model, tokenizer, device, orchestrator=self)
            self.loaded_models[model_key] = wrapper
            self._touch_model(model_key)
            return wrapper
        else:
            raise Exception(f"Unsupported model format for '{model_key}': {model_path}")

    def _call_model(self, model, prompt, max_tokens=512, temperature=0.7, system_prompt=None):
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("Generation cancelled by user.")
        
        # Acquire inference lock with timeout to prevent deadlock
        acquired = self.inference_lock.acquire(timeout=60.0)
        if not acquired:
            raise RuntimeError("Inference lock acquisition timed out.")
        try:
            if self.cancel_event and self.cancel_event.is_set():
                raise RuntimeError("Generation cancelled by user.")

            # Run inference in a sub-thread with a hard timeout to prevent
            # infinite hangs when models silently fall back to CPU
            result_container = [None]
            error_container = [None]

            def _inference():
                try:
                    if hasattr(model, 'create_chat_completion'):
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        resp = model.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
                        result_container[0] = resp['choices'][0]['message']['content']
                    elif isinstance(model, TransformerWrapper):
                        result_container[0] = model(prompt, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt)
                    elif callable(model):
                        result_container[0] = model(prompt, max_tokens=max_tokens, temperature=temperature)
                    else:
                        error_container[0] = Exception("Unknown model callable signature.")
                except Exception as e:
                    error_container[0] = e

            inference_thread = threading.Thread(target=_inference, daemon=True)
            inference_thread.start()

            # Hard timeout: 180s default, scaled by token count
            timeout_secs = min(300, max(120, max_tokens * 0.04))
            inference_thread.join(timeout=timeout_secs)

            if inference_thread.is_alive():
                # Thread is still running — timed out
                if self.cancel_event:
                    self.cancel_event.set()  # Signal cancellation to stop the model
                raise RuntimeError(
                    f"Model inference timed out after {int(timeout_secs)}s "
                    f"(max_tokens={max_tokens}). The model may be running on CPU. "
                    f"Check GPU settings."
                )

            if error_container[0]:
                raise error_container[0]

            return result_container[0]
        finally:
            self.inference_lock.release()

    def _strip_thinking(self, text):
        from backend.orchestrator.dma import TransformerWrapper
        if not text:
            return text
        tag_patterns = [
            (r'<think>.*?</think>', r'</?think>'),
            (r'<thought>.*?</thought>', r'</?thought>'),
            (r'\[THINKING\].*?\[/THINKING\]', r'\[/?THINKING\]'),
        ]
        cleaned = text
        for block_pat, inline_pat in tag_patterns:
            if re.search(block_pat, cleaned, flags=re.DOTALL):
                res = re.sub(block_pat, '', cleaned, flags=re.DOTALL).strip()
                if res:
                    cleaned = res
                else:
                    cleaned = re.sub(inline_pat, '', cleaned).strip()

        for open_tag in ['<think>', '<thought>', '[THINKING]']:
            close_tag = open_tag.replace('<', '</').replace('[', '[/')
            if open_tag in cleaned and close_tag not in cleaned:
                parts = cleaned.split(open_tag, 1)
                before_think = parts[0].strip()
                after_think = parts[1].strip() if len(parts) > 1 else ""
                # Filter out internal reasoning from after_think
                if "\n\n" in after_think:
                    paragraphs = [p.strip() for p in after_think.split("\n\n") if p.strip()]
                    content_paragraphs = [p for p in paragraphs if not p.lower().startswith(("okay,", "so,", "i need to", "i should", "first, i", "let's see", "i'll start"))]
                    after_clean = "\n\n".join(content_paragraphs) if content_paragraphs else after_think
                else:
                    after_clean = after_think
                # Keep both before and after content (don't drop after_think)
                if before_think and after_clean:
                    cleaned = before_think + "\n\n" + after_clean
                elif before_think:
                    cleaned = before_think
                elif after_clean:
                    cleaned = after_clean

        lines = cleaned.split("\n")
        dedup_lines = []
        prev_stripped = None
        consec_count = 0
        for line in lines:
            stripped_line = line.strip()
            if len(stripped_line) > 15 and stripped_line == prev_stripped:
                consec_count += 1
                if consec_count >= 3:
                    continue  # Skip 4th+ consecutive identical line
            else:
                consec_count = 0
            prev_stripped = stripped_line
            dedup_lines.append(line)
        return "\n".join(dedup_lines).strip()

    def _clean_cutoff_notes(self, text):
        if not text:
            return text
        patterns = [
            r'Knowledge cutoff:.*?\n',
            r'Note: My knowledge cutoff is.*?\n',
            r'As an AI developed by.*?\n'
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Compact extra blank lines between markdown table rows
        cleaned = re.sub(r'(\|[^\n]+\|)\n\s*\n+(\|[^\n]+\|)', r'\1\n\2', cleaned)
        return cleaned.strip()

    def _compute_headroom(self):
        ds_ctx = 8192
        oc_ctx = 8192
        router_ctx = 2048
        gen_tokens = self.max_tokens
        gen_temp = self.temperature
        return ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp

    def _crunch_prompt(self, prompt, model_key, budget, status_callback=None, router_llm=None):
        if len(prompt) // 4 <= budget:
            return prompt
        return prompt[:budget * 4]

    def _get_display_model_name(self, model_key):
        """Get the actual user-friendly display name of the model on disk / assigned role."""
        resolved_key = resolve_model_key(model_key)
        model_path = None
        try:
            model_path = get_model_path(resolved_key)
        except Exception:
            pass
            
        if model_path and os.path.exists(model_path):
            filename = os.path.basename(model_path)
            return filename.replace('.gguf', '').replace('.safetensors', '')
        return MODEL_DEFINITIONS.get(resolved_key, {}).get("name", MODEL_DEFINITIONS.get(model_key, {}).get("name", model_key))

    def _check_cancelled(self, stage_name=""):
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError(f"Generation cancelled at stage '{stage_name}'.")

    def _is_playground_applicable(self, router_llm, prompt):
        return TaskRouter.is_playground_applicable(self, router_llm, prompt)

    def _classify_task(self, router_llm, prompt):
        return TaskRouter.classify_task(self, router_llm, prompt)

    def _run_playground(self, model, hypothesis, purpose="logic", status_callback=None, model_key=None, original_prompt=None):
        if status_callback:
            status_callback("Reasoning Sandbox: Verifying logic...", "info", model_key, 35)
        coder_llm = self._get_model("ornith", required_ctx=4096)
        script_p = f"Write a short Python validation script to verify this logic:\n{hypothesis}\n\nWrap script in ```python```."
        code = Sandbox.extract_code(self._strip_thinking(self._call_model(coder_llm, script_p, 1024, 0.2)))
        ok, output = self.sandbox.execute(code, language="python")
        return ok, output, code

    def _synthesize_coding_response(self, prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback=None, req_lang="python", execution_passed=True):
        if execution_passed:
            exec_header = "### ⚙️ Sandbox Execution Output (Passed ✅)"
            code_header = "### 💻 Verified Working Code"
        else:
            exec_header = "### ⚙️ Sandbox Execution Diagnostic (Review Required ⚠️)"
            code_header = "### 💻 Generated Implementation"

        return f"### 💡 Logic & Architectural Plan\n\n{compiled_plan}\n\n{exec_header}\n```\n{output[:3000]}\n```\n\n{code_header}\n\n```{req_lang}\n{code}\n```"

    def _generate_3d_visualization(self, prompt, coder_llm, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        viz_prompt = (
            "Write a COMPLETE HTML page rendering an interactive 3D visualization using Three.js or Plotly.js.\n\n"
            "RULES:\n"
            "1. Three.js r128 CDN: https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\n"
            "2. OrbitControls CDN: https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\n"
            "3. Plotly CDN: https://cdn.plot.ly/plotly-2.24.1.min.js\n"
            "4. Dark background #0d0d0d.\n"
            "5. NO ES6 imports. Use global THREE or Plotly.\n"
            "6. STRICT JS RULES: NEVER reference non-existent JavaScript classes (like PolynomialRidgeRegression, scikit-learn). Implement mathematical formulas in pure JavaScript loops or plot 3D surfaces/scatters with Plotly.newPlot.\n"
            "7. DOM CONTAINER MANDATE: Always place <div id=\"plot\" style=\"width:100vw; height:100vh;\"></div> in the <body> BEFORE any <script> tag. Wrap Plotly.newPlot('plot', ...) inside document.addEventListener('DOMContentLoaded', function() { ... }) to ensure the container element is never null.\n\n"
            f"Topic: {prompt}\n\n"
            "Output ONLY complete HTML in ```html``` blocks."
        )
        viz_resp = self._call_model(coder_llm, viz_prompt, max_tokens=2048, temperature=0.2)
        html_extract = Sandbox.extract_code(self._strip_thinking(viz_resp))
        if html_extract and ("THREE" in html_extract or "Plotly" in html_extract or "<script" in html_extract):
            return f"<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"
        return ""

    def _extreme_websearch_pipeline(self, prompt, status_callback=None):
        if status_callback:
            status_callback("🔬 Extreme Research [Stage 1/3]: Ingesting multi-source web & data feeds...", "info", "system", 15)

        raw_contexts = []

        # 1. Check for real-time financial market data
        try:
            fin_table = self.web_search.fetch_financial_quote(prompt)
            if fin_table:
                raw_contexts.append(fin_table)
        except Exception:
            pass

        self._check_cancelled("extreme:web_scrape")

        # 2. Deep multi-source web scrape (Primary Query) — reduced from 8/5 to 6/3
        try:
            scrape_res = self.web_search.search_and_scrape(prompt, max_results=6, max_scrapes=3)
            if isinstance(scrape_res, dict) and not scrape_res.get("empty", True):
                raw_contexts.append(scrape_res.get("context", ""))
            else:
                search_res = self.web_search.search(prompt, max_results=6)
                if isinstance(search_res, list) and search_res:
                    formatted = [f"[{i+1}] {item.get('title', '')} ({item.get('link', item.get('url', ''))}):\n{item.get('snippet', '')}" for i, item in enumerate(search_res[:5])]
                    raw_contexts.append("\n\n".join(formatted))
        except Exception:
            pass

        self._check_cancelled("extreme:sub_query")

        # 3. Targeted Sub-Query snippet search (no deep scrape — just snippets)
        try:
            sub_query = re.sub(r"(tell me|explain|give me|what is|how to|a complete guide for)", "", prompt, flags=re.I).strip()
            if sub_query and len(sub_query) > 5 and sub_query.lower() != prompt.lower():
                sub_res = self.web_search.search(f"{sub_query} latest analysis overview", max_results=3)
                if isinstance(sub_res, list) and sub_res:
                    formatted_sub = [f"[Ref {i+1}] {item.get('title', '')} ({item.get('link', item.get('url', ''))}):\n{item.get('snippet', '')}" for i, item in enumerate(sub_res[:3])]
                    raw_contexts.append("\n\n".join(formatted_sub))
        except Exception:
            pass

        aggregated_web_corpus = "\n\n---\n\n".join(raw_contexts)

        if status_callback:
            status_callback("🔬 Extreme Research [Stage 2/3]: Cross-referencing empirical data & resolving facts...", "info", "vibethinker", 50)

        self._check_cancelled("extreme:synthesis")

        # Stage 3: Deep Technical Synthesis with DeepSeek-R1
        if status_callback:
            status_callback("🔬 Extreme Research [Stage 3/3]: Synthesizing academic research paper with DeepSeek-R1...", "info", "deepseek_r1", 75)

        ds_llm = self._get_model("deepseek_r1", required_ctx=8192)
        is_quantum = any(k in prompt.lower() for k in ["qubit", "quantum", "surface code", "color code", "fault-tolerant", "qec"])
        domain_constraints = ""
        if is_quantum:
            domain_constraints = (
                "STRICT QUANTUM DOMAIN ACCURACY CONSTRAINTS:\n"
                "1. Surface Code Fault-Tolerant Error Threshold is ~0.7% - 1.0% (MWPM decoder). Color Code Threshold is ~0.1% - 0.7%.\n"
                "2. Surface Code requires 2d^2 physical qubits per logical qubit. Color Code requires 7d^2 physical qubits per logical qubit.\n"
                "3. Include a Markdown comparison table with columns: | Code Type | Space Dim | Threshold (%) | Physical Qubit Overhead | Transversal Gates | Decoder |.\n\n"
            )

        research_prompt = (
            "You are a distinguished principal research scientist and technical domain expert.\n"
            "Synthesize an exhaustive, academic-grade technical research report/survey based on the raw multi-source web corpus below.\n\n"
            f"USER INQUIRY: {prompt}\n\n"
            f"RAW MULTI-SOURCE INGESTED RESEARCH DATA:\n{aggregated_web_corpus}\n\n"
            f"{domain_constraints}"
            "MANDATORY REPORT STRUCTURE & GUIDELINES:\n"
            "1. EXECUTIVE SUMMARY: High-level overview, key takeaways, and core metrics/findings.\n"
            "2. IN-DEPTH TECHNICAL ANALYSIS: Detailed breakdown of underlying mechanisms, chronological events/data, statistics, and domain-specific principles.\n"
            "3. EMPIRICAL COMPARISON TABLE: A structured Markdown comparison table contrasting key components, metrics, models, or entities.\n"
            "4. TRADEOFFS & STRATEGIC INSIGHTS: Concrete pros/cons, best practices, or forward-looking projections.\n"
            "5. NUMBERED CITATIONS & SOURCES: Include [1], [2], [3] matching the live sources in a dedicated References section.\n"
            "6. ACCURACY: Rely strictly on real empirical figures from the context without placeholder variables."
        )

        res = self._strip_thinking(self._call_model(ds_llm, research_prompt, max_tokens=3072, temperature=0.25))
        return f"# 🔬 Extreme Web Search & Academic Technical Survey\n\n{res}"

    # ── Multimodal Vision Engine Entrypoint ────────────────────────────────
    def transcribe_image(self, image_input, user_prompt=None, status_callback=None):
        return VisionEngine.transcribe_image(self, image_input, user_prompt, status_callback)

    # ── Main Entrypoint: Process User Query ────────────────────────────────
    def process_query(self, prompt, mode="auto", selected_models=None, status_callback=None, attached_image=None):
        self._check_cancelled("start_query")

        # 1. Determine search mode setting (off, simple/search, prediction, extreme, study)
        search_mode = str(getattr(self, "search_mode", "off")).lower()

        if search_mode in ["study", "stu..."] or (isinstance(mode, str) and mode.upper() == "STUDY"):
            task_type = "STUDY"
        elif search_mode in ["extreme", "ext..."] or (isinstance(mode, str) and mode.upper() == "EXTREME_WEBSEARCH"):
            task_type = "EXTREME_WEBSEARCH"
        elif search_mode in ["prediction", "pre..."] or (isinstance(mode, str) and mode.upper() == "PREDICTION"):
            task_type = "PREDICTION"
        elif search_mode in ["simple", "search", "se..."]:
            task_type = "SIMPLE"
        elif isinstance(mode, str) and mode.upper() in ["SIMPLE", "CODING", "REASONING", "CHIP_DESIGN", "STUDY"]:
            task_type = mode.upper()
        else:
            router_llm = self._get_model("router", required_ctx=2048)
            task_type = TaskRouter.classify_task(self, router_llm, prompt)
            if task_type in ["EXTREME_WEBSEARCH", "PREDICTION", "STUDY"]:
                task_type = "SIMPLE"

        if status_callback:
            status_callback(f"Task classified as: {task_type}", "info", "router", 12)

        if task_type == "STUDY":
            res = StudyPipeline.execute(self, prompt, mode, selected_models, status_callback, attached_image=attached_image)
            return self._clean_cutoff_notes(res)

        # 2. Simple Web Search context retrieval if search_mode == simple/search
        web_context = ""
        if search_mode in ["simple", "search", "se..."]:
            if status_callback:
                status_callback("🔍 Simple WebSearch: Scraping live web sources...", "info", "system", 25)
            try:
                # Try deep search and scrape first for rich content
                scrape_res = self.web_search.search_and_scrape(prompt, max_results=5, max_scrapes=3)
                if isinstance(scrape_res, dict) and not scrape_res.get("empty", True):
                    web_context = scrape_res.get("context", "")
                else:
                    search_res = self.web_search.search(prompt, max_results=6)
                    if isinstance(search_res, list) and search_res:
                        formatted = []
                        for idx, item in enumerate(search_res[:6]):
                            title = item.get("title", f"Source {idx+1}")
                            url = item.get("link", item.get("url", ""))
                            snippet = item.get("snippet", item.get("content", ""))
                            if snippet:
                                formatted.append(f"[{idx+1}] {title} ({url}):\n{snippet}")
                        web_context = "\n\n".join(formatted)
                    elif isinstance(search_res, str):
                        web_context = search_res
            except Exception:
                pass

        if task_type == "SIMPLE":
            try:
                vt = self._get_model("vibethinker", required_ctx=2048)
                ans_model = "vibethinker" if self._is_model_valid(vt) else "router"
            except (FileNotFoundError, Exception):
                ans_model = "router"
            llm = self._get_model(ans_model, required_ctx=2048)
            final_p = prompt
            sys_prompt = None
            if web_context:
                sys_prompt = (
                    "You are a helpful, knowledgeable AI assistant with direct access to live real-time internet search and meteorological feeds.\n"
                    "Provide a thorough, accurate, and helpful response to the user's inquiry based on the live data context.\n"
                    "Always present the live metrics, current conditions, exact numbers, and temperatures directly.\n"
                    "Never say 'I cannot provide real-time information' or tell the user to check external apps because live data is provided to you."
                )
                final_p = (
                    f"LIVE REAL-TIME DATA CONTEXT:\n{web_context}\n\n"
                    f"USER QUESTION: {prompt}\n\n"
                    f"Synthesize a clear, comprehensive, and directly informative answer incorporating all key facts and live metrics from the context above."
                )
            res = self._strip_thinking(self._call_model(llm, final_p, max_tokens=max(1024, self.max_tokens), temperature=0.2, system_prompt=sys_prompt))
            return self._clean_cutoff_notes(res)

        elif task_type == "CODING":
            res = CodingPipeline.execute(self, prompt, mode, selected_models, status_callback)
            return self._clean_cutoff_notes(res)

        elif task_type == "REASONING":
            res = ReasoningPipeline.execute(self, prompt, mode, selected_models, status_callback)
            return self._clean_cutoff_notes(res)

        elif task_type == "PREDICTION":
            res = PredictionPipeline.execute(self, prompt, mode, selected_models, status_callback)
            return self._clean_cutoff_notes(res)

        elif task_type == "CHIP_DESIGN":
            res = ChipDesignPipeline.execute(self, prompt, mode, selected_models, status_callback)
            return self._clean_cutoff_notes(res)

        elif task_type == "EXTREME_WEBSEARCH":
            res = self._extreme_websearch_pipeline(prompt, status_callback)
            return self._clean_cutoff_notes(res)

        else:
            router_llm = self._get_model("router", required_ctx=2048)
            res = self._call_model(router_llm, prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            return self._clean_cutoff_notes(res)
