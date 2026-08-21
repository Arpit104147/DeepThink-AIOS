import os
import sys
import re
import subprocess

# Standard backend setup
os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "0")

# Auto-release ports 8000 & 8080 on Linux/Kaggle environments before initializing
if sys.platform != 'win32':
    for _port in [8000, 8080]:
        try:
            subprocess.run(f"fuser -k -9 {_port}/tcp", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception:
            pass

# Add root folder to sys.path to resolve backend package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import json
import asyncio
import gc
import shutil
try:
    import psutil
except ImportError:
    class _MockVM:
        def __init__(self):
            self.total = 16 * (1024 ** 3)
            self.available = 8 * (1024 ** 3)
            self.used = 8 * (1024 ** 3)
            self.percent = 50.0
            if sys.platform == 'win32':
                try:
                    import ctypes
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                        ]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        self.total = stat.ullTotalPhys
                        self.available = stat.ullAvailPhys
                    self.used = self.total - self.available
                    self.percent = round((self.used / self.total) * 100, 1) if self.total else 50.0
                except Exception:
                    pass
            else:
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                self.total = int(line.split()[1]) * 1024
                            elif line.startswith('MemAvailable:'):
                                self.available = int(line.split()[1]) * 1024
                    self.used = self.total - self.available
                    self.percent = round((self.used / self.total) * 100, 1) if self.total else 50.0
                except Exception:
                    pass
    class _MockPsutil:
        def virtual_memory(self):
            return _MockVM()
        def cpu_percent(self):
            return 0.0
    psutil = _MockPsutil()
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from fastapi.responses import HTMLResponse
from backend.memory import Memory
from backend.downloader import (
    check_models_status, download_model, MODEL_DEFINITIONS,
    add_custom_model, remove_custom_model, delete_downloaded_model_files,
    ROLE_ASSIGNMENTS, save_role_assignments, cancel_model_download
)
from backend.orchestrator import AgentOrchestrator
from backend.benchmarks.runner import BENCHMARK_STATE, run_benchmark_suite, stop_benchmark, STATE_LOCK
from backend.vulkan_engine import (
    check_vulkan_engine_status, start_vulkan_update_background, get_vulkan_gpu_diagnostics
)

# Phase 3: Security & Air-Gap
try:
    from backend.security import (
        air_gap, AUTH_ENABLED, create_jwt_token,
        verify_jwt_token, get_user_id_from_token, scan_code_sast
    )
except ImportError:
    from security import (
        air_gap, AUTH_ENABLED, create_jwt_token,
        verify_jwt_token, get_user_id_from_token, scan_code_sast
    )

# Phase 4: Git Agent
try:
    from backend.git_agent import GitAgent
    git_agent = GitAgent()
except ImportError:
    try:
        from git_agent import GitAgent
        git_agent = GitAgent()
    except ImportError:
        git_agent = None

app = FastAPI(title="Local Multi-Agent XPU System API")

# Allow CORS for React frontend (development and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

os.makedirs("workspaces", exist_ok=True)
app.mount("/workspaces", StaticFiles(directory="workspaces"), name="workspaces")

# Cancellation flag — set when user clicks Cancel to stop generation (models stay loaded)
generation_cancel = threading.Event()

# Global orchestrator instance — receives the cancel flag for mid-generation stopping
orchestrator = AgentOrchestrator(cancel_event=generation_cancel)

# Lock and progress tracker for downloads
download_lock = threading.Lock()
download_progress = {}

# Global lock to prevent multi-user orchestrator state collisions
chat_lock = threading.Lock()

class ChatRequest(BaseModel):
    prompt: str
    mode: str  # reasoning, coding, writing, searching, auto
    context_length: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.7
    selected_models: Optional[List[str]] = None
    image: Optional[str] = None
    device_mode: Optional[str] = None
    gpu_layers: Optional[int] = None
    search_mode: str = "off"  # off, simple, prediction, extreme
    history: Optional[List[dict]] = None

class SettingsRequest(BaseModel):
    context_length: int
    max_tokens: int
    temperature: float
    device_mode: Optional[str] = "gpu"
    gpu_layers: Optional[int] = -1
    search_mode: str = "off"  # off, simple, prediction, extreme

def bg_download_task(model_key: str):
    global download_progress
    with download_lock:
        download_progress[model_key] = {"status": "downloading", "progress": 0}
        
    try:
        # Perform download
        download_model(model_key)
        with download_lock:
            download_progress[model_key] = {"status": "completed", "progress": 100}
    except Exception as e:
        with download_lock:
            download_progress[model_key] = {"status": "failed", "error": str(e)}

@app.get("/api/status")
def get_system_status():
    """Retrieve system health and model statuses."""
    # Check downloaded models
    models_status = check_models_status()
    
    # Merge with active downloads
    with download_lock:
        for key, dl in download_progress.items():
            if key in models_status:
                models_status[key]["download_task"] = dl

    # System Resources
    ram = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    # Check GPU VRAM if Vulkan / PyTorch is available or active
    gpu_info = "N/A"
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_total_gb = round(total_b / (1024 ** 3), 2)
            vram_free_gb = round(free_b / (1024 ** 3), 2)
            gpu_info = f"{gpu_name} ({vram_free_gb}/{vram_total_gb} GB VRAM Free)"
        else:
            loaded_keys = list(orchestrator.loaded_models.keys())
            if loaded_keys:
                gpu_info = f"Active ({len(loaded_keys)} models cached)"
            else:
                gpu_info = "Standby (Vulkan device ready)"
    except Exception:
        pass
        
    return {
        "models": models_status,
        "system": {
            "cpu": f"{cpu_percent}%",
            "ram_used": f"{ram.used / (1024**3):.1f} GB",
            "ram_total": f"{ram.total / (1024**3):.1f} GB",
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_available_gb": round(ram.available / (1024**3), 2),
            "ram_percent": f"{ram.percent}%",
            "gpu": gpu_info,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_free_gb": vram_free_gb,
            "evm_active": getattr(orchestrator, 'kaggle_hotswap_mode', False)
        },
        "settings": {
            "context_length": orchestrator.context_length,
            "max_tokens": orchestrator.max_tokens,
            "temperature": orchestrator.temperature,
            "device_mode": getattr(orchestrator, "device_mode", "gpu"),
            "gpu_layers": getattr(orchestrator, "gpu_layers", -1),
            "search_mode": getattr(orchestrator, "search_mode", "off")
        },
        "security": {
            "auth_enabled": AUTH_ENABLED,
            "air_gap": air_gap.to_dict(),
            "sast_enabled": True,
            "isolation_available": getattr(orchestrator, '_sandbox_isolation', False),
        }
    }

@app.post("/api/download/{model_key}")
def trigger_download(model_key: str, background_tasks: BackgroundTasks):
    """Trigger background download of a model."""
    if model_key not in MODEL_DEFINITIONS:
        raise HTTPException(status_code=400, detail="Invalid model key")
        
    status = check_models_status()
    if status[model_key]["downloaded"]:
        return {"status": "already_downloaded"}
        
    with download_lock:
        if model_key in download_progress and download_progress[model_key]["status"] == "downloading":
            return {"status": "in_progress"}
            
    background_tasks.add_task(bg_download_task, model_key)
    return {"status": "started"}

class AddCustomModelRequest(BaseModel):
    key: str
    name: str
    repo_id: str
    filename: str
    type: str = "text"
    mmproj_filename: Optional[str] = None

@app.post("/api/models/custom/add")
def api_add_custom_model(req: AddCustomModelRequest):
    """Add or update a custom GGUF model entry in the Model Hub."""
    if not req.key or not req.repo_id or not req.filename:
        raise HTTPException(status_code=400, detail="key, repo_id, and filename are required")
    key_clean = req.key.lower().replace(" ", "_")
    defn = add_custom_model(
        model_key=key_clean,
        name=req.name,
        repo_id=req.repo_id,
        filename=req.filename,
        model_type=req.type,
        mmproj_filename=req.mmproj_filename
    )
    return {"status": "success", "key": key_clean, "definition": defn}

@app.delete("/api/models/custom/{model_key}")
def api_remove_custom_model(model_key: str):
    """Remove a custom model entry from the Model Hub."""
    success = remove_custom_model(model_key)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot remove default system model or key not found")
    return {"status": "success", "removed": model_key}

@app.delete("/api/models/delete/{model_key}")
def api_delete_downloaded_model(model_key: str):
    """Physically deletes downloaded .gguf file(s) for a model to free disk storage."""
    # Unload model from active memory cache if loaded
    try:
        if orchestrator and hasattr(orchestrator, "loaded_models"):
            if model_key in orchestrator.loaded_models:
                model_obj = orchestrator.loaded_models.pop(model_key, None)
                if model_obj:
                    orchestrator._close_model(model_obj, model_key)
                gc.collect()
    except Exception:
        pass

    result = delete_downloaded_model_files(model_key)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete model files"))
    return {"status": "success", **result}

@app.get("/api/models/hf_scan")
def scan_hf_repo(repo_id: str):
    """Scan a HuggingFace repository for .gguf files."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        return {"repo_id": repo_id, "gguf_files": gguf_files}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scan repo: {str(e)}")

@app.get("/api/models/search")
def search_hf_models(q: str = "", limit: int = 20):
    """Search HuggingFace Hub for GGUF models (LM Studio style)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        query = q.strip() if q else "gguf"
        models_iter = api.list_models(search=query, filter="gguf", limit=limit, sort="downloads")
        results = []
        for m in models_iter:
            author = getattr(m, 'author', None) or (m.id.split('/')[0] if '/' in m.id else 'community')
            model_name = m.id.split('/')[-1] if '/' in m.id else m.id
            downloads = getattr(m, 'downloads', 0) or 0
            likes = getattr(m, 'likes', 0) or 0
            last_mod = getattr(m, 'last_modified', None)
            mod_str = "recently"
            if last_mod:
                try:
                    mod_str = str(last_mod).split('T')[0]
                except Exception:
                    pass
            results.append({
                "id": m.id,
                "model_name": model_name,
                "author": author,
                "downloads": downloads,
                "likes": likes,
                "last_modified": mod_str
            })
        return {"query": query, "models": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace search error: {str(e)}")

@app.get("/api/models/repo_details")
@app.get("/api/models/repo_files")
def get_hf_repo_details(repo_id: str):
    """Get GGUF quantization files with size in GB and README for a HF repo."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        files = []
        for sibling in info.siblings:
            if sibling.rfilename.endswith(".gguf"):
                size_gb = round(sibling.size / (1024 ** 3), 2) if sibling.size else 0
                quant = sibling.rfilename.split('/')[-1].replace('.gguf', '')
                files.append({
                    "filename": sibling.rfilename,
                    "size_gb": size_gb,
                    "quant": quant
                })
        
        # Sort files by size ascending
        files.sort(key=lambda x: x["size_gb"])
        
        # Try fetching README.md
        readme_text = "No README available for this model."
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                readme_text = f.read(5000) # first 5KB
        except Exception:
            pass

        return {
            "repo_id": repo_id,
            "downloads": getattr(info, 'downloads', 0) or 0,
            "likes": getattr(info, 'likes', 0) or 0,
            "files": files,
            "readme": readme_text
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch repo details: {str(e)}")

@app.post("/api/models/cancel_download/{model_key}")
def api_cancel_model_download(model_key: str):
    """Cancel an in-progress model download."""
    cancelled = cancel_model_download(model_key)
    return {"status": "cancelled" if cancelled else "not_found", "key": model_key}

@app.post("/api/models/download_hf")
def download_hf_quant_model(payload: dict = Body(...)):
    """Add GGUF model dynamically from HF search and trigger background download."""
    repo_id = payload.get("repo_id")
    filename = payload.get("filename")
    name = payload.get("name") or repo_id.split('/')[-1]
    if not repo_id or not filename:
        raise HTTPException(status_code=400, detail="repo_id and filename are required")

    key_clean = re.sub(r'[^a-zA-Z0-9_]', '_', filename.replace('.gguf', '')).lower()
    defn = add_custom_model(
        model_key=key_clean,
        name=name,
        repo_id=repo_id,
        filename=filename,
        model_type="text"
    )
    # Trigger background download
    t = threading.Thread(target=download_model, args=(key_clean,))
    t.daemon = True
    t.start()
    return {"status": "started", "key": key_clean, "name": name, "filename": filename}

@app.get("/api/vulkan/status")
def get_vulkan_status():
    """Get pre-compiled Vulkan engine status and version info."""
    return check_vulkan_engine_status()

@app.get("/api/vulkan/diagnostics")
def vulkan_diagnostics():
    """Perform GPU hardware diagnostics and confirm Vulkan GPU offloading status."""
    return get_vulkan_gpu_diagnostics()

@app.post("/api/vulkan/update")
@app.post("/api/gpu/setup")
def update_vulkan(engine: str = "auto"):
    """Trigger background setup of chosen GPU acceleration engine (cuda, vulkan, metal)."""
    started = start_vulkan_update_background(engine_target=engine)
    return {"status": "started" if started else "already_running", "engine": engine}

@app.get("/api/models/roles")
def get_model_roles():
    """Retrieve current system role assignments."""
    return {"roles": ROLE_ASSIGNMENTS}

@app.post("/api/models/roles")
def update_model_roles(mapping: dict = Body(...)):
    """Update role assignments (LM Studio style role mapping)."""
    updated = save_role_assignments(mapping)
    return {"status": "success", "roles": updated}

@app.post("/api/settings")
def update_settings(settings: SettingsRequest):
    """Update settings on the orchestrator."""
    if hasattr(orchestrator, "update_settings"):
        orchestrator.update_settings(
            context_length=settings.context_length,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            device_mode=settings.device_mode,
            gpu_layers=settings.gpu_layers,
            search_mode=settings.search_mode
        )
    else:
        orchestrator.context_length = settings.context_length
        orchestrator.max_tokens = settings.max_tokens
        orchestrator.temperature = settings.temperature
        orchestrator.device_mode = settings.device_mode
        orchestrator.gpu_layers = settings.gpu_layers
        orchestrator.search_mode = settings.search_mode
    return {"status": "updated"}

# Audio transcription endpoint removed — no speech model in the Dual-Core pipeline.

@app.post("/api/cancel")
async def cancel_generation():
    """Stop active generation & active benchmarks while keeping models loaded in memory."""
    generation_cancel.set()
    stop_benchmark(orchestrator)
    return {"status": "success", "message": "Generation and benchmarks cancelled."}

@app.post("/api/offload")
async def offload_memory():
    """Fully unload all models from RAM/VRAM."""
    try:
        # Signal any running generation to stop first
        generation_cancel.set()
        
        # Wait for the active generation thread to actually finish.
        # The generation thread holds chat_lock; once it's released, we know
        # the inference_lock inside _call_model is also released and it's safe
        # to call unload_all_models() without deadlocking.
        deadline = time.time() + 15  # 15s timeout
        while time.time() < deadline:
            if chat_lock.acquire(blocking=False):
                chat_lock.release()
                break
            await asyncio.sleep(0.3)
        
        # Small grace period to ensure all locks are fully released
        await asyncio.sleep(0.5)
        
        orchestrator.unload_all_models()
        generation_cancel.clear()
        return {"status": "success", "message": "All models offloaded from memory."}
    except Exception as e:
        generation_cancel.clear()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load_all")
def load_all_models():
    """Pre-load ALL core models directly into System RAM (CPU mode) at startup.
    
    This keeps VRAM 100% empty, maximizing available space for the active model's KV cache.
    During conversation, EVM hot-swaps models from RAM → VRAM one at a time.
    """
    try:
        models_status = check_models_status()
        downloaded = [k for k, v in models_status.items() if v.get("downloaded")]
        
        benchmark_models = ["router", "deepseek_r1", "vibethinker", "ornith"]
        loaded = []
        
        # Suspend EVM hot-swap during this initialization phase
        evm_was_active = getattr(orchestrator, 'kaggle_hotswap_mode', False)
        orchestrator.kaggle_hotswap_mode = False
        
        try:
            for model_key in benchmark_models:
                if model_key in downloaded:
                    try:
                        print(f"📥 EVM Pre-Load: Instantiating '{model_key}' in System RAM (CPU)...")
                        # Enforce force_cpu=True to load it into CPU RAM
                        # Use a uniform, safe context (2048) to minimize RAM footprint while warm
                        cpu_ctx = 2048
                        
                        orchestrator._get_model(model_key, required_ctx=cpu_ctx, force_cpu=True)
                        loaded.append(model_key)
                        gc.collect()
                    except Exception as e:
                        print(f"⚠️ Failed to pre-load '{model_key}' to RAM: {e}")
        finally:
            # Restore EVM hot-swap mode
            orchestrator.kaggle_hotswap_mode = evm_was_active
            
        print(f"🚀 EVM Pre-Load Complete: {len(loaded)} models instantiated in System RAM. VRAM is 100% free!")
        
        return {
            "status": "success", 
            "message": f"Instantiated {len(loaded)} models in System RAM. VRAM is 100% free for KV cache."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main Chat Endpoint. Streams status updates as agents work,
    and then streams the final output.
    """
    # 1. Update request-specific settings
    orchestrator.update_settings(
        context_length=request.context_length,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        device_mode=request.device_mode if request.device_mode else orchestrator.device_mode,
        gpu_layers=request.gpu_layers if request.gpu_layers is not None else orchestrator.gpu_layers,
        search_mode=request.search_mode
    )
    
    # 2. Check if any models are downloaded
    models_status = check_models_status()
    any_downloaded = any(info.get("downloaded", False) for info in models_status.values())
    if not any_downloaded:
        raise HTTPException(
            status_code=400, 
            detail="No models are downloaded on your system. Please open Model Hub & Discovery to download a GGUF model."
        )

    # 3. Stream generator
    async def response_generator():
        owns_chat = False
        try:
            yield json.dumps({"type": "status", "message": "Initiating agent core...", "level": "info", "model": "coordinator", "progress": 0}) + "\n"
            
            import queue
            q = queue.Queue()
            
            def thread_cb(msg, lvl="info", model=None, progress=None):
                payload = {"type": "status", "message": msg, "level": lvl}
                if model:
                    payload["model"] = model
                if progress is not None:
                    try:
                        prog_val = int(progress)
                        if lvl == "success":
                            prog_val = min(100, max(0, prog_val))
                        else:
                            prog_val = min(99, max(0, prog_val))
                        payload["progress"] = prog_val
                    except (ValueError, TypeError):
                        payload["progress"] = progress
                q.put(payload)

            def run_orchestrator():
                nonlocal owns_chat
                # Acquire global lock to prevent state mixing between simultaneous requests
                if not chat_lock.acquire(blocking=False):
                    q.put({"type": "error", "message": "The AI is currently processing another request. Please wait until it finishes."})
                    q.put(None)
                    return
                owns_chat = True
                try:
                    # Clear any stale cancel signal before starting
                    generation_cancel.clear()
                    
                    # If an image was uploaded, run Qwen 2.5 VL vision parsing first
                    final_prompt = request.prompt
                    if request.image and not (request.search_mode == "study" and "application/pdf" in str(request.image)):
                        if generation_cancel.is_set():
                            q.put({"type": "error", "message": "Generation cancelled."})
                            return
                        try:
                            ocr_text = orchestrator.transcribe_image(request.image, user_prompt=request.prompt, status_callback=thread_cb)
                            if isinstance(ocr_text, str) and (ocr_text.startswith("⚠️ Qwen") or ocr_text.startswith("⚠️ Vision")):
                                thread_cb("Vision model not downloaded", "warning")
                                q.put({"type": "chunk", "text": ocr_text})
                                q.put({"type": "done"})
                                q.put(None)
                                return

                            if isinstance(ocr_text, str) and ocr_text.startswith("📄 Extracted PDF"):
                                final_prompt = (
                                    f"[Uploaded PDF Document Content]:\n"
                                    f"{ocr_text}\n\n"
                                    f"User Question:\n{request.prompt}"
                                )
                            else:
                                final_prompt = (
                                    f"[Transcribed Image Content via Qwen 2.5-VL 7B]:\n"
                                    f"{ocr_text}\n\n"
                                    f"User Prompt: {request.prompt}"
                                )
                        except Exception as ex:
                            thread_cb(f"File transcription failed: {str(ex)}. Proceeding with prompt only.", "warning")
                    
                    # Prepend recent conversation history to preserve multi-turn context
                    if request.history and len(request.history) > 0:
                        hist_text = ""
                        for item in request.history[-6:]:
                            role = item.get("role", "user")
                            content = item.get("content", "").strip()
                            if content:
                                label = "User" if role == "user" else "Assistant"
                                hist_text += f"{label}: {content}\n\n"
                        if hist_text:
                            final_prompt = f"Previous Conversation Context:\n{hist_text}Current User Request:\n{final_prompt}"

                    if generation_cancel.is_set():
                        q.put({"type": "error", "message": "Generation cancelled."})
                        return
                            
                    res = orchestrator.process_query(
                        final_prompt, 
                        request.mode, 
                        selected_models=request.selected_models,
                        status_callback=thread_cb,
                        attached_image=request.image
                    )
                    if not generation_cancel.is_set():
                        q.put({"type": "final_response", "text": res})
                    else:
                        q.put({"type": "error", "message": "Generation cancelled."})
                except Exception as ex:
                    if not generation_cancel.is_set():
                        q.put({"type": "error", "message": str(ex)})
                finally:
                    chat_lock.release()
                    q.put(None)  # Sentinel

            thread = threading.Thread(target=run_orchestrator)
            thread.start()
            
            while True:
                try:
                    item = q.get(timeout=0.1)
                    if item is None:
                        break
                    yield f"data: {json.dumps(item)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'keep_alive'})}\n\n"
                    await asyncio.sleep(0.5)
                    
            thread.join()

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Only cancel if this request actually owned the generation
            if owns_chat:
                generation_cancel.set()
            if 'thread' in locals() and thread.is_alive():
                thread.join()

    return StreamingResponse(response_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/memory/count")
def get_memory_count():
    """Get total number of memories stored."""
    return {"count": orchestrator.memory.count()}

@app.post("/api/memory/clear")
def clear_all_memory():
    """Reset vector database / SQLite store."""
    try:
        db_path = orchestrator.memory.db_path
        # Remove the entire persistent DB directory (ChromaDB + SQLite)
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        orchestrator.memory = Memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/unload")
def unload_models():
    """Unload all models to free system RAM."""
    orchestrator.unload_all_models()
    return {"status": "unloaded"}

# ── Benchmark Endpoints for TPU v5e-8 ─────────────────────────────────────

class BenchmarkStartRequest(BaseModel):
    category: str = "HumanEval"
    sample_size: Optional[int] = None

@app.get("/benchmark", response_class=HTMLResponse)
def get_benchmark_dashboard():
    """Serve the gorgeous glassmorphic TPU v5e-8 Benchmark Dashboard."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.html")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)

@app.get("/api/benchmark/status")
def get_benchmark_status():
    """Retrieve the real-time status of all 8 TPU worker cores and active KPIs."""
    with STATE_LOCK:
        # Return a copy to prevent race conditions during serialization
        return dict(BENCHMARK_STATE)

@app.post("/api/benchmark/start")
@app.post("/api/benchmark/run")
def start_benchmark_suite(req: BenchmarkStartRequest, background_tasks: BackgroundTasks):
    """Start the parallel benchmark evaluation in the background."""
    with STATE_LOCK:
        if BENCHMARK_STATE["active"]:
            return {"status": "already_running"}
    
    # run_benchmark_suite is async — we must bridge it into a sync wrapper
    # for FastAPI's BackgroundTasks, which runs tasks in a plain thread.
    def _run_sync():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models_status = check_models_status()
            has_downloaded = any(info.get("downloaded", False) for info in models_status.values())
            active_orchestrator = orchestrator if has_downloaded else None
            loop.run_until_complete(run_benchmark_suite(req.category, orchestrator=active_orchestrator))
            loop.close()
        except Exception as e:
            import traceback
            print(f"❌ Benchmark runner crashed: {traceback.format_exc()}")
            with STATE_LOCK:
                BENCHMARK_STATE["active"] = False
                BENCHMARK_STATE["logs"].append(f"[ERROR] Benchmark crashed: {str(e)}")
    
    background_tasks.add_task(_run_sync)
    return {"status": "started"}

@app.post("/api/benchmark/stop")
@app.post("/api/benchmark/cancel")
def stop_benchmark_suite():
    """Stop the active benchmark evaluation and release the workers."""
    return stop_benchmark(orchestrator)

# ── Phase 3: Authentication endpoint ─────────────────────────────────────

@app.post("/api/auth/login")
def auth_login(username: str = Body(...), password: str = Body(...)):
    """Generate a JWT token for API access (when auth is enabled)."""
    if not AUTH_ENABLED:
        return {"token": "auth_disabled", "message": "Authentication not enabled"}
    # Simple credential check — in production, use a proper user database
    # For now, accept any username with password matching AIOS_ADMIN_PASSWORD env var
    admin_password = os.environ.get("AIOS_ADMIN_PASSWORD", "admin")
    if password == admin_password:
        token = create_jwt_token(user_id=username)
        return {"token": token, "user_id": username, "expires_in": "24h"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/security/status")
def security_status():
    """Get the current security configuration status."""
    return {
        "auth_enabled": AUTH_ENABLED,
        "air_gap": air_gap.to_dict(),
        "sast_enabled": True,
        "git_agent_available": git_agent is not None and git_agent.available,
    }

# ── Phase 4: Git Workspace endpoints ─────────────────────────────────────

@app.get("/api/workspace/status")
def workspace_status():
    """Get git workspace status."""
    if git_agent is None or not git_agent.available:
        return {"available": False, "message": "Git agent not available"}
    return {"available": True, "workspace_base": git_agent.workspace_base}

@app.post("/api/workspace/clone")
def workspace_clone(repo_url: str = Body(...), workspace_name: str = Body(None)):
    """Clone or pull a git repository into the workspace."""
    if git_agent is None or not git_agent.available:
        raise HTTPException(status_code=503, detail="Git agent not available")
    return git_agent.clone_or_pull(repo_url, workspace_name)

@app.post("/api/workspace/commit")
def workspace_commit(
    workspace_path: str = Body(...),
    files: dict = Body(...),
    message: str = Body(...)
):
    """Commit files to a git workspace."""
    if git_agent is None or not git_agent.available:
        raise HTTPException(status_code=503, detail="Git agent not available")
    return git_agent.commit_files(workspace_path, files, message)

def free_port(port: int):
    """Cleanly terminate any stale process holding the specified port."""
    import subprocess
    try:
        if sys.platform != 'win32':
            subprocess.run(f"fuser -k {port}/tcp", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run("pkill -f 'backend/app.py'", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            time.sleep(0.5)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    free_port(port)
    print(f"🚀 Starting FastAPI backend on http://0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
