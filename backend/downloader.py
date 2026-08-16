import os
import sys
import shutil
import requests
import time
import re
import threading
# Enable fast multi-threaded Rust transfer engine for HuggingFace Hub downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download

import json

# Default local models directory
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
CUSTOM_MODELS_FILE = os.path.join(MODELS_DIR, "custom_models.json")
ROLE_ASSIGNMENTS_FILE = os.path.join(MODELS_DIR, "role_assignments.json")

DEFAULT_ROLE_ASSIGNMENTS = {
    'router': 'router',
    'coding': 'ornith',
    'reasoning': 'deepseek_r1',
    'linter': 'vibethinker',
    'vision': 'qwen_vl'
}

ROLE_ASSIGNMENTS = dict(DEFAULT_ROLE_ASSIGNMENTS)

def load_role_assignments():
    """Load role assignments from role_assignments.json."""
    global ROLE_ASSIGNMENTS
    ROLE_ASSIGNMENTS = dict(DEFAULT_ROLE_ASSIGNMENTS)
    if os.path.exists(ROLE_ASSIGNMENTS_FILE):
        try:
            with open(ROLE_ASSIGNMENTS_FILE, 'r') as f:
                saved = json.load(f)
                for r, m in saved.items():
                    ROLE_ASSIGNMENTS[r] = m
        except Exception as e:
            print(f"⚠️ Error loading role_assignments.json: {e}")

def save_role_assignments(new_mapping):
    """Save user-configured role assignments."""
    global ROLE_ASSIGNMENTS
    os.makedirs(MODELS_DIR, exist_ok=True)
    for r, m in new_mapping.items():
        if r in DEFAULT_ROLE_ASSIGNMENTS:
            ROLE_ASSIGNMENTS[r] = m
    with open(ROLE_ASSIGNMENTS_FILE, 'w') as f:
        json.dump(ROLE_ASSIGNMENTS, f, indent=2)
    return ROLE_ASSIGNMENTS

def resolve_model_key(key_or_role):
    """Resolve a system role (or raw key) to the mapped target model key, falling back to any downloaded model if not present."""
    role_aliases = {
        'router': 'router',
        'coding': 'coding',
        'ornith': 'coding',
        'reasoning': 'reasoning',
        'deepseek_r1': 'reasoning',
        'linter': 'linter',
        'vibethinker': 'linter',
        'vision': 'vision',
        'qwen_vl': 'vision'
    }
    target_role = role_aliases.get(key_or_role)
    resolved_key = ROLE_ASSIGNMENTS.get(target_role, key_or_role) if target_role else key_or_role
    
    if is_model_downloaded(resolved_key):
        return resolved_key
        
    fallback = get_any_available_model_key()
    if fallback:
        return fallback
        
    return resolved_key

# Default definitions of local LLMs
DEFAULT_MODEL_DEFINITIONS = {
    'qwen_vl': {
        'repo_id': 'unsloth/Qwen2.5-VL-7B-Instruct-GGUF',
        'filename': 'Qwen2.5-VL-7B-Instruct-UD-Q6_K_XL.gguf',
        'mmproj_filename': 'mmproj-BF16.gguf',
        'name': 'Qwen-2.5-VL-7B (Vision/Doc Parsing)',
        'type': 'image_to_text',
    },
    'router': {
        'repo_id': 'bartowski/Phi-3.5-mini-instruct-GGUF',
        'filename': 'Phi-3.5-mini-instruct-Q6_K.gguf',
        'name': 'Phi-3.5-Mini (Master Router)',
        'type': 'text',
    },
    'deepseek_r1': {
        'repo_id': 'unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF',
        'filename': 'DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf',
        'name': 'DeepSeek-R1-7B (Reasoning Engine)',
        'type': 'text',
    },
    'ornith': {
        'repo_id': 'deepreinforce-ai/Ornith-1.0-9B-GGUF',
        'filename': 'ornith-1.0-9b-Q6_K.gguf',
        'name': 'Ornith 1.0-9B (Agentic Coder)',
        'type': 'text',
    },
    'vibethinker': {
        'repo_id': 'prithivMLmods/VibeThinker-3B-GGUF',
        'filename': 'VibeThinker-3B.Q6_K.gguf',
        'name': 'VibeThinker 3B (Math/Logic Engine)',
        'type': 'text',
    }
}

MODEL_DEFINITIONS = dict(DEFAULT_MODEL_DEFINITIONS)

def load_custom_models():
    """Load custom user models from custom_models.json and merge into MODEL_DEFINITIONS."""
    global MODEL_DEFINITIONS
    MODEL_DEFINITIONS = dict(DEFAULT_MODEL_DEFINITIONS)
    if os.path.exists(CUSTOM_MODELS_FILE):
        try:
            with open(CUSTOM_MODELS_FILE, 'r') as f:
                custom_data = json.load(f)
                for key, defn in custom_data.items():
                    MODEL_DEFINITIONS[key] = defn
        except Exception as e:
            print(f"⚠️ Error loading custom_models.json: {e}")

def save_custom_models():
    """Save user-added custom models to custom_models.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    custom_data = {k: v for k, v in MODEL_DEFINITIONS.items() if k not in DEFAULT_MODEL_DEFINITIONS}
    with open(CUSTOM_MODELS_FILE, 'w') as f:
        json.dump(custom_data, f, indent=2)

def add_custom_model(model_key, name, repo_id, filename, model_type="text", mmproj_filename=None):
    """Add or update a custom model dynamically."""
    defn = {
        'repo_id': repo_id,
        'filename': filename,
        'name': name,
        'type': model_type
    }
    if mmproj_filename:
        defn['mmproj_filename'] = mmproj_filename
    MODEL_DEFINITIONS[model_key] = defn
    save_custom_models()
    return defn

def remove_custom_model(model_key):
    """Remove a custom model entry."""
    if model_key in MODEL_DEFINITIONS and model_key not in DEFAULT_MODEL_DEFINITIONS:
        del MODEL_DEFINITIONS[model_key]
        save_custom_models()
        return True
    return False

# Initialize custom models and role assignments on load
load_custom_models()
load_role_assignments()

def get_model_filenames(definition):
    filenames = []
    filename = definition["filename"]
    match = re.search(r'^(.*?)(\d+)-of-(\d+)(.*?)$', filename)
    if match:
        prefix, start, total, suffix = match.groups()
        total_shards = int(total)
        width = len(start)
        for i in range(1, total_shards + 1):
            shard_num = str(i).zfill(width)
            filenames.append(f"{prefix}{shard_num}-of-{total}{suffix}")
    else:
        filenames.append(filename)
    
    if "mmproj_filename" in definition:
        filenames.append(definition["mmproj_filename"])
    return filenames

def auto_discover_local_gguf_models():
    """Scans MODELS_DIR recursively for any .gguf files on disk and registers them into MODEL_DEFINITIONS."""
    if not os.path.exists(MODELS_DIR):
        return

    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".gguf") and not file.endswith(".incomplete") and not file.startswith("mmproj"):
                # Match existing definition
                found = False
                for k, d in MODEL_DEFINITIONS.items():
                    if d.get("filename") == file:
                        found = True
                        break
                if not found:
                    folder_name = os.path.basename(root)
                    key_name = re.sub(r'[^a-zA-Z0-9_]', '_', file.replace(".gguf", "")).lower()[:32]
                    MODEL_DEFINITIONS[key_name] = {
                        "repo_id": f"local/{folder_name}",
                        "filename": file,
                        "name": file.replace(".gguf", "").replace("-", " ").title(),
                        "type": "text"
                    }

def is_model_downloaded(model_key):
    """Check if all files for a model key exist on disk (checking standard, flat, or recursive paths)."""
    auto_discover_local_gguf_models()
    if model_key not in MODEL_DEFINITIONS:
        return False
    definition = MODEL_DEFINITIONS[model_key]
    
    # Check if main model file exists anywhere in MODELS_DIR
    main_found = False
    for root, dirs, files in os.walk(MODELS_DIR):
        if definition["filename"] in files:
            main_found = True
            break
            
    if not main_found:
        return False

    # For qwen_vl vision model, verify that at least one mmproj projector file exists
    if model_key == "qwen_vl":
        mmproj_found = False
        for root, dirs, files in os.walk(MODELS_DIR):
            if any("mmproj" in f.lower() and f.endswith(".gguf") and not f.endswith(".incomplete") for f in files):
                mmproj_found = True
                break
        return mmproj_found

    return True

def get_any_available_model_key():
    """Find and return any model key that is currently downloaded on disk."""
    auto_discover_local_gguf_models()
    status = check_models_status()
    for key, info in status.items():
        if info.get("downloaded", False):
            return key
    return None

def get_model_path(model_key):
    """Get the local path for a model key, finding it recursively if present."""
    auto_discover_local_gguf_models()
    if model_key not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model key: {model_key}")
        
    definition = MODEL_DEFINITIONS[model_key]
    subfolder = definition.get("type", "text")
    if "/" in subfolder:
        subfolder = subfolder.split("/")[-1]
        
    # Standard path
    target_dir = os.path.join(MODELS_DIR, subfolder, model_key)
    standard_path = os.path.join(target_dir, definition["filename"])
    if os.path.exists(standard_path):
        return standard_path

    # Check if it exists in a flat structure (Kaggle datasets)
    flat_path = os.path.join(MODELS_DIR, definition["filename"])
    if os.path.exists(flat_path):
        return flat_path

    # Recursive disk search
    for root, dirs, files in os.walk(MODELS_DIR):
        if definition["filename"] in files:
            return os.path.join(root, definition["filename"])

    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError:
        pass
    
    return standard_path

DOWNLOAD_PROGRESS = {}
CANCEL_DOWNLOAD_EVENTS = {}

def cancel_model_download(model_key):
    """Cancel an active model download by setting its cancellation event."""
    if model_key in CANCEL_DOWNLOAD_EVENTS:
        CANCEL_DOWNLOAD_EVENTS[model_key].set()
        DOWNLOAD_PROGRESS[model_key] = {"status": "cancelled", "percent": 0}
        return True
    return False

def check_models_status():
    """Check which models are downloaded and ready."""
    auto_discover_local_gguf_models()
    status = {}
    for key, definition in MODEL_DEFINITIONS.items():
        all_downloaded = is_model_downloaded(key)
        first_path = get_model_path(key) if all_downloaded else None
        total_size = 0
        if all_downloaded and first_path and os.path.exists(first_path):
            try:
                total_size = os.path.getsize(first_path)
            except Exception:
                total_size = 0

        prog_info = None
        if not all_downloaded:
            prog_info = DOWNLOAD_PROGRESS.get(key, None)

        status[key] = {
            "name": definition["name"],
            "filename": definition["filename"],
            "repo_id": definition["repo_id"],
            "downloaded": all_downloaded,
            "path": first_path if all_downloaded else None,
            "size": f"{total_size / (1024**3):.2f} GB" if all_downloaded else "N/A",
            "progress": prog_info
        }
    return status

def download_model(model_key, progress_callback=None):
    """Download a model from Hugging Face (downloads all shards/files if sharded)."""
    if model_key not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model key: {model_key}")
        
    definition = MODEL_DEFINITIONS[model_key]
    filenames = get_model_filenames(definition)
    
    subfolder = definition.get("type", "text")
    if "/" in subfolder:
        subfolder = subfolder.split("/")[-1]
    target_dir = os.path.join(MODELS_DIR, subfolder, model_key)
    cancel_evt = threading.Event()
    CANCEL_DOWNLOAD_EVENTS[model_key] = cancel_evt
    
    for idx, fname in enumerate(filenames):
        local_path = os.path.join(target_dir, fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Check if already complete
        if os.path.exists(local_path):
            print(f"[{idx+1}/{len(filenames)}] {fname} already exists at {local_path}.")
            continue
            
        print(f"[{idx+1}/{len(filenames)}] Downloading {fname} (Repo: {definition['repo_id']})...")
        
        # Resumable chunked stream download with live progress tracking
        temp_path = local_path + ".incomplete"
        url = f"https://huggingface.co/{definition['repo_id']}/resolve/main/{fname}"
        max_retries = 15
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                try:
                    import huggingface_hub
                    token = huggingface_hub.get_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                except Exception:
                    pass
                
                initial_pos = 0
                if os.path.exists(temp_path):
                    initial_pos = os.path.getsize(temp_path)
                    headers["Range"] = f"bytes={initial_pos}-"
                    print(f"\nResuming download from {initial_pos / (1024**2):.2f} MB...")
                    
                response = requests.get(url, stream=True, headers=headers, timeout=(15, 60))
                
                if response.status_code == 416:
                    print("\nRange error, starting download from scratch...")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    initial_pos = 0
                    headers_scratch = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    if "Authorization" in headers:
                        headers_scratch["Authorization"] = headers["Authorization"]
                    response = requests.get(url, stream=True, headers=headers_scratch, timeout=(15, 60))
                    
                response.raise_for_status()
                
                mode = "ab" if (response.status_code == 206 and initial_pos > 0) else "wb"
                if mode == "wb":
                    initial_pos = 0
                    
                total_size = int(response.headers.get('content-length', 0)) + initial_pos
                print(f"Total Size: {total_size / (1024**2):.2f} MB")
                
                downloaded = initial_pos
                chunk_size = 4 * 1024 * 1024  # 4MB chunks for faster throughput
                
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if cancel_evt.is_set():
                            print(f"🛑 Download of '{model_key}' cancelled by user.")
                            DOWNLOAD_PROGRESS[model_key] = {"status": "cancelled", "percent": 0}
                            return None
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = round((downloaded / total_size) * 100, 1) if total_size > 0 else 0
                            DOWNLOAD_PROGRESS[model_key] = {
                                "status": "downloading",
                                "downloaded_gb": round(downloaded / (1024**3), 2),
                                "total_gb": round(total_size / (1024**3), 2),
                                "percent": percent
                            }
                            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB)")
                            sys.stdout.flush()
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                sys.stdout.write("\n")
                break
            except Exception as e:
                print(f"\nDownload attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 3 * (attempt + 1)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
                        
        if os.path.exists(temp_path):
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(temp_path, local_path)
            DOWNLOAD_PROGRESS[model_key] = {
                "status": "completed",
                "downloaded_gb": round(os.path.getsize(local_path) / (1024**3), 2),
                "total_gb": round(os.path.getsize(local_path) / (1024**3), 2),
                "percent": 100
            }
            print(f"Successfully downloaded {fname} to {local_path}")
        
    return os.path.join(target_dir, filenames[0])

def migrate_existing_models():
    """Migrate existing model files to their new flat subcategory model-specific folders."""
    if not os.path.exists(MODELS_DIR):
        return
        
    for key, definition in MODEL_DEFINITIONS.items():
        new_path = get_model_path(key)
        if os.path.exists(new_path):
            continue
            
        # 1. Candidate 1: directly under models/<subcategory>/filename (from the flat subcategory step)
        subfolder = definition.get("type", "text")
        if "/" in subfolder:
            subfolder = subfolder.split("/")[-1]
        old_path_flat = os.path.join(MODELS_DIR, subfolder, definition["filename"])
        if os.path.exists(old_path_flat) and os.path.isfile(old_path_flat) and old_path_flat != new_path:
            try:
                print(f"Migrating {definition['filename']} (flat) -> {new_path}...")
                shutil.move(old_path_flat, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")
                
        # 2. Candidate 2: old nested type path (e.g. models/natural_language_processing_nlp/sentence_similarity/file)
        old_nested_folder = os.path.join(MODELS_DIR, definition.get("type", "text"))
        old_path_nested = os.path.join(old_nested_folder, definition["filename"])
        if os.path.exists(old_path_nested) and os.path.isfile(old_path_nested) and old_path_nested != new_path:
            try:
                print(f"Migrating {definition['filename']} (nested) -> {new_path}...")
                shutil.move(old_path_nested, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")

        # 3. Candidate 3: root models/filename
        old_path_root = os.path.join(MODELS_DIR, definition["filename"])
        if os.path.exists(old_path_root) and os.path.isfile(old_path_root) and old_path_root != new_path:
            try:
                print(f"Migrating {definition['filename']} (root) -> {new_path}...")
                shutil.move(old_path_root, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")

# Run model file categorization migration on load
migrate_existing_models()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Download specific models
        for arg in sys.argv[1:]:
            if arg in MODEL_DEFINITIONS:
                try:
                    download_model(arg)
                except Exception as e:
                    print(f"Error downloading {arg}: {str(e)}")
            else:
                print(f"Unknown model: {arg}")
    else:
        # Test script to check status
        print("Checking local models status...")
        status = check_models_status()
        for key, info in status.items():
            status_str = "[READY]" if info["downloaded"] else "[MISSING]"
            print(f"- {info['name']}: {status_str} ({info['filename']})")
