import os
import gc
import weakref

try:
    import psutil
except ImportError:
    import sys
    if sys.platform != 'linux' and not os.path.exists('/proc/meminfo'):
        print(f"⚠️ [Startup Warning] psutil is not installed and /proc/meminfo is not available on {sys.platform}. "
              "Memory manager will use static virtual memory defaults (16GB total, 8GB available). "
              "Run 'pip install psutil' to enable active memory tracking.")
    else:
        print("⚠️ [Startup Warning] psutil is not installed. Falling back to reading /proc/meminfo. "
              "Run 'pip install psutil' to enable more robust memory tracking.")

    class MockVirtualMemory:
        def __init__(self):
            self.total = 16 * (1024 ** 3)
            self.available = 8 * (1024 ** 3)
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
                except Exception:
                    pass
    class MockPsutil:
        def virtual_memory(self):
            return MockVirtualMemory()
    psutil = MockPsutil()

try:
    import torch
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
except ImportError:
    torch = None


class TransformerWrapper:
    """Wrapper that holds model + tokenizer refs so the Dynamic Memory Allocator
    can deterministically delete them and free GPU/RAM on eviction."""
    def __init__(self, model, tokenizer, device, orchestrator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._orchestrator_ref = weakref.ref(orchestrator) if orchestrator is not None else None

    @property
    def cancel_event(self):
        if self._orchestrator_ref:
            orch = self._orchestrator_ref()
            if orch:
                return getattr(orch, "cancel_event", None)
        return None

    @property
    def _n_gpu_layers(self):
        return -1 if self.device in ["cuda", "xpu"] else 0

    def __call__(self, prompt, max_tokens=512, temperature=0.7, system_prompt=None):
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        stopping_criteria = []
        if self.cancel_event:
            from transformers import StoppingCriteria
            class CancelCriteria(StoppingCriteria):
                def __init__(self, event):
                    self.event = event
                def __call__(self, input_ids, scores, **kwargs):
                    return self.event.is_set()
            stopping_criteria = [CancelCriteria(self.cancel_event)]
        
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria
                )
        except RuntimeError as e:
            print(f"⚠️ GPU compute error ({e}). Attempting one-time fallback to CPU for this prompt...")
            original_device = self.device
            self.model = self.model.to("cpu")
            self.device = "cpu"
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria
                )
            try:
                self.model = self.model.to(original_device)
                self.device = original_device
                print(f"✅ Restored model to {original_device} after CPU fallback.")
            except Exception as restore_err:
                print(f"⚠️ Could not restore model to {original_device} ({restore_err}), staying on CPU.")
        
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("Generation cancelled by user.")
            
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

    def close(self):
        """Deterministically free GPU/CPU memory held by this model."""
        if hasattr(self, 'model') and hasattr(self.model, 'to'):
            try:
                self.model.to("cpu")
            except Exception:
                pass
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass
