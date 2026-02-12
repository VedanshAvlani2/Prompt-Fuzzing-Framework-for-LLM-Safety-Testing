import asyncio
import queue
import threading
from typing import Any, Dict

class GPUExecutor:
    """Single-threaded executor for GPU operations to avoid context conflicts."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.results = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.task_id = 0
    
    def _worker(self):
        """Worker thread that processes GPU tasks sequentially."""
        while True:
            task_id, func, args, kwargs = self.queue.get()
            try:
                result = func(*args, **kwargs)
                self.results[task_id] = ("success", result)
            except Exception as e:
                self.results[task_id] = ("error", e)
            self.queue.task_done()
    
    async def submit(self, func, *args, **kwargs):
        """Submit a function to run on the GPU thread."""
        self.task_id += 1
        task_id = self.task_id
        self.queue.put((task_id, func, args, kwargs))
        
        # Poll for result
        while task_id not in self.results:
            await asyncio.sleep(0.1)
        
        status, result = self.results.pop(task_id)
        if status == "error":
            raise result
        return result

# Global GPU executor instance
_gpu_executor = None

def get_gpu_executor():
    global _gpu_executor
    if _gpu_executor is None:
        _gpu_executor = GPUExecutor()
    return _gpu_executor