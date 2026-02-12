# # engine/sandbox_runner.py
# import asyncio
# import concurrent.futures
# from typing import Any, Dict

# async def sandbox_invoke(adapter, prompt: str, timeout: float = 300.0) -> Dict[str, Any]:
#     print(f"[sandbox] invoking {adapter.__class__.__name__}")

#     """
#     Asynchronously run adapter.run(prompt) in a thread pool and return the adapter result dict.
#     Times out after `timeout` seconds with a safe timeout object.
#     """
#     loop = asyncio.get_event_loop()
#     # Use a long-lived executor if you want; using context manager is fine for simple runs
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
#         fut = loop.run_in_executor(ex, adapter.run, prompt)
#         try:
#             result = await asyncio.wait_for(fut, timeout=timeout)
#             print(f"[sandbox] {adapter.__class__.__name__} finished.")
#             return result if isinstance(result, dict) else {"text": str(result)}
#         except asyncio.TimeoutError:
#             print(f"⏰ [sandbox] Timeout while waiting for {adapter.__class__.__name__}")
#             return {"text": "[TIMEOUT] model did not respond", "status": "timeout"}
#         except Exception as e:
#             print(f"❌ [sandbox] Exception: {repr(e)}")
#             return {"text": f"[SANDBOX ERROR] {e}", "status": "error"}

import asyncio
from typing import Any, Dict
from engine.gpu_executor import get_gpu_executor

async def sandbox_invoke(adapter, prompt: str, timeout: float = 2000.0) -> Dict[str, Any]:
    print(f"[sandbox] invoking {adapter.__class__.__name__}")
    
    executor = get_gpu_executor()
    
    try:
        result = await asyncio.wait_for(
            executor.submit(adapter.run, prompt),
            timeout=timeout
        )
        print(f"[sandbox] {adapter.__class__.__name__} finished.")
        return result if isinstance(result, dict) else {"text": str(result)}
    except asyncio.TimeoutError:
        print(f"⏰ [sandbox] Timeout while waiting for {adapter.__class__.__name__}")
        return {"text": "[TIMEOUT] model did not respond", "status": "timeout"}
    except Exception as e:
        print(f"❌ [sandbox] Exception: {repr(e)}")
        return {"text": f"[SANDBOX ERROR] {e}", "status": "error"}