# adapters/lmstudio_adapter.py
"""
LM Studio Adapter
Synchronous .run(prompt) interface so it works with your existing sandbox_invoke + GPUExecutor.
Talks to LM Studio Local Server (default: http://localhost:1234).
"""

import requests
from typing import Dict, Any

class LMStudioAdapter:
    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = "local-model", timeout: int = 600):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.timeout = timeout

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Synchronous run method returning a dict similar to your other adapters:
        { "provider": "lmstudio", "model": "<name>", "text": "<response>", "status": "ok" }
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 5000,
            "stream": False
        }

        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # Typical LM Studio response contains choices[0].message.content
            text = ""
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                # Fallback if structure differs
                text = str(data)
            return {
                "provider": "lmstudio",
                "model": self.model_name,
                "text": text,
                "status": "ok",
            }

        except requests.exceptions.RequestException as e:
            return {
                "provider": "lmstudio",
                "model": self.model_name,
                "text": f"[ERROR] Connection/HTTP error: {e}",
                "status": "error"
            }
        except Exception as e:
            return {
                "provider": "lmstudio",
                "model": self.model_name,
                "text": f"[ERROR] Unexpected response format: {e}",
                "status": "error"
            }

    def get_adapter_name(self) -> str:
        return f"LMStudioAdapter:{self.model_name}"
