# adapters/api_adapter.py
"""
Groq API Adapter (with embedded API key + model fallback).
For local/private testing only — do NOT push to public repos.
"""

import time
import logging
from typing import Dict, Any

import groq
from groq import BadRequestError, RateLimitError  # exception class used by groq SDK

LOG = logging.getLogger(__name__)

class ApiAdapter:
    def __init__(self, model_candidates=None):
        # ✅ Embed your Groq API key directly here
        self.api_key = "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"

        # Initialize Groq client
        if not self.api_key or not self.api_key.startswith("gsk_"):
            LOG.warning("⚠️ No valid Groq API key found — running in simulated mode.")
            self.client = None
        else:
            self.client = groq.Client(api_key=self.api_key)

        # Prioritized model list — automatically tries next if one fails
        self.model_candidates = model_candidates or [
            "llama-3.3-70b-versatile",   # replacement for old llama3-70b
            "llama-3.1-8b-instant",      # smaller, faster model
            "mixtral-8x7b-32768",        # another supported Groq model
        ]

    def run(self, prompt: str):
        """Send prompt to Groq API with rate-limit and error handling."""
        if not self.client:
            return {
                "provider": "groq",
                "model": "simulated",
                "text": "[No valid API key configured] simulated response",
                "status": "simulated"
            }

        for model in self.model_candidates:
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a safe red-teaming assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )
                text = response.choices[0].message.content.strip()
                return {
                    "provider": "groq",
                    "model": model,
                    "text": text,
                    "latency_s": round(time.time() - start, 3),
                    "status": "ok",
                }

            except RateLimitError as e:
                # Handle 429 errors — wait a bit and retry
                LOG.warning(f"⚠️ Rate limit hit for {model}. Waiting 60 seconds before retry...")
                time.sleep(60)
                continue

            except BadRequestError as e:
                LOG.warning(f"Bad request for model {model}: {e}")
                continue

            except Exception as e:
                LOG.exception(f"Groq API error for {model}: {e}")
                continue

        # All models failed
        return {
            "provider": "groq",
            "model": None,
            "text": "[Error contacting Groq — please check API key or limits]",
            "status": "error",
        }