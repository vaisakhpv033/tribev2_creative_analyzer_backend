"""
LLM Client — Gemini API Wrapper
=================================
Thin wrapper around the Google GenAI SDK for calling Gemini.

Responsibilities:
  - Initialize Gemini client with API key
  - Send system + user prompts
  - Parse JSON from response
  - Handle retries and errors

This module does NOT know about brain features, CTR, or the database.
It only knows how to call Gemini and return structured JSON.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("llm_client")


class GeminiClient:
    """Wrapper around Google GenAI SDK for structured JSON generation.

    Usage:
        client = GeminiClient()
        result = client.generate_json(system_prompt, user_message)
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Model name. Defaults to gemini-2.0-flash.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key to GeminiClient()."
            )

        self._model = model or self.DEFAULT_MODEL
        self._client = genai.Client(api_key=self._api_key)
        logger.info(f"GeminiClient initialized with model={self._model}")

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate structured JSON output from Gemini.

        Args:
            system_prompt: System-level instructions.
            user_message: User-level data/question.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

        Returns:
            Parsed JSON dict from the model response.

        Raises:
            LLMError: If the model fails after all retries.
        """
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"Gemini API call attempt {attempt}/{self.MAX_RETRIES}")

                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )

                # Extract text from response
                raw_text = response.text
                if not raw_text:
                    raise LLMError("Gemini returned empty response")

                # Parse JSON
                parsed = json.loads(raw_text)
                logger.info("Gemini response parsed successfully")
                return parsed

            except json.JSONDecodeError as e:
                last_error = LLMError(f"Failed to parse Gemini JSON: {e}\nRaw: {raw_text[:500]}")
                logger.warning(f"JSON parse error (attempt {attempt}): {e}")

            except Exception as e:
                last_error = LLMError(f"Gemini API error: {e}")
                logger.warning(f"Gemini error (attempt {attempt}): {e}")

            # Wait before retry
            if attempt < self.MAX_RETRIES:
                wait = self.RETRY_DELAY_SECONDS * attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

        raise last_error


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""
    pass
