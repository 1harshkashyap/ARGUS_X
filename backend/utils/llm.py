import asyncio
import threading
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import google.generativeai as genai
from openai import AsyncOpenAI
from utils.logger import logger
from config import settings


# ── Key type detection ────────────────────────────────────────────────

def detect_key_type(api_key: Optional[str]) -> str:
    """
    Detect what type of API key was provided.
    Returns: GEMINI | OPENAI | NONE | UNKNOWN
    NEVER logs the key value itself.
    """
    if not api_key or not api_key.strip():
        return "NONE"
    key = api_key.strip()
    if key.startswith("AI") and len(key) > 30:
        return "GEMINI"
    if key.startswith("sk-") and len(key) > 40:
        return "OPENAI"
    return "UNKNOWN"


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class LLMResult:
    content: str = ""
    mode: str = "NONE"
    latency_ms: float = 0.0
    error: Optional[str] = None
    model_used: str = ""


# ── Mock responses ────────────────────────────────────────────────────

_MOCK_RESPONSES = [
    "I can help you with that. Based on company policy, employees are entitled to 20 days of annual leave.",
    "Thank you for your question. Our HR system processes requests within 2-3 business days.",
    "According to our guidelines, that process requires manager approval first.",
    "I understand your concern. Please contact HR directly for personalized assistance.",
    "That information is available in the employee handbook, section 4.2.",
    "Our policy on this matter was updated last quarter. The new guidelines state...",
    "I can see your account details. Is there anything specific you need help with today?",
    "For security reasons, I can only discuss general policy information in this channel.",
]


def _mock_response(prompt: str) -> str:
    """
    Deterministic mock response selected by prompt hash.
    Same prompt always returns same response — useful for testing.
    """
    idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(_MOCK_RESPONSES)
    return _MOCK_RESPONSES[idx]


# ── LLM Wrapper ───────────────────────────────────────────────────────

class LLMWrapper:
    """
    BYOAK-compliant LLM wrapper.
    Priority: User Gemini -> System Gemini (dev only) -> User OpenAI -> Mock

    SECURITY INVARIANTS:
      - Never logs key values
      - Never stores key beyond function scope
      - System key never used for user requests in production
    """

    def __init__(self):
        self._openai_client: Optional[AsyncOpenAI] = None
        self._gemini_lock = threading.Lock()

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, honest, and secure AI assistant.",
        user_api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResult:
        """
        Generate a response using the best available LLM.
        Falls through priority chain without crashing.
        """
        key_type = detect_key_type(user_api_key)
        start = time.time()

        # Log key type only — NEVER the value
        logger.info(f"LLM request — key_type: {key_type}, env: {settings.ENVIRONMENT}")

        # ── Priority 1: User's own Gemini key ────────────────────────
        if key_type == "GEMINI":
            result = await self._try_gemini(
                prompt, system_prompt, user_api_key, temperature, max_tokens
            )
            if result:
                result.latency_ms = round((time.time() - start) * 1000, 2)
                return result

        # ── Priority 2: User's own OpenAI key ────────────────────────
        if key_type == "OPENAI":
            result = await self._try_openai(
                prompt, system_prompt, user_api_key, temperature, max_tokens
            )
            if result:
                result.latency_ms = round((time.time() - start) * 1000, 2)
                return result

        # ── Priority 3: Unknown key format ────────────────────────────
        if key_type == "UNKNOWN":
            logger.warning("Invalid API key format provided")
            return LLMResult(
                content="",
                mode="ERROR",
                error="INVALID_API_KEY",
                latency_ms=round((time.time() - start) * 1000, 2)
            )

        # ── Priority 4: No key provided ───────────────────────────────
        if key_type == "NONE":
            if settings.is_production:
                # Production: require user key — no free rides
                logger.warning("No API key provided in production mode")
                return LLMResult(
                    content="",
                    mode="ERROR",
                    error="API_KEY_REQUIRED",
                    latency_ms=round((time.time() - start) * 1000, 2)
                )
            # Development: try system key first, then fall through to mock
            if settings.GEMINI_API_KEY:
                logger.info("Dev mode: using system Gemini key (development only)")
                result = await self._try_gemini(
                    prompt, system_prompt, settings.GEMINI_API_KEY,
                    temperature, max_tokens
                )
                if result:
                    result.latency_ms = round((time.time() - start) * 1000, 2)
                    return result
            # Dev mode: no system key or Gemini failed — fall through to mock

        # ── Final fallback: Mock mode ──────────────────────────────────
        logger.info("Falling back to mock mode")
        return LLMResult(
            content=_mock_response(prompt),
            mode="MOCK",
            latency_ms=round((time.time() - start) * 1000, 2),
            model_used="mock"
        )

    async def _try_gemini(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[LLMResult]:
        """
        Attempt Gemini API call with 30-second timeout.
        Returns None on any failure — never raises.
        """
        try:
            def _call():
                with self._gemini_lock:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name="gemini-2.0-flash",
                        system_instruction=system_prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    response = model.generate_content(prompt)
                    return response.text

            loop = asyncio.get_running_loop()
            content = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=settings.GEMINI_TIMEOUT
            )

            if content and content.strip():
                logger.info("Gemini response received successfully")
                return LLMResult(
                    content=content.strip(),
                    mode="GEMINI_FLASH",
                    model_used="gemini-1.5-flash"
                )
            return None

        except asyncio.TimeoutError:
            logger.warning(f"Gemini timeout after {settings.GEMINI_TIMEOUT}s")
            return None
        except Exception as e:
            logger.warning(f"Gemini call failed: {type(e).__name__}")
            return None

    async def _try_openai(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[LLMResult]:
        """
        Attempt OpenAI API call with 30-second timeout.
        Returns None on any failure — never raises.
        """
        try:
            client = AsyncOpenAI(api_key=api_key, timeout=settings.OPENAI_TIMEOUT)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                ),
                timeout=settings.OPENAI_TIMEOUT
            )
            content = response.choices[0].message.content
            if content and content.strip():
                logger.info("OpenAI response received successfully")
                return LLMResult(
                    content=content.strip(),
                    mode="OPENAI",
                    model_used="gpt-4o-mini"
                )
            return None

        except asyncio.TimeoutError:
            logger.warning(f"OpenAI timeout after {settings.OPENAI_TIMEOUT}s")
            return None
        except Exception as e:
            logger.warning(f"OpenAI call failed: {type(e).__name__}")
            return None


# ── Global singleton ──────────────────────────────────────────────────
llm = LLMWrapper()
