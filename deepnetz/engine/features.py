"""
DeepNetz Advanced Features — Vision, Reasoning, Tool Calling, Speculative Decoding.

These features work across all backends that support them.
"""

import re
import base64
import os
from typing import List, Dict, Optional, Tuple


# ── Vision / Multimodal ───────────────────────────────────────────────

def prepare_vision_message(prompt: str, image_paths: List[str] = None,
                           image_base64: List[str] = None) -> Dict:
    """
    Prepare a multimodal message with text + images.

    Supports:
        - Local file paths → auto-converted to base64
        - Base64 strings
        - URLs (passed through)

    Returns OpenAI-compatible message format:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }
    """
    content = [{"type": "text", "text": prompt}]

    # Add images from file paths
    if image_paths:
        for path in image_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                ext = path.rsplit(".", 1)[-1].lower()
                mime = {"png": "image/png", "jpg": "image/jpeg",
                        "jpeg": "image/jpeg", "gif": "image/gif",
                        "webp": "image/webp"}.get(ext, "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"}
                })

    # Add pre-encoded base64 images
    if image_base64:
        for data in image_base64:
            if data.startswith("data:"):
                url = data
            else:
                url = f"data:image/png;base64,{data}"
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    return {"role": "user", "content": content}


def is_vision_model(model_name: str) -> bool:
    """Check if model likely supports vision based on name."""
    name = model_name.lower()
    vision_keywords = ["vl", "vision", "visual", "multimodal", "omni",
                       "llava", "minicpm-v", "gemma-3", "gemma-4",
                       "qwen3-vl", "qwen2.5-vl", "pixtral"]
    return any(kw in name for kw in vision_keywords)


# ── Reasoning Mode ─────────────────────────────────────────────────────

def parse_reasoning(text: str) -> Tuple[str, str]:
    """
    Parse reasoning/thinking from model output.

    Supports:
        - <think>...</think> (DeepSeek-R1)
        - <reasoning>...</reasoning>
        - ```thinking\n...\n```

    Returns: (thinking, answer)
    """
    # <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
        return thinking, answer

    # <reasoning>...</reasoning>
    reason_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if reason_match:
        thinking = reason_match.group(1).strip()
        answer = text[reason_match.end():].strip()
        return thinking, answer

    # Thinking Process: ... (common pattern)
    think_proc = re.search(r'(?:Thinking Process|Thinking|思考过程)[:\s]*(.*?)(?:\n\n|\n---)', text, re.DOTALL)
    if think_proc:
        thinking = think_proc.group(1).strip()
        answer = text[think_proc.end():].strip()
        return thinking, answer

    return "", text


def format_reasoning_prompt(prompt: str, enable_reasoning: bool = False) -> str:
    """Add reasoning instructions to prompt if enabled."""
    if not enable_reasoning:
        return prompt
    return (
        f"{prompt}\n\n"
        "Please think step by step. Show your reasoning in <think>...</think> tags, "
        "then provide your final answer."
    )


def is_reasoning_model(model_name: str) -> bool:
    """Check if model supports native reasoning."""
    name = model_name.lower()
    return any(kw in name for kw in ["r1", "reasoning", "think", "qwq", "o1", "o3"])


# ── Tool Calling ───────────────────────────────────────────────────────

def parse_tool_calls(text: str) -> List[Dict]:
    """
    Parse tool/function calls from model output.

    Supports multiple formats:
        - OpenAI: {"name": "func", "arguments": {...}}
        - Qwen: <tool_call>{"name":...}</tool_call>
        - Generic JSON blocks with "name" + "arguments"
    """
    import json
    calls = []

    # Qwen format: <tool_call>...</tool_call>
    qwen_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for match in qwen_matches:
        try:
            data = json.loads(match.strip())
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            pass

    # Generic JSON with function_call or tool_calls
    json_matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', text)
    for match in json_matches:
        try:
            data = json.loads(match)
            if "name" in data and data not in calls:
                calls.append(data)
        except json.JSONDecodeError:
            pass

    return calls


# ── Speculative Decoding ───────────────────────────────────────────────

class SpeculativeDecoder:
    """
    Speculative decoding: draft model proposes, target verifies.

    2-3x speedup for memory-bound inference.

    Usage:
        spec = SpeculativeDecoder(target_model, draft_model, k=5)
        for token in spec.generate(prompt):
            print(token, end="")
    """

    def __init__(self, target_backend, draft_backend, k: int = 5):
        """
        Args:
            target_backend: The large model backend
            draft_backend: Small fast model backend (e.g. 3B)
            k: Number of draft tokens per step
        """
        self.target = target_backend
        self.draft = draft_backend
        self.k = k
        self._stats = {"drafted": 0, "accepted": 0}

    def generate(self, messages: List[Dict], max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        Speculative generation with draft+verify.

        For each step:
        1. Draft model generates k tokens autoregressively
        2. Target model verifies all k tokens in one forward pass
        3. Accept longest matching prefix
        4. Yield accepted tokens
        """
        from deepnetz.backends.base import GenerationConfig

        draft_config = GenerationConfig(max_tokens=1, temperature=temperature)
        verify_config = GenerationConfig(max_tokens=self.k + 1, temperature=temperature)

        current_messages = list(messages)
        generated = []
        total = 0

        while total < max_tokens:
            # 1. Draft k tokens
            draft_tokens = []
            draft_msgs = list(current_messages)
            for _ in range(self.k):
                try:
                    tok = self.draft.chat(draft_msgs, draft_config)
                    if not tok:
                        break
                    draft_tokens.append(tok)
                    draft_msgs.append({"role": "assistant", "content": tok})
                    draft_msgs.append({"role": "user", "content": "continue"})
                except Exception:
                    break

            if not draft_tokens:
                # Draft failed, fall back to target
                for token in self.target.stream(current_messages,
                                                GenerationConfig(max_tokens=max_tokens - total,
                                                                 temperature=temperature)):
                    yield token
                    total += 1
                return

            # 2. Verify with target (single forward pass with all draft tokens)
            verify_prompt = current_messages + [
                {"role": "assistant", "content": "".join(draft_tokens)}
            ]
            try:
                target_response = self.target.chat(
                    current_messages,
                    GenerationConfig(max_tokens=len(draft_tokens) + 1, temperature=temperature)
                )
            except Exception:
                target_response = ""

            # 3. Find longest matching prefix
            accepted = 0
            for i, dt in enumerate(draft_tokens):
                if i < len(target_response) and target_response[i:i+len(dt)] == dt:
                    accepted += 1
                else:
                    break

            self._stats["drafted"] += len(draft_tokens)
            self._stats["accepted"] += accepted

            # 4. Yield accepted tokens
            accepted_text = "".join(draft_tokens[:accepted])
            if accepted_text:
                yield accepted_text
                total += accepted
                current_messages.append({"role": "assistant", "content": accepted_text})

            # If nothing accepted, yield one target token
            if accepted == 0 and target_response:
                yield target_response[0] if target_response else ""
                total += 1
                current_messages.append({"role": "assistant", "content": target_response[:1]})

    @property
    def acceptance_rate(self) -> float:
        if self._stats["drafted"] == 0:
            return 0.0
        return self._stats["accepted"] / self._stats["drafted"]
