"""
HuggingFace Backend — runs models via transformers pipeline.

For models that don't exist as GGUF. Requires transformers + torch.
Security: only safetensors format, no pickle.
"""

from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class HuggingFaceBackend(BackendAdapter):
    """HuggingFace transformers pipeline backend."""

    def __init__(self):
        self._pipeline = None
        self._model_name = ""

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def detect(self) -> BackendInfo:
        try:
            import transformers
            version = transformers.__version__
            return BackendInfo(name="huggingface", version=version,
                               available=True, details=f"transformers {version}")
        except ImportError:
            return BackendInfo(name="huggingface", version="", available=False,
                              details="pip install transformers torch")

    def list_models(self) -> List[ModelEntry]:
        # HF has millions of models, we don't list them all
        return []

    def pull(self, model_name: str, progress_callback=None) -> str:
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_name)
            return path
        except Exception:
            return model_name

    def load(self, model_ref: str, n_ctx: int = 4096, **kwargs) -> None:
        try:
            import transformers
            import torch
        except ImportError:
            raise RuntimeError("pip install transformers torch")

        self._model_name = model_ref

        # Security: prefer safetensors
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=model_ref,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            model_kwargs={"use_safetensors": True},
        )

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[GenerationConfig] = None) -> str:
        if not self._pipeline:
            raise RuntimeError("No model loaded")
        cfg = config or GenerationConfig()

        result = self._pipeline(
            messages,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.temperature > 0,
        )
        return result[0]["generated_text"][-1]["content"]

    def stream(self, messages: List[Dict[str, str]],
               config: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        # transformers TextIteratorStreamer
        try:
            from transformers import TextIteratorStreamer
            import threading

            streamer = TextIteratorStreamer(
                self._pipeline.tokenizer, skip_special_tokens=True
            )
            cfg = config or GenerationConfig()

            inputs = self._pipeline.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self._pipeline.device)

            thread = threading.Thread(
                target=self._pipeline.model.generate,
                kwargs={"input_ids": inputs, "max_new_tokens": cfg.max_tokens,
                        "streamer": streamer, "temperature": cfg.temperature}
            )
            thread.start()
            for text in streamer:
                yield text
            thread.join()
        except Exception:
            # Fallback: non-streaming
            yield self.chat(messages, config)

    def unload(self) -> None:
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def stats(self) -> BackendStats:
        return BackendStats(model_name=self._model_name)
