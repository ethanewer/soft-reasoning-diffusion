from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


class SoftReasoningDiffusionLLM(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        begin_thinking_token: str = "<think>",
        end_thinking_token: str = "</think>",
        eot_token: str = "<|im_end|>",
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.begin_thinking_token = begin_thinking_token
        self.end_thinking_token = end_thinking_token
        self.begin_thinking_id = self.tokenizer.encode(begin_thinking_token)[0]
        self.end_thinking_id = self.tokenizer.encode(end_thinking_token)[-1]
        self.eot_id = self.tokenizer.encode(eot_token)[-1]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Tensor:
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        ).logits

    def embed_logits(self, logits: Tensor):
        probabilities = logits.float().softmax(dim=-1).to(self.model.dtype)
        return probabilities @ self.model.model.embed_tokens.weight

    def denoise(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        past_key_values: DynamicCache,
    ) -> Tensor:
        prev_seq_length = past_key_values.get_seq_length()
        assert prev_seq_length + inputs_embeds.shape[1] == attention_mask.shape[1]
        logits = self(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        past_key_values.crop(prev_seq_length)
        return self.embed_logits(logits)

    def generate(
        self,
        input_text: list[str],
        num_thinking_tokens: int = 512,
        num_denoising_steps: int = 128,
        **generation_kwargs,
    ) -> list[str]:
        inputs = self.tokenizer(
            input_text,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(self.model.device)
        input_ids = F.pad(inputs.input_ids, pad=(0, 1), value=self.begin_thinking_id)
        attention_mask = F.pad(inputs.attention_mask, pad=(0, 1), value=1)
        past_key_values = DynamicCache()

        # prefill input tokens
        with torch.no_grad():
            self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )[:, -1:]

        # diffusion soft thinking
        input_embeds = self.embed_logits(
            torch.randn(
                input_ids.shape[0],
                num_thinking_tokens,
                self.model.config.vocab_size,
                dtype=self.model.dtype,
                device=self.model.device,
            )
        )
        attention_mask = F.pad(attention_mask, pad=(0, num_thinking_tokens), value=1)
        for _ in range(num_denoising_steps):
            with torch.no_grad():
                input_embeds = self.denoise(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

        # prefill soft thinking tokens
        with torch.no_grad():
            self(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # answer generation
        end_thinking_ids = torch.full(
            (input_ids.shape[0], 1),
            self.end_thinking_id,
            device=self.model.device,
        )
        sequences = self.model.generate(
            input_ids=end_thinking_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        # decode outputs
        output_text = []
        for seq in sequences:
            text = self.tokenizer.decode(seq.tolist())
            output_text.append(text[: text.find(self.eot_id)])

        return output_text
