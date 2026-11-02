# Models/liver_adapter_model.py
"""Map liver CT embeddings to prefix tokens for report generation."""

import os
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiverAdapter(nn.Module):
    """Project Merlin CT embeddings into learnable prefix tokens."""

    def __init__(
            self,
            image_dim: int = 512,
            prompt_len: int = 16,
            llm_hidden_size: int = 4096,
            adapter_hidden_dim: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.llm_hidden_size = llm_hidden_size

        self.proj = nn.Sequential(
            nn.Linear(image_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, prompt_len * llm_hidden_size),
        )
        self.prompt_norm = nn.LayerNorm(llm_hidden_size)

        self.residual_prompt = nn.Parameter(
            torch.randn(prompt_len, llm_hidden_size) * 0.02
        )

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embedding: [B, 512]
        Returns:
            prefix_embeds: [B, prompt_len, hidden_size]
        """
        batch_size = image_embedding.size(0)
        x = self.proj(image_embedding)
        x = x.view(batch_size, self.prompt_len, self.llm_hidden_size)
        x = x + self.residual_prompt.unsqueeze(0)
        x = self.prompt_norm(x)
        return x


class LiverReportQwen(nn.Module):
    """Generate liver reports from CT features with a Qwen language model."""

    def __init__(
            self,
            llm: nn.Module,
            image_dim: int = 512,
            prompt_len: int = 16,
            adapter_hidden_dim: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.llm = llm
        self.config = getattr(llm, "config", None)
        hidden_size = llm.config.hidden_size

        self.adapter = LiverAdapter(
            image_dim=image_dim,
            prompt_len=prompt_len,
            llm_hidden_size=hidden_size,
            adapter_hidden_dim=adapter_hidden_dim,
            dropout=dropout,
        )

    def _concat_prefix(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_feat: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ):
        """ concat prefix."""
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        prefix_embeds = self.adapter(image_feat).to(token_embeds.dtype)

        batch_size = input_ids.size(0)
        prefix_len = prefix_embeds.size(1)

        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, prefix_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        extended_labels = None
        if labels is not None:
            prefix_ignore = torch.full(
                (batch_size, prefix_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            extended_labels = torch.cat([prefix_ignore, labels], dim=1)

        return inputs_embeds, extended_attention_mask, extended_labels

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_feat: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        inputs_embeds, extended_attention_mask, extended_labels = self._concat_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_feat=image_feat,
            labels=labels,
        )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            return_dict=True,
            **kwargs,
        )

        return {
            "loss": llm_outputs.loss,
            "logits": llm_outputs.logits,
        }

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_feat: torch.Tensor,
            **generate_kwargs,
    ):
        inputs_embeds, extended_attention_mask, _ = self._concat_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_feat=image_feat,
            labels=None,
        )

        input_length = inputs_embeds.size(1)
        batch_size = input_ids.size(0)


        bos_id = self.llm.config.bos_token_id
        if bos_id is None:

            if hasattr(self.llm, 'base_model') and hasattr(self.llm.base_model, 'config'):
                bos_id = getattr(self.llm.base_model.config, 'bos_token_id', None)

            if bos_id is None:
                bos_id = getattr(self.llm.config, 'eod_id', None)
            if bos_id is None and hasattr(self.llm, 'base_model'):
                bos_id = getattr(self.llm.base_model.config, 'eod_id', None)

            if bos_id is None:
                bos_id = 0
                print(f"[Warning] bos_token_id not found, using 0 as fallback")


        dummy_input_ids = torch.full(
            (batch_size, input_length),
            bos_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        outputs = self.llm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            **generate_kwargs,
        )

        return outputs[:, input_length:]

    def save_adapter(self, output_dir: str):
        """Save adapter."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.adapter.state_dict(),
            os.path.join(output_dir, "liver_adapter.pt"),
        )

    def load_adapter(self, checkpoint_path: str):
        """Load adapter."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.adapter.load_state_dict(state_dict)
