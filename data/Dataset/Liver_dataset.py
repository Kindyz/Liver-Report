# data/Dataset/Liver_dataset.py
import torch
from typing import Any, Dict, List
import h5py, json


class LiverDataset(torch.utils.data.Dataset):
    """Load Merlin embeddings and reports using a predefined split."""

    def __init__(
            self,
            h5_path: str,
            split_json: str,
            split: str,
            tokenizer,
            max_length: int,
            report_prompts: List[str],
            system_prompt: str,
            use_chat_template: bool = True,
    ):
        self.h5_path = h5_path
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.report_prompts = report_prompts
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template and hasattr(tokenizer, "apply_chat_template")


        with open(split_json, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        self.indices = split_data[split]["indices"]
        self.pids = split_data[split]["pids"]


        with h5py.File(h5_path, 'r') as f:
            all_embeds = f['image_embeds'][:]
            all_reports = f['report_desc'][:]

            self.image_embeds = all_embeds[self.indices]
            self.reports = [
                r.decode('utf-8') if isinstance(r, bytes) else str(r)
                for r in all_reports[self.indices]
            ]

        print(f"[{split}] Loaded {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def _build_prompt(self) -> str:
        import random
        return random.choice(self.report_prompts)

    def _build_input_ids(self, prompt_text: str, report_text: str) -> tuple:
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt_text_full = f"System: {self.system_prompt}\nUser: {prompt_text}\nAssistant:"
            prompt_ids = self.tokenizer(prompt_text_full, add_special_tokens=True)["input_ids"]

        eos_text = self.tokenizer.eos_token or ""
        answer_ids = self.tokenizer(report_text + eos_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + len(answer_ids) > self.max_length:
            max_prompt_len = max(1, self.max_length - len(answer_ids))
            prompt_ids = prompt_ids[-max_prompt_len:]
            if len(prompt_ids) + len(answer_ids) > self.max_length:
                answer_ids = answer_ids[: self.max_length - len(prompt_ids)]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        prompt_len = len(prompt_ids)

        return input_ids, attention_mask, labels, prompt_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        report_text = self.reports[idx]
        prompt_text = self._build_prompt()

        input_ids, attention_mask, labels, prompt_len = self._build_input_ids(prompt_text, report_text)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_feat": torch.tensor(self.image_embeds[idx], dtype=torch.float32),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }

    def subset(self, indices):
        """Return a subset using dataset-relative indices."""
        from copy import copy
        new_ds = copy(self)
        new_ds.indices = [self.indices[i] for i in indices]
        new_ds.pids = [self.pids[i] for i in indices]
        new_ds.image_embeds = self.image_embeds[indices]
        new_ds.reports = [self.reports[i] for i in indices]
        return new_ds


class LiverDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is None")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        from torch.nn.utils.rnn import pad_sequence

        input_ids = pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        )
        image_feat = torch.stack([f["image_feat"] for f in features], dim=0)


        prompt_len = torch.stack([f["prompt_len"] for f in features], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_feat": image_feat,
            "prompt_len": prompt_len,
        }
