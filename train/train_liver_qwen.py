# train/train_liver_qwen.py
"""Train a Qwen model to generate liver CT reports."""
import os, sys, json, logging, argparse, time, warnings, setproctitle
from os.path import join
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")

# util
from util.train_utils import load_config, save_config, EpochCSVLogger
# Dataset
from data.Dataset.Liver_dataset import LiverDataset, LiverDataCollator
from Models.liver_adapter_model import LiverReportQwen


def setup_seed(seed: int):
    """Setup seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Build optimizer."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )
    return optimizer


def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float, scheduler_type: str = "cosine"):
    """Build scheduler."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0,
                       float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def save_checkpoint(
        accelerator: Accelerator,
        model: LiverReportQwen,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        ckpt_dir: str,
):
    """Save checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)


    lora_dir = join(ckpt_dir, f"llm_lora_epoch{epoch}")
    unwrapped_model.llm.save_pretrained(lora_dir)


    adapter_path = join(ckpt_dir, f"liver_adapter_epoch{epoch}.pt")
    torch.save(unwrapped_model.adapter.state_dict(), adapter_path)


    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(state, join(ckpt_dir, f"training_state_epoch{epoch}.pt"))

    if accelerator.is_main_process:
        print(f"[Checkpoint] Saved to {ckpt_dir} (epoch {epoch})")


def find_latest_checkpoint(ckpt_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Find latest checkpoint."""
    import glob
    state_files = glob.glob(join(ckpt_dir, "training_state_epoch*.pt"))
    if not state_files:
        return None, None

    epochs = [int(os.path.basename(f).split("epoch")[-1].replace(".pt", "")) for f in state_files]
    latest_epoch = max(epochs)
    state_path = join(ckpt_dir, f"training_state_epoch{latest_epoch}.pt")
    return latest_epoch, state_path


def load_checkpoint(
        ckpt_dir: str,
        base_model: nn.Module,
        adapter: nn.Module,  # Image Adapter
        optimizer: torch.optim.Optimizer,
        scheduler,
) -> Tuple[int, int]:
    """Load checkpoint."""
    latest_epoch, state_path = find_latest_checkpoint(ckpt_dir)
    if latest_epoch is None:
        return 1, 0


    state = torch.load(state_path, map_location="cpu")


    lora_dir = join(ckpt_dir, f"llm_lora_epoch{latest_epoch}")
    if os.path.exists(lora_dir):
        from safetensors.torch import load_file
        adapter_weights_path = join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_weights_path):
            adapter_weights = load_file(adapter_weights_path)

            base_model_state = base_model.state_dict()
            filtered_weights = {k: v for k, v in adapter_weights.items() if k in base_model_state}
            missing_keys, unexpected_keys = base_model.load_state_dict(filtered_weights, strict=False)
            if missing_keys:
                print(f"[Warning] LoRA missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[Warning] LoRA unexpected keys: {unexpected_keys}")
        else:
            print(f"[Warning] LoRA weights not found at {lora_dir}")
    else:
        print(f"[Warning] LoRA directory not found: {lora_dir}")


    adapter_path = join(ckpt_dir, f"liver_adapter_epoch{latest_epoch}.pt")
    if os.path.exists(adapter_path):
        adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        print(f"[Checkpoint] Loaded adapter from epoch {latest_epoch}")
    else:
        print(f"[Warning] Adapter not found: {adapter_path}")


    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])

    print(f"[Checkpoint] Resumed from epoch {latest_epoch}, step {state['global_step']}")
    return state["epoch"] + 1, state["global_step"]


def train_one_epoch(
        accelerator: Accelerator,
        model: LiverReportQwen,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_dl: DataLoader,
        epoch: int,
        global_step: int,
        max_grad_norm: float,
        start_time: float,
        total_steps: int,
) -> Tuple[float, int]:
    """Train one epoch."""
    model.train()
    epoch_loss = 0.0
    steps_in_epoch = 0

    pbar = tqdm(
        train_dl,
        disable=not accelerator.is_main_process,
        desc=f"Train Epoch {epoch}",
        ncols=120,
        leave=False,
    )

    for batch in pbar:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        image_feat = batch["image_feat"]


        with accelerator.accumulate(model):

            with accelerator.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_feat=image_feat,
                    labels=labels,
                )
                loss = outputs["loss"]


            accelerator.backward(loss)


            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)


            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()


        batch_loss = loss.item()
        epoch_loss += batch_loss
        steps_in_epoch += 1
        global_step += 1


        elapsed = time.time() - start_time
        avg_time_per_step = elapsed / global_step if global_step > 0 else 0
        steps_left = total_steps - global_step
        eta_seconds = int(steps_left * avg_time_per_step)
        eta_str = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))

        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "ETA": eta_str,
        })

    avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
    return avg_loss, global_step


@torch.no_grad()
def eval_one_epoch(
        accelerator: Accelerator,
        model: LiverReportQwen,
        eval_dl: DataLoader,
        tokenizer,
        epoch: int,
        output_dir: str,
        num_vis_samples: int = 3,
        gen_config: dict = None,
) -> Tuple[float, Dict[str, float]]:
    """Eval one epoch."""
    model.eval()

    val_loss = 0.0
    val_steps = 0

    all_generated = []
    all_references = []
    vis_samples = []


    max_new_tokens = gen_config.get('max_new_tokens', 256) if gen_config else 256
    do_sample = gen_config.get('do_sample', False) if gen_config else False
    temperature = gen_config.get('temperature', 1.0) if gen_config else 1.0
    top_p = gen_config.get('top_p', 1.0) if gen_config else 1.0
    repetition_penalty = gen_config.get('repetition_penalty', 1.0) if gen_config else 1.0

    pbar = tqdm(
        eval_dl,
        disable=not accelerator.is_main_process,
        desc=f"Eval Epoch {epoch}",
        ncols=120,
        leave=False,
    )

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        image_feat = batch["image_feat"]


        with accelerator.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_feat=image_feat,
                labels=labels,
            )
            loss = outputs["loss"]

        val_loss += loss.item()
        val_steps += 1


        batch_size = input_ids.size(0)

        for b in range(batch_size):
            prompt_len = batch["prompt_len"][b].item()

            prompt_ids = input_ids[b, :prompt_len].unsqueeze(0)
            prompt_mask = attention_mask[b, :prompt_len].unsqueeze(0)
            image_feat_b = image_feat[b].unsqueeze(0)

            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                image_feat=image_feat_b,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )


            # if accelerator.is_main_process and b == 0:
            #     print(f"[Debug] Generated length: {generated_ids.shape[1]}")

            generated_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )


            label_b = labels[b]
            non_ignore_mask = label_b != -100
            ref_ids = label_b[non_ignore_mask]
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            all_generated.append(generated_text)
            all_references.append(ref_text)

            if len(vis_samples) < num_vis_samples:
                vis_samples.append({
                    "generated": generated_text,
                    "reference": ref_text,
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                })

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0


    metrics = {"val_loss": avg_val_loss}

    if len(all_generated) > 0:

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for gen, ref in zip(all_generated, all_references):
            scores = scorer.score(ref, gen)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        metrics["rouge1"] = sum(rouge1_scores) / len(rouge1_scores)
        metrics["rouge2"] = sum(rouge2_scores) / len(rouge2_scores)
        metrics["rougeL"] = sum(rougeL_scores) / len(rougeL_scores)


        try:
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(all_generated, [all_references])
            metrics["bleu"] = bleu.score
        except ImportError:
            print("[Warning] sacrebleu not installed, skipping BLEU calculation")
            metrics["bleu"] = 0.0


    if accelerator.is_main_process and len(vis_samples) > 0:
        vis_dir = join(output_dir, "output_samples")
        os.makedirs(vis_dir, exist_ok=True)

        vis_file = join(vis_dir, f"epoch{epoch:04d}_samples.txt")
        with open(vis_file, "w", encoding="utf-8") as f:
            f.write(f"=== Epoch {epoch} Generated Samples ===\n\n")
            for i, sample in enumerate(vis_samples):
                f.write(f"--- Sample {i + 1} (batch {sample['batch_idx']}, idx {sample['sample_idx']}) ---\n")
                f.write(f"[Generated]:\n{sample['generated']}\n\n")
                f.write(f"[Reference]:\n{sample['reference']}\n\n")
                f.write("-" * 80 + "\n\n")

        print(f"[Vis] Saved samples to {vis_file}")

    return avg_val_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    setproctitle.setproctitle('liver-report-training')
    project_root = PROJECT_ROOT
    parser.add_argument("--config", type=str,
                        default=str(project_root / 'config' / '260712_liver_qwen.yaml'))
    parser.add_argument("--resume", help="Resume training from a checkpoint", type=bool,
                        default=False)
    args = parser.parse_args()


    config = load_config(args.config)
    config['_config_path'] = args.config
    for key, value in config['paths'].items():
        config['paths'][key] = str((project_root / value).resolve())


    save_dir = join(config['paths']['output_dir'], config['save_name'])
    os.makedirs(save_dir, exist_ok=True)


    config_save_path = save_config(args.config, save_dir, config['save_name'])
    logging.info(f"[Config] Saved to: {config_save_path}")


    if config['device'] != 'cuda':
        raise ValueError("Set device to 'cuda'; select a GPU with CUDA_VISIBLE_DEVICES.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. A CUDA-enabled PyTorch installation and GPU are required.")

    use_tf32 = config['runtime'].get('use_tf32', False)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    mixed_precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    )
    device = accelerator.device
    if device.type != 'cuda':
        raise RuntimeError(f"Accelerate selected {device}; CUDA training is required.")

    if accelerator.is_main_process:
        print(f"[INFO] Rank {accelerator.process_index} using device: {device}, precision: {mixed_precision}")
        print(f"[Config] {json.dumps(dict(config), indent=2, ensure_ascii=False)}")


    setup_seed(config['training']['seed'])


    model_path = config['paths']['model_path']

    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    from tokenization_qwen import QWenTokenizer

    tokenizer = QWenTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eod_id


    torch_dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float16

    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=config['runtime']['low_cpu_mem_usage'],
        local_files_only=True,
    )
    base_model.config.use_cache = False


    if base_model.config.bos_token_id is None and base_model.config.eos_token_id is not None:
        base_model.config.bos_token_id = base_model.config.eos_token_id
        if accelerator.is_main_process:
            print(f"[Fix] Set bos_token_id = {base_model.config.bos_token_id}")

    # Gradient checkpointing
    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        base_model.gradient_checkpointing_enable()

    # LoRA
    lora_cfg = config['lora']
    lora_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, lora_config)


    eod_id = tokenizer.eod_id


    if base_model.config.bos_token_id is None:
        base_model.config.bos_token_id = eod_id
        if accelerator.is_main_process:
            print(f"[Fix] Set PeftModel bos_token_id = {eod_id}")


    if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'config'):
        real_config = base_model.base_model.config
        if getattr(real_config, 'bos_token_id', None) is None:
            real_config.bos_token_id = eod_id
            if accelerator.is_main_process:
                print(f"[Fix] Set base_model bos_token_id = {eod_id}")

    if accelerator.is_main_process:
        base_model.print_trainable_parameters()


    train_dataset = LiverDataset(
        h5_path=config['paths']['h5_path'],
        split_json=config['paths']['split_json'],
        split="train",
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        report_prompts=config['prompt']['report_prompts'],
        system_prompt=config['prompt']['system_prompt'],
        use_chat_template=False,
    )
    eval_dataset = LiverDataset(
        h5_path=config['paths']['h5_path'],
        split_json=config['paths']['split_json'],
        split="val",
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        report_prompts=config['prompt']['report_prompts'],
        system_prompt=config['prompt']['system_prompt'],
        use_chat_template=False,
    )


    train_cfg = config['training']
    train_dl = DataLoader(
        train_dataset,
        batch_size=train_cfg['per_device_train_batch_size'],
        shuffle=True,
        num_workers=train_cfg['dataloader_num_workers'],
        collate_fn=LiverDataCollator(tokenizer),
        pin_memory=True,
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=train_cfg['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=LiverDataCollator(tokenizer),
        pin_memory=True,
    )


    model_cfg = config['model']
    model = LiverReportQwen(
        llm=base_model,
        image_dim=model_cfg['image_dim'],
        prompt_len=model_cfg['prompt_len'],
        adapter_hidden_dim=model_cfg['adapter_hidden_dim'],
        dropout=model_cfg['dropout'],
    )


    num_epochs = int(train_cfg['num_train_epochs'])
    steps_per_epoch = len(train_dl) // train_cfg['gradient_accumulation_steps']
    if len(train_dl) % train_cfg['gradient_accumulation_steps'] != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * num_epochs

    optimizer = build_optimizer(model, train_cfg['learning_rate'], train_cfg['weight_decay'])
    scheduler = build_scheduler(
        optimizer, total_steps, train_cfg['warmup_ratio'], train_cfg['lr_scheduler_type']
    )


    start_epoch = 1
    global_step = 0
    ckpt_dir = join(save_dir, "checkpoints")

    if args.resume:

        start_epoch, global_step = load_checkpoint(
            ckpt_dir=ckpt_dir,
            base_model=base_model,
            adapter=model.adapter,
            optimizer=optimizer,
            scheduler=scheduler,
        )


    model, optimizer, train_dl, eval_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, scheduler
    )

    # ========== CSV Logger ==========
    csv_writer = None
    if accelerator.is_main_process:
        csv_writer = EpochCSVLogger(
            save_path=save_dir,
            filename=f"{config['save_name']}.csv",
            resume=args.resume,
        )


    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        # ---- Train ----
        avg_train_loss, global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            epoch=epoch,
            global_step=global_step,
            max_grad_norm=train_cfg['max_grad_norm'],
            start_time=start_time,
            total_steps=total_steps,
        )

        if accelerator.is_main_process:
            print(
                f"\n[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")


            csv_writer.add_epoch(
                epoch=epoch,
                train_loss=f"{avg_train_loss:.6f}",
                learning_rate=f"{optimizer.param_groups[0]['lr']:.8f}",
            )

        # ---- Eval ----
        eval_freq = train_cfg.get('eval_freq', 1)
        vis_freq = train_cfg.get('vis_freq', 2)

        if epoch % eval_freq == 0:
            avg_val_loss, metrics = eval_one_epoch(
                accelerator=accelerator,
                model=model,
                eval_dl=eval_dl,
                tokenizer=tokenizer,
                epoch=epoch,
                output_dir=save_dir,
                num_vis_samples=train_cfg.get('num_vis_samples', 3),
                gen_config=config['generation']
            )

            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f}")
                for k, v in metrics.items():
                    if k != "val_loss":
                        print(f"  {k}: {v:.4f}")


                log_dict = {
                    "epoch": epoch,
                    "train_loss": f"{avg_train_loss:.6f}",
                    "val_loss": f"{avg_val_loss:.6f}",
                    "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}",
                }
                for k, v in metrics.items():
                    if k != "val_loss":
                        log_dict[k] = f"{v:.6f}"
                csv_writer.add_epoch(**log_dict)
        else:
            if accelerator.is_main_process:
                csv_writer.add_epoch(
                    epoch=epoch,
                    train_loss=f"{avg_train_loss:.6f}",
                    learning_rate=f"{optimizer.param_groups[0]['lr']:.8f}",
                )

        # ---- Save checkpoint ----
        if epoch % train_cfg['save_ckpt_freq'] == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                ckpt_dir=ckpt_dir,
            )


    if accelerator.is_main_process:
        final_dir = join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.llm.save_pretrained(join(final_dir, "llm_lora"))
        tokenizer.save_pretrained(join(final_dir, "llm_lora"))
        unwrapped_model.save_adapter(final_dir)

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(f"LoRA weights: {join(final_dir, 'llm_lora')}")
        print(f"Adapter weights: {join(final_dir, 'liver_adapter.pt')}")
        print(f"Training log: {join(save_dir, config['save_name'] + '.csv')}")
        print(f"{'=' * 60}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
