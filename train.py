import argparse
import os
from functools import partial

import torch
import torch.nn.functional as F
import yaml  # type: ignore
from accelerate import Accelerator  # type: ignore
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm, trange  # type: ignore
from transformers.cache_utils import DynamicCache

from soft_reasoning_diffusion_llm import SoftReasoningDiffusionLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingDataset(Dataset):
    def __init__(self, data: list[dict[str, Tensor]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return self.data[i]


def collate_fn(
    batch: list[dict[str, Tensor]],
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    input_ids = nn.utils.rnn.pad_sequence(
        [example["input_ids"] for example in batch],
        batch_first=True,
        padding_value=pad_id,
        padding_side="left",
    )
    attention_mask = (input_ids != pad_id).long()
    embeds = torch.stack([example["embeds"] for example in batch])
    return input_ids, attention_mask, embeds


def compute_metrics(true: Tensor, pred: Tensor) -> dict[str, float]:
    t = true.detach().cpu().reshape(-1)
    p = pred.detach().cpu().reshape(-1)
    diff = p - t

    mse = float((diff**2).mean())
    mae = float(diff.abs().mean())

    mean_t = t.mean()
    ss_tot = ((t - mean_t) ** 2).sum()
    ss_res = (diff**2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "mae": mae, "r2": r2}


def compute_loss(
    model: SoftReasoningDiffusionLLM,
    scheduler: DDPMScheduler,
    input_ids: Tensor,
    attention_mask: Tensor,
    target_embeds: Tensor,
) -> Tensor:
    # build cache
    past_key_values = DynamicCache()
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=F.pad(attention_mask, (0, target_embeds.shape[1]), value=1),
            past_key_values=past_key_values,
        )[:, -1]

    # sample noise
    batch_size = target_embeds.shape[0]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,  # type: ignore
        (batch_size,),
        device=input_ids.device,
    )
    noise = torch.randn_like(target_embeds)
    noisy = scheduler.add_noise(target_embeds, noise, timesteps)  # type: ignore

    # predict noise
    pred = model.denoise(
        inputs_embeds=noisy.to(target_embeds.dtype),
        attention_mask=F.pad(attention_mask, (0, target_embeds.shape[1]), value=1),
        past_key_values=past_key_values,
    )

    loss = F.mse_loss(pred, noise)
    return loss


def evaluate(
    model: SoftReasoningDiffusionLLM,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_true = []
    all_pred = []

    torch.manual_seed(0)

    T = scheduler.config.num_train_timesteps  # type: ignore
    timesteps = [
        0,
        T // 4,
        T // 2,
        3 * T // 4,
        T - 1,
    ]

    with torch.no_grad():
        for input_ids, attention_mask, target_embeds in dataloader:
            input_ids = input_ids.to(device)
            target_embeds = target_embeds.to(device)
            attention_mask = F.pad(
                attention_mask,
                (0, target_embeds.shape[1]),
                value=1,
            ).to(device)

            for t in timesteps:
                t_batch = torch.full(
                    (target_embeds.size(0),),
                    t,
                    device=device,
                    dtype=torch.long,
                )

                noise = torch.randn_like(target_embeds)

                noisy = scheduler.add_noise(target_embeds, noise, t_batch)  # type: ignore

                past_key_values = DynamicCache()
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )[:, -1]
                pred = model.denoise(
                    inputs_embeds=noisy.to(target_embeds.dtype),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

                all_true.append(noise)
                all_pred.append(pred)

    true = torch.cat(all_true, dim=0)
    pred = torch.cat(all_pred, dim=0)
    return compute_metrics(true, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # scheduler, model, optimizer
    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["T"]),
        beta_start=float(cfg["diffusion"]["beta_schedule"]["start"]),
        beta_end=float(cfg["diffusion"]["beta_schedule"]["end"]),
    )

    model = SoftReasoningDiffusionLLM(**cfg["model"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
    )

    # full dataset
    full_data = torch.load(cfg["data_path"])
    dataset = EmbeddingDataset(full_data)

    # train/test split
    test_size = int(cfg["training"].get("test_size", 64))
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=model.tokenizer.pad_token_id),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=model.tokenizer.pad_token_id),
    )

    # prepare everything
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    model.train()

    num_epochs = cfg["training"]["num_epochs"]
    checkpoint_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in trange(num_epochs, desc="Training"):
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for input_ids, attention_mask, target_embeds in pbar:
            optimizer.zero_grad()
            loss = compute_loss(
                model,
                scheduler,
                input_ids,
                attention_mask,
                target_embeds,
            )
            accelerator.backward(loss)

            optimizer.step()

            bs = input_ids.shape[0]
            running_loss += loss.item() * bs
            seen += bs
            pbar.set_description(f"Train MSE: {running_loss / seen:.6f}")

        # evaluate on test set
        test_metrics = evaluate(model, scheduler, test_loader, device)
        print(
            f"\nEpoch {epoch + 1} -> "
            f"Test MSE: {test_metrics['mse']:.6f}, "
            f"MAE: {test_metrics['mae']:.6f}, "
            f"R^2: {test_metrics['r2']:.6f}"
        )

        # save checkpoint after each epoch
        if accelerator.is_main_process:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            accelerator.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
