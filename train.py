import argparse

import torch
import torch.nn.functional as F
import yaml  # type: ignore
from torch import Tensor, nn, optim
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from tqdm import trange  # type: ignore
from transformers.cache_utils import DynamicCache

from soft_reasoning_diffusion_llm import SoftReasoningDiffusionLLM


class EmbeddingDataset(Dataset):
    def __init__(self, data: list[dict[str, Tensor]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return self.data[i]


def collate_fn(batch: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
    input_ids = nn.utils.rnn.pad_sequence(
        [example["input_ids"] for example in batch],
        batch_first=True,
        padding_value=-1,
        padding_side="left",
    )
    attention_mask = (input_ids >= 0).long()
    embeds = torch.stack([example["embeds"] for example in batch])
    return input_ids.relu(), attention_mask, embeds


def make_diffusion_schedule(cfg: dict[str, dict], device: Device) -> Tensor:
    T = cfg["diffusion"]["T"]
    beta_start = cfg["diffusion"]["beta_schedule"]["start"]
    beta_end = cfg["diffusion"]["beta_schedule"]["end"]
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def train_step(
    model: SoftReasoningDiffusionLLM,
    optimizer: optim.Optimizer,
    input_ids: Tensor,
    attention_mask: Tensor,
    target_embeds: Tensor,
    diffusion_schedule: Tensor,
    device: Device,
):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_embeds = target_embeds.to(device)
    past_key_values = DynamicCache()

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )[:, -1]

    timesteps = torch.randint(
        low=0,
        high=len(diffusion_schedule),
        size=(target_embeds.shape[0],),
        device=device,
    )
    alpha_bars = diffusion_schedule[timesteps][:, None, None]

    noise = torch.randn_like(target_embeds)
    noisy_embeds = alpha_bars.sqrt() * target_embeds + (1 - alpha_bars).sqrt() * noise

    pred_embeds = model.denoise(
        input_embeds=noisy_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )

    loss = F.mse_loss(pred_embeds, target_embeds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to YAML config file")
    with open(p.parse_args().config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmbeddingDataset(torch.load(cfg["data_path"]))
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    diffusion_schedule = make_diffusion_schedule(cfg, device)

    model = SoftReasoningDiffusionLLM(
        model_name=cfg["model"]["name"],
        tokenizer_name=cfg["model"]["tokenizer"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    num_epochs = cfg["training"]["num_epochs"]
    for epoch in trange(num_epochs):
        running_loss = 0.0
        for input_ids, attention_mask, target_embeds in loader:
            loss = train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_embeds=target_embeds,
                diffusion_schedule=diffusion_schedule,
                device=device,
            )
            running_loss += loss.item() * input_ids.shape[0]

        avg_loss = running_loss / len(dataset)
        print(f"[Epoch {epoch + 1:2d}/{num_epochs}]  Avg MSE Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
