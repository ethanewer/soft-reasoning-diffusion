#!/usr/bin/env python
import argparse

import torch
import torch.nn.functional as F
import yaml  # type: ignore
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers.cache_utils import DynamicCache

# Import your model implementation
from soft_reasoning_diffusion_llm import SoftReasoningDiffusionLLM


class ReasoningDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of dicts, each with
          - "input_ids": LongTensor [seq_len]
          - "embeds": FloatTensor [num_thinking_tokens, embed_dim]
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx]["embeds"]


def collate_fn(batch):
    input_ids, embeds = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    embeds = torch.stack(embeds, dim=0)
    return input_ids, embeds


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SoftReasoningDiffusionLLM with YAML config"
    )
    p.add_argument("config", type=str, help="Path to YAML config file")
    return p.parse_args()


def make_diffusion_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def main():
    args = parse_args()
    # --- 1) Load config ---------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 2) Prepare device ------------------------------
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- 3) Load data ----------------------------------
    # expects a Torch-saved list[dict] at cfg["data_path"]
    data_list = torch.load(cfg["data_path"])
    dataset = ReasoningDataset(data_list)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # --- 4) Build diffusion schedule -------------------
    T = cfg["diffusion"]["T"]
    beta_start = cfg["diffusion"]["beta_schedule"]["start"]
    beta_end = cfg["diffusion"]["beta_schedule"]["end"]
    betas, alphas, alpha_bars = make_diffusion_schedule(T, beta_start, beta_end, device)

    # --- 5) Init model & optimizer ---------------------
    model = SoftReasoningDiffusionLLM(
        model_name=cfg["model"]["name"],
        tokenizer_name=cfg["model"]["tokenizer"],
        begin_thinking_token=cfg["model"].get("begin_thinking_token", "<think>"),
        end_thinking_token=cfg["model"].get("end_thinking_token", "</think>"),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # --- 6) Training loop -------------------------------
    num_epochs = cfg["training"]["num_epochs"]
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for input_ids, target_embeds in loader:
            B, S, D = target_embeds.shape
            input_ids = input_ids.to(device)
            target_embeds = target_embeds.to(device)

            # a) sample random timesteps
            timesteps = torch.randint(0, T, (B,), device=device)
            a_b = alpha_bars[timesteps].view(B, 1, 1)

            # b) add noise
            noise = torch.randn_like(target_embeds)
            noisy_embeds = a_b.sqrt() * target_embeds + (1 - a_b).sqrt() * noise

            # c) build input_embeds with zero “<think>” prefix
            prefix = torch.zeros(B, 1, D, device=device)
            input_embeds = torch.cat([prefix, noisy_embeds], dim=1)
            attention_mask = torch.ones(B, 1 + S, device=device)

            past = DynamicCache()
            # d) denoise one step
            pred = model.denoise(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                past_key_values=past,
            )
            pred_embeds = pred[:, 1:, :]  # drop prefix

            # e) MSE loss
            loss = F.mse_loss(pred_embeds, target_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

        avg_loss = running_loss / len(dataset)
        print(f"[Epoch {epoch:2d}/{num_epochs}]  Avg MSE Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
