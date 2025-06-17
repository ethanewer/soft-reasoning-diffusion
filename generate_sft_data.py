import torch
from datasets import load_dataset
from tqdm.notebook import trange
from transformers import (
    AutoModelForCausalLM,  # type: ignore
    AutoTokenizer,  # type: ignore
    DynamicCache,  # type: ignore
)

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
embed_layer = model.get_input_embeddings()

ds = load_dataset("openai/gsm8k", name="main", split="train")
question_messages = [
    [{"role": "user", "content": question}]
    for question in ds["question"]  # type: ignore
]
question_text = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    for messages in question_messages
]

batch_size = 32
num_thinking_tokens = 1024

sft_data = []

with torch.no_grad():
    for i in trange(0, len(question_text), batch_size, desc="Building SFT Data"):
        inputs = tokenizer(
            question_text[i : i + batch_size],
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(model.device)
        thinking_tag = tokenizer.encode("<think>")[0]
        input_ids = torch.nn.functional.pad(
            inputs.input_ids, pad=(0, 1), value=thinking_tag
        )
        attention_mask = torch.nn.functional.pad(
            inputs.attention_mask, pad=(0, 1), value=1
        )
        past_key_values = DynamicCache()

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            ).logits[:, -1:]
            attention_mask = torch.nn.functional.pad(
                attention_mask, pad=(0, 1), value=1
            )

        embeds = []

        for _ in trange(num_thinking_tokens, desc="Soft Thinking", leave=False):
            probabilities = logits.float().softmax(dim=-1).to(model.dtype)
            next_embeds = probabilities @ embed_layer.weight
            with torch.no_grad():
                logits = model(
                    inputs_embeds=next_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits

            attention_mask = torch.nn.functional.pad(
                attention_mask, pad=(0, 1), value=1
            )
            embeds.append(next_embeds.cpu())

        torch.cuda.empty_cache()

        for j in range(input_ids.shape[0]):
            sft_data.append(
                {
                    "input_ids": input_ids[
                        j, attention_mask[j, : input_ids.shape[1]] == 1
                    ].cpu(),
                    "embeds": torch.cat(embeds, dim=1),
                }
            )

        if i % (10 * batch_size) == 0:
            torch.save(sft_data, "sft_data.pt")


torch.save(sft_data, "sft_data.pt")
