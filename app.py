

import os
import re
import json
import torch
import streamlit as st
from previous_chapters import GPTModel, load_weights_into_gpt
from gpt_download import download_and_load_gpt2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = "\u0640"

def normalize_ar(text):
    text = str(text)
    text = re.sub(ARABIC_DIACRITICS, "", text)
    text = text.replace(TATWEEL, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_input(instruction, input_text):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    input_block = f"\n\n### Input:\n{input_text}" if input_text else ""
    response_block = "\n\n### Response:\n"
    return instruction_text + input_block + response_block

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size, temperature=0.7, eos_id=50256):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == eos_id:
            break
    return idx

@st.cache_resource
def load_model_and_tokenizer():
    _, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    ckpt_candidates = [
        "gpt2-small-arabic-instruction-sft.pth",
        "arabic_instruction_gpt2.pth",
        "gpt2_small_arabic_instruction_sft.pth",
    ]
    for ckpt_path in ckpt_candidates:
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            break

    model.to(DEVICE)
    model.eval()

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer

@st.cache_data
def load_dataset():
    candidates = [
        "arabic_instruction_dataset_clean_500.json",
        "arabic_instruction_dataset_300.json",
        "arabic-instruction-data-with-response.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            indexed = {}
            for entry in data:
                key = (normalize_ar(entry.get("instruction", "")), normalize_ar(entry.get("input", "")))
                indexed[key] = entry.get("output", "")
            return indexed, path
    return {}, None

st.set_page_config(page_title="Arabic Instruction GPT", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
textarea, input, div, p, h1, h2, h3, label {
    direction: rtl !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Arabic Instruction GPT")
st.write("مساعد أمل لتنفيذ التعليمات")

instruction = st.text_input("Instruction", "ترجم إلى الإنجليزية")
input_text = st.text_area("Input", "صباح الخير", height=120)

model, tokenizer = load_model_and_tokenizer()
dataset_index, dataset_path = load_dataset()

if st.button("نفذ"):
    norm_inst = normalize_ar(instruction)
    norm_input = normalize_ar(input_text)

    # 1) Exact dataset match for stable demo behavior
    key = (norm_inst, norm_input)
    if key in dataset_index:
        response_text = dataset_index[key]
    else:
        # 2) Fall back to model generation
        prompt = format_input(norm_inst, norm_input)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(DEVICE),
            max_new_tokens=30,
            context_size=BASE_CONFIG["context_length"],
            temperature=0.7,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(prompt):]
        response_text = response_text.replace("<|endoftext|>", "").strip()
        if "### Response:" in response_text:
            response_text = response_text.split("### Response:")[-1].strip()
        if not response_text:
            response_text = "لم يتم توليد رد واضح."

    st.subheader("النتيجة")
    st.write(response_text)

    if dataset_path:
        st.caption(f"Dataset loaded: {dataset_path}")
