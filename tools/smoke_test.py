# P0 smoke test: load Llama-3.2-1B (ModelScope mirror), verify forward pass,
# verify hidden_states[-1] == lm_head input (post-final-RMSNorm), report memory.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
DATA_FILE = r"E:\Trash_things\llm_codec\data\babylm_data\babylm_10M\aochildes.train"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    print(f"embeddings tied (same tensor object): {tied}")
    print(f"embed table: {tuple(model.model.embed_tokens.weight.shape)}")

    with open(DATA_FILE, encoding="utf-8") as f:
        line = next(f).strip()
    print(f"sample line: {line[:80]}")
    ids = tok(line, return_tensors="pt").input_ids.cuda()
    print(f"tokens: {ids.shape[1]}")

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    h = out.hidden_states[-1]
    print(f"logits: {tuple(out.logits.shape)}, hidden[-1]: {tuple(h.shape)}")

    # verify hidden_states[-1] is exactly what lm_head consumes
    recomputed = model.lm_head(h)
    diff = (recomputed - out.logits).abs().max().item()
    print(f"max |lm_head(hidden[-1]) - logits|: {diff:.6f}")

    # sanity: top-1 next token after the line
    nxt = out.logits[0, -1].argmax().item()
    print(f"teacher top-1 next token: {tok.decode([nxt])!r}")

    # top-40 mass coverage on this position
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    top40 = probs.topk(40).values.sum().item()
    print(f"top-40 mass coverage (1 position, anecdotal): {top40:.4f}")

    print(f"GPU mem allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

if __name__ == "__main__":
    main()
