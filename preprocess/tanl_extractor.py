# run_tanl_scico_extraction.py
import json, argparse, torch, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def load_tanl(model_dir, tokenizer_dir='t5-base', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    return tok, model, device

def tanl_generate(tok, model, device, text, task="scierc_joint_er",
                  prefix=None, num_beams=4, max_in=512, max_out=1024):
    task_prefix = f"{task}: " if prefix is None else prefix
    inp = task_prefix + text.strip()
    enc = tok(inp, return_tensors="pt", truncation=True, max_length=max_in)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=num_beams,
            max_length=max_out,
            early_stopping=True,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True)

# very light detok so we don't get weird spaces before punctuation
_punct = re.compile(r"\s+([,.;:%)\]])")
_openp = re.compile(r"([\[(])\s+")
def detok(tokens):
    s = " ".join(tokens)
    s = _punct.sub(r"\1", s)
    s = _openp.sub(r"\1", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_dir", default="t5-base")
    ap.add_argument("--split", default="test", choices=["train","validation","test","dev","val","development","test"])
    ap.add_argument("--out", default="scico_tanl_extraction.jsonl")
    ap.add_argument("--task", default="scierc_joint_er")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_input_len", type=int, default=1024)
    ap.add_argument("--max_output_len", type=int, default=1024)
    ap.add_argument("--max_topics", type=int, default=-1)
    args = ap.parse_args()

    # map split aliases
    split = {"dev":"validation","val":"validation","development":"validation"}.get(args.split, args.split)
    ds = load_dataset("allenai/scico")[split]

    tok, model, device = load_tanl(args.model_dir, args.tokenizer_dir)

    with open(args.out, "w", encoding="utf-8") as fout:
        n_topics = len(ds) if args.max_topics < 0 else min(args.max_topics, len(ds))
        for t in tqdm(range(n_topics), desc=f"Extracting {split}"):
            row = ds[t]
            topic_id = row["id"]
            token_paras = row["tokens"]             # list[list[str]]
            doc_ids = row["doc_ids"]                # list[int], aligned with paragraphs
            para_sent_spans = row.get("sentences")  # list[list[[start,end]]], optional
            # precompute cumulative flatten offsets in case we need them later
            cum = 0
            flatten_offsets = []
            for toks in token_paras:
                flatten_offsets.append(cum)
                cum += len(toks)

            for pidx, toks in enumerate(token_paras):
                text = detok(toks)

                # === call TANL exactly like your CLI would ===
                tanl_text = tanl_generate(
                    tok, model, device, text,
                    task=args.task, prefix=args.prefix,
                    num_beams=args.num_beams,
                    max_in=args.max_input_len, max_out=args.max_output_len
                )

                rec = {
                    "topic_id": int(topic_id),
                    "para_idx": int(pidx),
                    "doc_id": int(doc_ids[pidx]),
                    "flatten_offset": int(flatten_offsets[pidx]),
                    "n_tokens": int(len(toks)),
                    "text": text,
                    # Treat the raw TANL-augmented paragraph as our initial “signature”.
                    "tanl_output": tanl_text,
                    # Optional: sentence spans so you can window later without reloading SciCo
                    "sentences": para_sent_spans[pidx] if para_sent_spans else None
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
