import argparse
import json
import os

import sacrebleu
import whisper
from jiwer import wer
from transformers import MarianMTModel, MarianTokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="ASR-BLEU/WER comparing baseline vs. fine-tuned outputs against "
        "a PT->EN pseudo-reference (Whisper transcript of the source audio, machine-"
        "translated). No human reference translation exists for these samples -- "
        "this is a proxy, and the dissertation text must say so explicitly."
    )
    parser.add_argument("--videos", nargs="+", default=["video1", "video2", "video3", "video4"])
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="_relatorio_dissertacao")
    parser.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large")
    parser.add_argument("--mt-model", default="Helsinki-NLP/opus-mt-pt-en")
    return parser


def transcribe(asr_model, path, language=None):
    result = asr_model.transcribe(path, language=language)
    return result["text"].strip()


def translate_pt_en(tokenizer, model, text):
    if not text:
        return ""
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**batch)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    args = get_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading Whisper ({args.whisper_model})...")
    asr_model = whisper.load_model(args.whisper_model)

    print(f"Loading MT model ({args.mt_model})...")
    mt_tokenizer = MarianTokenizer.from_pretrained(args.mt_model)
    mt_model = MarianMTModel.from_pretrained(args.mt_model)

    rows = []
    for v in args.videos:
        src_path = os.path.join(args.samples_dir, f"{v}.mp4")
        baseline_path = os.path.join(args.results_dir, f"{v}_pt2en.mp4")
        finetuned_path = os.path.join(args.results_dir, f"{v}_pt2en_finetuned.mp4")

        print(f"\n=== {v} ===")
        print("Transcribing PT source...")
        pt_text = transcribe(asr_model, src_path, language="pt")
        print(f"  PT: {pt_text}")

        print("Translating PT -> EN (pseudo-reference)...")
        pseudo_ref = translate_pt_en(mt_tokenizer, mt_model, pt_text)
        print(f"  pseudo-ref EN: {pseudo_ref}")

        print("Transcribing baseline (pre-trained checkpoint) output...")
        baseline_hyp = transcribe(asr_model, baseline_path, language="en")
        print(f"  baseline hyp: {baseline_hyp}")

        print("Transcribing fine-tuned checkpoint output...")
        finetuned_hyp = transcribe(asr_model, finetuned_path, language="en")
        print(f"  finetuned hyp: {finetuned_hyp}")

        bleu_baseline = sacrebleu.corpus_bleu([baseline_hyp], [[pseudo_ref]]).score
        bleu_finetuned = sacrebleu.corpus_bleu([finetuned_hyp], [[pseudo_ref]]).score
        wer_baseline = wer(pseudo_ref, baseline_hyp) if pseudo_ref else None
        wer_finetuned = wer(pseudo_ref, finetuned_hyp) if pseudo_ref else None

        print(
            f"  BLEU: baseline={bleu_baseline:.2f} finetuned={bleu_finetuned:.2f} | "
            f"WER: baseline={wer_baseline:.3f} finetuned={wer_finetuned:.3f}"
        )

        rows.append({
            "video": v,
            "pt_transcript": pt_text,
            "pseudo_reference_en": pseudo_ref,
            "baseline_hyp": baseline_hyp,
            "finetuned_hyp": finetuned_hyp,
            "bleu_baseline": bleu_baseline,
            "bleu_finetuned": bleu_finetuned,
            "wer_baseline": wer_baseline,
            "wer_finetuned": wer_finetuned,
        })

    out_json = os.path.join(args.out_dir, "asr_bleu_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nPer-video results saved to {out_json}")

    all_refs = [r["pseudo_reference_en"] for r in rows]
    all_baseline = [r["baseline_hyp"] for r in rows]
    all_finetuned = [r["finetuned_hyp"] for r in rows]
    corpus_bleu_baseline = sacrebleu.corpus_bleu(all_baseline, [all_refs]).score
    corpus_bleu_finetuned = sacrebleu.corpus_bleu(all_finetuned, [all_refs]).score
    print(
        f"\nCorpus BLEU (todos os {len(rows)} vídeos): "
        f"baseline={corpus_bleu_baseline:.2f}  finetuned={corpus_bleu_finetuned:.2f}"
    )


if __name__ == "__main__":
    main()
