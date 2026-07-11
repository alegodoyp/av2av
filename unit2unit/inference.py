import argparse
import os
import numpy as np
import torch

from fairseq import checkpoint_utils, utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from unit2unit.task import UTUTPretrainingTask
from util import process_units, save_unit

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Must match scripts/train_full_pipeline.py's create_extended_dict() exactly:
# this order was reverse-engineered to match utut_sts_ft.pt's embedding table.
# fairseq's MultilingualDenoisingTask.setup_task (which UTUTPretrainingTask
# inherits) only adds dictionary symbols for cfg.task.langs (e.g. "pt,en" for
# a checkpoint fine-tuned with --langs pt,en). Loading with the plain,
# unit-only dict.txt would dynamically add just those 2 lang tokens at the
# wrong indices, mismatching the checkpoint's embedding rows. Using this
# extended dict (all 19 langs pre-baked) makes that add a no-op lookup that
# lands on the same indices used at training time, for both the original
# pretrained checkpoint and any fine-tune derived from it.
_LANG_ORDER = ["en", "es", "fr", "it", "pt", "el", "ru", "cs", "da", "de",
               "fi", "hr", "hu", "lt", "nl", "pl", "ro", "sk", "sl"]

def _ensure_extended_dict_dir():
    dict_dir = os.path.join(REPO, "dict_data")
    dict_path = os.path.join(dict_dir, "dict.txt")
    if not os.path.exists(dict_path):
        os.makedirs(dict_dir, exist_ok=True)
        with open(dict_path, "w") as f:
            for i in range(1000):
                f.write(f"{i} 1\n")
            for lang in _LANG_ORDER:
                f.write(f"[{lang}] 1\n")
            f.write("<mask> 1\n")
    return dict_dir

def load_model(model_path, src_lang, tgt_lang, use_cuda=False):
    # Checkpoints store the training-time task.data path (e.g. a scratch dir
    # on the training machine), which doesn't exist at inference time.
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"user_dir": "unit2unit", "data": _ensure_extended_dict_dir()}
    )

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    for model in models:
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    task.source_language = src_lang
    task.target_language = tgt_lang

    # setup_task already added <mask> via UTUTPretrainingTask.__init__; this is a no-op.
    task.dictionary.add_symbol("<mask>")
    task.mask_idx = task.dictionary.index("<mask>")

    # Override Fairseq defaults to permit inference outputs over 2.0 seconds (i.e., 200 tokens @100hz)
    cfg.generation.max_len_a = 1.2
    cfg.generation.max_len_b = 100
    cfg.generation.beam = 5

    generator = task.build_generator(
        models, cfg.generation
    )

    return task, generator

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    task, generator = load_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)

    with open(args.in_unit_path) as f:
        unit = list(map(int, f.readline().strip().split()))
    unit = task.source_dictionary.encode_line(
        " ".join(map(lambda x: str(x), process_units(unit, reduce=True))),
        add_if_not_exist=False,
        append_eos=True,
    ).long()
    unit = torch.cat([
        unit.new([task.source_dictionary.bos()]),
        unit,
        unit.new([task.source_dictionary.index("[{}]".format(task.source_language))])
    ])

    sample = {"net_input": {
        "src_tokens": torch.LongTensor(unit).view(1,-1),
    }}
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    pred = task.inference_step(
        generator,
        None,
        sample,
    )[0][0]

    pred_str = task.target_dictionary.string(
        pred["tokens"].int().cpu(),
        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator)
    )

    save_unit(pred_str, args.out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit-path", type=str, required=True, help="File path of source unit input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="File path of target unit output"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
