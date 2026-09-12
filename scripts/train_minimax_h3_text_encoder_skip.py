"""Run a real MiniMax-H3 training job without ever loading the ~32B text
encoder, PROVIDED every prompt the run could need is already cached to disk.

load_model() normally loads the transformer, text encoder, and VAE together,
so the text encoder sits in VRAM/RAM alongside the ~33B transformer for the
whole setup phase (network build, optimizer, dataset caching) even when it
turns out to do nothing -- if every dataset item is already cached, the
per-item pass never touches it. This script checks, upfront and before
anything is loaded, whether that's actually true; if so it runs the job with
model_kwargs.text_encoder_precached set (see minimax_h3.py's
_load_text_encoder, which then loads a lightweight stub instead of the real
32B model). If anything is missing, it aborts loudly instead of silently
falling back to loading the text encoder -- which could blow the VRAM budget
of a box sized only for the transformer.

Only covers configs where nothing else needs live text encoding: sampling
must be disabled, and no trigger_word or diff_output_preservation. (Sample
previews, trigger-word prompts, and DOP class prompts aren't cached by
scripts/cache_minimax_h3_text_embeds.py, so this script refuses to skip the
text encoder if any of those are in play -- it would otherwise crash deep
into a real training run instead of failing the upfront check.)

Usage:
    1. python scripts/cache_minimax_h3_text_embeds.py path/to/config.yaml
    2. python scripts/train_minimax_h3_text_encoder_skip.py path/to/config.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.data_loader import AiToolkitDataset
from toolkit.job import get_job
from toolkit.print import print_acc
from toolkit.util.get_model import get_model_class

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "config_file_list",
    nargs="+",
    type=str,
    help="Training config file(s) -- the same ones you'd pass to run.py",
)
parser.add_argument(
    "-n", "--name", type=str, default=None, help="Name to replace [name] tag in the config"
)
args = parser.parse_args()


def check_feature_compatibility(process) -> list:
    reasons = []
    if not str(getattr(process.model_config, "arch", "")).startswith("minimax_h3"):
        reasons.append(f"arch '{process.model_config.arch}' is not a MiniMax-H3 arch")

    has_samples = (
        process.sample_config is not None
        and process.sample_config.samples is not None
        and len(process.sample_config.samples) > 0
    )
    if not process.train_config.disable_sampling and has_samples:
        reasons.append(
            "sampling is enabled with configured samples -- set "
            "train.disable_sampling: true (sample-preview prompts are not "
            "covered by the pre-cache)"
        )

    if any(ds.trigger_word is not None for ds in process.dataset_configs):
        reasons.append(
            "a trigger_word is set -- trigger-word prompts are not covered "
            "by the pre-cache"
        )

    if process.train_config.diff_output_preservation or any(
        getattr(ds, "diff_output_preservation", False) for ds in process.dataset_configs
    ):
        reasons.append(
            "diff_output_preservation is enabled -- DOP class prompts are "
            "not covered by the pre-cache"
        )
    return reasons


def check_cache_completeness(process, sd_probe) -> list:
    missing = []
    for ds_cfg in list(process.dataset_configs):
        batch_size = ds_cfg.batch_size if ds_cfg.batch_size is not None else process.train_config.batch_size
        ds = AiToolkitDataset(ds_cfg, batch_size=batch_size, sd=sd_probe, skip_setup_epoch=True)
        if not ds.text_embeddings_are_cached():
            missing.append(f"dataset '{ds_cfg.folder_path or ds_cfg.dataset_path}'")

    for static_prompt in {"", process.train_config.unconditional_prompt.strip()}:
        static_path = sd_probe._static_prompt_cache_path(static_prompt)
        if not os.path.exists(static_path):
            missing.append(f"static prompt {static_prompt!r} (expected at {static_path})")
    return missing


def run_process(process):
    print_acc(f"=== Checking cache completeness for process '{process.name}' ===")

    reasons = check_feature_compatibility(process)
    if reasons:
        raise SystemExit(
            f"{process.name}: cannot skip the text encoder for this run:\n - "
            + "\n - ".join(reasons)
        )

    static_cache_dir = os.path.join(process.save_root, "_static_te_cache")
    process.model_config.model_kwargs["static_text_embed_cache_dir"] = static_cache_dir

    ModelClass = get_model_class(process.model_config)
    # a probe instance only -- load_model() is never called on it, so it
    # costs nothing and never touches the disk or a GPU
    sd_probe = ModelClass(
        device=process.accelerator.device,
        model_config=process.model_config,
        dtype=process.train_config.dtype,
    )
    missing = check_cache_completeness(process, sd_probe)
    del sd_probe

    if missing:
        raise SystemExit(
            f"{process.name}: not all text embeddings are cached, aborting "
            "(run scripts/cache_minimax_h3_text_embeds.py first):\n - "
            + "\n - ".join(missing)
        )

    print_acc(
        f"=== All text embeddings cached for '{process.name}' -- running "
        "with the text encoder skipped ==="
    )
    process.model_config.model_kwargs["text_encoder_precached"] = True
    process.run()


for config_file in args.config_file_list:
    job = get_job(config_file, args.name)
    if not hasattr(job, "process"):
        raise ValueError(
            f"{config_file}: expected a 'job: extension' training config "
            "(the same one you'd train with)"
        )
    for process in job.process:
        run_process(process)
    job.cleanup()
