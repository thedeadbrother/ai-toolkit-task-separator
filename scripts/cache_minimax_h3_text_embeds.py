"""Pre-cache MiniMax-H3 text embeddings to disk without loading the ~33B DiT.

Takes the SAME training config you'd use for a real run (job: extension,
process: [{type: sd_trainer, model: {...}, datasets: [...]}]), builds the
model with model_kwargs.text_encode_only forced on (see
extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py's
_load_transformer, which swaps in a lightweight stub instead of the real
transformer when that flag is set), and runs each dataset's on-disk
text-embedding cache pass -- the same pass a normal training run does
automatically for a dataset with cache_text_embeddings: true. Dataset configs
are built through the trainer's own parsing (trigger words, diff output
preservation, caption dropout, ...) so the cache keys match exactly what a
later real training run will look up.

Also caches the blank string and the run's unconditional_prompt -- the static
embeds every run computes for itself once, outside the per-dataset-item cache
(see MinimaxH3Model._static_prompt_cache_path). Trigger-word, diff output
preservation, and sample-preview prompts are NOT cached here; run
scripts/train_minimax_h3_text_encoder_skip.py which requires those features
to be off before it will skip loading the text encoder for a real run.

The real training run afterwards still needs cache_text_embeddings: true (at
the train level or per-dataset) for it to read this cache instead of
re-encoding prompts live.

Usage:
    python scripts/cache_minimax_h3_text_embeds.py path/to/config.yaml [more_configs.yaml ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.basic import flush
from toolkit.data_loader import get_dataloader_from_datasets
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


def cache_text_embeds_for_process(process):
    print_acc(f"=== Caching text embeddings for process '{process.name}' ===")
    if not str(getattr(process.model_config, "arch", "")).startswith("minimax_h3"):
        print_acc(
            f" - WARNING: arch '{process.model_config.arch}' is not a MiniMax-H3 "
            "arch; model_kwargs.text_encode_only will be ignored and the full "
            "transformer will load"
        )

    # short-circuits transformer loading -- see minimax_h3.py's _load_transformer
    process.model_config.model_kwargs["text_encode_only"] = True
    # where the static (non-per-item) prompt cache lives -- must match what
    # train_minimax_h3_text_encoder_skip.py sets for the real run
    static_cache_dir = os.path.join(process.save_root, "_static_te_cache")
    process.model_config.model_kwargs["static_text_embed_cache_dir"] = static_cache_dir

    ModelClass = get_model_class(process.model_config)
    sampler = ModelClass.get_train_scheduler() if hasattr(ModelClass, "get_train_scheduler") else None
    process.sd = ModelClass(
        device=process.accelerator.device,
        model_config=process.model_config,
        dtype=process.train_config.dtype,
        noise_scheduler=sampler,
    )
    process.sd.load_model()
    flush()

    # write the cache regardless of what the config's cache_text_embeddings
    # flag says -- caching is the entire point of this script
    for dataset_config in process.dataset_configs:
        dataset_config.cache_text_embeddings = True

    # AiToolkitDataset runs its cache_text_embeddings() pass as a side effect
    # of construction (see toolkit/data_loader.py's setup_epoch)
    get_dataloader_from_datasets(process.datasets, process.train_config.batch_size, process.sd)
    if process.datasets_reg:
        get_dataloader_from_datasets(process.datasets_reg, process.train_config.batch_size, process.sd)

    # the static prompts hook_before_train_loop in SDTrainer.py always
    # computes for itself, regardless of dataset caching: the run's
    # unconditional_prompt (for self.unconditional_embeds) and a literal
    # blank string (for self.cached_blank_embeds) -- usually the same prompt,
    # deduped automatically since they hash to the same cache path
    for static_prompt in {"", process.train_config.unconditional_prompt.strip()}:
        static_path = process.sd._static_prompt_cache_path(static_prompt)
        print_acc(f"static prompt path :{static_path}")
        if not os.path.exists(static_path):
            print_acc(f" - Caching static prompt {static_prompt!r} at {static_path}")
            pe = process.sd.encode_prompt(static_prompt)
            pe.save(static_path)

    print_acc(f"=== Done caching text embeddings for process '{process.name}' ===")


for config_file in args.config_file_list:
    job = get_job(config_file, args.name)
    if not hasattr(job, "process"):
        raise ValueError(
            f"{config_file}: expected a 'job: extension' training config "
            "(the same one you'd train with)"
        )
    for process in job.process:
        cache_text_embeds_for_process(process)
