"""Keep only the current text-embedding cache per image, backup the rest.

Every time a dataset's captions (or caching-relevant config) change, ai-toolkit
computes a new content hash for the text embedding and writes a new
"<image_stem>_<hash>.safetensors" file into that dataset's "_t_e_cache"
folder (see FileItemDTO._build_text_embedding_path in
toolkit/dataloader_mixins.py) -- it never deletes the old one. Over many
caption edits / retraining runs this leaves several stale, orphaned cache
files sitting next to the current one for the same image.

There's no way to recompute the "correct" hash from outside a real training
run (it depends on the model's text_embedding_space_version, encode_control
flags, etc., not just the caption text). Instead this script groups cache
files per image by filename stem and uses on-disk mtime clustering: for each
image, the most-recent group of cache files (the ones written together by
the last caching pass -- typically the main embed plus its blank/dropout
variants) is treated as current and kept in "_t_e_cache"; every older file
for that image is moved into a sibling "_t_e_backup" folder instead of being
deleted, so nothing is lost if it turns out still needed.

Usage:
    python scripts/clean_text_embedding_cache.py path/to/dataset_or_parent [more...] [--gap-seconds 300] [--dry-run]

Each given path is searched recursively for "_t_e_cache" folders, so you can
point it at a single dataset folder or at a parent folder containing many.
"""
import argparse
import os
import re
import shutil
import sys

CACHE_DIRNAME = '_t_e_cache'
BACKUP_DIRNAME = '_t_e_backup'
FILENAME_RE = re.compile(r'^(?P<stem>.+)_(?P<hash>[A-Za-z0-9_-]{22})\.safetensors$')


def find_cache_dirs(root):
    cache_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if CACHE_DIRNAME in dirnames:
            cache_dirs.append(os.path.join(dirpath, CACHE_DIRNAME))
    return cache_dirs


def group_by_stem(cache_dir):
    groups = {}
    skipped = []
    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            continue
        match = FILENAME_RE.match(name)
        if not match:
            skipped.append(name)
            continue
        groups.setdefault(match.group('stem'), []).append(path)
    return groups, skipped


def split_current_vs_outdated(paths, gap_seconds):
    if len(paths) <= 1:
        return paths, []
    entries = sorted(((os.path.getmtime(p), p) for p in paths), key=lambda e: e[0])
    boundary = 0
    for i in range(len(entries) - 1, 0, -1):
        if entries[i][0] - entries[i - 1][0] > gap_seconds:
            boundary = i
            break
    current = [p for _, p in entries[boundary:]]
    outdated = [p for _, p in entries[:boundary]]
    return current, outdated


def process_cache_dir(cache_dir, gap_seconds, dry_run):
    groups, skipped = group_by_stem(cache_dir)
    for name in skipped:
        print(f"  [skip] unrecognized filename, leaving alone: {name}")

    backup_dir = os.path.join(os.path.dirname(cache_dir), BACKUP_DIRNAME)
    moved = 0
    moved_bytes = 0
    kept = 0

    for stem, paths in groups.items():
        current, outdated = split_current_vs_outdated(paths, gap_seconds)
        kept += len(current)
        for path in outdated:
            size = os.path.getsize(path)
            dest = os.path.join(backup_dir, os.path.basename(path))
            if os.path.exists(dest):
                print(f"  [skip] backup already has {os.path.basename(path)}, leaving in cache")
                kept += 1
                continue
            if dry_run:
                print(f"  [dry-run] would move {os.path.relpath(path, cache_dir)} -> {BACKUP_DIRNAME}/")
            else:
                os.makedirs(backup_dir, exist_ok=True)
                shutil.move(path, dest)
            moved += 1
            moved_bytes += size

    return kept, moved, moved_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs='+', help="Dataset folder(s), or a parent folder containing many datasets")
    parser.add_argument("--gap-seconds", type=float, default=300.0,
                         help="Time gap (seconds) between two cache files for the same image before they're "
                              "considered separate caching runs (default: 300)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be moved")
    args = parser.parse_args()

    cache_dirs = []
    for root in args.paths:
        if not os.path.isdir(root):
            print(f"Not a directory, skipping: {root}")
            continue
        cache_dirs.extend(find_cache_dirs(root))

    if not cache_dirs:
        print(f"No '{CACHE_DIRNAME}' folders found under: {', '.join(args.paths)}")
        return

    total_kept = total_moved = total_bytes = 0
    for cache_dir in cache_dirs:
        dataset_dir = os.path.dirname(cache_dir)
        print(f"{dataset_dir}")
        kept, moved, moved_bytes = process_cache_dir(cache_dir, args.gap_seconds, args.dry_run)
        print(f"  kept {kept} current, moved {moved} outdated ({moved_bytes / 1024 / 1024:.1f} MB)")
        total_kept += kept
        total_moved += moved
        total_bytes += moved_bytes

    verb = "would move" if args.dry_run else "moved"
    print(f"\nTotal: kept {total_kept} current cache files, {verb} {total_moved} outdated "
          f"({total_bytes / 1024 / 1024:.1f} MB) across {len(cache_dirs)} dataset(s)")


if __name__ == "__main__":
    sys.exit(main())
