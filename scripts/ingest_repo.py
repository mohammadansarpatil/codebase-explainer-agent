import argparse
from pathlib import Path

from ingest.clone import clone_repo
from ingest.filter import list_source_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="GitHub repository URL")
    parser.add_argument("--update", action="store_true", help="Pull latest changes if repo already exists")
    args = parser.parse_args()

    result = clone_repo(args.url, update_if_exists=args.update)
    print("Repo URL:", result.repo_url)
    print("Local path:", result.local_path)
    print("Was cloned:", result.was_cloned)
    print("Did update:", result.did_update)

    files, stats = list_source_files(Path(result.local_path))
    print("\n--- Filter Stats ---")
    print("Total files seen:", stats.total_files_seen)
    print("Kept files:", stats.kept_files)
    print("Skipped dirs:", stats.skipped_by_dir)
    print("Skipped ext:", stats.skipped_by_ext)
    print("Skipped by size:", stats.skipped_by_size)
    print("Skipped non-text:", stats.skipped_non_text)

    print("\nSample kept files:")
    for p in files[:20]:
        print("-", p)


if __name__ == "__main__":
    main()
