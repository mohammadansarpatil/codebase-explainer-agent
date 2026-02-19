import argparse
from pathlib import Path

from ingest.clone import clone_repo
from ingest.filter import list_source_files
from ingest.loader import load_documents
from ingest.repo_map import build_repo_map
from ingest.chunk import chunk_documents

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

    docs = load_documents(Path(result.local_path), files)
    print("\n--- Loaded Docs ---")
    print("Docs loaded:", len(docs))
    print("First doc rel_path:", docs[0].rel_path)
    print("First doc chars:", len(docs[0].text))

    repo_map = build_repo_map(Path(result.local_path), files)
    print("\n--- Repo Map ---")
    print("Top-level dirs:", repo_map.top_level_dirs)
    print("Important files:", repo_map.important_files[:10])
    print("Likely entrypoints:", repo_map.likely_entrypoints)
    print("Top extensions:", list(repo_map.ext_counts.items())[:8])

    chunks = chunk_documents(docs)
    print("\n--- Chunking (Python AST) ---")
    print("Total chunks:", len(chunks))
    if chunks:
        print("Sample chunk:", chunks[0].chunk_id)
        print("Sample chunk lines:", f"{chunks[0].start_line}-{chunks[0].end_line}")

    print("Sample chunks:")
    for c in chunks[:5]:
        print("-", c.chunk_id, "|", c.rel_path)

if __name__ == "__main__":
    main()
