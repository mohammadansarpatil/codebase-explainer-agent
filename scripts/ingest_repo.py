import argparse
from ingest.clone import clone_repo


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


if __name__ == "__main__":
    main()
