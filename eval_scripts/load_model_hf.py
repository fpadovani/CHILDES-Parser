from huggingface_hub import HfApi

api = HfApi()

repo_id = "fpadovani/cds_parser_roberta_stanza"

# create repo if it doesn't exist
api.create_repo(repo_id, exist_ok=True)

# upload entire folder
api.upload_folder(
    folder_path="/Users/frapadovani/Desktop/CHILDES-Parser/saved_models/depparse/roberta",
    repo_id=repo_id,
)