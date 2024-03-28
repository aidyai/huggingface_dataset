## creating a repository
from huggingface_hub import create_repo
from huggingface_hub import HfApi


#create_repo("aidystark/shoe41k", repo_type="dataset")

api = HfApi()
api.upload_file(
    path_or_fileobj="/content/footnet.tar.gz",
    path_in_repo="README.md",
    repo_id="aidystark/shoe41k",
    repo_type="dataset",
)
