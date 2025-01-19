from huggingface_hub import login, snapshot_download
import os
#model_name="Cosmos-Tokenizer-CI16x16" #you can download the other ones this way as well, make sure to clone the repo tho
model_name="Cosmos-Tokenizer-DI16x16"

hf_repo = "nvidia/" + model_name
local_dir = "pretrained_ckpts/" + model_name
os.makedirs(local_dir, exist_ok=True)
print(f"downloading {model_name}...")
snapshot_download(repo_id=hf_repo, local_dir=local_dir)