import os

REPO_ID = "chemahc94/Cephalometric-Landmark-CVM"

def ensure_model_exists(filename, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=save_dir)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    return local_path
