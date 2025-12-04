import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
api = HfApi(token=os.getenv("HF_token"))

try:
    api.upload_file(
        path_or_fileobj="../model/sarimax_model.pkl",
        path_in_repo="sarimax_model.pkl",
        repo_id="mongare70/medOptix-admission-model",
        repo_type="model",
    )
except Exception as e:
    print(f"Upload failed: {str(e)}")
