import os
from huggingface_hub import hf_hub_download
from huggingface_hub import login

# get huggingface files
if __name__ == "__main__":
    necessary_huggingface_files = ["model.ckpt", "model_config.json"]
    if all([os.path.exists(file) for file in necessary_huggingface_files]):
        print("The necessary model files have been found: ")
        for file in necessary_huggingface_files:
            print(f"\t{file}")
        print("Do you want to repopulate these files? (Y/n)")
        reply = input()
        if reply.lower() != 'y':
            print("Terminating without repopulating model files.")
        else:
            login()
            for file in necessary_huggingface_files:
                hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename=file, local_dir="./")
    else:
        login()
        for file in necessary_huggingface_files:
            hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename=file, local_dir="./")