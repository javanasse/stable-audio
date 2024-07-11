import os
import shutil
import glob
import zipfile
import json
import gdown

def repopulate_dataset(dataset_dir: str):

    os.makedirs(dataset_dir)

    # glitch data
    gdrive_ids = ["1h-Kqd3hOFqFrKTjJMzcocnmD2VtocNx1", "1b2lBExR47glLcltFyUEwgvT3170QCP-7"]
    data_zips = list()
    for n, id in enumerate(gdrive_ids):
        outname = f"data-hoard-{n}.zip"
        gdown.download(id=id, output=outname)
        data_zips.append(outname)

    # unzip
    for data_zip in data_zips:
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(data_zip)

    # make dataset config
    cfg = dict()
    cfg["dataset_type"] = "audio_dir"
    cfg["random_crop"] = False
    cfg["datasets"] = list()
    for n, dir in enumerate(glob.glob(os.path.join(dataset_dir, '*'))):
        cfg["datasets"].append(
            {
                "id": f"data{n}",
                "path": dir,
                "custom_metadata_module": "dataset.py"
            }
        )
    with open('dataset.json', 'w') as f:
        json.dump(cfg, f, indent=4)

if __name__ == "__main__":
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        print(f"The directory \"{dataset_dir}\" already exists. Would you like to repopulate it? (Y/n)")
        reply = input()
        if not reply.lower() == 'y':
            print(f'Terminated without repolulating \"{dataset_dir}\"')
        else:
            shutil.rmtree(dataset_dir)
            repopulate_dataset(dataset_dir)