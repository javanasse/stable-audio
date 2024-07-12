import os
import glob
import shutil
from pathlib import Path
import subprocess as sp

def unpack(ziptar: str, dest: str):
    dext = os.path.splitext(ziptar)[1]
    if dext == ".zip":
        sp.call(["unzip", "-j", str(ziptar), "-d", f"{dest}/"])
    elif dext in [".tar", ".gz", ".tgz", ".tar.gz"]:
        sp.call(["tar", "-xvaf", str(ziptar), "-C", f"{dest}/", "--strip-components=1"])
    else:
        raise Exception(
            "Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz', 'tgz' types of compression file."
        )

    # Removing junk files
    for item in ["__MACOSX", ".DS_Store"]:
        item_path = Path(f"{dest}/{item}")
        if item_path.exists():
            shutil.rmtree(str(item_path), ignore_errors=True)
            
def print_files(startpath: str):
    for root, dirnames, filenames in os.walk(startpath):
        for filename in filenames:
            print(os.path.join(root, filename))