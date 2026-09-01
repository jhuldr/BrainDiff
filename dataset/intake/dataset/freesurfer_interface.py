import subprocess

from tqdm import tqdm
from dataset.dataset_utils import *

"""
export FREESURFER_HOME=/home/apps/freesurfer/8.1.0 
source $FREESURFER_HOME/SetUpFreeSurfer.sh
"""


FREESURFER_HOME = "/home/apps/freesurfer/8.1.0"
PYTHON_BIN = "/path/to/.conda/envs/SurferTest/bin"

env = os.environ.copy()
env["FREESURFER_HOME"] = FREESURFER_HOME
env["PATH"] = f"{PYTHON_BIN}:{FREESURFER_HOME}/bin:{env['PATH']}"
env["CUDA_VISIBLE_DEVICES"] = "0"
env["TF_USE_LEGACY_KERAS"] = "1"

def freesurfer_segment(input_dir, output_dir):

    print("#######INPUT DIR#######")
    print(input_dir)

    working_files = []
    files_to_process = []
    counter = 1

    # --- Phase 1: collect files, skip already-processed ones ---
    for p in input_dir.rglob("*.nii*"):

        if p.name.startswith("x0") or "LESION" in str(p.name).upper() or "synthseg" in p.name:
            continue

        expected_output = Path(output_dir) / modify_name(p.name)

        if expected_output.exists():
            print(f"Skipping {p} - already processed")
            working_files.append(p)
            continue

        counter += 1
        files_to_process.append(p)

    if not files_to_process:
        return working_files

    # --- Phase 2: stage into temp dir and run in batch, retrying on error ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Copy all candidate files into the temp dir, remembering original paths
        tmp_to_original = {}
        for p in files_to_process:
            tmp_path = tmp_dir / p.name
            shutil.copy2(p, tmp_path)
            tmp_to_original[tmp_path.name] = p

        pbar = tqdm(total=len(files_to_process), desc="FreeSurfer segmentation")

        while True:
            remaining = sorted(tmp_dir.glob("*.nii*"))
            if not remaining:
                break

            command = [
                f"{PYTHON_BIN}/python3",
                f"{FREESURFER_HOME}/python/scripts/mri_synthseg",
                "--i", str(tmp_dir),
                "--o", str(output_dir),
                "--robust",
                "--parc",
            ]

            try:
                subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # All remaining files succeeded
                for tmp_file in remaining:
                    working_files.append(tmp_to_original[tmp_file.name])
                pbar.update(len(remaining))
                break  # Done — exit the retry loop

            except subprocess.CalledProcessError:
                # Identify which outputs now exist to infer what succeeded vs. failed
                newly_succeeded = []
                failed_tmp = None

                for tmp_file in remaining:
                    expected_output = Path(output_dir) / modify_name(tmp_file.name)
                    if expected_output.exists():
                        newly_succeeded.append(tmp_file)
                    else:
                        failed_tmp = tmp_file
                        break  # First file without output is the culprit

                # Credit successful files and remove them from temp dir
                for tmp_file in newly_succeeded:
                    working_files.append(tmp_to_original[tmp_file.name])
                    tmp_file.unlink()
                pbar.update(len(newly_succeeded))

                # Remove the failed file from temp dir and report it
                if failed_tmp is not None:
                    print(f"An error occurred while executing FreeSurfer, omitting {tmp_to_original[failed_tmp.name]}")
                    failed_tmp.unlink()
                    pbar.update(1)
                else:
                    # Can't identify the culprit — bail to avoid an infinite loop
                    print("An error occurred but no failed file could be identified; aborting remaining files.")
                    pbar.update(len(remaining) - len(newly_succeeded))
                    break

        pbar.close()

    return working_files

"""
cmd = [
    f"{PYTHON_BIN}/python3",
    f"{FREESURFER_HOME}/python/scripts/mri_synthseg",
    "--i", "/path/to/code/BrainDiff/augment_lesions2/epoch_10_brain/mask_0_1.nii.gz",
    "--o", "/path/to/freeTest",
    "--robust",
    "--parc",
]

subprocess.run(cmd, check=True, env=env)
"""

if __name__ == "__main__":
    input_scan = "generated_images/y0_0.nii.gz"
    output_directory = "output"    
    gpu = 1                                
    threads = 4
    freesurfer_segment(input_scan, output_directory)