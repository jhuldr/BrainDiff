import os
import random
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
from itertools import permutations
import subprocess

#from .USB_interface import cond_gen
from .lesionModification import shrink_to_fraction
from dataset.ants_interface import align_folder_target


class PairGen:
    """
    Generate synthetic MRI-lesion pairs from a single input MRI and lesion mask.
    Simulates progressive lesion reduction over time.
    """

    def genPairs(
        self,
        save_location: str,
        mri_path: str,
        lesion_path: str,
        numSynth: int,
        reduction_per_step_range: tuple[float, float] = (0.1, 0.2),
        pair_sample_ratio: float = 0.5
    ) -> list[tuple[str, str, str, str, float]]:
        """
        Generate synthetic MRI-lesion pairs with progressively shrinking lesions,
        then create all permutations of brain image pairs.

        Parameters
        ----------
        save_location : str
            Root directory where all generated data will be saved.
        mri_path : str
            Path to the input MRI scan (.nii.gz).
        lesion_path : str
            Path to the corresponding lesion mask (.nii.gz).
        numSynth : int
            Number of synthetic images to generate.
        reduction_per_step_range : tuple[float, float], optional
            Range for random reduction factor per step (default: 0.1 to 0.2).
        pair_sample_ratio : float, optional
            Fraction of permutation pairs to return (default: 0.5).

        Returns
        -------
        list[tuple[str, str, str, str, float]]
            List of tuples containing:
            - brain1: Path to first MRI
            - lesion1: Path to first lesion mask
            - brain2: Path to second MRI
            - lesion2: Path to second lesion mask
            - delta: Change in lesion size (lesion2 - lesion1) as decimal
        """
        save_location = Path(save_location).resolve()
        save_location.mkdir(parents=True, exist_ok=True)

        mri_dir = save_location / "mri"
        lesion_dir = save_location / "lesions"
        mri_dir.mkdir(parents=True, exist_ok=True)
        lesion_dir.mkdir(parents=True, exist_ok=True)

        mri_path = Path(mri_path).resolve()
        lesion_path = Path(lesion_path).resolve()

        if not mri_path.exists():
            raise FileNotFoundError(f"MRI file not found: {mri_path}")
        if not lesion_path.exists():
            raise FileNotFoundError(f"Lesion file not found: {lesion_path}")
        if numSynth <= 0:
            raise ValueError("numSynth must be a positive integer")

        original_mri_name = mri_path.name.replace(".nii.gz", "").replace(".nii", "")

        expected_mri_paths = [
            mri_dir / f"{original_mri_name}-modification{idx}.nii.gz"
            for idx in range(numSynth)
        ]
        expected_lesion_paths = [
            lesion_dir / f"{original_mri_name}-modification{idx}_lesion.nii.gz"
            for idx in range(numSynth)
        ]

        if all(p.exists() for p in expected_mri_paths) and all(p.exists() for p in expected_lesion_paths):
            original_volume = np.sum(nib.load(lesion_path).get_fdata() > 0)
            generated_images = []
            for idx in range(numSynth):
                lesion_p = str(expected_lesion_paths[idx])
                if original_volume > 0:
                    remaining_fraction = np.sum(nib.load(lesion_p).get_fdata() > 0) / original_volume
                else:
                    remaining_fraction = 0.0
                generated_images.append((str(expected_mri_paths[idx]), lesion_p, remaining_fraction))
            generated_images.sort(key=lambda x: x[2], reverse=True)
            return self.create_permutation_pairs(generated_images, pair_sample_ratio)

        temp_dir = tempfile.mkdtemp(prefix="pairgen_temp_")
        temp_lesion_dir = Path(temp_dir) / "temp_lesions"
        temp_synth_dir = Path(temp_dir) / "temp_synth"
        temp_lesion_dir.mkdir(parents=True, exist_ok=True)
        temp_synth_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_lesion_paths, cumulative_reductions = self.generate_progressive_lesions(
                lesion_path, temp_lesion_dir, numSynth, reduction_per_step_range
            )

            txt_path = self.create_lesion_txt(temp_lesion_paths, temp_dir)
            project_root = Path(__file__).parent.parent.parent


            subprocess.run(
                [
                    "conda", "run", "-n", "CLIPTRAIN",
                    "--cwd", str(project_root),
                    "python", "-m", "dataset.LesionModification.USB_interface",
                    "cond",
                    "--text_path", str(txt_path),
                    "--save_location", str(temp_synth_dir),
                ],
                env={**os.environ, "PYTHONPATH": str(project_root)},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            
            #synth_output_dir = cond_gen(str(txt_path), str(temp_synth_dir))
            synth_output_dir = Path(temp_synth_dir).resolve()

            self.filter_usb_output(synth_output_dir)

            generated_images = self.finalize_outputs(
                mri_path,
                synth_output_dir,
                temp_lesion_paths,
                cumulative_reductions,
                original_mri_name,
                mri_dir,
                lesion_dir
            )

            pairs = self.create_permutation_pairs(generated_images, pair_sample_ratio)

            return pairs

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_progressive_lesions(
        self,
        lesion_path: Path,
        output_dir: Path,
        num_lesions: int,
        reduction_per_step_range: tuple[float, float]
    ) -> tuple[list[str], list[float]]:
        """
        Generate progressively smaller versions of the input lesion mask.
        """
        img = nib.load(lesion_path)
        original_data = np.squeeze(img.get_fdata()).astype(np.uint8)
        affine = img.affine
        header = img.header

        original_volume = np.sum(original_data > 0)

        lesion_paths = []
        cumulative_reductions = []
        current_data = original_data.copy()

        #print(f"\nGenerating {num_lesions} progressively smaller lesions...")
        #print(f"Original lesion volume: {original_volume} voxels")

        for i in range(num_lesions):
            step_keep_fraction = 1.0 - random.uniform(
                reduction_per_step_range[0],
                reduction_per_step_range[1]
            )

            current_data = shrink_to_fraction(current_data.copy(), step_keep_fraction)
            current_volume = np.sum(current_data > 0)

            if original_volume > 0:
                remaining_fraction = current_volume / original_volume
                cumulative_reduction = 1.0 - remaining_fraction
            else:
                cumulative_reduction = 1.0

            cumulative_reductions.append(cumulative_reduction)

            fname = f"mask_{i}_lesion.nii.gz"
            out_path = output_dir / fname
            out_img = nib.Nifti1Image(current_data.astype(np.uint8), affine, header)
            nib.save(out_img, out_path)

            lesion_paths.append(str(out_path))
            #print(f"  Saved: {fname} | Volume: {current_volume} voxels | "
            #      f"Cumulative reduction: {cumulative_reduction:.1%} of original")

        return lesion_paths, cumulative_reductions

    def create_lesion_txt(self, lesion_paths: list[str], temp_dir: str) -> Path:
        """
        Create a text file listing all lesion paths for USB interface.
        """
        txt_path = Path(temp_dir) / "lesion_paths.txt"

        with open(txt_path, "w") as f:
            for p in lesion_paths:
                f.write(os.path.abspath(p) + "\n")

        return txt_path

    def filter_usb_output(self, folder_path: Path) -> None:
        """
        Filter generated files, keeping only y0_* files and removing others.
        """
        for fname in os.listdir(folder_path):
            fpath = folder_path / fname
            if not (fname.startswith("y0_") and fname.endswith(".gz")):
                os.remove(fpath)

    def finalize_outputs(
        self,
        target: Path,
        synth_dir: Path,
        temp_lesion_paths: list[str],
        cumulative_reductions: list[float],
        original_mri_name: str,
        mri_dir: Path,
        lesion_dir: Path
    ) -> list[tuple[str, str, float]]:
        """
        Move and rename files from temp to final location with proper naming.

        Returns
        -------
        list[tuple[str, str, float]]
            List of (mri_path, lesion_path, remaining_lesion_fraction) tuples.
        """
        lesion_map = {
            idx: (lpath, reduction)
            for idx, (lpath, reduction) in enumerate(zip(temp_lesion_paths, cumulative_reductions))
        }

        move_operations = []
        results = []

        for fname in os.listdir(synth_dir):
            if not fname.startswith("y0_"):
                continue

            fpath = synth_dir / fname

            try:
                name_part = fname[3:]
                parts = name_part.split("_")
                idx = int(parts[1])

                if idx not in lesion_map:
                    print(f"  Warning: No lesion match for index {idx}")
                    continue

                temp_lesion_path, reduction = lesion_map[idx]

                remaining_fraction = 1.0 - reduction

                # New naming convention: originalname-modification#.nii.gz
                final_mri_name = f"{original_mri_name}-modification{idx}.nii.gz"
                final_lesion_name = f"{original_mri_name}-modification{idx}_lesion.nii.gz"

                final_mri_path = mri_dir / final_mri_name
                final_lesion_path = lesion_dir / final_lesion_name

                move_operations.append((str(fpath), str(final_mri_path)))
                move_operations.append((temp_lesion_path, str(final_lesion_path)))

                results.append((str(final_mri_path), str(final_lesion_path), remaining_fraction))

                #print(f"  Matched idx {idx}: {fname} -> {final_mri_name}")

            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not process {fname}: {e}")
                continue

        for src, dst in move_operations:
            shutil.move(src, dst)
            #print(f"    Moved: {Path(src).name} -> {Path(dst).name}")

        # align the final (moved) files in place so the aligned outputs land on
        # the same paths that `results` references
        final_pairs = [(mri, lesion) for mri, lesion, _ in results]
        align_folder_target(target, final_pairs, mri_dir, lesion_dir)

        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def create_permutation_pairs(
        self,
        generated_images: list[tuple[str, str, float]],
        sample_ratio: float = 0.5
    ) -> list[tuple[str, str, str, str, float]]:
        """
        Create all permutations of image pairs and return a random sample.
        """
        all_pairs = []

        for img1, img2 in permutations(generated_images, 2):
            brain1, lesion1, fraction1 = img1
            brain2, lesion2, fraction2 = img2
            delta = fraction2 - fraction1
            all_pairs.append((brain1, lesion1, brain2, lesion2, delta))

        num_samples = max(1, int(len(all_pairs) * sample_ratio))
        sampled_pairs = random.sample(all_pairs, num_samples)

        #print(f"\nCreated {len(all_pairs)} permutation pairs, sampled {len(sampled_pairs)} ({sample_ratio:.0%})")

        return sampled_pairs


if __name__ == "__main__":
    mri_path = "/home/data/CLIPDATA/TRIAL_IMAGES/ALIGNED_MRI/sub-001_brain.nii.gz"
    lesion_path = "/home/data/CLIPDATA/TRIAL_IMAGES/ALIGNED_LESIONS/sub-001_lesion.nii.gz"
    save_location = "pair_gen_test"

    generator = PairGen()
    pairs = generator.genPairs(
        save_location=save_location,
        mri_path=mri_path,
        lesion_path=lesion_path,
        numSynth=5,
        reduction_per_step_range=(0.1, 0.2),
        pair_sample_ratio=0.5
    )