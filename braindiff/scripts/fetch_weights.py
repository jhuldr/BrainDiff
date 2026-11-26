"""Download BrainDiff weights into the HuggingFace cache.

  --release   the BrainDiff checkpoint only; prints its local path (pass to eval/generate.py --ckpt)
  default     also NeuroVFM's gated repos (not redistributed): mlinslab/neurovfm-encoder and mlinslab/neurovfm-llm.
              Accept their terms and `huggingface-cli login` first.

Offline afterwards: export HF_HUB_OFFLINE=1. BRAINDIFF_LLM_ROOT points at an existing neurovfm-llm copy instead.
"""
import argparse

REPO = "ORG/braindiff"          # placeholder: set when the weights are published
FILE = "braindiff.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="only the BrainDiff checkpoint")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    print(f"braindiff      : {hf_hub_download(a.repo, FILE)}")
    if a.release:
        return
    import braindiff.models.encoder as E
    cfg, weights = E.encoder_files()
    print(f"encoder config : {cfg}\nencoder weights: {weights}")
    print(f"llm snapshot   : {E.llm_root()}\ndecoder        : {E.decoder_dir()}\nconnector      : {E.connector_path()}")


if __name__ == "__main__":
    main()
