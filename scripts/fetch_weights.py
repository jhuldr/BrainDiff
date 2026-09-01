"""Pre-download the NeuroVFM weights this project builds on.

Nothing here is redistributed with BrainDiff: the vision backbone, the perceiver connector and
the Qwen3-14B decoder are NeuroVFM's, fetched from the Hub into the standard HuggingFace cache.
`models/paths.py` resolves them on demand, so this script is only a convenience -- run it once
up front rather than discovering a 30 GB download mid-run.

    python scripts/fetch_weights.py            # everything
    python scripts/fetch_weights.py --encoder  # just the ViT (~274 MB)

Both repositories are GATED. Accept their terms on the Hub and log in first:

    huggingface-cli login

Offline afterwards: export HF_HUB_OFFLINE=1 and the Hub is never contacted again. To point at a
copy you already have, set BRAINDIFF_LLM_ROOT to its directory.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="store_true",
                    help="fetch only the vision encoder, skipping the 30 GB decoder")
    a = ap.parse_args()

    import braindiff.models.paths as P

    cfg, weights = P.encoder_files()
    print(f"encoder config : {cfg}")
    print(f"encoder weights: {weights}")
    if a.encoder:
        return

    print(f"llm snapshot   : {P.llm_root()}")
    print(f"decoder        : {P.decoder_dir()}")
    print(f"connector      : {P.connector_path()}")


if __name__ == "__main__":
    main()
