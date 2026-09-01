# BRATS APADTER
def extract_modality_brats_men(f):
    """Extract modality from a BraTS filename."""
    name = f.stem.split("-")[-1].upper().split(".")[0]
    if "T1C" in name:
        return "T1ce"
    elif "T1N" in name:
        return "T1w"
    elif "T2W" in name:
        return "T2w"
    elif "T2F" in name:
        return "FLAIR"
    else:
        return "T1w"
    
def find_seg_brats_men(folder):
    """Find the segmentation file in a BraTS folder."""
    for f in folder.iterdir():
        if f.is_file() and "seg" in f.name.lower():
            return f
    return None


# YALE ADAPTER
def extract_modality_yale(f):
    """Extract modality from a Yale filename."""
    name = f.stem.split("_")[-1].upper().split(".")[0]
    if "T1C" in name:
        return "T1ce"
    elif "T1W" in name:
        return "T1w"
    elif "T2W" in name:
        return "T2w"
    elif "FLAIR" in name:
        return "FLAIR"
    else:
        return None
