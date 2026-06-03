"""
Build SDP-KIDNEY-FINAL COLLAB-NAMITHAA.ipynb from the original notebook,
adapted for Google Colab T4 GPU with 4-hour max runtime.
"""
import json, copy, re

with open(r"d:\SDP\SDP-KIDNEY-FINAL.IPYNB", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_nb = copy.deepcopy(nb)
new_nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.10.12"},
    "accelerator": "GPU",
    "gpuClass": "standard",
    "colab": {"provenance": [], "gpuType": "T4"}
}

cells = []

# Cell 0: Colab setup markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "# SDP Kidney Glomeruli Grading - COLLAB-NAMITHAA\n\n**Optimized for Google Colab T4 GPU (4-hour max runtime)**\n\nSelf-contained PyTorch workflow for 4-class ordinal kidney glomeruli grading.\n\nThis notebook includes manifest loading, slide-level leakage checks, split distribution reporting, medical preprocessing, augmentation, class imbalance handling, ordinal-aware training, staged fine-tuning, calibration, validation-only threshold/ensemble tuning, and final test reporting.\n\nPrimary model: ViT-B/16. Lighter complementary model: EfficientNet-B0.\n\n---\n**Setup:** Upload `kidney_dataset.json` and your image folders to Google Drive, then mount Drive below."
})

# Cell 1: Colab environment setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """# =========================
# 0. Colab Environment Setup
# =========================
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Install any missing packages (most are pre-installed on Colab)
!pip install -q seaborn scikit-learn pillow opencv-python-headless

# Verify GPU
import torch
assert torch.cuda.is_available(), "GPU not available! Go to Runtime > Change runtime type > T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.2f} GB")

import time
COLAB_START_TIME = time.time()
MAX_RUNTIME_SECONDS = 3.5 * 3600  # 3.5 hours safety margin for 4-hour limit

def check_time_budget(stage_name=""):
    elapsed = time.time() - COLAB_START_TIME
    remaining = MAX_RUNTIME_SECONDS - elapsed
    print(f"[TIME] {stage_name} | Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
    if remaining < 300:
        print("WARNING: Less than 5 minutes remaining! Consider saving checkpoints.")
    return remaining > 0
"""
})

# Cell 2: Imports - modify from original cell index 1
orig_cell1 = new_nb["cells"][1]["source"]

# Replace config section for T4
old_config = """# GTX 1650 4 GB safe defaults.
NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = True
GRAD_CLIP_NORM = 1.0"""

new_config = """# T4 GPU 16 GB Colab defaults.
NUM_WORKERS = 2
PIN_MEMORY = True
USE_AMP = True
GRAD_CLIP_NORM = 1.0"""

orig_cell1 = orig_cell1.replace(old_config, new_config)

# Replace manifest path for Drive
orig_cell1 = orig_cell1.replace(
    'MANIFEST_PATH = Path("kidney_dataset.json")',
    '# UPDATE THIS PATH to your Google Drive location\nMANIFEST_PATH = Path("/content/drive/MyDrive/SDP/kidney_dataset.json")'
)

# Replace output dir for Drive
orig_cell1 = orig_cell1.replace(
    'OUTPUT_DIR = Path("sdp_kidney_final_outputs")',
    'OUTPUT_DIR = Path("/content/drive/MyDrive/SDP/sdp_kidney_final_outputs")'
)

# Increase batch sizes for T4 (16GB VRAM)
orig_cell1 = orig_cell1.replace("BATCH_SIZE_VIT = 2", "BATCH_SIZE_VIT = 8")
orig_cell1 = orig_cell1.replace("GRAD_ACCUM_VIT = 8       # effective batch 16", "GRAD_ACCUM_VIT = 2       # effective batch 16")
orig_cell1 = orig_cell1.replace("BATCH_SIZE_EFFNET = 8", "BATCH_SIZE_EFFNET = 16")
orig_cell1 = orig_cell1.replace("GRAD_ACCUM_EFFNET = 4    # effective batch 32", "GRAD_ACCUM_EFFNET = 2    # effective batch 32")

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": orig_cell1
})

# Cell 3: Medical preprocessing (original cell index 2) - unchanged
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_nb["cells"][2]["source"]
})

# Cell 4: Manifest loading (original cell index 3) - add time check
orig_cell3 = new_nb["cells"][3]["source"]
orig_cell3 = orig_cell3.rstrip()
orig_cell3 += "\n\ncheck_time_budget('Data loading and splitting complete')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": orig_cell3
})

# Cell 5: Losses, metrics, calibration (original cell index 4) - unchanged
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_nb["cells"][4]["source"]
})

# Cell 6: Models and training (original cell index 5) - add time budget check in training loop
orig_cell5 = new_nb["cells"][5]["source"]

# Add time budget check in the epoch loop
orig_cell5 = orig_cell5.replace(
    '            if stale_epochs >= int(phase["patience"]):\n                print(f"Early stopping phase {phase[\'name\']} after {local_epoch} epochs without validation score improvement.")\n                break',
    '            if stale_epochs >= int(phase["patience"]):\n                print(f"Early stopping phase {phase[\'name\']} after {local_epoch} epochs without validation score improvement.")\n                break\n            if not check_time_budget(f"{cfg.name}/{phase[\'name\']}/epoch_{local_epoch}"):\n                print("TIME BUDGET EXHAUSTED - stopping training early to save results.")\n                break'
)

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": orig_cell5
})

# Cell 7: Experiment runner (original cell index 6) - unchanged
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_nb["cells"][6]["source"]
})

# Cell 8: Run experiments (original cell index 7) - add time check
orig_cell7 = new_nb["cells"][7]["source"]
orig_cell7 = orig_cell7.rstrip()
orig_cell7 += "\n\ncheck_time_budget('All experiments complete')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": orig_cell7
})

# Cell 9: Ensemble and final reporting (original cell index 8) - add final time report
orig_cell8 = new_nb["cells"][8]["source"]
orig_cell8 = orig_cell8.rstrip()
orig_cell8 += """

check_time_budget('Pipeline complete')
total_minutes = (time.time() - COLAB_START_TIME) / 60
print(f"\\nTotal runtime: {total_minutes:.1f} minutes ({total_minutes/60:.2f} hours)")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": orig_cell8
})

# Cell 10: Download results
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """# =========================
# 9. Download Results (Optional)
# =========================
# Results are saved to Google Drive automatically.
# You can also download specific files:
# from google.colab import files
# files.download(str(OUTPUT_DIR / "final_report.json"))

print("All outputs saved to:", OUTPUT_DIR.resolve())
print("\\nFiles:")
import os
for root, dirs, fnames in os.walk(OUTPUT_DIR):
    for fname in fnames:
        fpath = os.path.join(root, fname)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"  {os.path.relpath(fpath, OUTPUT_DIR):50s} {size_mb:.2f} MB")
"""
})

new_nb["cells"] = cells
out_path = r"d:\SDP\SDP-KIDNEY-FINAL COLLAB-NAMITHAA.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(new_nb, f, indent=1)
print(f"Created: {out_path}")
