# BALM·predict — Binding Affinity Prediction Web Tool

A local web application for running ESM-2 + LoRA binding affinity predictions
with per-residue attribution via Integrated Gradients.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **PyTorch:** Install the correct CUDA build from https://pytorch.org/get-started/locally/  
> Example (CUDA 12.1):  
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 2. Run the server
```bash
python app.py
# or specify a port:
python app.py 8080
```

### 3. Open the tool
Navigate to **http://localhost:5000** in your browser.

---

## Usage

1. **Load Model Weights**  
   Paste the full path to any `best_model_fold_N.pth` checkpoint saved by your
   training script. Set `pKd Lower/Upper` to match the bounds used during training
   (default 2.0 – 14.0).  
   Click **Load Model Weights**.

2. **Enter Sequences**  
   - *Seq A* = Target protein (receptor)  
   - *Seq B* = Binder/ligand protein  
   FASTA headers are automatically stripped; only amino acid characters are kept.

3. **Run Prediction**  
   Click **Run Prediction**.  
   With *Integrated Gradients* enabled, the tool also computes per-residue
   importance scores (adds ~30–120 s on CPU; <5 s on GPU).

4. **Interpret Results**  
   - **pKd** — predicted binding affinity  
   - **Cosine Similarity** — raw model output (scaled to pKd)  
   - **Affinity Bar** — visual strength indicator  
   - **3D Bead Model** — coarse Cα trace coloured by IG attribution  
     (drag to rotate, scroll to zoom)  
   - **Residue Attribution Panel** — per-residue heatmap + Top-10 bar chart  
     Switch between Target and Binder tabs.

---

## Architecture Notes

The tool reconstructs the exact model used during training:

```
ESM-2 (facebook/esm2_t33_650M_UR50D)
  └─ LoRA adapters (r=8, α=16, target: key/query/value)
  └─ BALMProjectionHead
       ├─ Linear(1280 → 256) × 2  (one per chain)
       ├─ L2 Normalisation
       └─ Cosine Similarity → pKd rescaling
```

**Integrated Gradients** are computed by attributing the cosine similarity
output w.r.t. the word-embedding layer of ESM-2, using 50 interpolation steps
from a zero-vector baseline.

---

## Notes

- Sequences are truncated to 1024 tokens (ESM-2 limit).
- The 3D viewer shows a coarse helical bead model — not a true 3D structure.
  For real structure visualisation, combine with AlphaFold2 PDB outputs via py3Dmol.
- To load multiple checkpoints (ensemble), run `python app.py` on different ports
  and aggregate predictions client-side.
