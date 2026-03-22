"""
BALM·predict — Streamlit App
================================
Users upload their .pth model weights directly in the browser.
No local server needed. Deploy to Streamlit Community Cloud for free.

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations
import io
import os
import sys
import tempfile
import traceback

import numpy as np
import streamlit as st
import requests
# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="BALM-PPI·predict",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme overrides */
:root { --acc: #39ff6e; --acc2: #00c4ff; --hot: #ff4d6d; }

[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #1f2733;
}
[data-testid="stSidebar"] * { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: #161b24; border: 1px solid #1f2733; border-radius: 8px;
    padding: 16px 20px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
.metric-lbl { font-size: 0.65rem; color: #5a6678; letter-spacing: 2px;
               text-transform: uppercase; margin-top: 4px; font-family: monospace; }
.green { color: #39ff6e; }
.blue  { color: #00c4ff; }

.status-ok  { color: #39ff6e; font-family: monospace; font-size: 0.8rem; }
.status-err { color: #ff4d6d; font-family: monospace; font-size: 0.8rem; }
.status-wrn { color: #ffd166; font-family: monospace; font-size: 0.8rem; }

.section-header {
    font-family: monospace; font-size: 1rem; letter-spacing: 3px;
    text-transform: uppercase; color: #39ff6e;
    border-bottom: 1px solid #1f2733; padding-bottom: 6px; margin-bottom: 12px;
    -webkit-text-stroke: 0.5px black;
    
    border-bottom: 1px solid #1f2733; 
    padding-bottom: 6px; 
    margin-bottom: 12px;
}
.res-cell {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 22px; border-radius: 3px;
    font-family: monospace; font-size: 13px; font-weight: bold;
    margin: 1px; cursor: default;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  PACKAGING FIX — same patch as Flask version
# ═══════════════════════════════════════════════════
def _patch_packaging():
    try:
        import packaging.version as pv
        _orig = pv.parse
        def _safe(v):
            try: return _orig(v)
            except: return _orig("0.0.0")
        pv.parse = _safe
    except Exception:
        pass
    try:
        import transformers.utils.versions as tv
        tv.require_version      = lambda *a, **kw: None
        tv.require_version_core = lambda *a, **kw: None
    except Exception:
        pass

_patch_packaging()

DEFAULT_MODEL_URL = "https://huggingface.co/Harshit494/BALM-PPI/resolve/main/best_model_fold_1.pth"

@st.cache_data(show_spinner="Downloading default weights from Hugging Face (this may take a minute)...")
def download_default_weights(url: str) -> bytes:
    """Downloads weights from a URL and returns the bytes."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    # Using a buffer to show progress could be done, 
    # but for simplicity, we return the content.
    return response.content

# ═══════════════════════════════════════════════════
#  MODEL ARCHITECTURE (identical to training)
# ═══════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ESM-2 base model…")
def _load_esm_base():
    """Cache the heavy ESM-2 base model so it's only downloaded once."""
    _patch_packaging()
    from transformers.models.esm.modeling_esm import EsmModel
    from transformers.models.esm.tokenization_esm import EsmTokenizer
    base = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    tok  = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    return base, tok


def build_and_load_model(weights_bytes: bytes, pkd_lower: float, pkd_upper: float):
    """
    Build BALMForLoRAFinetuning, load uploaded weights, return (model, device).
    Weights are passed as raw bytes from st.file_uploader.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Projection head ───────────────────────────────────────────────────────
    class BALMProjectionHead(nn.Module):
        def __init__(self, embedding_size, projected_size=256, dropout=0.1):
            super().__init__()
            self.protein_projection  = nn.Linear(embedding_size, projected_size)
            self.proteina_projection = nn.Linear(embedding_size, projected_size)
            self.dropout             = nn.Dropout(dropout)
            self.loss_fn             = nn.MSELoss()

        def forward(self, prot_emb, prot_a_emb, labels=None):
            p  = F.normalize(self.protein_projection(self.dropout(prot_emb)),  p=2, dim=1)
            pa = F.normalize(self.proteina_projection(self.dropout(prot_a_emb)), p=2, dim=1)
            cos = torch.clamp(F.cosine_similarity(p, pa), -0.9999, 0.9999)
            out = {"cosine_similarity": cos}
            if labels is not None:
                out["loss"] = self.loss_fn(cos, labels)
            return out

    # ── Full model ────────────────────────────────────────────────────────────
    class BALMForLoRAFinetuning(nn.Module):
        def __init__(self, esm_model, esm_tokenizer, projected_size,
                     projected_dropout, pkd_bounds):
            super().__init__()
            self.esm_model       = esm_model
            self.esm_tokenizer   = esm_tokenizer
            self.embedding_size  = self.esm_model.config.hidden_size
            self.projection_head = BALMProjectionHead(
                self.embedding_size, projected_size, projected_dropout)
            self.pkd_lower, self.pkd_upper = pkd_bounds
            self.pkd_range  = self.pkd_upper - self.pkd_lower
            self.cls_token  = self.esm_tokenizer.cls_token

        def _get_esm_embeddings(self, sequences):
            processed = [s.replace('|', f"{self.cls_token}{self.cls_token}")
                         for s in sequences]
            inputs = self.esm_tokenizer(
                processed, return_tensors="pt",
                padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.esm_model.device) for k, v in inputs.items()}
            outputs = self.esm_model(**inputs)
            h    = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(h.size()).float()
            return (torch.sum(h * mask, 1) /
                    torch.clamp(mask.sum(1), min=1e-9)).float()

        def forward(self, seq_a, seq_b):
            ea  = self._get_esm_embeddings([seq_a])
            eb  = self._get_esm_embeddings([seq_b])
            out = self.projection_head(ea, eb)
            cos = out["cosine_similarity"]
            pkd = ((cos + 1) / 2) * self.pkd_range + self.pkd_lower
            return pkd, cos

    # ── Build ESM-2 + LoRA ────────────────────────────────────────────────────
    base_esm, tok = _load_esm_base()
    base_esm = copy.deepcopy(base_esm)  # fresh copy per upload

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["key", "query", "value"])
    peft_esm = get_peft_model(base_esm, lora_cfg)

    model = BALMForLoRAFinetuning(
        esm_model=peft_esm, esm_tokenizer=tok,
        projected_size=256, projected_dropout=0.1,
        pkd_bounds=(pkd_lower, pkd_upper))

    # ── Load weights: write to temp file so torch.load handles 2GB+ safely ────
    # Using BytesIO on a 2GB file doubles RAM usage; temp file avoids this.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    try:
        tmp.write(weights_bytes)
        tmp.flush()
        tmp.close()
        raw_state = torch.load(tmp.name, map_location=device, weights_only=False)
    finally:
        try: os.unlink(tmp.name)
        except: pass

    ckpt_keys  = list(raw_state.keys())
    model_keys = list(model.state_dict().keys())

    result = model.load_state_dict(raw_state, strict=False)
    n_missing    = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)

    if n_missing == len(model_keys):
        raise ValueError(
            f"Key mismatch — no checkpoint keys matched.\n"
            f"Checkpoint: {ckpt_keys[:3]}\nModel: {model_keys[:3]}"
        )

    model.to(device)
    model.eval()
    return model, device, n_missing, n_unexpected


# ═══════════════════════════════════════════════════
#  INTEGRATED GRADIENTS
# ═══════════════════════════════════════════════════
def compute_ig(model, seq_a: str, seq_b: str, steps: int = 30):
    import torch

    device  = next(model.parameters()).device
    tok     = model.esm_tokenizer
    cls_tok = tok.cls_token
    esm     = model.esm_model

    def tokenise(seq):
        proc = seq.replace('|', f"{cls_tok}{cls_tok}")
        return tok(proc, return_tensors="pt",
                   padding=False, truncation=True, max_length=1024).to(device)

    word_embed = esm.base_model.model.embeddings.word_embeddings
    enc_a = tokenise(seq_a); mask_a = enc_a["attention_mask"]
    enc_b = tokenise(seq_b); mask_b = enc_b["attention_mask"]
    emb_a = word_embed(enc_a["input_ids"]).detach()
    emb_b = word_embed(enc_b["input_ids"]).detach()

    def encode(embs, mask):
        ext = esm.base_model.model.get_extended_attention_mask(mask, embs.shape[:2])
        h   = esm.base_model.model.encoder(embs, attention_mask=ext).last_hidden_state
        m   = mask.unsqueeze(-1).expand(h.size()).float()
        return (torch.sum(h * m, 1) / torch.clamp(m.sum(1), min=1e-9)).float()

    def fwd(e_a, e_b):
        return model.projection_head(encode(e_a, mask_a), encode(e_b, mask_b))["cosine_similarity"]

    def riemann(tgt, baseline, fixed, tgt_is_b):
        alphas = torch.linspace(0, 1, steps, device=device)
        grads  = []
        for a in alphas:
            x = (baseline + a * (tgt - baseline)).detach().requires_grad_(True)
            (fwd(fixed, x) if tgt_is_b else fwd(x, fixed)).sum().backward()
            grads.append(x.grad.detach().clone())
        avg = torch.stack(grads).mean(0)
        return (avg * (tgt - baseline)).abs().sum(-1).squeeze(0).cpu().numpy()

    attr_a = riemann(emb_a, torch.zeros_like(emb_a), emb_b, False)
    attr_b = riemann(emb_b, torch.zeros_like(emb_b), emb_a, True)

    def norm(a):
        lo, hi = a.min(), a.max()
        return ((a - lo) / (hi - lo + 1e-9)).tolist()

    return norm(attr_a[1:-1]), norm(attr_b[1:-1])


# ═══════════════════════════════════════════════════
#  NGL VIEWER  (rendered as HTML component)
# ═══════════════════════════════════════════════════
def ngl_viewer_html(uniprot_id: str | None, ig_scores: list | None,
                    pdb_content: str | None = None, height: int = 420) -> str:
    """
    Returns an HTML string that renders a real NGL 3D structure.
    If ig_scores is provided, residues are coloured by IG attribution.
    """
    ig_json = "null"
    if ig_scores:
        ig_json = "[" + ",".join(f"{v:.4f}" for v in ig_scores) + "]"

    if pdb_content:
        # Escape for embedding in JS string
        escaped = pdb_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        load_cmd = f"""
            var blob = new Blob([`{escaped}`], {{type:'text/plain'}});
            var url  = URL.createObjectURL(blob);
            stage.loadFile(url, {{ext:'pdb', name:'uploaded'}}).then(addRepr);
        """
    elif uniprot_id:
        af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id.upper()}-F1-model_v4.pdb"
        load_cmd = f"""
            stage.loadFile('{af_url}', {{ext:'pdb', name:'{uniprot_id}'}}).then(addRepr);
        """
    else:
        load_cmd = ""

    return f"""
<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<style>
  body {{margin:0;padding:0;background:#060809;overflow:hidden}}
  #vp  {{width:100%;height:{height}px}}
  #info{{position:absolute;top:8px;left:10px;font-family:monospace;font-size:11px;
         color:#5a6678;pointer-events:none}}
  #controls{{position:absolute;bottom:10px;right:10px;display:flex;flex-direction:column;gap:5px}}
  .cbtn{{padding:4px 10px;font-family:monospace;font-size:10px;background:rgba(15,18,24,.9);
         border:1px solid #1f2733;color:#5a6678;border-radius:4px;cursor:pointer}}
  .cbtn:hover{{border-color:#00c4ff;color:#00c4ff}}
  #legend{{position:absolute;bottom:10px;left:10px;font-family:monospace;font-size:10px;
           color:#5a6678;background:rgba(15,18,24,.9);border:1px solid #1f2733;
           border-radius:4px;padding:6px 10px;line-height:1.8}}
</style></head><body>
<div id="vp"></div>
<div id="info">🧬 Drag to rotate · Scroll to zoom · Right-drag to pan</div>
<div id="controls">
  <button class="cbtn" onclick="setRep('cartoon')">CARTOON</button>
  <button class="cbtn" onclick="setRep('surface')">SURFACE</button>
  <button class="cbtn" onclick="setRep('ball+stick')">BALL+STICK</button>
  <button class="cbtn" onclick="stage.autoView()">RESET</button>
</div>
<div id="legend"></div>
<script>
var stage = new NGL.Stage('vp', {{backgroundColor:'#060809', quality:'medium', tooltip:true}});
var comp  = null;
var curRep = 'cartoon';
var igScores = {ig_json};

function igColor(score) {{
  var s = Math.max(0, Math.min(1, score || 0));
  var r = Math.round(10  + s * 47);
  var g = Math.round(18  + s * 237);
  var b = Math.round(40  + s * 70);
  return (r << 16) | (g << 8) | b;
}}

function addRepr(c) {{
  comp = c;
  if (igScores) {{
    var schemeId = NGL.ColormakerRegistry.addScheme(function() {{
      this.atomColor = function(atom) {{
        var i = atom.resno - 1;
        return (i >= 0 && i < igScores.length) ? igColor(igScores[i]) : 0x1a2a3a;
      }};
    }});
    comp.addRepresentation(curRep, {{color: schemeId}});
    var leg = document.getElementById('legend');
    leg.innerHTML = '<span style="color:#39ff6e">■</span> High IG &nbsp; <span style="color:#0a1220">■</span> Low IG<br><span style="color:#5a6678;font-size:9px">residues coloured by attribution score</span>';
  }} else {{
    comp.addRepresentation(curRep, {{colorScheme:'bfactor', colorScale:'RdYlGn'}});
    var leg = document.getElementById('legend');
    leg.innerHTML = '<span style="color:#5a6678;font-size:9px">coloured by AlphaFold pLDDT confidence</span>';
  }}
  stage.autoView();
}}

function setRep(rep) {{
  curRep = rep;
  if (!comp) return;
  comp.removeAllRepresentations();
  addRepr(comp);
}}

window.addEventListener('resize', function() {{ stage.handleResize(); }});

{load_cmd}
</script></body></html>
"""


# ═══════════════════════════════════════════════════
#  PLOTLY HEATMAP
# ═══════════════════════════════════════════════════
def make_heatmap(seq: str, attr: list, title: str):
    import plotly.graph_objects as go

    W = 50
    rows, hover, ylabels = [], [], []
    for r in range(0, len(seq), W):
        chunk = seq[r:r+W]
        row   = [attr[r+c] if r+c < len(attr) else None for c in range(W)]
        htrow = [f"{seq[r+c]}{r+c+1} | IG: {attr[r+c]:.3f}" if r+c < len(seq) else ""
                 for c in range(W)]
        rows.append(row); hover.append(htrow)
        ylabels.append(f"{r+1}–{min(r+W, len(seq))}")

    colorscale = [
        [0.00, "#060809"], [0.15, "#0a1a2a"],
        [0.35, "#0d3a3a"], [0.55, "#116633"],
        [0.75, "#22cc55"], [1.00, "#39ff6e"],
    ]

    fig = go.Figure(go.Heatmap(
        z=rows, text=hover, hoverinfo="text",
        colorscale=colorscale, zmin=0, zmax=1,
        xgap=1, ygap=1,
        colorbar=dict(
            thickness=12, len=0.9,
            tickfont=dict(family="Space Mono", size=9, color="#5a6678"),
            title=dict(text="IG Score", font=dict(family="Space Mono", size=9, color="#5a6678")),
        )
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=11, color="#5a6678")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0c10",
        margin=dict(t=32, b=32, l=60, r=60),
        height=220,
        xaxis=dict(
            title=dict(text="Position in window",
                       font=dict(family="Space Mono", size=9, color="#5a6678")),
            tickfont=dict(family="Space Mono", size=8, color="#5a6678"),
            gridcolor="#1f2733",
        ),
        yaxis=dict(
            tickvals=list(range(len(ylabels))),
            ticktext=ylabels,
            tickfont=dict(family="Space Mono", size=8, color="#5a6678"),
            gridcolor="#1f2733",
        ),
    )
    return fig


# ═══════════════════════════════════════════════════
#  RESIDUE STRIP (HTML)
# ═══════════════════════════════════════════════════
def residue_strip_html(seq: str, attr: list) -> str:
    cells = []
    for i, aa in enumerate(seq):
        s = attr[i] if i < len(attr) else 0
        r = int(10 + s * 47)
        g = int(18 + s * 237)
        b = int(40 + s * 70)
        text_col = "#040804" if s > 0.5 else "#dde4ed"
        cells.append(
            f'<span class="res-cell" style="background:rgb({r},{g},{b});color:{text_col}" '
            f'title="{aa}{i+1} IG:{s:.3f}">{aa}</span>'
        )
    return f"""
<div style="font-family:monospace;font-size:10px;color:#5a6678;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
  Residue Attribution (hover for score)
</div>
<div style="line-height:1.8;word-break:break-all">
  {"".join(cells)}
</div>
"""


# ═══════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">
      <span style="font-size:2.5rem;font-weight:800;color:#39ff6e;letter-spacing:-1px; -webkit-text-stroke: 2px black; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;">BALM-PPI<span style="color:Black">-predict</span></span>
      <span style="font-family:monospace;font-size:1.5rem;color:#5a6678;letter-spacing:3px;text-transform:uppercase">    ESM-2 · LoRA · Integrated Gradients</span>
    </div>
    <hr style="border-color:#1f2733;margin-bottom:1rem">
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    for k, v in [("model", None), ("device", None), ("result", None),
                 ("ig_a", None), ("ig_b", None), ("pdb_a", None), ("pdb_b", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ═══════════════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════════════
    with st.sidebar:
        st.markdown('<div class="section-header">01 · Model Weights</div>', unsafe_allow_html=True)
        st.caption("Upload your trained .pth checkpoint (up to 4 GB supported)")

        model_source = st.radio(
            "Select Model Source",
            ["Default (Hugging Face)", "Upload Custom (.pth)"],
            help="Choose whether to use the pre-trained weights from Hugging Face or upload your own."
        )
        weights_bytes = None
        
        if model_source == "Upload Custom (.pth)":
            weights_file = st.file_uploader(
                "Checkpoint (.pth / .pt / .bin)",
                type=["pth", "pt", "bin"],
                help="Upload your trained model file."
            )
            if weights_file:
                weights_bytes = weights_file.read()
        else:
            st.info(f"Using weights from: Harshit494/BALM-PPI")

        col1, col2 = st.columns(2)
        pkd_lo = col1.number_input("pKd min", value=1.0, step=0.5, key="pkd_input_min_sidebar")
        pkd_hi = col2.number_input("pKd max", value=16.0, step=0.5, key="pkd_input_max_sidebar")

        # Updated Load Button Logic
        if st.button("⚡ Load Model", use_container_width=True, type="primary"):
            try:
                # Determine which bytes to use
                if model_source == "Default (Hugging Face)":
                    weights_to_load = download_default_weights(DEFAULT_MODEL_URL)
                else:
                    if weights_bytes is None:
                        st.error("Please upload a file first.")
                        return
                    weights_to_load = weights_bytes

                with st.spinner("Building ESM-2 + LoRA architecture…"):
                    model, device, n_miss, n_unexp = build_and_load_model(
                        weights_to_load, pkd_lo, pkd_hi
                    )
                    st.session_state.model  = model
                    st.session_state.device = str(device)
                    st.success(f"✅ Loaded on **{device}**")
                    if n_miss > 0: st.caption(f"Missing keys: {n_miss}")
            except Exception as e:
                st.error(f"❌ Loading failed: {e}")
                st.code(traceback.format_exc(), language="python")


        if st.session_state.model:
            st.markdown(f'<div class="status-ok">● MODEL READY · {st.session_state.device}</div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Structure source ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">02 · Structure Source</div>', unsafe_allow_html=True)
        st.caption("Optional — for 3D visualisation")

        uniprot_a = st.text_input("UniProt ID — Target (A)", placeholder="e.g. P0DTD1")
        uniprot_b = st.text_input("UniProt ID — Binder (B)",  placeholder="e.g. P0DTD3")

        pdb_file_a = st.file_uploader("Local PDB — Target (A)", type=["pdb", "cif"], key="pdb_up_a")
        pdb_file_b = st.file_uploader("Local PDB — Binder (B)",  type=["pdb", "cif"], key="pdb_up_b")

        if pdb_file_a: st.session_state.pdb_a = pdb_file_a.read().decode("utf-8", errors="ignore")
        if pdb_file_b: st.session_state.pdb_b = pdb_file_b.read().decode("utf-8", errors="ignore")

        st.divider()

        # ── Quick examples ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">03 · Examples</div>', unsafe_allow_html=True)
        if st.button("CR6261 × Influenza HA", use_container_width=True):
            st.session_state["_eg"] = "cr6261"
        if st.button("CR9114 × Influenza HA", use_container_width=True):
            st.session_state["_eg"] = "cr9114"
        if st.button("Clear", use_container_width=True):
            st.session_state["_eg"] = "blank"

    # ═══════════════════════════════════════════════
    #  MAIN CONTENT
    # ═══════════════════════════════════════════════
    EXAMPLES = {
        "cr6261": {
            "a": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQGLEWIGYINPYNDVTHYNQKFKDRVTITVDKSTSTAYMELSSLRSEDTAVYYCARGYDILTGYYRYGMDYWGQGTLVTVSS",
            "b": "MKTIIALSYIFCLVFA",
        },
        "cr9114": {
            "a": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYPDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS",
            "b": "MKTIIALSYIFCLVFAQKIPGQTKVNKTVTGENLTEYFNVKAVLNQVQLIIQSANSVTARIDGSQSVSVTGAKYGESIFKELSG",
        },
        "blank": {"a": "", "b": ""},
    }

    eg_key   = st.session_state.get("_eg", "")
    eg_vals  = EXAMPLES.get(eg_key, {"a": "", "b": ""})

    # ── Input sequences ───────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown('<div class="section-header">Target Protein — Seq A</div>', unsafe_allow_html=True)
            seq_a_raw = st.text_area("", value=eg_vals["a"], height=100,
                                    placeholder="Paste FASTA or raw amino acid sequence…",
                                    key="seq_a_input", label_visibility="collapsed")
    with col_b:
        with st.container(border=True):         
            st.markdown('<div class="section-header">Binder Protein — Seq B</div>', unsafe_allow_html=True)
            seq_b_raw = st.text_area("", value=eg_vals["b"], height=100,
                                    placeholder="Paste FASTA or raw amino acid sequence…",
                                    key="seq_b_input", label_visibility="collapsed")

    def clean(s):
        import re
        return re.sub(r'[^A-Za-z]', '', re.sub(r'>.*', '', s, flags=re.MULTILINE)).upper()

    seq_a = clean(seq_a_raw)
    seq_b = clean(seq_b_raw)

    run_ig = st.checkbox("Run Integrated Gradients (30–120 s on CPU, <5 s on GPU)", value=True)

    run_btn = st.button(
        "🔬 Run Prediction",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.model is None),
    )
    if st.session_state.model is None:
        st.caption("⬅ Load model weights in the sidebar first")

    # ── RUN ───────────────────────────────────────────────────────────────────
    if run_btn:
        if not seq_a or not seq_b:
            st.warning("Please provide both sequences.")
        else:
            import torch
            model = st.session_state.model

            with st.spinner("Running prediction…"):
                try:
                    with torch.no_grad():
                        pkd, cos = model(seq_a, seq_b)
                    st.session_state.result = {
                        "pkd": float(pkd.item()), "cosine": float(cos.item())
                    }
                    st.session_state.ig_a = None
                    st.session_state.ig_b = None
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.code(traceback.format_exc())

            if run_ig and st.session_state.result:
                with st.spinner("Computing Integrated Gradients…"):
                    try:
                        ig_a, ig_b = compute_ig(model, seq_a, seq_b)
                        st.session_state.ig_a = ig_a
                        st.session_state.ig_b = ig_b
                        st.success("✅ Integrated Gradients computed")
                    except Exception as e:
                        st.warning(f"IG failed: {e}")

    # ═══════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════
    if st.session_state.result:
        res   = st.session_state.result
        pkd   = res["pkd"]
        cos   = res["cosine"]
        pct   = max(0, min(100, ((pkd - pkd_lo) / (pkd_hi - pkd_lo)) * 100))
        strength = ("Weak" if pct < 30 else "Moderate" if pct < 55
                    else "Strong" if pct < 75 else "Very Strong")

        st.divider()
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.markdown(f"""
        <div class="metric-card">
          <div class="metric-val green">{pkd:.3f}</div>
          <div class="metric-lbl">Predicted pKd</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card">
          <div class="metric-val blue">{cos:.4f}</div>
          <div class="metric-lbl">Cosine Similarity</div>
        </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left">
              <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:10px;color:#5a6678;margin-bottom:6px">
                <span>Weak (pKd {pkd_lo})</span><span style="color:#39ff6e">{strength}</span><span>Strong (pKd {pkd_hi})</span>
              </div>
              <div style="height:8px;background:#1f2733;border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{pct}%;border-radius:4px;background:linear-gradient(90deg,#00c4ff,#39ff6e,#ffd166,#ff4d6d);transition:width 1s"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs: 3D structure | IG strip | Heatmap ───────────────────────────
        tab_3d, tab_strip, tab_heatmap = st.tabs(["🧬 3D Structure", "📊 IG Residue Strip", "🗺 IG Heatmap"])

        # ── 3D Structure ──────────────────────────────────────────────────────
        with tab_3d:
            view_a, view_b = st.columns(2)

            with view_a:
                st.markdown("**Target (Seq A)**")
                uid_a = uniprot_a.strip() or None
                pdb_a = st.session_state.pdb_a
                if uid_a or pdb_a:
                    html_a = ngl_viewer_html(
                        uniprot_id=uid_a if not pdb_a else None,
                        ig_scores=st.session_state.ig_a,
                        pdb_content=pdb_a,
                        height=400
                    )
                    st.components.v1.html(html_a, height=405, scrolling=False)
                    if st.session_state.ig_a:
                        st.caption("🎨 Coloured by IG attribution score (dark=low, green=high)")
                    else:
                        st.caption("Coloured by AlphaFold pLDDT confidence")
                else:
                    st.info("Enter a UniProt ID or upload a PDB file in the sidebar for Chain A")

            with view_b:
                st.markdown("**Binder (Seq B)**")
                uid_b = uniprot_b.strip() or None
                pdb_b = st.session_state.pdb_b
                if uid_b or pdb_b:
                    html_b = ngl_viewer_html(
                        uniprot_id=uid_b if not pdb_b else None,
                        ig_scores=st.session_state.ig_b,
                        pdb_content=pdb_b,
                        height=400
                    )
                    st.components.v1.html(html_b, height=405, scrolling=False)
                    if st.session_state.ig_b:
                        st.caption("🎨 Coloured by IG attribution score (dark=low, green=high)")
                    else:
                        st.caption("Coloured by AlphaFold pLDDT confidence")
                else:
                    st.info("Enter a UniProt ID or upload a PDB file in the sidebar for Chain B")

            if not (uid_a or pdb_a or uid_b or pdb_b):
                st.warning("No structure loaded. Enter UniProt IDs or upload PDB files in the sidebar.")

        # ── IG Residue Strip ──────────────────────────────────────────────────
        with tab_strip:
            if st.session_state.ig_a or st.session_state.ig_b:
                col_strip_a, col_strip_b = st.columns(2)

                with col_strip_a:
                    st.markdown("**Target Protein (Seq A)**")
                    if st.session_state.ig_a:
                        st.markdown(
                            residue_strip_html(seq_a, st.session_state.ig_a),
                            unsafe_allow_html=True
                        )
                        # Top-10 bar chart
                        ig_a = st.session_state.ig_a
                        indexed = sorted(
                            [{"aa": seq_a[i], "idx": i+1, "score": ig_a[i]}
                             for i in range(min(len(seq_a), len(ig_a)))],
                            key=lambda x: -x["score"]
                        )[:10]
                        import plotly.graph_objects as go
                        fig_top = go.Figure(go.Bar(
                            x=[x["score"] for x in indexed],
                            y=[f"{x['aa']}{x['idx']}" for x in indexed],
                            orientation="h",
                            marker_color="#39ff6e",
                            marker_line_width=0,
                        ))
                        fig_top.update_layout(
                            title=dict(text="Top 10 Contributing Residues",
                                       font=dict(family="Space Mono", size=11, color="#5a6678")),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#fefeff",
                            height=260,
                            margin=dict(t=30, b=20, l=55, r=20),
                            xaxis=dict(
                                       title=dict(text="IG Score", font=dict(family="Space Mono", size=9, color="#5a6678")),
                                       tickfont=dict(family="Space Mono", size=8, color="#5a6678"),
                                       gridcolor="#1f2733"),
                            yaxis=dict(tickfont=dict(family="Space Mono", size=9, color="#39ff6e"),
                                       autorange="reversed"),
                        )
                        st.plotly_chart(fig_top, use_container_width=True)
                    else:
                        st.info("IG not available for Chain A")

                with col_strip_b:
                    st.markdown("**Binder Protein (Seq B)**")
                    if st.session_state.ig_b:
                        st.markdown(
                            residue_strip_html(seq_b, st.session_state.ig_b),
                            unsafe_allow_html=True
                        )
                        ig_b = st.session_state.ig_b
                        indexed_b = sorted(
                            [{"aa": seq_b[i], "idx": i+1, "score": ig_b[i]}
                             for i in range(min(len(seq_b), len(ig_b)))],
                            key=lambda x: -x["score"]
                        )[:10]
                        fig_top_b = go.Figure(go.Bar(
                            x=[x["score"] for x in indexed_b],
                            y=[f"{x['aa']}{x['idx']}" for x in indexed_b],
                            orientation="h",
                            marker_color="#00c4ff",
                            marker_line_width=0,
                        ))
                        fig_top_b.update_layout(
                            title=dict(text="Top 10 Contributing Residues",
                                       font=dict(family="Space Mono", size=11, color="#5a6678")),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#f8f8f8",
                            height=260,
                            margin=dict(t=30, b=20, l=55, r=20),
                            xaxis=dict(
                                       title=dict(text="IG Score", font=dict(family="Space Mono", size=9, color="#5a6678")),
                                       tickfont=dict(family="Space Mono", size=8, color="#5a6678"),
                                       gridcolor="#1f2733"),
                            yaxis=dict(tickfont=dict(family="Space Mono", size=9, color="#00c4ff"),
                                       autorange="reversed"),
                        )
                        st.plotly_chart(fig_top_b, use_container_width=True)
                    else:
                        st.info("IG not available for Chain B")
            else:
                st.info("Run prediction with Integrated Gradients enabled to see residue attribution.")

        # ── IG Heatmap ────────────────────────────────────────────────────────
        with tab_heatmap:
            if st.session_state.ig_a or st.session_state.ig_b:
                hm_a, hm_b = st.columns(2)
                with hm_a:
                    if st.session_state.ig_a:
                        st.plotly_chart(
                            make_heatmap(seq_a, st.session_state.ig_a, "Target (A) · IG Heatmap"),
                            use_container_width=True
                        )
                with hm_b:
                    if st.session_state.ig_b:
                        st.plotly_chart(
                            make_heatmap(seq_b, st.session_state.ig_b, "Binder (B) · IG Heatmap"),
                            use_container_width=True
                        )
                st.caption("Each cell = one residue. Colour = normalised IG attribution score. "
                           "Rows are windows of 50 residues.")
            else:
                st.info("Run prediction with Integrated Gradients to generate heatmaps.")

        # ── Download results ──────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)

        # CSV of IG scores
        if st.session_state.ig_a or st.session_state.ig_b:
            import csv, io as sio
            buf = sio.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["chain", "position", "residue", "ig_score"])
            for i, (aa, sc) in enumerate(zip(seq_a, st.session_state.ig_a or [])):
                writer.writerow(["A", i+1, aa, f"{sc:.6f}"])
            for i, (aa, sc) in enumerate(zip(seq_b, st.session_state.ig_b or [])):
                writer.writerow(["B", i+1, aa, f"{sc:.6f}"])
            dl1.download_button(
                "📥 IG Scores CSV", buf.getvalue().encode(),
                "ig_scores.csv", "text/csv", use_container_width=True
            )

        # JSON summary
        import json
        summary = {
            "pkd": pkd, "cosine": cos,
            "seq_a_length": len(seq_a), "seq_b_length": len(seq_b),
            "top10_target": sorted(
                [{"residue": f"{seq_a[i]}{i+1}", "ig": st.session_state.ig_a[i]}
                 for i in range(min(len(seq_a), len(st.session_state.ig_a or [])))],
                key=lambda x: -x["ig"]
            )[:10] if st.session_state.ig_a else [],
            "top10_binder": sorted(
                [{"residue": f"{seq_b[i]}{i+1}", "ig": st.session_state.ig_b[i]}
                 for i in range(min(len(seq_b), len(st.session_state.ig_b or [])))],
                key=lambda x: -x["ig"]
            )[:10] if st.session_state.ig_b else [],
        }
        dl2.download_button(
            "📥 Summary JSON", json.dumps(summary, indent=2).encode(),
            "balm_result.json", "application/json", use_container_width=True
        )


if __name__ == "__main__":
    main()
