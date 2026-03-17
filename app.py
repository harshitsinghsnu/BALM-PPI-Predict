"""
BALM-LoRA Web Prediction Tool — app.py
Run:  python app.py
Open: http://localhost:5000

FIX: avoids the transformers packaging/__version__ crash by importing
     only the specific sub-modules we need, bypassing the version-check
     in transformers/__init__.py.
"""

import os
import sys
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/<path:p>", methods=["OPTIONS"])
def options_handler(p):
    return "", 204

# ── Globals ───────────────────────────────────────────────────────────────────
_model     = None
_pkd_lower = 2.0
_pkd_upper = 14.0


def _safe_import_transformers():
    """
    Import only the specific classes we need from transformers,
    bypassing the top-level __init__.py that crashes on broken `packaging`.
    This is the key fix for the 'Unable to compare versions for packaging>=20.0' error.
    """
    # Patch packaging.version.Version so transformers version-check doesn't crash
    try:
        import packaging.version as pv
        _orig_parse = pv.parse
        def _safe_parse(v):
            try:
                return _orig_parse(v)
            except Exception:
                return _orig_parse("0.0.0")
        pv.parse = _safe_parse

        import packaging.version
        _OrigVersion = packaging.version.Version
        class _SafeVersion(_OrigVersion):
            def __init__(self, v):
                try:
                    super().__init__(v)
                except Exception:
                    super().__init__("0.0.0")
        packaging.version.Version = _SafeVersion
    except Exception:
        pass

    # Now patch transformers.utils.versions so require_version never raises
    try:
        import transformers.utils.versions as tv
        tv.require_version      = lambda *a, **kw: None
        tv.require_version_core = lambda *a, **kw: None
    except Exception:
        pass

    # Safe to import now
    from transformers.models.esm.modeling_esm import EsmModel
    from transformers.models.esm.tokenization_esm import EsmTokenizer
    return EsmModel, EsmTokenizer


def _build_model(pkd_lower, pkd_upper):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType

    # ── safe transformers import ───────────────────────────────────────────
    EsmModel, EsmTokenizer = _safe_import_transformers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device : {device}")

    # ── exact same class/field names as training script ───────────────────
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

    # ── build ESM-2 + LoRA ────────────────────────────────────────────────
    print("  loading ESM-2 …")
    base_esm = EsmModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["key", "query", "value"])
    peft_esm = get_peft_model(base_esm, lora_cfg)
    tok = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    model = BALMForLoRAFinetuning(
        esm_model=peft_esm, esm_tokenizer=tok,
        projected_size=256, projected_dropout=0.1,
        pkd_bounds=(pkd_lower, pkd_upper))
    model.to(device)
    return model, tok, device


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded": _model is not None,
        "pkd_lower": _pkd_lower,
        "pkd_upper": _pkd_upper,
        "device": str(next(iter(_model.parameters())).device) if _model else None
    })


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    global _model, _pkd_lower, _pkd_upper

    data         = request.get_json(force=True)
    weights_path = data.get("weights_path", "").strip()
    pkd_lower    = float(data.get("pkd_lower", 2.0))
    pkd_upper    = float(data.get("pkd_upper", 14.0))

    if not weights_path:
        return jsonify({"error": "No weights path provided."}), 400
    if not os.path.exists(weights_path):
        return jsonify({"error": f"File not found: {weights_path}"}), 400

    try:
        import torch
        print(f"\n[load_model] building architecture …")
        model, tok, device = _build_model(pkd_lower, pkd_upper)

        print(f"[load_model] loading checkpoint …")
        raw_state  = torch.load(weights_path, map_location=device,
                                weights_only=False)
        ckpt_keys  = list(raw_state.keys())
        model_keys = list(model.state_dict().keys())
        print(f"  ckpt  {len(ckpt_keys)} keys  e.g. {ckpt_keys[:2]}")
        print(f"  model {len(model_keys)} keys  e.g. {model_keys[:2]}")

        res = model.load_state_dict(raw_state, strict=False)
        print(f"  missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}")

        if len(res.missing_keys) == len(model_keys):
            return jsonify({
                "error": (
                    "Key mismatch — no checkpoint keys matched the model.\n"
                    f"Checkpoint example: {ckpt_keys[:2]}\n"
                    f"Model example:      {model_keys[:2]}"
                )
            }), 500

        model.eval()
        _model     = model
        _pkd_lower = pkd_lower
        _pkd_upper = pkd_upper

        msg = (f"Loaded on {device} | "
               f"missing={len(res.missing_keys)} | "
               f"unexpected={len(res.unexpected_keys)}")
        print(f"[load_model] ✅ {msg}")
        return jsonify({"status": "ok", "message": msg,
                        "device": str(device),
                        "missing": len(res.missing_keys),
                        "unexpected": len(res.unexpected_keys)})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[load_model] ERROR:\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _model is None:
        return jsonify({"error": "Model not loaded."}), 400

    data   = request.get_json(force=True)
    seq_a  = "".join(c for c in data.get("sequence_a","").upper()
                     if c.isalpha())
    seq_b  = "".join(c for c in data.get("sequence_b","").upper()
                     if c.isalpha())
    run_ig = data.get("run_ig", True)

    if not seq_a: return jsonify({"error": "sequence_a is empty."}), 400
    if not seq_b: return jsonify({"error": "sequence_b is empty."}), 400

    try:
        import torch
        print(f"[predict] A={len(seq_a)}aa  B={len(seq_b)}aa  ig={run_ig}")

        with torch.no_grad():
            pkd, cos = _model(seq_a, seq_b)

        result = {
            "pkd": float(pkd.item()),
            "cosine": float(cos.item()),
            "ig_target": None,
            "ig_binder": None,
        }
        print(f"[predict] pKd={result['pkd']:.4f}  cos={result['cosine']:.4f}")

        if run_ig:
            try:
                ig_a, ig_b = _compute_ig(_model, seq_a, seq_b)
                result["ig_target"] = ig_a
                result["ig_binder"] = ig_b
                print(f"[predict] IG done {len(ig_a)}+{len(ig_b)} residues")
            except Exception as e:
                print(f"[predict] IG failed: {traceback.format_exc()}")
                result["ig_warning"] = f"IG failed: {e}"

        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict] ERROR:\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


# ── Integrated Gradients ──────────────────────────────────────────────────────

def _norm01(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-9)).tolist()


def _compute_ig(model, seq_a, seq_b, steps=30):
    import torch
    device  = next(model.parameters()).device
    tok     = model.esm_tokenizer
    cls_tok = tok.cls_token
    esm     = model.esm_model   # PeftModel

    def tokenise(seq):
        proc = seq.replace('|', f"{cls_tok}{cls_tok}")
        enc  = tok(proc, return_tensors="pt",
                   padding=False, truncation=True, max_length=1024).to(device)
        return enc

    word_embed = esm.base_model.model.embeddings.word_embeddings

    enc_a = tokenise(seq_a); mask_a = enc_a["attention_mask"]
    enc_b = tokenise(seq_b); mask_b = enc_b["attention_mask"]
    emb_a = word_embed(enc_a["input_ids"]).detach()
    emb_b = word_embed(enc_b["input_ids"]).detach()

    def encode(embs, mask):
        ext = esm.base_model.model.get_extended_attention_mask(
            mask, embs.shape[:2])
        h = esm.base_model.model.encoder(
            embs, attention_mask=ext).last_hidden_state
        m = mask.unsqueeze(-1).expand(h.size()).float()
        return (torch.sum(h * m, 1) /
                torch.clamp(m.sum(1), min=1e-9)).float()

    def forward_embs(e_a, e_b):
        out = model.projection_head(encode(e_a, mask_a), encode(e_b, mask_b))
        return out["cosine_similarity"]

    def riemann(tgt_emb, baseline, fixed_emb, tgt_is_b):
        alphas = torch.linspace(0, 1, steps, device=device)
        grads  = []
        for a in alphas:
            x = (baseline + a * (tgt_emb - baseline)).detach().requires_grad_(True)
            s = forward_embs(fixed_emb, x) if tgt_is_b else forward_embs(x, fixed_emb)
            s.sum().backward()
            grads.append(x.grad.detach().clone())
        avg_g = torch.stack(grads).mean(0)
        return (avg_g * (tgt_emb - baseline)).abs().sum(-1).squeeze(0).cpu().numpy()

    attr_a = riemann(emb_a, torch.zeros_like(emb_a), emb_b, tgt_is_b=False)
    attr_b = riemann(emb_b, torch.zeros_like(emb_b), emb_a, tgt_is_b=True)
    return _norm01(attr_a[1:-1]), _norm01(attr_b[1:-1])


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n{'='*55}")
    print(f"  BALM·predict  →  http://localhost:{port}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", debug=False, port=port, threaded=False)