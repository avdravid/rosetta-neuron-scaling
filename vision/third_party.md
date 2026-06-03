# Third-party model repositories

Several vision pipeline scripts call into external model repos that are NOT bundled with this release. Clone them next to the `vision/` directory and pass their paths via the relevant CLI flags. The default assumption in the example scripts is that each repo is cloned at `vision/<repo-name>/`.

## pMF — one-step generators

Used by [`match_pmf_vit_multigpu.py`](match_pmf_vit_multigpu.py) and [`visualize.py`](visualize.py).

```bash
git clone https://github.com/Lyy0725/pMF.git pMF
pip install -r pMF/requirements.txt
```

Pretrained checkpoints are pulled automatically from `huggingface.co/Lyy0725/pMF` (the `--pmf-hf-repo` flag).

## DINOv2 / DINOv3 — discriminative ViTs

Used by [`match_pmf_vit_multigpu.py`](match_pmf_vit_multigpu.py) and [`match_flux.py`](match_flux.py) (both DINOv3).

DINOv3 can be loaded either:
- directly from Hugging Face Transformers (default — no external repo needed): `--disc-arch facebook/dinov3-vitb16-pretrain-lvd1689m`, or
- via the official `dinov3` repo through `torch.hub`:

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3
# weights:
mkdir -p dinov3_checkpoints
wget -P dinov3_checkpoints https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

DINOv2 (legacy):

```bash
git clone https://github.com/facebookresearch/dinov2.git dinov2
pip install -r dinov2/requirements.txt
```

## OpenCLIP — discriminative ViTs

No external clone needed; pulled via the `open_clip_torch` package:

```bash
pip install open_clip_torch
```

Pass model names via `--disc-family openclip --disc-arch ViT-B-16 --disc-pretrained openai` (or any other supported weights tag).

## PixIO — vision tower

Used by [`match_flux.py`](match_flux.py) and [`match_pmf_vit_multigpu.py`](match_pmf_vit_multigpu.py).

```bash
git clone https://github.com/<pixio-upstream>/pixio.git pixio
pip install -r pixio/requirements.txt
```

(Substitute the actual PixIO upstream URL — fill in before release.)

## InternViT — vision tower

Used by [`match_flux.py`](match_flux.py).

InternViT can typically be loaded directly via Hugging Face Transformers (`AutoModel.from_pretrained("OpenGVLab/InternViT-...")`). No external clone needed.

## MAE — masked autoencoder

Used by [`match_pmf_vit_multigpu.py`](match_pmf_vit_multigpu.py) (via `--disc-family mae`).

```bash
git clone https://github.com/facebookresearch/mae.git mae
# weights — e.g. MAE-base ImageNet pretrain:
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

## Sana — efficient text-to-image transformer

Used by [`match_large_dit.py`](match_large_dit.py).

```bash
git clone https://github.com/NVlabs/Sana.git Sana
pip install -r Sana/requirements.txt
```

Sana weights are fetched from Hugging Face Hub (`Efficient-Large-Model/Sana_*`).

## FLUX.2-klein

Used by [`match_flux.py`](match_flux.py). Loaded via `diffusers`:

```bash
pip install --upgrade diffusers
```

Model weights are pulled from Hugging Face Hub (`black-forest-labs/FLUX.2-klein`) on first run — accept the model license on the Hugging Face page first.
