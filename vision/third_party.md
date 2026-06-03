# Third-party model repositories

Several vision pipeline scripts call into external model repos that are NOT bundled with this release. Clone them next to the `vision/` directory and pass their paths via the relevant CLI flags. The default assumption in the example scripts is that each repo is cloned at `vision/<repo-name>/`.

## pMF — one-step generators

```bash
git clone https://github.com/Lyy0725/pMF.git pMF
pip install -r pMF/requirements.txt
```

Pretrained checkpoints are pulled automatically from `huggingface.co/Lyy0725/pMF` (the `--pmf-hf-repo` flag).

## OpenCLIP — discriminative ViTs

No external clone needed; pulled via the `open_clip_torch` package:

```bash
pip install open_clip_torch
```

Pass model names via `--disc-family openclip --disc-arch ViT-B-16 --disc-pretrained openai` (or any other supported weights tag).


## DINOv2 / DINOv3 — discriminative ViTs

DINOv2:

```bash
git clone https://github.com/facebookresearch/dinov2.git dinov2
pip install -r dinov2/requirements.txt
```

DINOv3 can be loaded either:
- directly from Hugging Face Transformers (default — no external repo needed): `--disc-arch facebook/dinov3-vitb16-pretrain-lvd1689m`, or
- via the official `dinov3` repo through `torch.hub`:

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3
# weights:
mkdir -p dinov3_checkpoints
wget -P dinov3_checkpoints https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## Pixio — discriminative ViTs

```bash
git clone https://github.com/facebookresearch/pixio pixio
pip install -r pixio/requirements.txt
```

## Sana — efficient text-to-image transformer

```bash
git clone https://github.com/NVlabs/Sana.git Sana
pip install -r Sana/requirements.txt
```

Sana weights are fetched from Hugging Face Hub (`Efficient-Large-Model/Sana_*`).

## FLUX.2-klein

Loaded via `diffusers`:

```bash
pip install --upgrade diffusers
```

Model weights are pulled from Hugging Face Hub (`black-forest-labs/FLUX.2-klein`) on first run — accept the model license on the Hugging Face page first.
