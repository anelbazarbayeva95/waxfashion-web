---
license: mit
title: Wax Fashion StyleGAN
sdk: gradio
sdk_version: 6.14.0
emoji: ⚡
colorFrom: gray
colorTo: indigo
---


# WaxFashionStyleGAN

A generative AI space for creating **African wax-inspired textile patterns**, built by the [Pace AI Lab](https://huggingface.co/paceailab) at Pace University.

## About the model

The model is a fine-tuned **StyleGAN2-ADA** trained on a synthetic dataset of ~5,000 African Wax Print textile images. Given a random seed, it generates a unique 512×512 wax-inspired fabric pattern. It was developed as part of a research project exploring how generative AI can increase the representation of African textile design in digital creative tools.

- **Architecture:** StyleGAN2-ADA (PyTorch)
- **Training data:** [AfricanWaxPatterns_5KDataset](https://huggingface.co/datasets/paceailab/AfricanWaxPatterns_5KDataset)
- **Model weights:** [paceailab/Waxfashion_StyleGAN](https://huggingface.co/paceailab/Waxfashion_StyleGAN)
- **Resolution:** 512 × 512 px
- **GitHub:** [researchpace/waxfashion](https://github.com/researchpace/waxfashion)

## How to use

| Control | What it does |
|---|---|
| **Seed** | An integer that selects a specific pattern. Same seed + same settings = same image every time. |
| **Truncation Ψ** | Controls variety vs. quality. Low (0.1–0.5) = clean, typical patterns. High (0.9–1.5) = more unusual and experimental. |
| **Samples** | Number of patterns to generate at once (1–4). Keep it at 1 to save shared GPU time. |

Click **Generate** to create patterns. After generating, a download button appears: **↓ Download tile** saves a single PNG, and with more samples **↓ Download N tiles** saves all of them as a ZIP.

The **Examples** section shows patterns rendered by the model at start-up (seeds 21, 88, 305 and 1990), so they don't use your GPU quota.

## Note

This is an **unconditional** generative model — it does not respond to text prompts or style keywords. The only inputs that affect the output are the seed and truncation value. For a prompt-guided version, see [MSamyak/SDXL_AfricanWax](https://huggingface.co/MSamyak/SDXL_AfricanWax).

## Citation

```
@misc{waxfashionstylegan,
  title   = {WaxFashionStyleGAN},
  author  = {Pace AI Lab},
  year    = {2025},
  url     = {https://huggingface.co/paceailab/WaxFashionStyleGAN}
}
```