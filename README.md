<div align="center">

<h1>🌍 Boundless-World-Model </h1>

<p align="center">
    <a href="https://huggingface.co/spaces/W··orldArena/WorldArena"><img src="https://img.shields.io/badge/🏆_Leaderboard-WorldArena-yellow?style=flat"></a>  
</p>

</div>

> **BWM** is a physically consistent, action-conditioned video world model built upon Wan2.2-TI2V-5B, serving as a low-cost yet high-fidelity simulator for robotic manipulation.

## 🗞️ News

- **[2026-05]** 🚀 **Inference code released!** Generate action-conditioned robot manipulation videos with BWM. See [🛠️ Usage](#️-usage).
- **[2026-05]** 🎉 **Model definition released!** The BWM architecture and core model components are now available.

## Table of Contents
- [✅ TODO](#-todo)
- [🏗️ Framework](#️-framework)
- [🛠️ Usage](#️-usage)
- [🏋️ Training](#️-training)
- [📜 Citing](#-Citing)

---

## ✅ TODO

<input type="checkbox" checked disabled> Release inference code<br>
<input type="checkbox" checked disabled> Release model definition<br>
<input type="checkbox" disabled> Release model weights<br>
<input type="checkbox" disabled> Release training code<br>
<input type="checkbox" disabled> Release technical report

---

## 🏗️ Framework

Coming soon !

---

## 🛠️ Usage

### Quick Start: Video Generation Inference

#### Environment Setup

```bash
# Create conda environment
conda create -n BWM python=3.10.20
conda activate BWM

# Install PyTorch with CUDA support
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install DiffSynth-Studio
pip install diffsynth==2.0.11

# Install dependencies
pip install -r requirements.txt
```

#### Model Weights

Download the [Wan2.2-TI2V-5B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) base model from [ModelScope](https://www.modelscope.cn):

```bash
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Wan2.2-TI2V-5B
```

BWM checkpoint: Coming soon !

#### Run Inference

The demo metadata, videos, actions, and normalization statistics are already included under `demo/`.

Set local paths before running inference:

```bash
cp scripts/local.example.sh scripts/local.sh
```

Update `MODEL_PATHS` and `CKPT_PATH` in `scripts/local.sh`, then run:

```bash
bash scripts/infer_example.sh
```

## 🏋️ Training

Coming soon !

---


## 📜 Citing

If you find **BWM** is useful in your research or applications, please consider giving us a **star** 🌟.

<!-- If you find **ABot-PhysWorld** is useful in your research or applications, please consider giving us a **star** 🌟 and **citing** it by the following BibTeX entry:

```
@article{chen2026abotphysworld,
  title={ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment},
  author={Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, Feng Xiong, Xing Wei, Zhiheng Ma, Mu Xu},
  journal={arXiv preprint arXiv:2603.23376},
  year={2026}
}
``` -->

---
