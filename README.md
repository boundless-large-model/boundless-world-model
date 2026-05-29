<div align="center">

<h1>🌍 Boundless-World-Model </h1>

<p align="center">
    <a href="https://huggingface.co/spaces/WorldArena/WorldArena"><img src="https://img.shields.io/badge/🏆_Leaderboard-WorldArena-yellow?style=flat"></a>  
    <a href="https://huggingface.co/BLM-Lab/Boundless-World-Model"><img src="https://img.shields.io/badge/🤗_Model-BWM-blue?style=flat"></a>
</p>

</div>

> **BWM** is a physically consistent, action-conditioned video world model built upon Wan2.2-TI2V-5B, serving as a low-cost yet high-fidelity simulator for robotic manipulation.

## 🗞️ News

- **[2026-05]** 🚀 **Inference code released!** Generate action-conditioned robot manipulation videos with BWM. See [🛠️ Usage](#️-usage).
- **[2026-05]** 🎉 **Model definition released!** The BWM architecture and core model components are now available.

## Table of Contents
- [✅ TODO](#-todo)
- [🏗️ Framework](#️-framework)
- [🎬 Qualitative Results](#-qualitative-results)
- [🛠️ Usage](#️-usage)
- [🏋️ Training](#️-training)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 Citing](#-Citing)

---

## ✅ TODO

- [x] Release inference code
- [x] Release model definition
- [ ] Release model weights
- [ ] Release training code
- [ ] Release technical report

---

## 🏗️ Framework

Coming soon !

---

## 🎬 Qualitative Results

### **CVPR 2026 WorldArena Challenge**

> The following simulation scenes are generated autoregressively by **BWM** from initial frames and action sequences in the [**WorldArena test set**](https://github.com/tsinghua-fib-lab/WorldArena/), achieving high-fidelity visual realism while maintaining long-horizon physical consistency.

#### 🧩 Scene 1: Compositional Spatial Rearrangement

  <table align="center" >
    <tr>
      <td><img src="assets/blocks_ranking_size/episode228.gif" alt="blocks ranking size" width="260"></td>
      <td><img src="assets/stack_bowls_three/episode152.gif" alt="stack bowls three" width="260"></td>
    </tr>
  </table>

- **Task**: arrange blocks by size, stack bowls
- **Challenge**: Multi-object spatial ordering, stacking stability, and contact-rich placement
- **Ours**:
  - ✅ Preserves object identity and target layout
  - ✅ Maintains stable stacking contacts
  - ✅ Predicts adaptive gripper control

#### 🚪 Scene 2: Articulated Hinge Interaction

  <table align="center" >
    <tr>
      <td><img src="assets/open_microwave/episode347.gif" alt="open microwave" width="260"></td>
      <td><img src="assets/open_laptop/episode330.gif" alt="open laptop" width="260"></td>
    </tr>
  </table>

- **Task**: open microwave, open laptop
- **Challenge**: Articulated hinge motion, constrained rotation, and persistent object state
- **Ours**:
  - ✅ Captures hinge-constrained opening dynamics
  - ✅ Maintains coherent object geometry during rotation
  - ✅ Preserves opened states over long-horizon rollouts

#### 🕹️ Scene 3: Fine-Grained Affordance Interaction

  <table align="center" >
    <tr>
      <td><img src="assets/turn_switch/episode674.gif" alt="turn switch" width="260"></td>
      <td><img src="assets/hanging_mug/episode373.gif" alt="hanging mug" width="260"></td>
    </tr>
    <tr>
      <td><img src="assets/click_bell/episode796.gif" alt="click bell" width="260"></td>
      <td><img src="assets/stamp_seal/episode581.gif" alt="stamp seal" width="260"></td>
    </tr>
  </table>

- **Task**: turn switch, hang mug, click bell, stamp seal
- **Challenge**: Small contact regions, constrained placement, and precise state-changing interactions
- **Ours**:
  - ✅ Captures fine-grained affordance dynamics
  - ✅ Aligns contact with object affordances
  - ✅ Preserves state-changing interactions

#### 🤝 Scene 4: Bimanual Coordination and Handover

  <table align="center" >
    <tr>
      <td><img src="assets/handover_block/episode47.gif" alt="handover block" width="260"></td>
      <td><img src="assets/handover_mic/episode298.gif" alt="handover mic" width="260"></td>
    </tr>
  </table>

- **Task**: hand over block, hand over mic
- **Challenge**: Dual-arm synchronization, inter-arm occlusion, and coordinated grasp timing
- **Ours**:
  - ✅ Models synchronized dual-arm motion
  - ✅ Preserves object continuity
  - ✅ Avoids close-contact collisions

#### 📦 Scene 5: Long-Horizon Constrained Placement

  <table align="center" >
    <tr>
      <td><img src="assets/put_object_cabinet/episode33.gif" alt="put object cabinet" width="260"></td>
      <td><img src="assets/put_bottles_dustbin/episode1.gif" alt="put bottles dustbin" width="260"></td>
    </tr>
  </table>

- **Task**: put object in cabinet, put bottles in dustbin
- **Challenge**: Long-horizon transport, partial occlusion, and constrained final placement
- **Ours**:
  - ✅ Maintains long-horizon scene coherence
  - ✅ Handles occlusion without object drift
  - ✅ Produces stable constrained placement

### **Out-of-Distribution Generalization**

> To test generalization beyond benchmark initial states, we use **GPT-Image-2-created initial scenes** with original robot action sequences and let **BWM** autoregressively roll out the future under object appearance shifts.

  <table align="center" >
    <tr>
      <td><img src="assets/out_of_distribution/episode100.gif" alt="ood episode100" width="260"></td>
      <td><img src="assets/out_of_distribution/episode100-1.gif" alt="ood episode100 variant 1" width="260"></td>
      <td><img src="assets/out_of_distribution/episode100-3.gif" alt="ood episode100 variant 3" width="260"></td>
    </tr>
    <tr>
      <td><img src="assets/out_of_distribution/episode33.gif" alt="ood episode33" width="260"></td>
      <td><img src="assets/out_of_distribution/episode33-1.gif" alt="ood episode33 variant 1" width="260"></td>
      <td><img src="assets/out_of_distribution/episode33-5.gif" alt="ood episode33 variant 5" width="260"></td>
    </tr>
  </table>

- **Task**: shake bottle, put object in cabinet
- **Challenge**: Novel initial scenes and object appearance shifts
- **Ours**:
  - ✅ Generalizes to GPT-Image-2-created initial scenes
  - ✅ Preserves action-conditioned dynamics
  - ✅ Maintains coherent robot-object interaction

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

Download the [BWM checkpoint](https://huggingface.co/BLM-Lab/Boundless-World-Model) from [Hugging Face](https://huggingface.co):

```bash
hf download BLM-Lab/Boundless-World-Model step-12000.safetensors --local-dir ckpt/BLM
```

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

## 🙏 Acknowledgements

This project builds upon the following open-source projects and benchmarks.
We thank these teams for their contributions:

- Wan2.2: https://github.com/Wan-Video/Wan2.2
- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- WorldArena: https://github.com/tsinghua-fib-lab/WorldArena/
- ABot-PhysWorld: https://github.com/amap-cvlab/ABot-PhysWorld

We also acknowledge the following engineering contributions:

- Wentao Tan: basic architecture design · [Email](mailto:tan.wt.lucky@gmail.com) · [GitHub](https://github.com/FutureTwT)
- Zengrong Lin: core code implementation · [Email](mailto:zengronglin@tongji.edu.cn) · [GitHub](https://github.com/zzezze)
- Yang Sun: code refactoring and software maintainability · [Email](mailto:1006954899@qq.com) · [GitHub](https://github.com/DandelionWow)

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
