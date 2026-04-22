# UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

<p align="center">
  <a href="https://xpeng-robotics.github.io/unit/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-5b8dc9?style=flat-square&logo=github">
  </a>
  <a href="https://xpeng-robotics.github.io/unit/UniT.pdf">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-PDF-e63946?style=flat-square">
  </a>
</p>

<p align="center">
  <b>Boyu Chen</b><sup>1,2,*</sup> &nbsp;·&nbsp;
  <b>Yi Chen</b><sup>1,3,*</sup> &nbsp;·&nbsp;
  <b>Lu Qiu</b><sup>1,3</sup> &nbsp;·&nbsp;
  <b>Jerry Bai</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Yuying Ge</b><sup>1,&dagger;</sup> &nbsp;·&nbsp;
  <b>Yixiao Ge</b><sup>1</sup>
  <br>
  <sup>1</sup>XPENG Robotics &nbsp;·&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;·&nbsp;
  <sup>3</sup>The University of Hong Kong
  <br>
  <sub><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding author</sub>
  <br>
  <sub>Correspondence: <a href="mailto:yyge13@gmail.com">yyge13@gmail.com</a></sub>
</p>

<p align="center">
  <img src="assets/teaser.jpeg" alt="UniT teaser — from human demonstration to humanoid policy and world model" width="100%">
</p>

---

> **Project page:** <https://xpeng-robotics.github.io/unit/>

## Overview

Scaling humanoid foundation models is bottlenecked by the scarcity of robotic data.
While massive egocentric human data offers a scalable alternative, bridging the
cross-embodiment chasm remains a fundamental challenge due to kinematic mismatches.
We introduce **UniT** (**Uni**fied Latent Action **T**okenizer via Visual
Anchoring), a framework that establishes a unified physical language for
human-to-humanoid transfer. Grounded in the philosophy that heterogeneous
kinematics share universal visual consequences, UniT employs a tri-branch
cross-reconstruction mechanism: actions predict vision to anchor kinematics to
physical outcomes, while vision reconstructs actions to filter out irrelevant
visual confounders. Concurrently, a fusion branch synergizes these purified
modalities into a shared discrete latent space of embodiment-agnostic physical
intents.

We validate UniT across two paradigms:

1. **Policy Learning (VLA-UniT).** By predicting these unified tokens, it
   effectively leverages diverse human data to achieve state-of-the-art data
   efficiency and robust out-of-distribution (OOD) generalization on both a
   humanoid simulation benchmark and real-world deployments, notably
   demonstrating *zero-shot task transfer*.
2. **World Modeling (WM-UniT).** By aligning cross-embodiment dynamics via
   unified tokens as conditions, it realizes direct human-to-humanoid action
   transfer. This alignment ensures that human data seamlessly translates into
   enhanced action controllability for humanoid video generation.

Ultimately, by inducing a highly aligned cross-embodiment representation
(empirically verified by t-SNE visualizations revealing the convergence of
human and humanoid features into a shared manifold), UniT offers a scalable
path to distill vast human knowledge into general-purpose humanoid
capabilities.

## Status

> **The code release is in progress.** This repository currently hosts only the
> paper and project page link. Training code, inference code, pretrained
> checkpoints, and data preparation scripts will be released here progressively.
> Please watch the repository for updates.

Planned release order:

- [ ] Data preparation scripts
- [ ] Pretrained checkpoints
- [ ] UniT tokenizer — training & inference
- [ ] VLA-UniT — training & evaluation on RoboCasa GR1
- [ ] WM-UniT — training & sampling on RoboCasa GR1 and GR00T-Teleop mixtures
- [ ] Real-world deployment stack

## Contact

For questions about the paper or the upcoming release, please open an issue in
this repository, or reach out to:

- Yuying Ge &mdash; [yyge13@gmail.com](mailto:yyge13@gmail.com) *(corresponding author)*
- Boyu Chen &mdash; [boyuc448@gmail.com](mailto:boyuc448@gmail.com)
