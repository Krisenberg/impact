# IMPACT

**I**ntention-**M**odulated **P**ersonalized **ACT**-R &mdash; a hybrid sequential/cognitive-memory recommender system for music that beats SASRec (SOTA) by 114% NDCG@10 on the 4.8B-interaction Yambda-5B dataset. The full pipeline &mdash; download through training &mdash; runs within 16GB of RAM with no GPU and no OOM; the reported results were trained on GPU via Kaggle for speed though.

Built for master's thesis at Wrocław University of Science and Technology.

[![CI](https://github.com/Krisenberg/impact/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Krisenberg/impact/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/endpoint?url=https://docs.basedpyright.com/latest/badge.json)](https://docs.basedpyright.com)

<p align="center">
  <img src="assets/impact_architecture.svg"
       alt="IMPACT architecture: SASRec (session mean pooling + self-attention) and ACT-R (BLA+ memory signal) as independent pathways, merged at inference via a softmax-weighted late-fusion gate controlled by the hyperparameter alpha."
       width="860">
</p>
