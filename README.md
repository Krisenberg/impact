# IMPACT

**I**ntention-**M**odulated **P**ersonalized **ACT**-R &mdash; a hybrid sequential/cognitive-memory recommender system for music that beats SASRec (established baseline) by 114% NDCG@10 on the 4.8B-interaction Yambda-5B dataset. The full pipeline &mdash; download through training &mdash; runs within 16GB of RAM with no GPU and no OOM; the reported results were trained on GPU for speed though.

Built for master's thesis at Wrocław University of Science and Technology.

[![CI](https://github.com/Krisenberg/impact/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Krisenberg/impact/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/endpoint?url=https://docs.basedpyright.com/latest/badge.json)](https://docs.basedpyright.com)

Sequential recommender systems, such as SASRec [[1]](#1), model what a user wants *right now*, within a short window of recent sessions. However, music listening is dominated by repetition &mdash; people return to their favourite tracks for months or even years &mdash; and this long-term signal is completely outside the scope of a short attention window. **IMPACT** adds it back in by using a personalised popularity signal derived from **ACT-R** [[2]](#2), a cognitive architecture that models how human declarative memory strengthens and weakens with use. This signal is weighted by whether each listen was actively chosen or just passively served by the streaming platform. The two models operate and reason independently and are combined at inference time, so the final "blend" can be adjusted/retuned without retraining SASRec (ACT-R is purely deterministic either way).

Experiments showed that ACT-R achieves the highest ranking accuracy (NDCG@10), but can only recommend tracks already present in the user's listening history &mdash; its global exploration score is exactly 0.000 by definition. SASRec can recommend novel, previously unheard tracks, but at substantially lower ranking accuracy. **IMPACT** combines both capabilities, and on the full-scale **Yambda-5B** dataset (4.8 billion interactions) [[3]](#3) **improves NDCG@10 by +114%** relative to SASRec, while still being able to recommend tracks that the user has never heard before.

<p align="center">
  <img src="assets/impact_architecture.svg"
       alt="IMPACT architecture: SASRec (session mean pooling + self-attention) and ACT-R (BLA+ memory signal) as independent pathways, merged at inference via a softmax-weighted late-fusion gate controlled by the hyperparameter alpha."
       width="860">
</p>

## Citation

The accompanying master's thesis is undergoing formal review and defence at Wrocław University of Science and Technology, and cannot be shared publicly yet.

This repository serves as the primary citable artifact. It contains the complete implementation of IMPACT, together with the experimental pipelines, diagrams, and results required to reproduce the findings reported in the thesis.

If you use this software, please cite it using the entry provided in [`CITATION.cff`](CITATION.cff).

## References

<a id="1">[1]</a>
Wang-Cheng Kang and Julian McAuley, 2018.
Self-Attentive Sequential Recommendation.
DOI: [10.48550/arXiv.1808.09781](https://doi.org/10.48550/arXiv.1808.09781)

<a id="2">[2]</a>
John R. Anderson et al., 2004.
Psychological Review 111.4.
DOI: [10.1037/0033-295X.111.4.1036](https://doi.org/10.1037/0033-295X.111.4.1036)

<a id="3">[3]</a>
Alexander Ploshkin et al., 2025.
Yambda-5B — A Large-Scale Multi-Modal Dataset for Ranking and Retrieval.
DOI: [10.1145/3705328.3748163](https://doi.org/10.1145/3705328.3748163)

## License

MIT &mdash; see [`LICENSE`](LICENSE).
