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

## Architecture

The two pathways &mdash; SASRec and ACT-R &mdash; are fully decoupled and independent of each other. Fusion happens only at inference time, via a single hyperparameter alpha.

<p align="center">
    <img src="assets/impact_architecture.svg"
       alt="IMPACT architecture: SASRec (session mean pooling + self-attention) and ACT-R (BLA+ memory signal) as independent pathways, merged at inference via a softmax-weighted late-fusion gate controlled by the hyperparameter alpha."
       width="860">
</p>

## Results

Three configurations of IMPACT were evaluated on the held-out Yambda-5B test set: pure ACT-R (alpha = 0), the optimal (empirically found) hybrid (alpha = 0.4) and pure SASRec (alpha = 1).

| Metric | ACT-R (alpha=0) | **IMPACT (alpha=0.4)** | SASRec (alpha=1) |
|---|---|---|---|
| NDCG@10 | 0.0894 | **0.0650** | 0.0304 |
| Recall@10 | 0.0805 | **0.0624** | 0.0258 |

The results presented above, which use only overall metrics, tell a misleading story: ACT-R performs better than the hybrid with SASRec on both NDCG@10 and Recall@10, which might suggest that the SASRec component is unnecessary. To understand why ACT-R is superior and whether that superiority is desirable, the NDCG@10 metric is analysed along two dimensions &mdash; whether the recommended track is one the user already knows (*repetition*) or genuinely new to them (*exploration*), and whether the "knownness" is judged against the user's recent session window (*local*) or their entire listening history (global):

| NDCG@10, decomposed | ACT-R (alpha=0) | **IMPACT (alpha=0.4)** | SASRec (alpha=1) |
|---|---|---|---|
| Repetition, local | 0.1445 | **0.1024** | 0.0444 |
| Exploration, local | 0.0138 | **0.0098** | 0.0070 |
| Repetition, global | 0.1004 | **0.0720** | 0.0328 |
| Exploration, global | 0.0000 | **0.0022** | 0.0035 |

The NDCG@10 metric decomposition reveals the trade-off introduced earlier. ACT-R's accuracy advantage is entirely driven by repetition &mdash; it performs well because music consumption is repetitive by nature, and ACT-R is specifically designed to model such behaviour. However, the downside is that its global exploration score is 0.000, meaning it cannot recommend a track outside the user's listening history (which leads to a filtering bubble). SASRec is at the other extreme &mdash; it can recommend genuinely new content, but with only a fraction of ACT-R's accuracy. **IMPACT bridges this gap** by retaining most of ACT-R's repetition-driven accuracy while maintaining a non-zero exploration ability &mdash; the outcome the hybrid was designed to produce.

**Statistical significance.** To test whether the improvement is systematic rather than incidental, a Wilcoxon signed-rank test for paired data was applied to per-session NDCG@10 differences $d_i$ = NDCG@10(IMPACT) − NDCG@10(SASRec) across the full test set, under the following hypotheses:
- $H_0$: The median of $d_i$ is zero &mdash; IMPACT does not outperform SASRec systematically.
- $H_1$: The median of $d_i$ is positive &mdash; IMPACT systematically achieves a higher NDCG@10 than SASRec.

The test rejected $H_0$ in favour of $H_1$ with a p-value $<< 0.001$ and an effect size r = 0.771 (Cohen's $d$ $\approx$ 2.42).

Details that are worth stating explicitly:
- **Effect size scope:** an effect size of r = 0.771 describes the strength and consistency of the difference among the 5,960 pairs (26.8% of the test set) in which the two systems actually differed. For the remaining 73.2%, both models returned identical scores &mdash; predominantly NDCG@10 = 0 for both, which is typical in sparse recommendation settings. The effect is large where it exists, but not uniformly large across the full test set.
- **Direction:** of the 5,960 non-zero pairs, IMPACT outperformed SASRec in 5,182 cases (86.9%) and underperformed in 778 (13.1%) &mdash; a ratio of approximately 6.7:1 in favour of IMPACT, which is consistent with the one-sided hypothesis $H_1$. This could be understood like the following: **where introducing the personalised popularity signal changes the prediction at all, that change is nearly seven times more likely to help than to hurt.**

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
