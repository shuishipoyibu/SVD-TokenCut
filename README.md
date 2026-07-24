# SVD-Based Analysis of TokenCut

This repository presents an experimental analysis of SVD-based dimensionality reduction for TokenCut. It is built on the official TokenCut implementation; the original baseline code, paper, and attribution remain the work of the TokenCut authors. The extensions in this repository include centered PCA, uncentered SVD, cosine SVD, graph-consistency metrics, and the accompanying experiments on VOC12 and COCO.
## Upstream TokenCut Resources

- Official TokenCut repository: [YangtaoWANG95/TokenCut](https://github.com/YangtaoWANG95/TokenCut)
- Original paper: [Self-Supervised Transformers for Unsupervised Object Discovery Using Normalized Cut (CVPR 2022)](https://arxiv.org/abs/2202.11539)

This project retains and extends the official TokenCut codebase. The upstream implementation, baseline pipeline, and original method remain attributable to the TokenCut authors.

## Repository Guide

- [SVD_TOKENCUT_COMMANDS.md](SVD_TOKENCUT_COMMANDS.md): the operational reference for the current modified `main_tokencut.py`. It documents baseline runs, feature caching, centered PCA, uncentered SVD, cosine SVD, random projection, parameters, output paths, and known limitations.
- [DOWNLOAD_DATA.md](DOWNLOAD_DATA.md): dataset and DINO-checkpoint download instructions, together with the expected directory layouts.
- [TOKENCUT_ORIGINAL_README.md](TOKENCUT_ORIGINAL_README.md): the original TokenCut README retained for the upstream installation, inference, evaluation, and citation instructions.
- [Experimental_Procedure_and_Analysis_EN.md](Experimental_Procedure_and_Analysis_EN.md): a standalone copy of the full English experimental report reproduced below.
The complete English report follows. Its body is retained unchanged so that this README serves both as the project entry point and as the full research record.
# Investigating Whether SVD-Based Dimensionality Reduction Improves TokenCut Segmentation Accuracy

## Introduction

DINO is a self-supervised representation learning method based on the Vision Transformer (ViT) architecture and has demonstrated strong feature extraction capabilities relative to supervised learning methods. Building upon DINO, *Self-Supervised Transformers for Unsupervised Object Discovery Using Normalized Cut* proposes an unsupervised object discovery method that treats visual tokens as graph nodes, uses the pairwise cosine similarities between tokens as edge weights, and performs graph partitioning through a generalized eigenvalue decomposition.

However, subtle noise and distracting regions in an image are also represented as graph nodes and therefore participate in the partitioning process, potentially reducing segmentation accuracy. Principal component analysis (PCA) is a dimensionality reduction method based on singular value decomposition (SVD): it first centers the vectors and then performs SVD. In this project, PCA is applied while retaining the first $k$ principal-component directions. The feature vectors extracted by DINO are projected onto this low-dimensional subspace, after which the graph is reconstructed and TokenCut segmentation is performed. The purpose is to investigate whether this procedure improves segmentation accuracy.

The principal observations of this study are as follows:

1. Centering substantially changes the cosine similarities between vectors. In the present experiments, PCA-based dimensionality reduction involving centering markedly reduces object localization accuracy.
2. Applying SVD directly to uncentered features yields slightly better results than the original TokenCut method on some datasets; however, this improvement is not reproduced across all datasets.

## Related Work

*SoMA: Singular Value Decomposed Minor Components Adaptation for Domain Generalizable Representation Learning* first applies SVD to the weights of a pretrained model. It then freezes the principal singular subspace, which exhibits relatively strong cross-domain stability, and learns task-specific parameters only within the minor singular subspace, thereby balancing the preservation of general knowledge against task adaptation. The study reports that principal singular components tend to represent domain-stable shared semantics, whereas minor singular components are more likely to encode fine-grained, context-dependent, and task- or domain-specific information. This finding motivates the present study to retain the principal singular directions while discarding the minor ones, with the aim of preserving core information and suppressing noise.

## Experimental Procedure and Data Analysis

### Initial Experiments on the VOC12 Dataset

The experiments use the VOC12 dataset and a ViT-S/16 model, with CorLoc as the evaluation metric and the key vectors extracted by DINO as the input features. First, the TokenCut result reported in the original study is reproduced and used as the full-dimensional control. Next, full PCA is used to project the feature vectors to different dimensions; TokenCut is then applied, and the corresponding CorLoc is measured. It should be noted that the number of principal directions retained by SVD cannot exceed the number of patches.

| Method | 1 | 8 | 16 | 32 | 64 | 128 | Full dimension |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full PCA | 32.1 | 41.3 | 42.4 | 43.9 | 44.8 | 45.4 | / |
| Vanilla TokenCut | / | / | / | / | / | / | 72.1 |

Table 1. CorLoc on VOC12 at different retained dimensions (unit: %).

The results show that accuracy after PCA-based dimensionality reduction is substantially lower than that obtained without dimensionality reduction, and that accuracy increases as more dimensions are retained. This finding suggests that full PCA may disrupt information on which TokenCut graph construction depends. One plausible explanation is that centering substantially changes the cosine similarities between tokens, which constitute an important basis for TokenCut segmentation.

Let the original token features be $x_i$ and $x_j$, and let the mean of all tokens be
$$
\boldsymbol{\mu}=\frac{1}{N}\sum_{i=1}^{N}x_i.
$$
After centering, the token feature becomes $\tilde{x}_i=x_i-\boldsymbol{\mu}$. The cosine similarities computed from the original and centered features are, respectively,
$$
c_{ij}=\frac{x_i^{\mathsf T}x_j}{\lVert x_i\rVert_2\lVert x_j\rVert_2},
$$

$$
\tilde{c}_{ij}
=\frac{(x_i-\boldsymbol{\mu})^{\mathsf T}(x_j-\boldsymbol{\mu})}
{\lVert x_i-\boldsymbol{\mu}\rVert_2\lVert x_j-\boldsymbol{\mu}\rVert_2}
=\frac{x_i^{\mathsf T}x_j-\boldsymbol{\mu}^{\mathsf T}x_i-\boldsymbol{\mu}^{\mathsf T}x_j+\lVert\boldsymbol{\mu}\rVert_2^2}
{\lVert x_i-\boldsymbol{\mu}\rVert_2\lVert x_j-\boldsymbol{\mu}\rVert_2}.
$$
Because centering changes both the numerator and the denominator and introduces additional terms such as $-\boldsymbol{\mu}^{\mathsf T}x_i$ and $-\boldsymbol{\mu}^{\mathsf T}x_j$, in general,
$$
\tilde{c}_{ij}\neq c_{ij}.
$$
Centering is therefore not a cosine-similarity-preserving transformation. When these similarities are used to construct the TokenCut graph, the edge structure of the original graph may change accordingly.

To examine this hypothesis, two experimental conditions are established. In the first, the features are centered without dimensionality reduction before TokenCut is applied. In the second, SVD is applied directly without centering, followed by TokenCut.

| Method | 1 | 8 | 16 | 32 | 64 | 128 | Full dimension |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full PCA | 32.1 | 41.3 | 42.4 | 43.9 | 44.8 | 45.4 | / |
| Centering only | / | / | / | / | / | / | 45.5 |
| Uncentered SVD | 56.4 | 72.1 | 72.4 | 72.5 | 72.3 | 72.1 | / |
| Vanilla TokenCut | / | / | / | / | / | / | 72.1 |

Table 2. CorLoc of different methods on VOC12 (unit: %).

| Method | 1 | 8 | 16 | 32 | 64 | 128 | Full dimension |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full PCA | 2.5882 | 1.0408 | 0.8903 | 0.8184 | 0.7839 | 0.7689 | / |
| Centering only | / | / | / | / | / | / | 0.7643 |
| Uncentered SVD | 2.2418 | 0.5124 | 0.3590 | 0.2780 | 0.2272 | 0.0174 | / |
| Vanilla TokenCut | / | / | / | / | / | / | 0 |

Table 3. Mean relative cosine-graph error (`mean_relative_cosine_graph_error`; lower is better) of different methods on VOC12.

The cosine error is calculated as follows. Let $C\in\mathbb{R}^{N\times N}$ denote the cosine-similarity matrix constructed from the original token features, and let $\hat{C}$ denote the cosine-similarity matrix constructed from the dimensionally reduced token features. The mean relative cosine-graph error is defined as
$$
E_{\mathrm{cos}}
=\frac{\lVert\hat{C}-C\rVert_F}{\lVert C\rVert_F},
$$
where $\lVert\cdot\rVert_F$ denotes the Frobenius norm, namely, the square root of the sum of the squared matrix elements:
$$
\lVert A\rVert_F
=\sqrt{\sum_{i=1}^{N}\sum_{j=1}^{N}A_{ij}^{\,2}}.
$$
Thus, this metric quantifies the relative discrepancy between the entire cosine-similarity matrices before and after dimensionality reduction, rather than the mean relative error of individual edges. For vanilla TokenCut without dimensionality reduction, $\hat{C}=C$ and therefore $E_{\mathrm{cos}}=0$.

The accuracy obtained with centering alone is similar to that obtained with full PCA. By contrast, uncentered SVD yields slightly higher accuracy than direct TokenCut at several retained dimensions. Moreover, the relative cosine-graph error resulting from centering alone is substantially higher than that resulting from uncentered SVD, which is consistent with the hypothesis that centering changes the cosine-similarity information used by TokenCut. Under uncentered SVD, accuracy first increases and then decreases as the retained dimension grows. This trend indicates that preserving more information does not necessarily produce higher accuracy; on VOC12, moderate dimensionality reduction may help suppress information that is detrimental to segmentation.

Because TokenCut graph construction relies on cosine similarities between tokens, a dimensionality reduction method referred to here as cosine SVD is further introduced. It applies L2 normalization before SVD to remove vector-magnitude information, thereby encouraging the dimensionality reduction process to place greater emphasis on preserving pairwise cosine similarities. Uncentered SVD and cosine SVD are subsequently compared across different retained dimensions.

| Method | 1 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncentered SVD | 56.4 | 72.1 | 72.3 | 72.4 | 72.5 | 72.5 | 72.3 | 72.3 | 72.2 | 72.1 |
| Cosine SVD | 56.9 | 72.1 | 72.3 | 72.5 | 72.5 | 72.3 | 72.3 | 72.3 | 72.2 | 72.2 |

Table 4. CorLoc of uncentered SVD and cosine SVD at different retained dimensions (unit: %).

| Method | 1 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncentered SVD | 0.7306 | 0.9056 | 0.9280 | 0.9413 | 0.9504 | 0.9571 | 0.9624 | 0.9667 | 0.9846 | 0.9957 |
| Cosine SVD | 0.7312 | 0.9066 | 0.9288 | 0.9420 | 0.9510 | 0.9577 | 0.9629 | 0.9671 | 0.9848 | 0.9957 |

Table 5. Edge-selection agreement rates of uncentered SVD and cosine SVD at different retained dimensions.

TokenCut determines an edge value according to whether its cosine similarity lies above or below a threshold $\tau$. Consequently, the relative cosine-graph error does not fully characterize changes in graph structure. A small cosine-similarity error concentrated near $\tau$ may affect the graph more strongly than a larger error far from $\tau$. The edge-selection agreement rate is therefore introduced. It is defined as the proportion of edges in the graph constructed from the dimensionally reduced vectors whose values are identical to those in the original graph.

At the same retained dimension, cosine SVD achieves a slightly higher edge-selection agreement rate than uncentered SVD. At higher dimensions, however, uncentered SVD can attain agreement rates and accuracy values comparable to those achieved by cosine SVD at lower dimensions. No clear difference is observed between the peak accuracies of the two methods. These results suggest that the advantage of cosine SVD in preserving cosine similarity is limited and that cosine SVD does not exhibit a stronger ability to suppress task-irrelevant noise. Its primary effect may be to shift the peak accuracy toward a lower retained dimension rather than to increase the peak accuracy itself.

As an additional control, CorLoc is measured after Gaussian random projection with a random seed of 4. Its accuracy is substantially lower than that of the SVD-based methods, indicating that, under the present experimental setting, SVD is more effective than random projection at preserving information relevant to segmentation.

| Dataset | Retained dimension | CorLoc (%) |
|---|---:|---:|
| VOC12 | 1 | 21.4 |
| VOC12 | 8 | 58.3 |
| VOC12 | 16 | 61.3 |
| VOC12 | 32 | 65.2 |
| VOC12 | 64 | 68.9 |
| VOC12 | 128 | 71.7 |

Table 6. CorLoc obtained by combining Gaussian random projection with TokenCut.

### Evaluation on the COCO Dataset

To assess whether the change in accuracy resulting from SVD-based dimensionality reduction generalizes across datasets, additional experiments are conducted on COCO. Owing to memory constraints, only the first 10,000 images in the validation set are used. The TokenCut results reported here therefore differ from the official baseline. Because the preceding experiments indicate that cosine SVD and uncentered SVD perform similarly, only cosine SVD is evaluated in this experiment.

As shown in Table 7, TokenCut after SVD-based dimensionality reduction performs slightly worse on COCO than the full-dimensional baseline. Accuracy reaches its maximum of 58.4% at both 128 and 160 dimensions, remaining marginally below the baseline value of 58.5%. From 12 to 128 dimensions, the edge-selection agreement rate increases from approximately 92% to approximately 99%, while CorLoc generally increases and progressively approaches the baseline. These results indicate that, under the present COCO experimental setting, the loss of graph-structural information caused by dimensionality reduction has not been completely eliminated. Even when the edge-selection agreement rate is high, the remaining discrepancies may still affect the final segmentation result.

| Method | Retained dimension | CorLoc (%) | Edge-selection agreement rate |
|---|---:|---:|---:|
| Cosine SVD | 12 | 57.5 | 0.9222 |
| Cosine SVD | 16 | 57.4 | 0.9358 |
| Cosine SVD | 20 | 57.6 | 0.9452 |
| Cosine SVD | 24 | 57.7 | 0.9523 |
| Cosine SVD | 32 | 57.8 | 0.9623 |
| Cosine SVD | 64 | 58.2 | 0.9818 |
| Cosine SVD | 128 | 58.4 | 0.9945 |
| Cosine SVD | 160 | 58.4 | 0.9970 |
| Vanilla TokenCut | Full dimension | 58.5 | 1.0000 |

Table 7. CorLoc and edge-selection agreement rates of cosine SVD, together with the CorLoc of vanilla TokenCut, on COCO.

## Hypotheses and Directions for Further Investigation

1. CorLoc is a discrete metric that indicates only whether a prediction satisfies a specified correctness criterion; it may therefore be insufficient for a fine-grained assessment of segmentation quality. Continuous metrics such as intersection over union (IoU) could be introduced as complementary evaluation measures.
2. The principal singular directions may not fully correspond to the information most relevant to object localization in COCO, and some useful information may reside in minor singular directions. Future work could investigate intermediate-spectrum subspaces, low-spectrum subspaces, or combinations of multiple subspaces.
3. Priority should be given to analyzing COCO examples for which cosine SVD combined with TokenCut fails but vanilla TokenCut succeeds. The masks produced by the two methods can be visualized and compared to identify the regions with the greatest discrepancies. In addition, the vectors obtained from the generalized eigenvalue decompositions of the two methods can be reshaped into the $H\times W$ patch arrangement, upsampled to the original image resolution, and visualized as heatmaps to locate the source of their disagreement. Conversely, examples in which cosine SVD succeeds but vanilla TokenCut fails are also worthy of investigation.
