# ZSL-v2

With the development of deep learning approaches and convolutional neural networks (CNN) in particular, the task of recognising objects from a single image has become almost straightforward. However, not only do such image classifiers rely on large training datasets, but they can only recognise images whose class was used to train the network. Zero-Shot Learning (ZSL) addresses this issue by extending the range of recognisable objects by including object classes only described by textual attributes (i.e., without the need of image data) [1]. As textual attributes are not as rich as image ones, object annotations are usually provided with much lower confidence and often alternative labels are offered.

!["global recap of the project"](.\distribute\part3-FSL.png)

The aim of this project is to design and implement a novel deep learning pipeline to evaluate the annotations produced by a ZSL system for an unseen query image to eventually return a single annotation with high confidence. It is proposed that this pipeline will rely on the following steps:
1. Retrieve potential training images by using the potential labels returned by the ZSL system as query terms for a search engine
2. Validate the quality of the retrieved images using a Few-Shot Learning system as part of a leave-one-out approach [2]
3. Use validated images to conduct Few-Shot Learning recognition on the query image

[1] Zero-Shot Learning via Semantic Similarity Embedding, Z. Zhang, V. Saligrama, IEEE
International Conference on Computer Vision, https://doi.org/10.1109/ICCV.2015.474, 2015 \
[2] Generalizing from a Few Examples: A Survey on Few-Shot Learning, Y. Wang, Q. Yao, J.
Kwok, L. M. Ni, arXiv:1904.05046 [cs.LG], https://arxiv.org/pdf/1904.05046.pdf, 2020
