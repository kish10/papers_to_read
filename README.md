# Papers To Read

## Anomaly Detection
Holger Dette, Weichi Wu, Zhou Zhou
[Change point analysis of second order characteristics in non-stationary time series
(Mar 2015)](https://arxiv.org/abs/1503.08610)

Raghavendra Chalapathy, Sanjay Chawla.
[Deep Learning for Anomaly Detection: A Survey
(Jan 2019)](https://arxiv.org/abs/1901.03407)

## Machine Learning

### Automatic Differentiation
Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind
[Automatic differentiation in machine learning: a survey  
(Feb 2015)](https://arxiv.org/abs/1502.05767)


### Black Boxness of Neural Networks
Pang Wei Koh1Percy Liang1
[Understanding Black-box Predictions via Influence Functions.  
(July, 2017)](https://arxiv.org/pdf/1703.04730.pdf)
-- (Recommended by Daniel, June 11, 2019)

### Causal Inference - ML
Michel Besserve, Dominik Janzing, Bernhard Schoelkopf
[On The Difference Between Building and Ex-Tracting Patterns: A Causal Analysis of Deep Generative Models. 
(ICLR 2018) - open review](https://openreview.net/forum?id=SySisz-CW)
-- Rejected paper

Raphael Suter, Đorđe Miladinović, Bernhard Schölkopf, Stefan Bauer
[Robustly Disentangled Causal Mechanisms: Validating Deep Representations for Interventional Robustness
(ICML, May 2019)](https://arxiv.org/abs/1811.00007)


### Convergence & Generalization Properties of Neural Networks
Arthur Jacot, Franck Gabriel, Clément Hongler.  
[Neural Tangent Kernel: Convergence and Generalization in Neural Networks.  
(Nov 2018)](https://arxiv.org/abs/1806.07572)  
-- [write up](ml/convergence_and_generalization_properties_of_nns/Neural-Tangent-Kernel_Convergence-and-Generalization-in-Neural-Networks_nov-2018.md)
-- (Recommended by D.Roy, May 13, 2019)

Guillermo Valle-Pérez, Chico Q. Camargo, Ard A. Louis.  
[Deep learning generalizes because the parameter-function map is biased towards simple functions.  
(April, 2019)](https://arxiv.org/abs/1805.08522)  
-- (Recommended by D.Roy, May 13, 2019)

Simon S. Du, Jason D. Lee, Haochuan Li, Liwei Wang, Xiyu Zhai  
[Gradient Descent Finds Global Minima of Deep Neural Networks.  
(Nov 2018)](https://arxiv.org/abs/1811.03804)  
-- (Mentioned by D.Roy, Nov, 2018 ?)

Yazhen Wang  
[Asymptotic Analysis via Stochastic Differential Equations of Gradient Descent Algorithms in Statistical and Computational Paradigms.  
(Dec 2018)](https://arxiv.org/abs/1711.09514)


### Convolution Neural Networks
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
[ImageNet Classification with Deep Convolutional Neural Networks
(NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)

### Convolution Kernel Networks

Julien Mairal, Piotr Koniusz, Zaid Harchaoui, and CordeliaSchmid
[Convolutional Kernel Networks (June 2014, latest Nov 2014)](https://arxiv.org/pdf/1406.3332.pdf)

### Datasets

[Datasets on Antonio Torralba's site](http://web.mit.edu/torralba/www//)

### Disentangeled Representation
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem
[Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations
(Nov 2018)](https://arxiv.org/abs/1811.12359)

Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, Alexander Lerchner  
[Towards a Definition of Disentangled Representations  
(Dec 2018)](https://arxiv.org/abs/1812.02230v1)  
-- [write up](disentangled_representation/towards-a-definition-of-disentangled-representation.md)

Sten Sootlasetn.  
[Curated list of Disentangled Representation papers](https://github.com/sootlasten/disentangled-representation-papers)  
-- (Sent by Daniel, May 26, 2019)

### Natural Language Processing
David Blei, Andrew Ng, Michael Jordan
[Latent Dirichlet Allocation
(2003)](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)


### PAC-Bayes

Fran ̧cois Laviolette, Mario Marchand  
[PAC-Bayes Risk Bounds for Sample-Compressed Gibbs Classifiers]  
(http://www2.ift.ulaval.ca/~laviolette/Publications/ICML05_PacBayesBound.pdf)

Pascal Germain (INRIA Paris), Francis Bach (INRIA Paris), Alexandre Lacoste (Google), Simon Lacoste-Julien (INRIA Paris)  
[PAC-Bayesian Theory Meets Bayesian Inference (Feb, 2017)](https://arxiv.org/pdf/1605.08636.pdf)  
-- (Sent by Daniel, May, 2019)

### Generative Models

#### Variational AutoEncoders

D.P. Kingma, M. Welling
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
The International Conference on Learning Representations (ICLR), Banff, 2014

Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
[Semi-Supervised Learning with Deep Generative Models.  
Oct 2014](https://arxiv.org/abs/1406.5298)

Jack Klys, Jake Snell, Richard Zemel
[Learning Latent Subspaces inVariational Autoencoders.  
2018](http://www.cs.toronto.edu/~zemel/documents/Conditional_Subspace_VAE_all.pdf)
NIPS

Yang Li, Quan Pan, Suhang Wang, Haiyun Peng, Tao Yan, Erik Cambri
[Disentangled Variational Auto-Encoder for semi-supervised learning.  
May 2019](https://doi.org/10.1016/j.ins.2018.12.057)  
Information Sciences

Causal Effect Inference with Deep Latent-Variable Models
Christos Louizos, Uri Shalit, Joris Mooij, David Sontag, Richard Zemel, Max Welling
(Submitted on 24 May 2017 (v1), last revised 6 Nov 2017 (this version, v2))

#### Generative Advesarial Networks

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
(Submitted on 10 Jun 2014)

### Autoregressive Models

### Graphical Models

Bill Freeman andAntonio Torralba
[Lecture 7: graphical models and belief propagation
(2010)](http://helper.ipam.ucla.edu/publications/gss2013/gss2013_11344.pdf)

### Graph Neural Networks

Franco Scarcselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, Gabriele Monfardini
[The graph neural network model
(2009)](https://persagen.com/files/misc/scarselli2009graph.pdf)

Jiaxuan You, Bowen Liu, Rex Ying, Vijay Pande, Jure Leskovec
[Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation
(June 2018)](https://arxiv.org/abs/1806.02473)
-- (Recomended by Daniel July 2019)

Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec
[GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
(ICML 2018)](https://arxiv.org/abs/1802.08773)
-- (Recomended by Daniel Aug 2019)

Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, Maosong Sun
[Graph Neural Networks: A Review of Methods and Applications.  
Mar 2019](https://arxiv.org/abs/1812.08434)

Jure Leskovec (SNAP)
[WWW-18 Tutorial - Representation Learning on Graphs](http://snap.stanford.edu/proj/embeddings-www/)

Renjie Liao, Zhizhen Zhao, Raquel Urtasun, Richard S. Zemel
[LanczosNet: Multi-Scale Deep Graph Convolutional Networks
(Jan 2019)](https://arxiv.org/abs/1901.01484)

Thomas N. Kipf, Max Welling
[Semi-Supervised Classification with Graph Convolutional Networks
(Sep 2016)](https://arxiv.org/abs/1609.02907)

THUNLP - Natural Language Processing Lab at Tsinghua University
[Must-read papers on GNN](https://github.com/thunlp/GNNPapers)

Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel
[Gated Graph Sequence Neural Networks
(Nov 2019)](https://arxiv.org/abs/1511.05493)

- deep generation of graphs
- goal directed conv policy net

### Kernel Descriptors

Liefeng Bo, Xiaofeng Ren, Dieter Fox
[Kernel Descriptors for Visual Recognition (NEUROIPS Dec 2010)](https://proceedings.neurips.cc/paper/2010/file/4558dbb6f6f8bb2e16d03b85bde76e2c-Paper.pdf)

Liefeng Bo, Kevin Lai, Xiaofeng Ren, Dieter Fox,
[Object Recognition with Hierarchical Kernel Descriptors (CVPR 2011)](https://research.cs.washington.edu/istc/lfb/paper/cvpr11.pdf)

### Modular Neural Networks

### Recursive/Recurrent Neural Networks
Andrej Karpathy
[The Unreasonable Effectiveness of Recurrent Neural Networks
(May 2015)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Time Series

#### General Review Papers
John Cristian Borges Gamboa
[Deep Learning for Time-Series Analysis (Jan 2017)](https://arxiv.org/pdf/1701.01887.pdf)

#### Time Series Classification

Angus Dempster, François Petitjean, Geoffrey I. Webb
[ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels (Oct 2019)](https://arxiv.org/abs/1910.13051)

Anthony Bagnall, Aaron Bostrom, James Large, Jason Lines
[The Great Time Series Classification Bake Off: An Experimental Evaluation of Recently Proposed Algorithms. Extended Version (Feb 2016)](https://arxiv.org/pdf/1602.01711.pdf)

Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, Pierre-Alain Muller
[Deep learning for time series classification: a review (Sep 2018, latest May 2019)](https://arxiv.org/pdf/1809.04356v4.pdf)
Zhiguang Wang, Weizhong Yan, Tim Oates
[Time Series Classification from Scratch with Deep Neural Networks: A Strong (Nov 2016, latest Dec 2016)](https://arxiv.org/abs/1611.06455)


#### Time Series Unsupervised learning

Jean-Yves Franceschi (MLIA), Aymeric Dieuleveut (CMAP), Martin Jaggi
[Unsupervised Scalable Representation Learning for Multivariate Time Series (Jan 2019, latest Jan 2020)](https://arxiv.org/pdf/1901.10738v4.pdf)

### Random features

Ali Rahimi, Benjamin Recht
[Random Features for Large-Scale Kernel Machines (NEURIPS 2007](https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)

### Random shallow networks

Alyssa Morrow, Vaishaal Shankar, Devin Petersohn, Anthony Joseph, Benjamin Recht, Nir Yosef
[Convolutional Kitchen Sinks for Transcription FactorBinding Site Prediction (May 2017)](https://arxiv.org/abs/1706.00125)

Ali Rahimi, Benjamin Recht
[Weighted Sums of Random Kitchen Sinks: Replacingminimization with randomization in learning (NEUROIPS 2008)](https://papers.nips.cc/paper/2008/file/0efe32849d230d7f53049ddc4a4b0c60-Paper.pdf)

### Random weights

 Andrew M. Saxe , Pang Wei Koh , Zhenghao Chen , Maneesh Bh , Bipin Suresh , Andrew Y. Ng 
[On Random Weights and Unsupervised Feature Learning (ICML 2011)](http://ai.stanford.edu/~ang/papers/icml11-RandomWeights.pdf)

## Statistics

### Theory of Statistics - Relative Belief
Yanwu Gu, Weijun Li, Michael Evans, Berthold-Georg Englert  
[Very strong evidence in favor of quantum mechanics and against local hidden variables from a Bayesian analysis   
(Jan 2019)](https://arxiv.org/pdf/1808.06863.pdf)
-- [code](https://github.com/lrjconan/LanczosNetwork)
-- (Recomended by Danel July 2019 for code)

### Model Specifications

#### ROC Curves 

Nancy A Obuchowski, Jennifer A Bullen
[Receiver operating characteristic (ROC) curves: review of methods with applications in diagnostic medicine  
(March 2018)](https://iopscience.iop.org/article/10.1088/1361-6560/aab4b1)

