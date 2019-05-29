# Neural Tangent Kernel: Convergence and Generalization in Neural Networks
Arthur Jacot, Franck Gabriel, Clément Hongler  
(Submitted on 20 Jun 2018 (v1), last revised 26 Nov 2018 (this version, v3))  
(https://arxiv.org/abs/1806.07572)


## Questions

### General Questions

1. What is the main idea?
2. What is the significance of paper and the results?
3. What are the limitations/impracticalities?
4. What is the Neural Tangent Kernel?
### Abstract

1. What is the connection between Gaussian Processes & Kernel Methods?

2. What does "infinite width" or "limiting kernel" mean?  
2.1. What is the significance/consequences of these?

3. What is "Kernel Gradient Descent"?

4. Why is positive-definiteness important? Why do we care? And what does it tell us?

    > Convergence of the training can then be related to the *positive-definiteness* of the limiting NTK *when the data is supported on the sphere* and the non-linearity is non-polynomial.

5. Why is the the condition that the data needs to be supported on the sphere needed?

6. What are the "kernel principal components"?

    > The convergence is fastest along the largest *kernel principal components* of the input data with respect to the NTK, hence suggesting a theoretical motivation for early stopping.

7. What is the connection between kernel principal components & early stopping?

### 1. Introduction

1. Is this what motivated the paper ?

    > While it has long been known that ANNs can approximate any function with sufficiently many hidden nuerons (9;12), it is not known what the optimization of ANNs converges to.

    > [9] K. Hornik, M. Stinchcombe, and H. White.  Multilayer feed forward networks are universal approximators. Neural Networks, 2(5):359 – 366, 1989

    > [12] M. Leshno, V. Lin, A. Pinkus, and S. Schocken. Multilayer feedforward networks with a non-polynomial activation function can approximate any function. Neural Networks, 6(6):861–867,1993

2. How exactly (in a quantitative sense) do saddle points slow down the convergence?
    > Indeed the loss surface of neural networks optimization problems is highly non-convex: it has a high number of saddle points which may slow down the convergence (5).

3. Why?

    > A number of results (3;15;16) suggest that for wide enough networks, there are very few “bad” local minima ...

    > [3] A. Choromanska, M. Henaff, M. Mathieu, G. B. Arous, and Y. LeCun. The Loss Surfaces of Multilayer Networks. Journal of Machine Learning Research, 38:192–204, nov 2015.
    
    > [15] R. Pascanu,  Y. N. Dauphin,  S. Ganguli,  and Y. Bengio.   On the saddle point problem fornon-convex optimization. arXiv preprint, 2014.
    
    > [16] J. Pennington and Y. Bahri.   Geometry of neural network loss surfaces via random matrixtheory. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2798–2806, International ConventionCentre, Sydney, Australia, 06–11 Aug 2017. PMLR.

4. What does "dynamics of training" mean exactly?

5. Potentially need to investigate this with experiments?

    > A particularly mysterious feature of ANNs is their good generalization properties in spite of their usual over-parametrization (18). It seems paradoxical that a reasonably large neural network can fit random labels, while still obtaining good test accuracy when trained on real data (21). It can be notedthat in this case, kernel methods have the same properties (1).
    
    > [18] L. Sagun, U. Evci, V. U. G ̈uney, Y. Dauphin, and L. Bottou. Empirical analysis of the hessian of over-parametrized neural networks. CoRR, abs/1706.04454, 2017.
    
    > [21] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires rethinking generalization. ICLR 2017 proceedings, Feb 2017.
    
    > [1] M. Belkin, S. Ma, and S. Mandal. To understand deep learning we need to understand kernel learning. arXiv preprint, Feb 2018.

### 1.1 Contribution

1. Need to investigate this further. What are the consquences of this to BNNs?

    > We study the network functionfθof an ANN, which maps an input vector to an output vector, whereθis the vector of the parameters of the ANN. *In the limit as the widths of the hidden layers tend toinfinity, the network function at initialization, $f_θ$ converges to a Gaussian distribution (14; 11)*.
    
    > [14] R. M. Neal.Bayesian Learning for Neural Networks. Springer-Verlag New York, Inc., Secaucus,NJ, USA, 1996.
    
    > [11] J. H. Lee, Y. Bahri, R. Novak, S. S. Schoenholz, J. Pennington, and J. Sohl-Dickstein.  Deepneural networks as gaussian processes. ICLR, 2018.

2.   
2.1. What the implications of this to general NNs"  
2.2. What about BNNs?  
2.3. Is it actually a good measure/description of the generalization of NNs?

    > The convergence properties of ANNs during training can then be related to the positive-definiteness of the infinite-width limit NTK. In the case *when the dataset is supported on a sphere*, we prove this positive-definiteness using recent results on dual activation functions(4). *The values of the network function $f_θ$ outside the training set is described by the NTK, which is crucial to understand how ANN generalize*.
    
    > [4] A. Daniely, R. Frostig, and Y. Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity.  In D. D. Lee, M. Sugiyama, U. V.Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29, pages 2253–2261. Curran Associates, Inc., 2016.

3. Need to understand this better?

    > For a least-squares regression loss, the network function $f_θ$ follows a linear differential equation in the infinite-width limit, and the eigenfunctions of the Jacobian are the kernel principal components of the input data. This shows a direct connection to kernel methods and motivates the use of early stopping to reduce overfitting in the training of ANNs.