# Towards a Definition of Disentangled Representations
Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, Alexander Lerchner  
(Submitted on 5 Dec 2018)  
(https://arxiv.org/abs/1812.02230v1)


## Questions

### Abstract

1. How can this be done?

    > Here we propose that aprincipled solution to characterising disentangled representations can be foundby focusing on thetransformationproperties of the world.

2. How general/ appicable is this process?
3. This makes sense, how can we find it ?

    > In particular,we suggest that those transformations that change only some properties ofthe underlying world state, while leaving all other properties invariant, arewhat gives exploitable structure to any kind of data.

4. What are those points? And how does this resolve them?

    > Our new definition is in agreementwith many of the current intuitions about disentangling, while also providingprincipled resolutions to a number of previous points of contention.

5. That sounds reasonable, but any concrete evidence of this ?

    > While this work focuses on formally defining disentangling – as opposed to solvingthe learning problem – *we believe that the shift in perspective to studyingdata transformations can stimulate the development of better representationlearning algorithms*.

### 1. Introduction

1. Really? That sounds cool.  

    > A long standing idea in ML is that such shortcomings can be reduced by introducingcertain inductive biases into the model architecture that reflect the structure of theunderlying data [12,33,21,43].

    > [12] M. M. Botvinick, A. Weinstein, A. Solway, and A. Barto. Reinforcement learning,efficient coding, and the statistics of natural tasks. Current Opinion in BehaviouralSciences, 5, 2015.
    
    > [33] R. Gens and P. M. Domingos. Deep symmetry networks.NIPS, 2014.
    
    > [21] T. Cohen and M. Welling. Group equivariant convolutional networks.ICML, 2016.
    
    > [43] G. Hinton, A. Krizhevsky, and S. D. Wang. Transforming auto-encoders. International Conference on Artificial Neural Networks, 2011.  
    
    1.1. How & What?    
  1.2. Is there a general approach to doing this?

2. Need to look into this. Especially into Convolutions from this persepective.

    > One of the most impactful demonstration of this idea,which caused a step change in machine vision, is the convolutional neural network, inwhich the translation symmetry characteristic of visual observations is hard wired intothe network architecture through the convolution operator [54].

    > [54] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to hand written zip code recognition. Neural Computation, 1(4):541–551, 1989.

3. This perspective of working/incorporating bias or of representing the data better/differentley is important nad needs to be looked into.

    > An alternative to hard wiring inductive biases into the network architecture is to instead  learn a representation that is faithful to the underlying data structure [76,2,3,5,4,8,69].
    
    > [76] N. Tishby, F. C. Pereira, and W. Bialek. The information bottleneck method. InProceedings of the 37th Annual Allerton Conference on Communication, Control andComputing, pages 368–377, 1999.
    
    > [2] A. Achille and S. Soatto. Emergence of invariance and disentanglement in deeprepresentations.Journal of Machine Learning Research, 19(50):1–34, 2018.
    
    > [3] A. Achille and S. Soatto. Information dropout: Learning optimal representations through noisy computation. IEEE Transactions on Pattern Analysis and Machine Intelligence, PP(99):1–1, 2018.
    
    > [5] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.
    
    > [4] A. Alemi, B. Poole, I. Fischer, J. Dillon, R. A. Saurus, and K. Murphy.  Aninformation-theoretic analysis of deep latent-variable models.arXiv, 2018.
    
    > [8] F. Anselmi, L. Rosasco, and T. Poggio. On invariance and selectivity in representation learning. Information and Inference, 2016.
    
    > [69] S. Soatto. Steps toward a theory of visual information.Technical Report UCLA-CSD100028, 2010.

4. Is this part of a definition or disctinction of disentangled representation ?

    > This is the motivation for the work on disentangled representation learning, whichdiffers from other dimensionality reduction approaches through its explicit aim to learna representation that axis aligns with the generative factors of the data [10,67,25].
    
    > [10] Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review andnew perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013
    > [67]J. Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation, 4(6):863–869, 1992.
    > [25]J. J. DiCarlo and D. D. Cox. Untangling invariant object recognition. TRENDS in Cognitive Sciences, 11, 2007.

5. What are **symmetry transformations** ?  
5.1. What are some examples ?

    > In particular, our argument is based on the observation that many natural transformationswill change certain aspects of the world state, while keeping other aspects unchanged (or invariant). Such transformations are called **symmetry transformations**, and they can be described and characterised using group and representation theories in mathematics.
    
    > In particular, these approaches can be used to define a set of constraints on the decompositionof a vector space into independent subspaces to ensure that the vector space is reflectiveof the underlying structure of the corresponding symmetry group.

6. What exactly is their definition of **disentangled representation** ?

    > We apply these insightsto the vector space of the model representations and, through that, arrive at the firstprincipled definition of a disentangled representation. Intuitively, we define a vectorrepresentation as disentangled, if it can be decomposed into a number of subspaces, each one of which *is compatible with*, and can be transformed independently by a unique symmetry transformation.

7. How can/is this definition used in practice ?

8. What does 'is compatible with' mean ?

9. Even though there is no algorithmic solution. This is still usefull right ?  
9.1. What would it take to start putting these ideas into practice ?

    > Note that this paper only aims to make a theoretical contribution and does not providea recipe for a general algorithmic solution to disentangled representation learning. It builds a framework to establish a formal connection between symmetry groups andvector representations, which in turn helps resolve many outstanding points of contention surrounding disentangled representation learning.

10. How? And isn't this close to algorithmic ? or How far is this from algorithmic ?

    > For example, our insights can elucidate answers to questions like what are the “data generative factors”, which factors should in principle be possible to disentangle (and what form their representations may take), should each generative factor correspond to a single or multiple latent dimensions, and should a disentangled representation of a particular dataset have a unique basis (upto a permutation of axes). We hope that by gaining a better understanding of what disentangling is and what it is not, faster progress can be made in devising more robustand scalable approaches to disentangled representation learning.

11. How ?

    > We then present a high-level overview of our perspective on how the connections between symmetry transformations and vector representations can be used to define disentangled representations (Sec. 3),