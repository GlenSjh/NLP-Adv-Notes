<center> <font face="arial", size=6>Deep Text Classification Can be Fooled</font></center><br>

> [PAPER](https://arxiv.org/abs/1704.08006)

## Abstract

> The author propose an white & black box method to craft text adversarial samples. He designs three perturbation strategies: *insertion, modification, deletion*. The attack method is tested in SOTA character & word level DNN-based text classifiers.

## Goal

- Get ***hot characters***: apply **BP** to compute cost gradients  $\nabla_xJ(F,x,c)$ of every dimension in all character vectors. The characters with maximum highest **magnitude** is ***hot characters***
-  Identify ***HTPs***: Hot Training Phrases, Phrases that contain enough ***hot characters*** and occur the most frequently are chosen as ***HTPs***

- Identify ***HSPs***:  imply **where** to manipulate to craft an effectual adversarial sample. similar operations as identify ***HTPs*** , employ ***BP*** to locate hot phrases with significant contribution to the current classification

## White-box attack

### 1. Insertions

> Insert presuppositions and semantically empty phrases to perturb the target text sample

- presuppositions: implicit information that is well-known to readers (search keywords + some ***HTPs*** )

- forged fact: dispensable fact by reforming some real things related to the ***HTPs*** to make people believe it really happened.

### 2. Modificaitions

> affect the model output by slightly manipulating some ***HSPs*** in the input

**GOAL**: increase $J(F,t,c)$ and decrease $J(F,t,c')$

Typo-based watermarking techniques to modify ***HSPs***: 

1. replace with its common typos
2. some characters of it be changed to ones in similar visual appearance

### 3. Removals

> largely downgrade the confidence of the original class

## Black-box attack

> occlude words of samples one by one with a sequence of whitespaces (equal length as words)

![mark](http://pv4mhwy11.bkt.clouddn.com/occlude.png)



By comparing the classification result of a test sample with the seed, we can learn how much deviation an occluded word can cause. The larger the deviation is, the more importance the corresponding word attaches to the correct classification. The words that can bring largest deviations are identified as HSPs for the seed sample.

## Experiments

***HTPs*** generated by white box & black box  are 80% overlapped

Evaluation ways:

1. speed to generate ***HTPs***
2. how human evaluates & recognizes generated adversarial samples 
3. how adversarial samples fool the model
