
<center><font face="arial", size=6>Adversarial Training Methods For Semi-Supervised Text Classification</font></center>

[PAPER](https://arxiv.org/abs/1605.07725v1)
[CODE](https://github.com/tensorflow/models/tree/master/research/adversarial_text)

## Abstract

> Adversarial training provides a means of regularizing supervised learning algorithms while virtual adversarial training is able to extend supervised learning. We extend adversarial and virtual adversarial training to the text domain by applying perturbations to the word embeddings in a RNN rather than to the original input itself. 

## Concept

- ***Adversarial training***: the process of training a model to correctly classify both unmodified examples and adversarial examples.
- ***Virtual adversarial training***: extends the idea of adversarial training to the semi-supervised regime and unlabeled examples.

## Methods

### 1. Embedding Normalizations

> The model could trivially learn to make the perturbations insignificant by learning embeddings with very large norm.

$$
E(v)=\sum_{j=1}^kf_jv_j
$$

$$
Var(v)=\sum_{j=1}^kf_j(v_j-E(v))^2
$$

$$
\bar{v}_k=\frac{v_k-E(v)}{\sqrt{Var(v)}}
$$

### 2. Adversarial Training

> a novel regularization method for classifiers to improve robustness to small, approximately worst case perturbations.

Add the following term to cost function:
$$
-logp(y|x+r_{adv};\theta)
$$

$$
r_{adv}=\arg\min_{r,||r||<=\epsilon}log(p|x+r;\theta)
$$

Approximating by linearizing around $x$

$$
r_{adv} =-\epsilon g/||g||_2, g = \nabla_xlogp(y|x;\hat{\theta})
$$
Adversarial Loss:

$$
L_{adv}(\theta)=-\frac{1}{N}\sum_{i=1}^Nlogp(y_n|x_n+r_{adv,n};\theta)
$$

> Note: x represents inputs and thus embedding here

TF CODES:

```python
def adversarial_loss(embedded, loss, loss_fn):
  """Adds gradient to embedding and recomputes classification loss."""
  grad, = tf.gradients(
      loss,
      embedded,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  grad = tf.stop_gradient(grad)
  perturb = _scale_l2(grad, FLAGS.perturb_norm_length)
  return loss_fn(embedded + perturb)
```

## Pipelines for adversarial training

1. Generate $r_{adv}$ based on embedding
2. Throw $x+r_{adv}$ back to network (as the new embedding) to get $L_{adv}$
3. The new loss to minimize is $L = L_{ori} +L_{adv}$

### 3. Virtual adversarial training

> Virtual adversarial loss requires only the input x and does not require the actual label y while adversarial loss defined in Eq(4) requires the label y. This makes it possible to apply virtual adversarial training to semi-supervised learning. 

​	add the following term to cost function:

$$
KL[p(·|x;\hat{\theta}) || p(·|x+r_{adv};\hat{\theta})]
$$

$$r_{adv}=\arg\min_{r,||r||<=\epsilon}KL[p(·|x;\hat{\theta}) || p(·|x+r_{adv};\hat{\theta})]$$

Approximating by linearizing around $x$

$$
r_{v-adv} =-\epsilon g/||g||_2, g = \nabla_xKL[p(·|x;\hat{\theta}) || p(·|x+d;\hat{\theta})]
$$

Virtual  Adversarial Loss:

$$L_{v-adv}(\theta)=-\frac{1}{N}\sum_{i=1}^N KL[p(·|x;\hat{\theta}) || p(·|x+r_{v-adv};\hat{\theta})]$$


