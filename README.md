# Face-Recognition
Recognising faces in images using Python and identifying different persons


## The Eigenface Algorithm

**Input**: -$m$ $n$-dimenstional Datapoints: $m$ faces as imigas with width $w$ and height $h$ ($n=w \cdot h$), in grayscale. We assume that $m\leq n$.
- One $n$-dimensional query-image: Face with width $w$ and height $h$, in grayscale.

Output: The index $i\in \{1, \dots, m\} of the face image $g$, that best matches $g^*$.

##1. Calculating Avarage face image:
$
g = \frac{1}{m}\sum-{i=1}^m g_i \in R^n
$
