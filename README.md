# Partially supervised learning for diffuse lung abnormality pattern segmentation

This repository contains the source code for the following paper.

`TBA`


## Example
```python
import numpy as np
from psl_dld.loss import PartialCrossEntropyloss

partial_loss = PartialCrossEntropyloss(3, reduce='none')

y_true = np.array([[0, 1, 2, 3 ,3]]).astype(np.int64)
y_pred = np.array([[[.9, .1, 0], [0, .9, .1], [.1, 0, .9], [.9, 0, .1], [.1, 0, .9]]]).astype(np.float32)
print(partial_loss(y_true, y_pred))
```

```output
tf.Tensor([[0.6183691 0.6183691 0.618369  1.6171571 0.7050352]], shape=(1, 5), dtype=float32)
```

Above is an example with three classes.
Labels are encoded as follows {A:0, B:1, C:2, ¬A:3, ¬B:4, ¬C:5}.
In the example, true labels are [A, B, C, ¬A, ¬A]


## Cite
If you find the repository interesting, please consider citing the following paper.

```
TBA
```