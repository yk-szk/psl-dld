import numpy as np
from psl_dld.loss import PartialCrossEntropyloss

partial_loss = PartialCrossEntropyloss(3, reduce='none')

y_true = np.array([[0, 1, 2, 3 ,3]]).astype(np.int64)
y_pred = np.array([[[.9, .1, 0], [0, .9, .1], [.1, 0, .9], [.9, 0, .1], [.1, 0, .9]]]).astype(np.float32)
print(partial_loss(y_true, y_pred))
