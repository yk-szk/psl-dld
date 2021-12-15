from tensorflow.keras import backend as K


class PartialCrossEntropyloss:
    def __init__(self, num_classes, alpha=0.5, reduce='mean'):
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduce = reduce

    def __call__(self, y_true, y_pred):
        # y_true = y_true[:,:,:,0] # squeeze
        mask = K.cast(y_true < self.num_classes,
                      K.floatx())  # annotated pixels
        inv_mask = K.cast(y_true >= self.num_classes,
                          K.floatx())  # unannotated pixels
        y_true = y_true - K.cast(inv_mask * self.num_classes,
                                 y_true.dtype)  # offset unannotated pixels
        a = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        u = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = (1-self.alpha) * mask * a + inv_mask * self.alpha / u
        if self.reduce == 'mean':
            return K.mean(loss)
        else:
            return loss
