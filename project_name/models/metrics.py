# from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss, accuracy_score
# from sklearn.preprocessing import label_binarize

# class MultiClassMetrics:
#     """
#     Not yet implemented completely.
#     """
#     def confusion(self, y_true, y_pred, labels):
#         cm = confusion_matrix(y_true, y_pred, labels=labels)
#         return cm

#     def auroc(self, y_true, y_probs):
#         # y_probs: shape (n_samples, n_classes)
#         n_classes = y_probs.shape[1]
#         y_true_bin = label_binarize(y_true, classes=range(n_classes))
#         auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
#         return auc

#     def accuracy(self, y_true, y_pred):
#         return accuracy_score(y_true, y_pred)


from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import label_binarize
import torch

class MultiClassMetrics:
    """
    Metrics for multi-class classification.
    """
    
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def confusion(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def auroc(self, y_true, y_probs):
        y_true = self._to_numpy(y_true)
        y_probs = self._to_numpy(y_probs)
        n_classes = y_probs.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        return auc

    def accuracy(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return accuracy_score(y_true, y_pred)
