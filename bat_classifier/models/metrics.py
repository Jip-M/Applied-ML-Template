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
import numpy as np

class MultiClassMetrics:
    """
    Metrics for multi-class classification.
    """
    def _to_numpy(self, x: object) -> object:
        """
        Convert torch.Tensor to numpy array if needed.
        Args:
            x: Input array or tensor.
        Returns:
            Numpy array or original input if not a tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def confusion(self, y_true: object, y_pred: object) -> np.ndarray:
        """
        Compute the confusion matrix.
        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
        Returns:
            Confusion matrix as a numpy array.
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def auroc(self, y_true: object, y_probs: object) -> float:
        """
        Compute the multiclass ROC AUC using One-vs-Rest (OvR).
        Args:
            y_true: True class labels.
            y_probs: Predicted probabilities
        Returns:
            ROC AUC score.
        """
        y_true = self._to_numpy(y_true)
        y_probs = self._to_numpy(y_probs)
        n_classes = y_probs.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        return auc

    def accuracy(self, y_true: object, y_pred: object) -> float:
        """
        Compute the accuracy score.
        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
        Returns:
            Accuracy as a float.
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return accuracy_score(y_true, y_pred)
