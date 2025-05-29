from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import label_binarize

class MultiClassMetrics:
    """
    Not yet implemented completely.
    """
    def confusion(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def auroc(self, y_true, y_probs):
        # y_probs: shape (n_samples, n_classes)
        n_classes = y_probs.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        return auc

    def cross_entropy(self, y_true, y_probs):
        # y_probs: shape (n_samples, n_classes)
        loss = log_loss(y_true, y_probs)
        return loss

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
