import numpy as np

def accuracy(y_true, y_pred):
    """Accuracy classification score"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average='binary', labels=None):
    """Precision score for binary and multiclass classification"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if average == 'binary':
        positive_label = 1 if labels is None else labels[0]
        tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
        fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        for label in labels:
            tp = np.sum((y_pred == label) & (y_true == label))
            fp = np.sum((y_pred == label) & (y_true != label))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)

def recall(y_true, y_pred, average='binary', labels=None):
    """Recall score for binary and multiclass classification"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if average == 'binary':
        positive_label = 1 if labels is None else labels[0]
        tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
        fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        for label in labels:
            tp = np.sum((y_pred == label) & (y_true == label))
            fn = np.sum((y_pred != label) & (y_true == label))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)

def f1_score(y_true, y_pred, average='binary', labels=None):
    """F1 score for binary and multiclass classification"""
    if average == 'binary':
        prec = precision(y_true, y_pred, average=average, labels=labels)
        rec = recall(y_true, y_pred, average=average, labels=labels)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for label in labels:
            prec = precision(y_true, y_pred, average='binary', labels=[label])
            rec = recall(y_true, y_pred, average='binary', labels=[label])
            f1s.append(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0)
        return np.mean(f1s)
