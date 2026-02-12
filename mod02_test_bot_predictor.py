# packages
import pandas as pd
from mod02_build_bot_predictor import train_model

def predict_bot(df, model=None):
    """
    Predict whether each account is a bot (1) or human (0).
    """
    if model is None:
        model = train_model()

    preds = model.predict(df)
    return pd.Series(preds, index=df.index)

def confusion_matrix_and_metrics(y_true, y_pred):
    """
    Computes confusion matrix and common error rates for binary classification.

    Assumes labels:
      0 = negative class
      1 = positive class

    Returns:
      dict with:
        tn, fp, fn, tp
        misclassification_rate
        false_positive_rate
        false_negative_rate
    """
    tn = fp = fn = tp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 1 and yp == 1:
            tp += 1
        else:
            raise ValueError("Labels must be 0 or 1")

    total = tn + fp + fn + tp

    misclassification_rate = (fp + fn) / total if total > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "misclassification_rate": misclassification_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

TRAIN_PATH = "mod02_data/train.csv"
train = pd.read_csv(TRAIN_PATH)

TEST_PATH = "mod02_data/test.csv"
test = pd.read_csv(TEST_PATH)
X_train = train.drop(columns=["is_bot"])
y_train = train['is_bot']

X_test = test.drop(columns=["is_bot"])
y_test = test['is_bot']
model = train_model(X_train, y_train)
y_pred_train = predict_bot(X_train, model)
y_pred_test = predict_bot(X_test, model)
print(confusion_matrix_and_metrics(y_train, y_pred_train))
print(confusion_matrix_and_metrics(y_test, y_pred_test))