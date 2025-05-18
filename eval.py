import numpy as np
from collections import defaultdict
from sklearn.metrics import auc, roc_curve
import torch
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC
    """
    score = np.nan_to_num(score, nan=0.0)
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    try:
        fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    except:
        fpr, tpr, auc, acc = 0,0,0,0
    low = tpr[np.where(fpr<.01)[0][-1]]
    low05 = tpr[np.where(fpr<.05)[0][-1]]
    
    print('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f, TPR@0.5%%FPR of%.4f\n'%(legend, auc, acc, low, low05))

    return legend, auc,acc, low, low05


def fig_fpr_tpr(all_output, output_dir, mod='w'):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            metric2predictions[metric].append(ex["pred"][metric])

    with open(f"{output_dir}/auc.txt",mod) as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low,low05 = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f, TPR@0.5%%FPR of%.4f\n'%(legend, auc, acc, low, low05))


def compute_metrics(scores, labels):
    """Compute evaluation metrics: AUROC, FPR@95, TPR@1, TPR@5, Accuracy."""
    scores = np.nan_to_num(np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in scores]))
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    tpr01 = tpr[np.where(fpr <= 0.01)[0][-1]]
    tpr05 = tpr[np.where(fpr <= 0.05)[0][-1]]

    return auroc, fpr95, tpr01, tpr05, acc


