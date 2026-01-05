import json
from pathlib import Path

def compute_metrics(y_true, y_pred):
    # label: "authentic" / "tampered"
    tp = sum(1 for t,p in zip(y_true,y_pred) if t=="tampered" and p=="tampered")
    tn = sum(1 for t,p in zip(y_true,y_pred) if t=="authentic" and p=="authentic")
    fp = sum(1 for t,p in zip(y_true,y_pred) if t=="authentic" and p=="tampered")
    fn = sum(1 for t,p in zip(y_true,y_pred) if t=="tampered" and p=="authentic")

    acc = (tp+tn) / max(1,(tp+tn+fp+fn))
    prec = tp / max(1,(tp+fp))
    rec  = tp / max(1,(tp+fn))
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    return {
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def save_metrics(metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
