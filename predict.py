"""Command-line sentiment prediction using the trained model.

Usage:
    python predict.py "I love this product!"
    python predict.py            # interactive mode (type text, empty line to quit)

NOTE: clean_text() must stay identical to the one in sentiment_analysis.ipynb.
"""
import re
import sys
import joblib

URL_RE     = r"https?://\S+|www\.\S+"
MENTION_RE = r"@\w+"
EMO_RE     = r"(?::|;|=)-?[\)\(DPp/\\]|[\)\(]-?(?::|;|=)|:'\(|<3"


def clean_text(s: str) -> str:
    s = re.sub(URL_RE, " ", s)
    s = re.sub(MENTION_RE, " ", s)
    s = re.sub(EMO_RE, " ", s)
    s = s.lower()
    s = s.replace("&amp;", " and ").replace("#", " ")
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def predict(texts, model_path="models/sentiment_model.joblib"):
    model = joblib.load(model_path)
    cleaned = [clean_text(t) for t in texts]
    labels = model.predict(cleaned)
    scores = model.predict_proba(cleaned)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(cleaned)
    return [("positive" if l == 1 else "negative", float(s)) for l, s in zip(labels, scores)]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
        for t, (lab, sc) in zip(texts, predict(texts)):
            print(f"{lab.upper():9s} (score={sc:.3f})  <- {t}")
    else:
        print("Type a sentence (empty line to quit):")
        while True:
            t = input("> ").strip()
            if not t:
                break
            lab, sc = predict([t])[0]
            print(f"  {lab.upper()} (score={sc:.3f})")
