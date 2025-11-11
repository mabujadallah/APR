# AIDEV_Agent_exploration.py
# Robust exploration of hao-li/AIDev dataset (improved version)

from datasets import load_dataset
import pandas as pd
import re, json
from ast import literal_eval
import numpy as np

def safe_parse_as_json_like(x):
    # Try to interpret x as a JSON-like structure or Python literal.
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    s = str(x).strip()
    if s == "":
        return None
    try:
        if (s[0] == "{" and s[-1] == "}") or (s[0] == "[" and s[-1] == "]"):
            return json.loads(s)
    except Exception:
        pass
    try:
        return literal_eval(s)
    except Exception:
        pass
    return None

def session_length(value):
    parsed = safe_parse_as_json_like(value)
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        return len(parsed)
    s = "" if value is None else str(value)
    lines = s.splitlines()
    if len(lines) <= 1:
        tokens = re.split(r"[;\|]|->", s)
        return max(1, sum(1 for t in tokens if t.strip()))
    return len(lines)

def has_bug_kw(text):
    if text is None:
        return 0
    return 1 if re.search(r"\b(fix|bug|error|crash|patch|fail|exception)\b", str(text), re.I) else 0

def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    print("Loading dataset hao-li/AIDev ...")
    dataset = load_dataset("hao-li/AIDev")
    print("Available splits:", list(dataset.keys()))
    split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    print("Using split:", split)
    df = pd.DataFrame(dataset[split])
    print("Rows:", len(df))
    print("\nColumns:\n", df.columns.tolist())

    session_candidates = ['view_session', 'view_sessions', 'session', 'sessions', 'body', 'actions', 'action_history', 'viewed_actions']
    session_col = detect_column(df, session_candidates)
    if session_col is None:
        probable = [c for c in df.columns if df[c].dtype == object and df[c].map(lambda x: isinstance(x, str) and len(str(x))>20).sum() > 0]
        session_col = probable[0] if probable else df.columns[0]
        print(f"No known session column found. Picking '{session_col}' as fallback.")
    else:
        print(f"Detected session column: {session_col}")

    title_candidates = ['title', 'summary', 'subject']
    title_col = detect_column(df, title_candidates)
    print("Detected title column:", title_col)

    label_candidates = ['status', 'label', 'verdict', 'accepted', 'is_accepted', 'merged']
    label_col = detect_column(df, label_candidates)
    if label_col:
        print("Detected label/status column:", label_col)
    else:
        print("No label/status column detected. RQ2 (accepted vs rejected) will be skipped unless you provide labels.")

    working = df.copy()
    if title_col is None:
        working['title_for_analysis'] = ""
    else:
        working['title_for_analysis'] = working[title_col].astype(str)

    working['session_for_analysis'] = working[session_col]
    print("\nComputing features (this may take a while on large data)...")
    working['n_actions'] = working['session_for_analysis'].apply(session_length)
    working['has_bug_kw'] = (working['title_for_analysis'].apply(has_bug_kw) | working['session_for_analysis'].apply(has_bug_kw)).astype(int)

    if label_col:
        def label_to_binary(x):
            if x is None:
                return np.nan
            s = str(x).lower()
            if s in ('1','true','t','yes','y','accepted','merged','fix','patch','correct'):
                return 1
            if s in ('0','false','f','no','n','rejected','incorrect'):
                return 0
            return np.nan
        working['label'] = working[label_col].apply(label_to_binary)
    else:
        working['label'] = np.nan

    print("\n===== RQ1: Context Characteristics =====")
    print("Average actions per session:", round(working['n_actions'].mean(), 2))
    print("Median actions per session:", working['n_actions'].median())
    print("Bug keyword mentions (%):", round(working['has_bug_kw'].mean()*100, 2), "%")

    if working['label'].notna().sum() > 0:
        print("\n===== RQ2: Accepted vs Rejected =====")
        group_stats = working.groupby('label')[['n_actions','has_bug_kw']].mean()
        print(group_stats)
    else:
        print("\n===== RQ2: Skipped =====")
        print("No reliable binary label column found in dataset. You can supply or derive labels to run RQ2.")

    print("\nSample rows for manual inspection (first 10):")
    display_cols = ['title_for_analysis', 'session_for_analysis', 'n_actions', 'has_bug_kw', 'label']
    print(working[display_cols].head(10).to_string(index=False))

    print("\nSuggested next steps:")
    print("- Inspect the sample rows above to refine parsing of session structure.")
    print("- If you can find or create a label column, rerun RQ2.")
    print("- Consider saving a small subset to disk for manual annotation.")

if __name__ == "__main__":
    main()
