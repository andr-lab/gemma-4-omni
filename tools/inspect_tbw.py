import sys
sys.path.insert(0, '/workspace')
import kaggle_pipeline as KP

user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
print(f"tbw type: {type(tbw)}, len: {len(tbw) if tbw is not None else 'None'}")
if tbw:
    for idx in [0, 1, 2, 10, 100]:
        if idx < len(tbw):
            print(f"tbw[{idx}] (type {type(tbw[idx])}): {repr(tbw[idx])}")
