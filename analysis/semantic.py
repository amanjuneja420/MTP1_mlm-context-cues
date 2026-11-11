"""Semantic relatedness of MLM predictions (Kalyaeva et al. 2021 style), using all-MiniLM-L6-v2 (SBERT)."""
import sys, os, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(os.environ.get('MINILM', 'sentence-transformers/all-MiniLM-L6-v2')); mdl = AutoModel.from_pretrained(os.environ.get('MINILM', 'sentence-transformers/all-MiniLM-L6-v2')).eval()
_cache = {}
def emb(words):
    new = [w for w in dict.fromkeys(words) if w not in _cache]
    for i in range(0, len(new), 256):
        b = new[i:i+256]; e = tok(b, padding=True, return_tensors='pt')
        with torch.no_grad(): h = mdl(**e).last_hidden_state
        m = e['attention_mask'].unsqueeze(-1); v = (h*m).sum(1)/m.sum(1); v = torch.nn.functional.normalize(v, dim=-1)
        for w, x in zip(b, v.numpy()): _cache[w] = x
    return np.stack([_cache[w] for w in words])
DOM = {'GENERAL ENGLISH': 'everyday life', 'MEDICINE': 'medicine', 'LAW': 'law', 'FINANCE': 'finance',
       'CLIMATE SCIENCE': 'climate science', 'CYBERSECURITY': 'cybersecurity', 'EDUCATION': 'education',
       'BIOLOGY': 'biology', 'TECHNOLOGY': 'technology', 'GEOGRAPHY': 'geography'}
def analyse(final_csv):
    f = pd.read_csv(final_csv)
    keys = list(DOM); D = emb([DOM[k] for k in keys])
    clean = lambda w: str(w).replace('##', '').strip().lower()
    out = []
    for _, r in f.iterrows():
        top = [clean(w) for w in str(r.top10).split(', ')]
        T = emb([clean(r.target)]); P = emb(top)
        rel = P @ T[0]
        dom_of = lambda V: [keys[i] for i in (V @ D.T).argmax(1)]
        out.append(dict(domain=r.domain, idx=r.idx, hit1=r.hit1,
                        rel_top1=float(rel[0]), rel_top10_mean=float(rel.mean()),
                        target_in_domain=dom_of(T)[0] == r.domain,
                        top1_in_domain=dom_of(P[:1])[0] == r.domain,
                        top10_in_domain=float(np.mean([d == r.domain for d in dom_of(P)]))))
    return pd.DataFrame(out)
if __name__ == '__main__':
    s = analyse(sys.argv[1])
    print(s.drop(columns=['idx']).groupby('domain').mean(numeric_only=True).round(3).to_string())
    w = s[~s.hit1]
    print('\nWRONG top-1 only (n=%d): rel_top1 %.3f, top1_in_domain %.3f, target_in_domain %.3f' % (len(w), w.rel_top1.mean(), w.top1_in_domain.mean(), w.target_in_domain.mean()))
    print('ALL: top10_in_domain %.3f' % s.top10_in_domain.mean())
