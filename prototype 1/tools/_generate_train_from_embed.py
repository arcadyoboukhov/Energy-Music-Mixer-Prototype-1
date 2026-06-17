#!/usr/bin/env python3
import runpy, csv, os, sys

# try to load the parse_nl_prompt module by path
p = os.path.join(os.path.dirname(__file__), 'parse_nl_prompt.py')
mod = runpy.run_path(p)
EX = mod.get('EMBED_EXAMPLES') or []
out = os.path.join(os.path.dirname(__file__), '_autogen_nl_train.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text','label'])
    for e in EX:
        if not isinstance(e, dict):
            continue
        text = e.get('text') or ''
        meta = e.get('meta') or {}
        labels = meta.get('preferred_genres') or []
        label = labels[0] if labels else ''
        if text:
            writer.writerow([text, label])

print('WROTE', out, 'rows=', sum(1 for _ in open(out, 'r', encoding='utf-8'))-1)
