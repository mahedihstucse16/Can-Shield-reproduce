# -*- coding: utf-8 -*-
"""OTIDS impersonation label audit. Usage: python3 -I audit_impersonation.py <capture.txt>"""
import re, sys
pat = re.compile(r'Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)')
t0 = None; n = bad = 0
pre = post = 0; pre164 = post164 = 0; last = None
ids_pre, ids_post = set(), set()
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    m = pat.search(line)
    if not m:
        bad += 1; continue
    t = float(m.group(1)); i = int(m.group(2), 16)
    if t0 is None: t0 = t
    last = t; n += 1
    if t - t0 < 250.0:
        pre += 1; ids_pre.add(i); pre164 += (i == 0x164)
    else:
        post += 1; ids_post.add(i); post164 += (i == 0x164)
dur = last - t0
print('frames', n, 'unparsed', bad, 'duration %.1f s' % dur)
print('before 250 s: %d frames, 0x164 %d, %.3f /s' % (pre, pre164, pre164 / 250.0))
print('after  250 s: %d frames, 0x164 %d, %.3f /s' % (post, post164, post164 / (dur - 250.0)))
print('ids before %d, after %d, new after %s' % (len(ids_pre), len(ids_post),
      sorted(hex(x) for x in ids_post - ids_pre)))
