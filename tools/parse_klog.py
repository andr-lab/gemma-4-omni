# -*- coding: utf-8 -*-
"""Parser dei log kernel Kaggle (lista JSON di record stream).
Uso: python _parse_klog.py <file.log> [<file.txt>]
"""
import sys
import re
import json


def parse_klog(raw: str) -> str:
    """Estrae il testo stdout+stderr dal log JSON di Kaggle (lista di record)."""
    # Il log e' una lista JSON: [{"stream_name": ..., "time": ..., "data": "..."}, ...]
    # Parsing robusto: regex sui record "data" con escape JSON.
    out = []
    # pattern: "data":"..." con escape JSON (\n \u003e \" \\ ...)
    pat = re.compile(r'"data":"((?:[^"\\]|\\.)*)"')
    for m in pat.finditer(raw):
        chunk = m.group(1)
        try:
            chunk = json.loads('"' + chunk + '"')
        except Exception:
            chunk = chunk.encode("utf-8", "replace").decode("unicode_escape", "replace")
        out.append(chunk)
    return "".join(out)


if __name__ == "__main__":
    src = sys.argv[1]
    raw = open(src, encoding="utf-8", errors="replace").read()
    text = parse_klog(raw)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(text)
        print(f"scritto {len(text)} char in {sys.argv[2]}")
    else:
        sys.stdout.write(text)
