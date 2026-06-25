#!/usr/bin/env python3
"""HS-01 — render the frozen pool to a self-contained HTML quality report.

Reads pool_frozen/itempool.json and writes pool_frozen/report.html. Open it in a
browser (it lives next to assets/ so relative image paths resolve). Groups items
by phase -> stratum, sorts closest-boundary-first, shows each image + the VERBATIM
prompt (non-ASCII / homoglyph chars highlighted) + quality metrics.
"""
import html
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pool_frozen"
pool = json.loads((OUT / "itempool.json").read_text())
src_by = {s["source_id"]: s for s in pool["sources"]}

PHASES = [
    ("text", ["clean", "low_drift", "medium_drift", "high_drift"]),
    ("image", ["raw", "roundtrip", "boundary_joint", "image_heavy"]),
    ("pair", ["baseline", "image_heavy", "text_heavy", "balanced"]),
]
groups, attn = defaultdict(list), []
for it in pool["items"]:
    s = src_by[it["source_id"]]
    if it.get("is_attention_check"):
        attn.append((it, s)); continue
    groups[(it["kind"], s["strata"][it["kind"]])].append((it, s))


def hl(t):
    """Verbatim text with non-ASCII chars highlighted (homoglyphs / odd unicode)."""
    return "".join(f'<span class="hg" title="U+{ord(c):04X}">{html.escape(c)}</span>'
                   if ord(c) > 127 else html.escape(c) for c in (t or ""))


def fmt(v, sci=True):
    if v is None:
        return "—"
    return f"{v:.1e}" if sci else f"{v:.3f}"


def tb_cls(v):
    if v is None:
        return ""
    return "g" if v <= 1e-4 else ("g2" if v <= 1e-3 else ("a" if v <= 1e-2 else "r"))


def tg_cls(v):
    if v is None:
        return ""
    return "g" if v <= 4 else ("a" if v <= 7 else "r")


def card(it, s):
    a = s["assets"]; sr = s.get("search") or {}; dr = s.get("drift") or {}
    cell = s.get("cell") or {}
    img = a.get("image")
    imghtml = (f'<a href="{img["uri"]}" target="_blank"><img src="{img["uri"]}" '
               f'loading="lazy"><span class="dim">{img["width"]}×{img["height"]}</span></a>'
               if img else '<div class="noimg">text only</div>')
    p = a.get("prompt")
    prompt = f'<div class="prompt">{hl(p["text"])}</div>' if p else ""
    orig = (f'<div class="orig">clean: {html.escape(p["original_text"])}</div>'
            if p and p.get("original_text") else "")
    opts = ""
    if it["kind"] == "pair" and cell:
        opts = (f'<div class="opts"><b>A</b> {html.escape(cell["anchor_word"])} '
                f'· <b>B</b> {html.escape(cell["target_word"])} '
                f'<span class="exp">(expect A)</span></div>')
    tb = sr.get("tgtbal"); di = dr.get("d_img"); tg = dr.get("active_text_genes")
    sut = (s.get("sut") or {}).get("model_id", "")
    sut = sut.split("/")[-1] if sut else "—"
    badges = []
    if tb is not None:
        badges.append(f'<span class="b {tb_cls(tb)}">|lpA−lpB| {fmt(tb)}</span>')
    if di is not None:
        badges.append(f'<span class="b">Δimg {fmt(di, sci=False)}</span>')
    if tg is not None:
        badges.append(f'<span class="b {tg_cls(tg)}">{tg} text edits</span>')
    if sr.get("modality"):
        badges.append(f'<span class="b">{sr["modality"]}</span>')
    badges.append(f'<span class="b sut">{html.escape(sut)}</span>')
    if cell.get("anchor_class"):
        badges.append(f'<span class="b pr">{html.escape(cell["anchor_class"])}→'
                      f'{html.escape(cell["target_class"])}</span>')
    chk = (f'<span class="b chk">ATTENTION: {it["check_rule"]["metric"]} '
           f'{html.escape(str(it["check_rule"]["value"]))}</span>'
           if it.get("is_attention_check") else "")
    return (f'<div class="card">{imghtml}<div class="meta">{opts}{prompt}{orig}'
            f'<div class="badges">{"".join(badges)}{chk}</div>'
            f'<div class="sid">{html.escape(s["source_id"])}</div></div></div>')


# summary rows
sumrows = []
for phase, strata in PHASES:
    for st in strata:
        items = groups.get((phase, st), [])
        tbs = [ (i[1].get("search") or {}).get("tgtbal") for i in items ]
        tbs = [t for t in tbs if t is not None]
        suts = sorted({(i[1].get("sut") or {}).get("model_id", "ctrl").split("/")[-1] for i in items})
        rng = f"{min(tbs):.1e}–{max(tbs):.1e}" if tbs else "—"
        sumrows.append(f"<tr><td>{phase}/{st}</td><td>{len(items)}</td>"
                       f"<td>{rng}</td><td>{','.join(suts)}</td></tr>")

CSS = """
:root{--bd:#e2e2e6;--mut:#777;--bg:#fafafa}
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:var(--bg)}
header{padding:20px 28px;background:#fff;border-bottom:1px solid var(--bd)}
h1{font-size:20px;margin:0 0 4px}h2{font-size:17px;margin:26px 28px 10px;padding-top:8px}
.sub{color:var(--mut);font-size:13px}.legend{font-size:12px;color:var(--mut);margin-top:8px}
.legend .hg{padding:0 1px}
nav{padding:10px 28px;background:#fff;border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:5;font-size:12px}
nav a{margin-right:12px;color:#2456c8;text-decoration:none}
table.sum{margin:14px 28px;border-collapse:collapse;font-size:13px;background:#fff}
table.sum td,table.sum th{border:1px solid var(--bd);padding:4px 10px;text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px;padding:0 28px}
.card{background:#fff;border:1px solid var(--bd);border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.card img{width:100%;display:block;background:#eee}
.card a{position:relative;display:block}
.dim{position:absolute;right:4px;bottom:4px;background:rgba(0,0,0,.6);color:#fff;font-size:10px;padding:1px 5px;border-radius:3px}
.noimg{padding:18px;text-align:center;color:var(--mut);background:#f3f3f5;font-style:italic}
.meta{padding:9px 11px}
.opts{font-size:12.5px;margin-bottom:5px}.opts .exp{color:var(--mut)}
.prompt{font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;background:#f7f7f9;border:1px solid #ededf0;border-radius:5px;padding:6px 8px}
.orig{font-size:11.5px;color:var(--mut);margin-top:4px;font-style:italic}
.badges{margin-top:7px;display:flex;flex-wrap:wrap;gap:4px}
.b{font-size:11px;padding:1px 6px;border-radius:10px;background:#eef;border:1px solid #dde}
.b.g{background:#d8f5d8;border-color:#b3e6b3}.b.g2{background:#eaf7d8;border-color:#cfe9a8}
.b.a{background:#fdf0cf;border-color:#f0d999}.b.r{background:#fadbd8;border-color:#f0b3ab}
.b.sut{background:#e8e8ef}.b.pr{background:#eef3fb;border-color:#cdddf5}
.b.chk{background:#ffe0b2;border-color:#f0c070;font-weight:600}
.hg{background:#ffd2d2;border-radius:2px;outline:1px solid #f3a3a3}
.sid{font-size:10px;color:#aaa;margin-top:6px;font-family:ui-monospace,monospace}
"""

parts = [f"<!doctype html><meta charset=utf-8><title>HS-01 frozen pool</title><style>{CSS}</style>"]
parts.append(f"<header><h1>HS-01 — Frozen Stimulus Pool</h1>"
             f"<div class='sub'>{pool['pool_id']} · created {pool['created']} · "
             f"{len(pool['sources'])} sources · {len(pool['items'])} items · frozen={pool['frozen']}</div>"
             f"<div class='legend'>Sorted closest-boundary-first. "
             f"<b>|lpA−lpB|</b> = boundary closeness (lower = tighter). "
             f"<b>Δimg</b> = image change vs seed. <b>text edits</b> = active text genes "
             f"(green ≤4 / amber ≤7). Highlighted <span class='hg'>chars</span> = non-ASCII "
             f"(homoglyphs / odd unicode) — hover for codepoint. Click an image for full size.</div></header>")
nav = " ".join(f"<a href='#{p}-{st}'>{p[:3]}/{st}</a>" for p, ss in PHASES for st in ss)
parts.append(f"<nav>{nav} <a href='#attn'>checks</a></nav>")
parts.append("<table class='sum'><tr><th>phase/stratum</th><th>n</th>"
             "<th>tgtbal range</th><th>SUTs</th></tr>" + "".join(sumrows) + "</table>")

for phase, strata in PHASES:
    for st in strata:
        items = sorted(groups.get((phase, st), []),
                       key=lambda x: ((x[1].get("search") or {}).get("tgtbal") or 0))
        parts.append(f"<h2 id='{phase}-{st}'>{phase} / {st} "
                     f"<span class='sub'>({len(items)} items)</span></h2>")
        parts.append("<div class='grid'>" + "".join(card(it, s) for it, s in items) + "</div>")

parts.append("<h2 id='attn'>attention checks</h2><div class='grid'>"
             + "".join(card(it, s) for it, s in attn) + "</div>")

(OUT / "report.html").write_text("".join(parts))
print(f"wrote {OUT/'report.html'} ({(OUT/'report.html').stat().st_size/1024:.0f} KB)")
print(f"open: file://{OUT/'report.html'}")
