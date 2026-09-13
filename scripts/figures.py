import json
import math
from pathlib import Path

import networkx as nx
from xml.sax.saxutils import escape

BG = "#0f172a"
PANEL = "#1b2336"
FG = "#f8fafc"
DIM = "#94a3b8"
GRID = "#1e293b"
ACCENT = "#22d3ee"
DANGER = "#f43f5e"
MONO = "font-family='Fira Code, ui-monospace, monospace'"

METHOD_COLORS = {
    "gnn": "#22d3ee",
    "gnn_shuffled": "#94a3b8",
    "ocsvm": "#f472b6",
    "mlp_ae": "#facc15",
}

EDGE_COLORS = {
    "EXEC": "#7dd3fc",
    "READ": "#4ade80",
    "WRITE": "#facc15",
    "CONNECT": "#818cf8",
    "UNLINK": "#fb923c",
    "ACCEPT": "#c084fc",
}


def _svg_open(w, h, body_classes=""):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">'
        f'<rect width="{w}" height="{h}" fill="{BG}"/>'
    )


def _svg_close():
    return "</svg>"


def _axes(x0, y0, w, h, label):
    parts = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="{PANEL}" rx="6"/>']
    parts.append(
        f'<text x="{x0 + 12}" y="{y0 + 18}" fill="{DIM}" font-size="11" {MONO}>{label}</text>'
    )
    parts.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{GRID}"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="{GRID}"/>')
    return parts


def _error_bar(cx, mean, std, scale, ax_top, color):
    y = ax_top - mean * scale
    y_hi = ax_top - (mean + std) * scale
    y_lo = ax_top - max(0.0, mean - std) * scale
    return [
        f'<line x1="{cx}" y1="{y_lo}" x2="{cx}" y2="{y_hi}" stroke="{color}" stroke-width="2"/>',
        f'<line x1="{cx - 4}" y1="{y_hi}" x2="{cx + 4}" y2="{y_hi}" stroke="{color}" stroke-width="2"/>',
        f'<line x1="{cx - 4}" y1="{y_lo}" x2="{cx + 4}" y2="{y_lo}" stroke="{color}" stroke-width="2"/>',
    ]


def _baseline(x0, y0, w, h, aux):
    parts = [f'<line x1="{x0}" y1="{y0 + aux}" x2="{x0 + w}" y2="{y0 + aux}" stroke="{DIM}" stroke-dasharray="4 4"/>']
    parts.append(f'<text x="{x0 + w}" y="{y0 + aux - 4}" fill="{DIM}" font-size="9" text-anchor="end" {MONO}>0.5</text>')
    return parts


def results_svg(summary, seeds, epochs, w=880, h=460):
    rows = {}
    for key, value in summary.items():
        space = key.rfind(" ")
        method = key[:space]
        metric = key[space + 1 :]
        rows.setdefault(method, {})[metric] = value
    methods = sorted(rows)

    out = [_svg_open(w, h)]
    out.append(
        f'<text x="20" y="26" fill="{FG}" font-size="15" font-weight="600" {MONO}>'
        f"Comparativa de métodos</text>"
    )
    out.append(
        f'<text x="20" y="44" fill="{DIM}" font-size="11" {MONO}>'
        f"{seeds} seeds · GAE 50 épocas · barra = media ± desviación</text>"
    )
    if len(methods) > 4:
        out.append(
            f'<text x="20" y="60" fill="{DANGER}" font-size="10" {MONO}>'
            f"⚠ hay {len(methods)} métodos; se muestran las 4 primeras barras</text>"
        )
        methods = methods[:4]

    meta = [("auc_roc", "AUC ROC", 5, 6), ("recall_at_1pct", "Recall@1%", 5, 6)]
    panels = [(70, 70, 360, 300, meta[0]), (460, 70, 360, 300, meta[1])]

    for x0, y0, pw, ph, (metric, label, max_y, minor) in panels:
        out.extend(_axes(x0, y0, pw, ph, label))
        scale = (ph - 30) / max_y
        ax_top = y0 + ph - 30
        out.extend(_baseline(x0 + 44, y0, pw - 44, ph, (ph - 30) / 2))
        n = len(methods)
        slot = (pw - 44) / n
        for i, method in enumerate(methods):
            m = rows[method].get(metric)
            if m is None:
                continue
            cx = x0 + 44 + slot * (i + 0.5)
            bw = min(30.0, slot * 0.42)
            bar_h = m["mean"] * scale
            color = METHOD_COLORS.get(method, "#818cf8")
            out.append(
                f'<rect x="{cx - bw / 2}" y="{ax_top - bar_h}" width="{bw}" height="{max(1.5, bar_h)}" '
                f'fill="{color}" opacity="0.9" rx="2"/>'
            )
            out.extend(_error_bar(cx, m["mean"], m["std"], scale, ax_top, color))
            if bar_h < 26 or method in ("gnn_shuffled",):
                ty = ax_top - bar_h - 22
            else:
                ty = ax_top - bar_h + 16
            out.append(
                f'<text x="{cx}" y="{ty:g}" fill="{FG}" font-size="10" text-anchor="middle" {MONO}>'
                f"{m['mean']:.2f}</text>"
            )
            use_method = method.replace("_", " ")
            out.append(
                f'<text x="{cx}" y="{ax_top + 20}" fill="{DIM}" font-size="9" text-anchor="middle" {MONO}>'
                f"{escape(use_method)}</text>"
            )
    out.append(_svg_close())
    return "\n".join(out)


def _kind_shape(kind, cx, cy, r, fill):
    if kind == "file":
        w, h = r * 1.7, r * 1.3
        return (
            f'<rect x="{cx - w / 2:g}" y="{cy - h / 2:g}" width="{w:g}" height="{h:g}" '
            f'rx="2" fill="{fill}" opacity="0.92"/>'
        )
    if kind == "net":
        return (
            f'<path d="M{cx},{cy - r * 1.4} L{cx + r * 1.4},{cy} L{cx},{cy + r * 1.4} '
            f'L{cx - r * 1.4},{cy} Z" fill="{fill}" opacity="0.92"/>'
        )
    return f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{fill}" opacity="0.92"/>'


def graph_svg(snapshot, title, w=940, h=560):
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    n_anom = sum(1 for n in nodes if n.get("anomalous"))
    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"])
    for e in edges:
        g.add_edge(e["src"], e["dst"])
    pos = nx.spring_layout(g, k=0.62, iterations=90, seed=7)
    pad_x, pad_y = 70, 70
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    scale = min((w - 2 * pad_x) / span_x, (h - 2 * pad_y) / span_y)
    for nid in pos:
        pos[nid] = (
            pad_x + (pos[nid][0] - min(xs)) * scale + (w - 2 * pad_x - span_x * scale) / 2,
            pad_y + (pos[nid][1] - min(ys)) * scale + (h - 2 * pad_y - span_y * scale) / 2,
        )

    out = [_svg_open(w, h)]
    out.append(
        f'<text x="20" y="26" fill="{FG}" font-size="15" font-weight="600" {MONO}>'
        f"Grafo de procedencia anotado</text>"
    )
    out.append(
        f'<text x="20" y="44" fill="{DIM}" font-size="11" {MONO}>'
        f"{escape(title)} · {len(nodes)} nodos · {len(edges)} aristas · {n_anom} anómalos</text>"
    )
    by_id = {n["id"]: n for n in nodes}
    radii = {
        n["id"]: 3.0 + 6.0 * max(0.0, min(1.0, n.get("score", 0.0))) for n in nodes
    }
    for e in edges:
        sx, sy = pos[e["src"]]
        tx, ty = pos[e["dst"]]
        color = EDGE_COLORS.get(e["label"], "#64748b")
        r_s = radii.get(e["src"], 0)
        r_t = radii.get(e["dst"], 0)
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1.0
        sx2 = sx + dx / length * (r_s + 3.2)
        sy2 = sy + dy / length * (r_s + 3.2)
        tx2 = tx - dx / length * (r_t + 3.2)
        ty2 = ty - dy / length * (r_t + 3.2)
        out.append(
            f'<line x1="{sx2:g}" y1="{sy2:g}" x2="{tx2:g}" y2="{ty2:g}" '
            f'stroke="{color}" stroke-opacity="0.65" stroke-width="1.4"/>'
        )

    for n in nodes:
        cx = pos[n["id"]][0]
        cy = pos[n["id"]][1]
        r = radii[n["id"]]
        anom = n.get("anomalous")
        color = DANGER if anom else ACCENT
        if anom:
            out.append(
                f'<circle cx="{cx:g}" cy="{cy:g}" r="{r + 5:g}" fill="none" '
                f'stroke="{DANGER}" stroke-opacity="0.55" stroke-width="1.5"/>'
            )
        kind = n.get("kind", "process")
        out.append(_kind_shape(kind, cx, cy, r, color))
        out.append(
            f'<text x="{cx:g}" y="{cy + r + 12:g}" fill="{DANGER if anom else DIM}" '
            f'font-size="9" text-anchor="middle" {MONO}>{escape(n.get("label") or n["id"])}</text>'
        )

    lx, ly = w - 70, h - 78
    out.append(f'<text x="{lx}" y="{ly}" fill="{DIM}" font-size="10" {MONO}>nodo</text>')
    out.append(f'<circle cx="{lx - 62}" cy="{ly - 4}" r="5" fill="{ACCENT}"/>')
    out.append(f'<text x="{lx}" y="{ly + 18}" fill="{DIM}" font-size="10" {MONO}>anómalo</text>')
    out.append(
        f'<circle cx="{lx - 62}" cy="{ly + 14}" r="5" fill="{DANGER}" '
        f'stroke="{DANGER}" stroke-width="2"/>'
    )
    out.append(f'<text x="{lx - 14}" y="{ly + 34}" fill="{DIM}" font-size="10" {MONO}>aristas</text>')
    e_legend = EDGE_COLORS
    ly = ly + 52
    for label, color in e_legend.items():
        out.append(f'<line x1="{lx - 80}" y1="{ly}" x2="{lx - 40}" y2="{ly}" stroke="{color}" stroke-opacity="0.8"/>')
        out.append(f'<text x="{lx - 32}" y="{ly + 3}" fill="{DIM}" font-size="9" {MONO}>{label}</text>')
        ly += 15
    out.append(_svg_close())
    return "\n".join(out)


def generate_figures(report_path, annotated_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(report_path).read_text())
    seeds = len(report.get("per_seed", [])) // 4
    render_results = results_svg(report.get("summary", {}), seeds, 50)
    (out / "results.svg").write_text(render_results)

    ann = Path(annotated_dir)
    files = sorted(ann.glob("*.json")) if ann.is_dir() else []
    pick = [f for f in files if "_" in f.name]
    attack = [f for f in pick if f.name.startswith("attack")]
    chosen = (attack or pick or files)[0] if files else None
    if chosen:
        data = json.loads(chosen.read_text())
        last = data["snapshots"][-1] if data.get("snapshots") else {"nodes": [], "edges": []}
        title = f"run {data.get('run_id', chosen.stem)} · snapshot final"
        (out / "graph.svg").write_text(graph_svg(last, title))
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Genera figuras SVG del README")
    p.add_argument("--report", required=True, help="ruta a report.json")
    p.add_argument("--annotations", required=True, help="directorio de anotaciones")
    p.add_argument("--output", default="assets", help="directorio de salida")
    args = p.parse_args()
    generate_figures(args.report, args.annotations, args.output)
    print("figuras generadas en", args.output)