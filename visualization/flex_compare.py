# visualization/flex_compare.py
from __future__ import annotations
import os, re, json, math, unicodedata
from typing import Any, Dict, List, Tuple, Callable, Optional, Sequence
import matplotlib.pyplot as plt

# ── Color mapping ─────────────────────────────────────────────────────────────
_TAB_COLORS = plt.get_cmap("tab10").colors

def _color_map(run_ids: List[str]):
    """Stable color per run id."""
    cmap = {}
    for i, rid in enumerate(run_ids):
        cmap[rid] = _TAB_COLORS[i % len(_TAB_COLORS)]
    return cmap

# ── Filesystem helpers ────────────────────────────────────────────────────────
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_summary(runs_dir: str, run_id: str) -> Dict[str, Any]:
    path = os.path.join(runs_dir, run_id, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary.json for run '{run_id}' at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _list_run_ids(runs_dir: str) -> List[str]:
    if not os.path.exists(runs_dir):
        return []
    return [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]

# ── Slugging ──────────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "comparison"

def _comparison_dir(base_dir: str, name: str) -> str:
    return os.path.join(base_dir, "comparisons", _slugify(name))

# ── Run selectors ────────────────────────────────────────────────────────────
def resolve_run_ids(runs_dir: str, selectors: Sequence[str]) -> List[str]:
    all_ids = set(_list_run_ids(runs_dir))
    chosen: List[str] = []
    for sel in selectors:
        if sel.startswith("re:"):
            pat = sel[3:]
            rx = re.compile(pat)
            for rid in sorted(all_ids):
                if rx.search(rid) and rid not in chosen:
                    chosen.append(rid)
        else:
            if sel in all_ids and sel not in chosen:
                chosen.append(sel)
    return chosen

# ── Metric helpers ───────────────────────────────────────────────────────────
def _get_from_path(obj: Any, path: str, default=None):
    if path == "run_id":
        return obj.get("summary", {}).get("run_id")
    if '.' not in path:
        return obj.get(path) or obj.get("summary", {}).get(path)
    cur = obj.get("summary", {})
    for part in path.split('.'):
        if part == "history":
            cur = cur.get("history", {})
        else:
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                return default
    return cur

def _epoch_series(history: Dict[Any, Dict[str, Any]], key: str) -> Tuple[List[int], List[Any]]:
    if not history: return [], []
    epochs = sorted(int(e) for e in history.keys())
    ys = [(history.get(str(e)) or history.get(e) or {}).get(key) for e in epochs]
    return epochs, ys

def _label_for_run(run_id: str, name_map: Optional[Dict[str,str]]) -> str:
    return (name_map or {}).get(run_id, run_id)

def _resolve_scalar(run_info: Dict[str,Any], key: str,
                    custom_funcs: Dict[str, Callable[[Dict[str,Any]], Any]]):
    if key.startswith("func:"):
        fn = custom_funcs.get(key.split(":",1)[1])
        val = fn(run_info) if fn else None
    else:
        if key.startswith("history."):
            sub = key.split(".",1)[1]
            hist = run_info["summary"].get("history",{}) or {}
            _, ys = _epoch_series(hist, sub)
            val = ys[-1] if ys else None
        else:
            val = _get_from_path(run_info, key)

    # convert list -> string
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return str(val[0])
        return ",".join(str(v) for v in val)

    return val

# ── Plotting ─────────────────────────────────────────────────────────────────
def _render_panels(out_dir: str,
                   panels: List[Dict[str, Any]],
                   runs_info: List[Dict[str,Any]],
                   name_map: Optional[Dict[str,str]],
                   style: Optional[Dict[str,Any]],
                   custom_funcs: Dict[str, Callable[[Dict[str,Any]], Any]],
                   color_map: Dict[str,Any]) -> List[str]:

    style = style or {}
    dpi = int(style.get("dpi", 140))
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    figsize = style.get("figsize", [6 * cols, 4 * rows])

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes = [axes] if rows * cols == 1 else list(axes.flat)
    img_paths = []

    for i, spec in enumerate(panels):
        ax = axes[i]
        x_key = spec.get("x")
        y_key = spec.get("y")
        plot_type = (spec.get("plot") or "").lower()
        title = spec.get("title") or f"{y_key} vs {x_key}"

        if not x_key or not y_key:
            raise ValueError("Each panel must specify both 'x' and 'y'.")

        # -------- PER-EPOCH LINE PLOT --------
        if x_key == "epoch":
            if plot_type and plot_type != "line":
                print(f"[flex_compare] Warning: Overriding plot='{plot_type}' to 'line' due to x='epoch'.")

            for r in runs_info:
                rid = r["id"]
                hist = r["summary"].get("history", {}) or {}
                metric = y_key.split(".",1)[1] if y_key.startswith("history.") else y_key
                xs, ys = _epoch_series(hist, metric)
                if ys and any(v is not None for v in ys):
                    ax.plot(xs, ys, linewidth=2, color=color_map[rid])

            ax.set_xlabel("epoch")
            ax.set_ylabel(y_key)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

        # -------- PER-RUN (SCALAR) PLOTS --------
        else:
            xs, ys = [], []
            run_order = []

            for r in runs_info:
                rid = r["id"]
                xv = _resolve_scalar(r, x_key, custom_funcs)
                yv = _resolve_scalar(r, y_key, custom_funcs)
                if yv is None:
                    continue
                xs.append(xv)
                ys.append(yv)
                run_order.append(rid)

            # ----- BAR -----
            if plot_type == "bar":
                bar_colors = [color_map[rid] for rid in run_order]
                ax.bar(run_order, ys, color=bar_colors)
                ax.set_xlabel("run")
                ax.set_ylabel(y_key)
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.25)

            # ----- SCATTER -----
            else:
                for xv, yv, rid in zip(xs, ys, run_order):
                    ax.scatter([xv], [yv], s=70, color=color_map[rid])
                ax.set_xlabel(x_key)
                ax.set_ylabel(y_key)
                ax.set_title(title)
                ax.grid(True, alpha=0.25)

    # Remove empty axes if grid > panels
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # -------- GLOBAL LEGEND --------
    handles = []
    labels = []
    for rid in color_map:
        handles.append(plt.Line2D([0],[0], color=color_map[rid], lw=3, marker='o'))
        labels.append(_label_for_run(rid, name_map))

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),   # big bottom spacing
        ncol=min(len(labels), 6),
        frameon=False,
        fontsize=10
    )

    # Extra space for global legend
    fig.tight_layout(rect=[0, 0.15, 1, 1])

    out_name = "panels.png" if n > 1 else f"{panels[0]['y']}_vs_{panels[0]['x']}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")  # ensure legend not cut
    plt.close(fig)

    img_paths.append(out_path)
    return img_paths


# ── CSV helpers ──────────────────────────────────────────────────────────────
def _iter_rows(runs_info: List[Dict[str,Any]], spread_epochs: bool) -> List[Dict[str,Any]]:
    rows=[]
    for r in runs_info:
        summary = r["summary"]
        base_ctx={
            "run_id": summary.get("run_id"),
            "status": summary.get("status"),
            "config": summary.get("config", {}) or {},
            "final": summary.get("final", {}) or {},
            "meta": summary.get("meta", {}) or {},
            "summary": summary,
        }
        hist = summary.get("history", {}) or {}
        if spread_epochs and hist:
            for e in sorted(int(e) for e in hist.keys()):
                d = hist.get(str(e)) or hist.get(e) or {}
                ctx = dict(base_ctx)
                ctx.update({"epoch":e,"timestamp":d.get("timestamp"),"history_row":d})
                rows.append({"__run_info__":r,"__ctx__":ctx})
        else:
            rows.append({"__run_info__":r,"__ctx__":base_ctx})
    return rows

def _value_from_column_spec(col, ctx, run_info, custom_funcs):
    if "template" in col:
        try: return col["template"].format(**ctx)
        except: return None
    if "func" in col:
        fn = custom_funcs.get(col["func"])
        return fn(run_info) if fn else None
    if "value" in col:
        key = col["value"]
        if key == "pretty_name":
            return _label_for_run(run_info["id"], run_info.get("name_map"))
        if key == "epoch": return ctx.get("epoch")
        if key.startswith("history."):
            sub = key.split(".",1)[1]
            hr = ctx.get("history_row")
            if hr: return (hr or {}).get(sub)
            hist = run_info["summary"].get("history", {}) or {}
            _, ys = _epoch_series(hist, sub)
            return ys[-1] if ys else None
        return _deep_get_with_ctx(run_info, ctx, key)
    return None

def _deep_get_with_ctx(run_info, ctx, key):
    if key in ctx: return ctx[key]
    cur = run_info.get("summary", {})
    for part in key.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur

def write_csv(out_path_stem, columns, runs_info, spread_epochs, custom_funcs):
    out_path = out_path_stem + ".csv" if not out_path_stem.endswith(".csv") else out_path_stem
    _ensure_dir(os.path.dirname(out_path))
    rows = _iter_rows(runs_info, spread_epochs)
    headers = [c["name"] for c in columns]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            run_info = row["__run_info__"]
            ctx = row["__ctx__"]
            vals=[]
            for col in columns:
                v = _value_from_column_spec(col, ctx, run_info, custom_funcs)
                vals.append("" if v is None else str(v).replace("\n"," ").replace(",", ";"))
            f.write(",".join(vals) + "\n")
    return out_path

# ── Orchestrator ─────────────────────────────────────────────────────────────
def _collect_runs(base_dir, selectors, name_map):
    runs_dir = os.path.join(base_dir, "runs")
    ids = resolve_run_ids(runs_dir, selectors)
    out=[]
    for rid in ids:
        summary = _load_summary(runs_dir, rid)
        out.append({"id":rid,"summary":summary,"name_map":name_map or {}})
    return out

def run_comparisons(base_dir, comparisons, custom_funcs=None):
    custom_funcs = custom_funcs or {}
    results={}
    for comp in comparisons:
        comp_name = comp.get("name","comparison")
        comp_dir = _comparison_dir(base_dir, comp_name)
        _ensure_dir(comp_dir)

        selectors = comp.get("runs",[])
        name_map = comp.get("name_map",{})
        runs_info = _collect_runs(base_dir, selectors, name_map)

        # stable colors
        run_ids = [r["id"] for r in runs_info]
        color_map = _color_map(run_ids)

        dest = comp.get("dest",{})
        dtype = dest.get("type")
        produced={}

        inputs_json = os.path.join(comp_dir,"inputs.json")
        with open(inputs_json,"w",encoding="utf-8") as f:
            json.dump(comp,f,indent=2)
        produced["inputs"]=inputs_json

        if dtype=="plot":
            panels = dest.get("panels",[])
            style = dest.get("style")
            produced["plots"] = _render_panels(comp_dir, panels, runs_info, name_map, style, custom_funcs, color_map)

        elif dtype=="csv":
            file_stem = dest.get("file_stem","table")
            out_stem = os.path.join(comp_dir,file_stem)
            columns = dest.get("columns",[])
            spread_epochs = bool(dest.get("spread_epochs",False))
            produced["csv"] = write_csv(out_stem, columns, runs_info, spread_epochs, custom_funcs)

        else:
            produced["error"] = f"Unknown type {dtype}"

        results[comp_name]=produced

    return results
