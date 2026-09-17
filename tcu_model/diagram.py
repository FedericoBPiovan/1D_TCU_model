"""Vista statica dello schema a blocchi del circuito (stile P&ID/Simscape).

Non e' un editor grafico: e' una funzione che disegna, a partire dalla
topologia di un ``Circuit`` gia' assemblato, uno schema con i nodi (punti
di giunzione) e i blocchi dei componenti che li collegano, opzionalmente
annotato con portate/temperature di un punto di funzionamento risolto.

Layout: i nodi sono disposti su livelli verticali (BFS non orientata a
partire dal nodo di riferimento/serbatoio), coerente per la maggior parte
delle topologie TCU (serie con eventuali rami paralleli/bypass). Rami
paralleli tra la stessa coppia di nodi vengono automaticamente scostati
per non sovrapporsi.
"""

from __future__ import annotations

from collections import defaultdict, deque

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch

from .components import Heater, HeatExchanger, HydraulicResistance, Pipe, Pump

# Colori categorici (palette validata, ordine fisso per tipo di blocco:
# vedi skill "dataviz" - ordine mai ciclico, un colore per identita').
_BLOCK_STYLE = {
    Pump: ("#2a78d6", "Pompa"),
    Heater: ("#eb6834", "Riscaldatore"),
    HeatExchanger: ("#1baf7a", "Scambiatore"),
    HydraulicResistance: ("#eda100", "Valvola/resistenza"),
    Pipe: ("#e87ba4", "Tubazione"),
}
_DEFAULT_STYLE = ("#8a8a86", "Componente")

_INK = "#0b0b0b"
_MUTED = "#52514e"
_NODE_FILL = "#f0efec"
_NODE_EDGE = "#52514e"


def _block_style(component) -> tuple[str, str]:
    for cls, style in _BLOCK_STYLE.items():
        if isinstance(component, cls):
            return style
    return _DEFAULT_STYLE


def _layered_layout(circuit) -> dict[str, tuple[float, float]]:
    """BFS non orientata dal nodo di riferimento -> (livello, posizione
    verticale) per ogni nodo."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for br in circuit.branches:
        adjacency[br.from_node].append(br.to_node)
        adjacency[br.to_node].append(br.from_node)

    start = circuit.reference_node.name
    level = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in level:
                level[v] = level[u] + 1
                queue.append(v)
    for n in circuit.nodes:  # nodi isolati/non raggiunti: livello 0 di fallback
        level.setdefault(n, 0)

    by_level: dict[int, list[str]] = defaultdict(list)
    for n, lv in level.items():
        by_level[lv].append(n)

    scale = 2.4  # spaziatura tra nodi, abbastanza larga da ospitare i blocchi
    positions: dict[str, tuple[float, float]] = {}
    for lv, names in by_level.items():
        names.sort()
        n = len(names)
        for i, name in enumerate(names):
            y = (i - (n - 1) / 2.0)
            positions[name] = (float(lv) * scale, float(y) * scale)
    return positions


def draw_circuit(circuit, mdots: dict[str, float] | None = None,
                  node_temperatures: dict[str, float] | None = None,
                  ax=None, title: str | None = None):
    """Disegna lo schema a blocchi del circuito.

    Parameters
    ----------
    circuit : network.Circuit
        Il circuito assemblato (nodi + rami).
    mdots : dict[str, float], optional
        Portate risolte per ramo (es. output di ``circuit.solve_hydraulics()``):
        se fornite, orientano le frecce secondo il verso di flusso reale e
        annotano ogni blocco con la portata.
    node_temperatures : dict[str, float], optional
        Temperature dei nodi in K (es. output di ``circuit.thermal_step()``):
        se fornite, annotano ogni nodo con la temperatura in degC.
    """
    positions = _layered_layout(circuit)
    n_layers = len({p[0] for p in positions.values()})
    n_rows = len({p[1] for p in positions.values()})

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, 2.6 * n_layers),
                                         max(4.5, 1.8 * n_rows)))

    # --- rami: linea nodo -> blocco -> nodo, con offset per rami paralleli ---
    parallel_groups: dict[frozenset, list] = defaultdict(list)
    for br in circuit.branches:
        parallel_groups[frozenset((br.from_node, br.to_node))].append(br)

    box_w, box_h = 1.05, 0.38
    for group in parallel_groups.values():
        n = len(group)
        for i, br in enumerate(group):
            x0, y0 = positions[br.from_node]
            x1, y1 = positions[br.to_node]
            dx, dy = x1 - x0, y1 - y0
            length = max((dx**2 + dy**2) ** 0.5, 1e-9)
            perp = (-dy / length, dx / length)
            offset = (i - (n - 1) / 2.0) * max(1.1, box_w * 1.3)
            cx = (x0 + x1) / 2.0 + perp[0] * offset
            cy = (y0 + y1) / 2.0 + perp[1] * offset

            forward = True
            if mdots is not None:
                forward = mdots.get(br.name, 0.0) >= 0.0
            tail, head = (x0, y0), (x1, y1)
            if not forward:
                tail, head = head, tail

            ax.plot([tail[0], cx], [tail[1], cy], color=_MUTED, lw=1.4, zorder=1)
            arrow_dx, arrow_dy = (head[0] - cx) * 0.92, (head[1] - cy) * 0.92
            ax.add_patch(FancyArrow(cx, cy, arrow_dx, arrow_dy, width=0.006,
                                     head_width=0.10, head_length=0.10,
                                     length_includes_head=True,
                                     color=_MUTED, zorder=1))

            color, type_label = _block_style(br.component)
            ax.add_patch(FancyBboxPatch(
                (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=color, edgecolor=_INK, linewidth=1.0, zorder=2))
            ax.text(cx, cy + 0.06, br.name, ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color="white", zorder=3)
            ax.text(cx, cy - 0.08, type_label, ha="center", va="center",
                     fontsize=6.5, color="white", zorder=3)

            annot = []
            if mdots is not None and br.name in mdots:
                annot.append(f"ṁ={abs(mdots[br.name]):.3f} kg/s")
            if annot:
                ax.text(cx, cy - box_h / 2 - 0.12, "\n".join(annot), ha="center",
                        va="top", fontsize=7, color=_MUTED, zorder=3)

    # --- nodi ---
    for name, (x, y) in positions.items():
        ax.add_patch(Circle((x, y), 0.13, facecolor=_NODE_FILL,
                             edgecolor=_NODE_EDGE, linewidth=1.2, zorder=4))
        label = name
        if node_temperatures is not None and name in node_temperatures:
            label += f"\n{node_temperatures[name] - 273.15:.1f} °C"
        ax.text(x, y + 0.32, label, ha="center", va="bottom", fontsize=8,
                color=_INK, zorder=4)

    # --- legenda categorica (colore = tipo di blocco) ---
    seen = {}
    for br in circuit.branches:
        color, type_label = _block_style(br.component)
        seen[type_label] = color
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=_INK)
               for c in seen.values()]
    ax.legend(handles, seen.keys(), loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=min(len(seen), 4),
              frameon=False, fontsize=8)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color=_INK, fontsize=11)

    return fig
