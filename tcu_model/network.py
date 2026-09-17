"""Rete idraulica-termica: assembla i blocchi in un circuito e lo risolve.

Approccio "ibrido" (vedi discussione col progettista del modello):
- i blocchi (``components.py``) espongono porte fisiche (pressione,
  portata, temperatura) come farebbe Simscape;
- la rete pero' li risolve con un ordine causale esplicito, non con un
  assemblatore simbolico generico di equazioni:

  1. **Idraulica** (quasi-stazionaria): ad ogni passo di controllo si
     risolve un sistema algebrico non lineare generico (bilancio di massa
     ai nodi + equazione costitutiva pressione-portata di ogni ramo) via
     ``scipy.optimize.root``. Questa formulazione "nodale" funziona per
     qualsiasi topologia (serie, rami in parallelo, bypass, reti
     multi-loop) senza dover enumerare a mano gli anelli (a differenza del
     metodo di Hardy-Cross).
  2. **Termica** (dinamica): fissata la distribuzione di portata del
     punto 1, si integra nel tempo un sistema di ODE (una per ogni
     componente con massa termica propria + una per il serbatoio) usando
     ``scipy.integrate.solve_ivp``. Il mescolamento ai nodi puramente
     idraulici (senza massa termica) e' calcolato algebricamente
     percorrendo la rete in ordine topologico (dal serbatoio verso valle,
     secondo la direzione di flusso corrente).

Limite noto: la direzione di flusso e la topologia usata per il
mescolamento termico sono "congelate" per la durata di un passo di
controllo (coerente con l'ipotesi idraulica quasi-stazionaria); se la
portata si annulla o inverte segno all'interno del passo, l'integrazione
termica di quel passo resta un'approssimazione.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .components import Component, Tank
from .fluid_properties import FluidProperties


@dataclass
class Node:
    name: str
    reference: bool = False
    p_ref: float = 0.0
    tank: Tank | None = None
    T: float = 293.15


@dataclass
class Branch:
    name: str
    component: Component
    from_node: str
    to_node: str


class Circuit:
    """Contenitore del "netlist": nodi + rami, con i solutori idraulico e
    termico."""

    def __init__(self, fluid: FluidProperties):
        self.fluid = fluid
        self.nodes: dict[str, Node] = {}
        self.branches: list[Branch] = []
        self._last_hydraulic_x: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Assemblaggio topologia
    # ------------------------------------------------------------------
    def add_node(self, name: str, reference: bool = False, p_ref: float = 0.0,
                 tank: Tank | None = None, T0: float = 293.15) -> Node:
        if name in self.nodes:
            raise ValueError(f"Nodo '{name}' gia' definito")
        node = Node(name=name, reference=reference, p_ref=p_ref, tank=tank,
                    T=(tank.T if tank is not None else T0))
        self.nodes[name] = node
        return node

    def add_branch(self, name: str, component: Component, from_node: str,
                    to_node: str) -> Branch:
        for nm in (from_node, to_node):
            if nm not in self.nodes:
                raise ValueError(f"Nodo '{nm}' non definito (ramo '{name}')")
        branch = Branch(name=name, component=component, from_node=from_node,
                         to_node=to_node)
        self.branches.append(branch)
        return branch

    @property
    def reference_node(self) -> Node:
        refs = [n for n in self.nodes.values() if n.reference]
        if len(refs) != 1:
            raise ValueError("Lo scheletro attuale richiede esattamente un "
                              "nodo di riferimento (serbatoio)")
        return refs[0]

    # ------------------------------------------------------------------
    # Solutore idraulico: bilancio di massa ai nodi + equazioni di ramo
    # ------------------------------------------------------------------
    def solve_hydraulics(self, x0: np.ndarray | None = None) -> dict[str, float]:
        node_names = [n for n in self.nodes if not self.nodes[n].reference]
        idx_p = {nm: i for i, nm in enumerate(node_names)}
        n_p = len(node_names)
        n_b = len(self.branches)

        def unpack(x):
            p = {nm: (self.nodes[nm].p_ref if self.nodes[nm].reference
                      else x[idx_p[nm]]) for nm in self.nodes}
            mdots = x[n_p:n_p + n_b]
            return p, mdots

        def residuals(x):
            p, mdots = unpack(x)
            res = np.zeros(n_p + n_b)
            for i, br in enumerate(self.branches):
                T_avg = 0.5 * (self.nodes[br.from_node].T
                                + self.nodes[br.to_node].T)
                res[n_p + i] = br.component.hydraulic_residual(
                    mdots[i], p[br.from_node], p[br.to_node], self.fluid, T_avg)
            balance = {nm: 0.0 for nm in node_names}
            for i, br in enumerate(self.branches):
                if br.from_node in balance:
                    balance[br.from_node] -= mdots[i]
                if br.to_node in balance:
                    balance[br.to_node] += mdots[i]
            for nm in node_names:
                res[idx_p[nm]] = balance[nm]
            return res

        if x0 is None:
            x0 = self._last_hydraulic_x
        if x0 is None:
            x0 = np.concatenate([np.full(n_p, 5.0e4), np.full(n_b, 0.1)])

        sol = root(residuals, x0, method="hybr")
        if not sol.success:
            raise RuntimeError(f"Risoluzione idraulica non convergente: {sol.message}")
        self._last_hydraulic_x = sol.x

        _, mdots = unpack(sol.x)
        return {br.name: mdots[i] for i, br in enumerate(self.branches)}

    # ------------------------------------------------------------------
    # Topologia di flusso risolta (per il mescolamento termico)
    # ------------------------------------------------------------------
    def _resolve_flow_topology(self, mdots: dict[str, float]):
        """Ordina i nodi (Kahn) seguendo la direzione di flusso corrente e
        raggruppa i rami uscenti/entranti per nodo."""
        out_edges: dict[str, list[tuple[int, str]]] = {n: [] for n in self.nodes}
        in_degree = {n: 0 for n in self.nodes}
        resolved = []  # (branch_index, upstream, downstream)
        for i, br in enumerate(self.branches):
            m = mdots[br.name]
            upstream, downstream = (br.from_node, br.to_node) if m >= 0 \
                else (br.to_node, br.from_node)
            resolved.append((i, upstream, downstream))
            out_edges[upstream].append((i, downstream))
            in_degree[downstream] += 1

        # Il nodo di riferimento (serbatoio) ha una temperatura di stato gia'
        # nota all'inizio del passo (integrata separatamente): non deve
        # aspettare i suoi predecessori per essere "pronto". Lo seminamo
        # esplicitamente nella coda: e' proprio questo che spezza il ciclo
        # di flusso del circuito chiuso (tank -> ... -> tank) e rende
        # possibile un ordinamento topologico del resto della rete.
        ref_name = self.reference_node.name
        visited = {ref_name}
        queue = deque([ref_name])
        queue.extend(n for n in self.nodes if n != ref_name and in_degree[n] == 0)
        visited.update(queue)
        topo_order = []
        indeg = dict(in_degree)
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for _, v in out_edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0 and v not in visited:
                    visited.add(v)
                    queue.append(v)

        if len(topo_order) != len(self.nodes):
            missing = set(self.nodes) - set(topo_order)
            raise RuntimeError(
                "Impossibile ordinare la rete: ciclo di flusso non spezzato "
                f"da un nodo con massa termica (nodi coinvolti: {missing}). "
                "Verifica che il serbatoio/nodo di riferimento chiuda ogni "
                "anello del circuito.")

        return topo_order, out_edges, resolved

    # ------------------------------------------------------------------
    # Solutore termico: ODE sui componenti con massa termica + serbatoio
    # ------------------------------------------------------------------
    def thermal_step(self, dt: float, mdots: dict[str, float]) -> dict[str, float]:
        ref_node = self.reference_node
        tank = ref_node.tank
        if tank is None:
            raise ValueError("Il nodo di riferimento deve avere un Tank "
                              "associato per fornire la massa termica del ciclo")

        thermal_branches = [br for br in self.branches if br.component.has_thermal_state]

        # layout dello stato: [T_tank, T_comp_1, T_comp_2, ...]
        y0 = np.array([tank.T] + [br.component.T for br in thermal_branches])

        topo_order, out_edges, resolved = self._resolve_flow_topology(mdots)
        resolved_by_branch = {i: (u, v) for i, u, v in resolved}

        def rhs(t, y):
            tank.T = y[0]
            for br, T in zip(thermal_branches, y[1:]):
                br.component.T = T

            node_T = {ref_node.name: tank.T}
            branch_T_out = {}

            for u in topo_order:
                if u != ref_node.name and u not in node_T:
                    # nodo puramente idraulico: media pesata sulla portata
                    # dei rami entranti (gia' calcolati, essendo u successivo
                    # ai suoi predecessori nell'ordine topologico)
                    inflow_m, inflow_mT = 0.0, 0.0
                    for i, (up, down) in resolved_by_branch.items():
                        if down == u:
                            m_abs = abs(mdots[self.branches[i].name])
                            inflow_m += m_abs
                            inflow_mT += m_abs * branch_T_out[i]
                    node_T[u] = inflow_mT / inflow_m if inflow_m > 0 else node_T.get(u, ref_node.T)

                for i, v in out_edges[u]:
                    comp = self.branches[i].component
                    T_in = node_T[u]
                    branch_T_out[i] = comp.T if comp.has_thermal_state else T_in

            dydt = np.zeros_like(y)

            # bilancio energetico del serbatoio (miscelamento dei ritorni)
            inflow_m, inflow_mT = 0.0, 0.0
            for i, (up, down) in resolved_by_branch.items():
                if down == ref_node.name:
                    m_abs = abs(mdots[self.branches[i].name])
                    inflow_m += m_abs
                    inflow_mT += m_abs * branch_T_out[i]
            m_cp_tank = tank.thermal_mass(self.fluid, tank.T)
            if m_cp_tank > 0 and inflow_m > 0:
                cp = self.fluid.cp(tank.T)
                dydt[0] = (inflow_m * cp * (inflow_mT / inflow_m - tank.T)) / m_cp_tank

            # bilancio energetico di ogni componente con massa termica
            for k, br in enumerate(thermal_branches, start=1):
                i = self.branches.index(br)
                up, _ = resolved_by_branch[i]
                T_in = node_T[up]
                comp = br.component
                m_abs = abs(mdots[br.name])
                cp = self.fluid.cp(comp.T)
                m_cp = comp.thermal_mass(self.fluid, comp.T)
                Q = comp.thermal_source(comp.T, T_in, mdots[br.name], self.fluid)
                dydt[k] = (m_abs * cp * (T_in - comp.T) + Q) / m_cp if m_cp > 0 else 0.0

            return dydt

        sol = solve_ivp(rhs, (0.0, dt), y0, method="BDF")
        if not sol.success:
            raise RuntimeError(f"Integrazione termica fallita: {sol.message}")

        y_end = sol.y[:, -1]
        tank.T = y_end[0]
        for br, T in zip(thermal_branches, y_end[1:]):
            br.component.T = T

        # ricalcola la mescolanza finale ai nodi puramente idraulici per
        # aggiornare node.T (usato come guess T_avg dal prossimo solve idraulico)
        node_T = {ref_node.name: tank.T}
        branch_T_out = {}
        for u in topo_order:
            if u != ref_node.name and u not in node_T:
                inflow_m, inflow_mT = 0.0, 0.0
                for i, (up, down) in resolved_by_branch.items():
                    if down == u:
                        m_abs = abs(mdots[self.branches[i].name])
                        inflow_m += m_abs
                        inflow_mT += m_abs * branch_T_out[i]
                node_T[u] = inflow_mT / inflow_m if inflow_m > 0 else self.nodes[u].T
            for i, v in out_edges[u]:
                comp = self.branches[i].component
                branch_T_out[i] = comp.T if comp.has_thermal_state else node_T[u]
        for nm, T in node_T.items():
            self.nodes[nm].T = T

        return {nm: node.T for nm, node in self.nodes.items()}
