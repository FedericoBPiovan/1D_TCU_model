"""Blocchi fisici della TCU (stile Simscape): Pompa, Tubazione, Valvola,
Riscaldatore, Scambiatore, Serbatoio.

Ogni componente e' un "ramo" (branch) che collega due nodi (porte) di una
rete idraulica-termica. L'interfaccia e' volutamente uniforme in modo che
la rete (vedi ``network.py``) possa assemblare automaticamente il sistema
di equazioni a partire dalla topologia (approccio ibrido: porte fisiche +
risoluzione causale interna), senza dover conoscere il tipo concreto di
ogni blocco.

Ogni componente espone:
- ``hydraulic_residual(mdot, p_from, p_to, fluid)``: residuo dell'equazione
  di bilancio di pressione sul ramo. Il solver idraulico lo porta a zero.
- ``thermal_mass(fluid)``: capacita' termica concentrata del volume di
  fluido nel componente [J/K]. Se zero, il componente e' considerato senza
  inerzia termica propria: la sua T di uscita e' calcolata algebricamente.
- ``thermal_source(T_out, mdot, fluid)``: potenza termica scambiata dal
  componente con l'esterno (positiva se riscalda il fluido) [W], esclusa
  la parte convettiva mdot*cp*(T_in-T_out) che la rete aggiunge da sola.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .fluid_properties import FluidProperties
from .hydraulics import pipe_pressure_drop, quadratic_resistance_drop


class Component(ABC):
    """Blocco base. Sottoclassi implementano fisica idraulica + termica."""

    name: str = ""

    @abstractmethod
    def hydraulic_residual(self, mdot: float, p_from: float, p_to: float,
                            fluid: FluidProperties, T_avg: float) -> float:
        """Residuo dP(mdot) - (p_from - p_to) [Pa]. Zero quando l'equazione
        costitutiva del ramo e' soddisfatta."""

    def thermal_mass(self, fluid: FluidProperties, T_ref: float) -> float:
        """Capacita' termica concentrata del volume interno [J/K]. Default 0
        (componente idraulico puro, senza dinamica termica propria)."""
        return 0.0

    def thermal_source(self, T_out: float, T_in: float, mdot: float,
                        fluid: FluidProperties) -> float:
        """Potenza termica netta scambiata con l'esterno [W] (default nulla)."""
        return 0.0

    #: True se il componente ha una massa termica propria (stato dinamico
    #: T_out integrato nel tempo). False = componente "istantaneo"
    #: (T_out = T_in, nessuno stato): tipicamente valvole e pompe.
    has_thermal_state: bool = False


class Pipe(Component):
    """Tratto di tubazione: perdita di carico distribuita + dispersione
    termica verso l'ambiente (modello U*A)."""

    has_thermal_state = True

    def __init__(self, name: str, length: float, diameter: float,
                 roughness: float = 1.5e-5, U_amb: float = 5.0,
                 T_amb: float = 293.15, T0: float = 293.15):
        self.name = name
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.U_amb = U_amb  # coefficiente scambio globale verso ambiente [W/(m2 K)]
        self.T_amb = T_amb
        self.T = T0  # stato: temperatura di uscita del volume di fluido nel tubo

    def hydraulic_residual(self, mdot, p_from, p_to, fluid, T_avg):
        rho = fluid.rho(T_avg)
        mu = fluid.mu(T_avg)
        dP = pipe_pressure_drop(mdot, self.length, self.diameter,
                                 self.roughness, rho, mu)
        return dP - (p_from - p_to)

    def _volume(self) -> float:
        return math.pi / 4.0 * self.diameter**2 * self.length

    def thermal_mass(self, fluid, T_ref):
        return self._volume() * fluid.rho(T_ref) * fluid.cp(T_ref)

    def thermal_source(self, T_out, T_in, mdot, fluid):
        area_ext = math.pi * self.diameter * self.length
        return -self.U_amb * area_ext * (T_out - self.T_amb)


class HydraulicResistance(Component):
    """Valvola / perdita di carico concentrata pura (nessuno scambio termico).

    Legge quadratica dP = K/opening^2 * mdot*|mdot|. ``opening`` in [0,1]
    (1 = tutta aperta) e' pensato per essere pilotato da un controllore
    (es. valvola di bypass regolata da PID).
    """

    def __init__(self, name: str, K: float, opening: float = 1.0,
                 opening_min: float = 0.02):
        self.name = name
        self.K = K
        self.opening = opening
        self.opening_min = opening_min

    def set_opening(self, opening: float) -> None:
        self.opening = min(max(opening, 0.0), 1.0)

    def hydraulic_residual(self, mdot, p_from, p_to, fluid, T_avg):
        eff_opening = max(self.opening, self.opening_min)
        K_eff = self.K / eff_opening**2
        dP = quadratic_resistance_drop(mdot, K_eff)
        return dP - (p_from - p_to)


class Pump(Component):
    """Pompa centrifuga: curva caratteristica H(Q) quadratica.

    H(Q) = H0 - a*Q - b*Q^2   [m], Q = mdot/rho [m3/s]
    Il punto di lavoro (portata) e' il risultato dell'intersezione di
    questa curva con la curva di resistenza del resto del circuito,
    trovata dal solver idraulico della rete (non qui).
    ``speed`` in [0,1] scala la curva per affinita' (H ~ speed^2, Q ~ speed)
    per simulare una pompa a velocita' variabile.
    """

    def __init__(self, name: str, H0: float, a: float, b: float,
                 speed: float = 1.0):
        self.name = name
        self.H0 = H0
        self.a = a
        self.b = b
        self.speed = speed

    def set_speed(self, speed: float) -> None:
        self.speed = min(max(speed, 0.0), 1.0)

    def head(self, Q_vol: float) -> float:
        """Curva caratteristica affine alla velocita' [m]."""
        s = max(self.speed, 1e-6)
        Q_ref = Q_vol / s
        H_ref = self.H0 - self.a * Q_ref - self.b * Q_ref**2
        return max(s**2 * H_ref, 0.0)

    def hydraulic_residual(self, mdot, p_from, p_to, fluid, T_avg):
        rho = fluid.rho(T_avg)
        Q_vol = mdot / rho
        dP_rise = rho * 9.81 * self.head(Q_vol)
        # La pompa aumenta la pressione da from a to: p_to - p_from = dP_rise
        return dP_rise - (p_to - p_from)


class Heater(Component):
    """Riscaldatore elettrico (resistenza): converte potenza elettrica in
    calore ceduto al fluido. Perdita di carico trascurabile per default
    (piccolo coefficiente K residuo, tipico di un corpo scaldante in linea).
    """

    has_thermal_state = True

    def __init__(self, name: str, volume: float, power: float = 0.0,
                 efficiency: float = 0.98, K: float = 1.0e3, T0: float = 293.15):
        self.name = name
        self.volume = volume
        self.power = power  # potenza elettrica comandata [W]
        self.efficiency = efficiency
        self.K = K
        self.T = T0

    def set_power(self, power: float) -> None:
        self.power = max(power, 0.0)

    def hydraulic_residual(self, mdot, p_from, p_to, fluid, T_avg):
        dP = quadratic_resistance_drop(mdot, self.K)
        return dP - (p_from - p_to)

    def thermal_mass(self, fluid, T_ref):
        return self.volume * fluid.rho(T_ref) * fluid.cp(T_ref)

    def thermal_source(self, T_out, T_in, mdot, fluid):
        return self.efficiency * self.power


class HeatExchanger(Component):
    """Scambiatore semplificato a efficienza costante verso un pozzo/sorgente
    secondario a temperatura fissata (es. acqua di raffreddamento impianto,
    o processo/stampo lato utenza).

    Q = epsilon * mdot * cp(T_in) * (T_in - T_sink)   [W], rimosso dal
    fluido primario se T_in > T_sink (raffreddamento).

    Approssimazione: l'efficienza (NTU-epsilon) e' presa costante invece di
    essere ricalcolata da mdot, geometria e proprieta' lato secondario.
    """

    has_thermal_state = True

    def __init__(self, name: str, volume: float, epsilon: float,
                 T_sink: float, K: float = 5.0e3, T0: float = 293.15):
        self.name = name
        self.volume = volume
        self.epsilon = epsilon
        self.T_sink = T_sink
        self.K = K
        self.T = T0

    def hydraulic_residual(self, mdot, p_from, p_to, fluid, T_avg):
        dP = quadratic_resistance_drop(mdot, self.K)
        return dP - (p_from - p_to)

    def thermal_mass(self, fluid, T_ref):
        return self.volume * fluid.rho(T_ref) * fluid.cp(T_ref)

    def thermal_source(self, T_out, T_in, mdot, fluid):
        cp = fluid.cp(T_in)
        return -self.epsilon * abs(mdot) * cp * (T_in - self.T_sink)


class Tank:
    """Serbatoio/vaso di espansione: nodo di riferimento di pressione e
    grande capacita' termica di miscelamento (rappresenta il volume di
    accumulo della TCU). Va collegato come nodo di riferimento nella rete,
    non come ramo tra due nodi (vedi ``network.Circuit.add_node``,
    ``reference=True``): non e' un ``Component`` (non e' un ramo con
    ingresso/uscita, e' un punto di miscelamento con condizione al
    contorno di pressione), ma porta la stessa interfaccia
    ``thermal_mass`` usata dai rami per l'integrazione dello stato T_tank.
    """

    def __init__(self, name: str, volume: float, T0: float):
        self.name = name
        self.volume = volume
        self.T = T0

    def thermal_mass(self, fluid: FluidProperties, T_ref: float) -> float:
        return self.volume * fluid.rho(T_ref) * fluid.cp(T_ref)
