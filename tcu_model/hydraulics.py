"""Correlazioni idrauliche di base (perdite di carico distribuite).

Implementazioni auto-contenute (nessuna dipendenza esterna) cosi' lo
scheletro gira subito. Per maggiore fedelta' sostituire con le funzioni
gia' validate delle librerie ``fluids``/``ht`` (Caleb Bell):
https://github.com/CalebBell/fluids
"""

from __future__ import annotations

import math


def reynolds(mdot: float, diameter: float, mu: float) -> float:
    """Numero di Reynolds per flusso in tubo circolare pieno.

    Re = rho*v*D/mu = (mdot/area)*D/mu: rho si semplifica perche' mdot = rho*v*area.
    """
    area = math.pi / 4.0 * diameter**2
    if area <= 0 or mu <= 0:
        return 0.0
    return abs(mdot) * diameter / (area * mu)


def darcy_friction_factor(Re: float, roughness: float, diameter: float) -> float:
    """Fattore di attrito di Darcy. Laminare esatto, turbolento (Swamee-Jain)."""
    if Re < 1e-6:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re
    rel_rough = roughness / diameter
    return 0.25 / (math.log10(rel_rough / 3.7 + 5.74 / Re**0.9)) ** 2


def pipe_pressure_drop(mdot: float, length: float, diameter: float,
                        roughness: float, rho: float, mu: float) -> float:
    """Perdita di carico distribuita (Darcy-Weisbach) [Pa]. Segno = segno(mdot)."""
    area = math.pi / 4.0 * diameter**2
    if area <= 0:
        return 0.0
    velocity = mdot / (rho * area)
    Re = abs(velocity) * diameter * rho / mu if mu > 0 else 0.0
    f = darcy_friction_factor(Re, roughness, diameter)
    dP = f * (length / diameter) * (rho * velocity * abs(velocity) / 2.0)
    return dP


def quadratic_resistance_drop(mdot: float, K: float) -> float:
    """Perdita di carico concentrata generica: dP = K * mdot * |mdot|  [Pa].

    K [Pa / (kg/s)^2] raggruppa geometria + coefficiente di perdita
    concentrata (valvole, raccordi, scambiatori lato perdita di carico).
    """
    return K * mdot * abs(mdot)
