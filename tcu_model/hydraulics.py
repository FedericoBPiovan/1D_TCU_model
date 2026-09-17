"""Correlazioni idrauliche e di scambio termico, basate sulle librerie
``fluids`` (perdite di carico) e ``ht`` (coefficienti di scambio termico)
di Caleb Bell: https://github.com/CalebBell/fluids , https://github.com/CalebBell/ht

Sostituiscono le correlazioni auto-contenute (Swamee-Jain, U_amb fisso)
della prima versione dello scheletro con implementazioni validate e
piu' complete (es. Colebrook esatto invece di un'approssimazione
esplicita, scelta automatica della migliore correlazione di scambio
termico interno disponibile).
"""

from __future__ import annotations

import math

from fluids.core import Grashof, Prandtl, Reynolds
from fluids.friction import friction_factor
from ht.conv_free_immersed import Nu_horizontal_cylinder
from ht.conv_internal import Nu_conv_internal

from .fluid_properties import Air, FluidProperties

_AIR = Air()


def pipe_pressure_drop(mdot: float, length: float, diameter: float,
                        roughness: float, rho: float, mu: float) -> float:
    """Perdita di carico distribuita (Darcy-Weisbach) [Pa]. Segno = segno(mdot)."""
    area = math.pi / 4.0 * diameter**2
    if area <= 0 or mdot == 0.0:
        return 0.0
    velocity = mdot / (rho * area)
    Re = Reynolds(V=abs(velocity), D=diameter, rho=rho, mu=mu)
    f = friction_factor(Re=Re, eD=roughness / diameter)
    return f * (length / diameter) * (rho * velocity * abs(velocity) / 2.0)


def quadratic_resistance_drop(mdot: float, K: float) -> float:
    """Perdita di carico concentrata generica: dP = K * mdot * |mdot|  [Pa].

    K [Pa / (kg/s)^2] raggruppa geometria + coefficiente di perdita
    concentrata (valvole, raccordi, scambiatori lato perdita di carico).
    """
    return K * mdot * abs(mdot)


def valve_K_from_Kv(Kv: float, rho: float) -> float:
    """Converte un coefficiente di valvola Kv [m3/h a dP=1 bar, acqua a
    densita' di riferimento 1000 kg/m3] nel coefficiente K [Pa/(kg/s)^2]
    usato da ``quadratic_resistance_drop`` (formula standard IEC 60534,
    dP[bar] = SG * (Q[m3/h] / Kv)^2, SG = rho/1000).

    Permette di specificare le valvole con il parametro di targa reale
    (Kv/Cv da datasheet) invece di un coefficiente K astratto.
    """
    if Kv <= 0:
        raise ValueError("Kv deve essere positivo")
    # dP[Pa] = 1e5 * (rho/1000) * (mdot*3600/(rho*Kv))^2
    return 1.0e5 * 3600.0**2 / (1000.0 * rho * Kv**2)


def pipe_overall_heat_transfer_coefficient(
        mdot: float, diameter: float, fluid: FluidProperties, T_fluid: float,
        T_ambient: float, roughness: float = 1.5e-5) -> float:
    """Coefficiente di scambio termico globale U [W/(m2 K)] (riferito alla
    superficie esterna del tubo) tra fluido interno e aria ambiente in
    convezione naturale, come resistenze in serie:

        1/U = 1/h_interno + 1/h_esterno

    Approssimazioni: spessore/conducibilita' della parete del tubo e
    isolamento trascurati (resistenza di parete assunta nulla: da
    aggiungere esplicitamente se il tubo e' isolato). Diametro esterno
    approssimato uguale al diametro interno (parete sottile).
    """
    area = math.pi / 4.0 * diameter**2
    if area <= 0:
        return 0.0

    # --- convezione forzata interna ---
    rho_f = fluid.rho(T_fluid)
    mu_f = fluid.mu(T_fluid)
    velocity = abs(mdot) / (rho_f * area) if rho_f > 0 else 0.0
    if velocity > 1e-9:
        Re = Reynolds(V=velocity, D=diameter, rho=rho_f, mu=mu_f)
        Pr = Prandtl(Cp=fluid.cp(T_fluid), k=fluid.k(T_fluid), mu=mu_f)
        Nu_in = Nu_conv_internal(Re=Re, Pr=Pr, eD=roughness / diameter)
        h_in = Nu_in * fluid.k(T_fluid) / diameter
    else:
        h_in = 0.0  # fluido fermo: nessuna convezione forzata (approssimazione)

    # --- convezione naturale esterna (cilindro orizzontale in aria) ---
    # A dT=0 la convezione naturale (spinta di galleggiamento) e'
    # fisicamente nulla: la correlazione di Nu diverge in quel limite
    # (Gr=0), quindi il caso va gestito esplicitamente invece di
    # dividere per un h_out numericamente nullo.
    if abs(T_fluid - T_ambient) < 1e-9:
        h_out = 0.0
    else:
        T_film = 0.5 * (T_fluid + T_ambient)
        rho_a = _AIR.rho(T_film)
        mu_a = _AIR.mu(T_film)
        Pr_a = Prandtl(Cp=_AIR.cp(T_film), k=_AIR.k(T_film), mu=mu_a)
        Gr_a = Grashof(L=diameter, beta=_AIR.beta(T_film), T1=T_fluid, T2=T_ambient,
                        rho=rho_a, mu=mu_a)
        Nu_out = Nu_horizontal_cylinder(Pr=Pr_a, Gr=Gr_a)
        h_out = Nu_out * _AIR.k(T_film) / diameter

    if h_in <= 0.0 or h_out <= 0.0:
        return 0.0
    return 1.0 / (1.0 / h_in + 1.0 / h_out)
