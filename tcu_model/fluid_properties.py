"""Proprieta' del fluido di processo.

Correlazioni approssimate per acqua liquida (0-100 degC, ~1-3 bar), pensate
per iniziare a far girare il modello senza dipendenze esterne pesanti.

Per accuratezza da validazione / altri fluidi (olio termico, glicole,
refrigeranti) sostituire questa classe con un wrapper su CoolProp
(https://pypi.org/project/CoolProp/) o iapws, mantenendo la stessa
interfaccia (rho, cp, mu, k in funzione di T [K]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class FluidProperties(ABC):
    """Interfaccia comune: tutte le proprieta' sono funzione della sola T [K]."""

    @abstractmethod
    def rho(self, T: float) -> float:
        """Densita' [kg/m3]."""

    @abstractmethod
    def cp(self, T: float) -> float:
        """Calore specifico a pressione costante [J/(kg K)]."""

    @abstractmethod
    def mu(self, T: float) -> float:
        """Viscosita' dinamica [Pa s]."""

    @abstractmethod
    def k(self, T: float) -> float:
        """Conducibilita' termica [W/(m K)]."""

    def Pr(self, T: float) -> float:
        """Numero di Prandtl [-]."""
        return self.cp(T) * self.mu(T) / self.k(T)


class Water(FluidProperties):
    """Correlazioni polinomiali/empiriche di uso comune per acqua liquida.

    Validita' indicativa: 0-100 degC a pressione vicina all'atmosferica.
    Non usare vicino alla pressione di vapore o in regime bifase.
    """

    def rho(self, T: float) -> float:
        Tc = T - 273.15
        return 1000.0 - 0.0605 * Tc - 0.0035 * Tc**2

    def cp(self, T: float) -> float:
        Tc = T - 273.15
        return 4180.0 + 0.15 * (Tc - 20.0)

    def mu(self, T: float) -> float:
        # Correlazione di Vogel/Andrade
        Tc = T - 273.15
        return 2.414e-5 * 10 ** (247.8 / (Tc + 133.15))

    def k(self, T: float) -> float:
        Tc = T - 273.15
        return 0.5706 + 0.0017 * Tc - 6e-6 * Tc**2
