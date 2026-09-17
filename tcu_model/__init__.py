"""Modello 1D modulare di una TCU (Temperature Control Unit).

Blocchi (``components.py``) assemblati in una rete (``network.Circuit``)
stile "netlist", risolti con un approccio ibrido porte-fisiche +
ordine-causale (vedi docstring di ``network.py`` per i dettagli e le
approssimazioni).
"""

from .components import Heater, HeatExchanger, HydraulicResistance, Pipe, Pump, Tank
from .controller import PID
from .fluid_properties import Air, Water
from .network import Circuit

__all__ = [
    "Heater",
    "HeatExchanger",
    "HydraulicResistance",
    "Pipe",
    "Pump",
    "Tank",
    "PID",
    "Air",
    "Water",
    "Circuit",
]
