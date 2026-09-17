"""Esempio: TCU con anello di bypass (rete a rami multipli).

Topologia (dimostra il risolutore idraulico generico su una rete
non a singolo anello, come richiesto):

    tank --Pump--> A --Heater--> B --+--HeatExchanger(load)--+--> C --Pipe--> tank
                                       +--HydraulicResistance--+
                                          (valvola di bypass)

Un PID regola la potenza del riscaldatore per portare la temperatura in
uscita dal riscaldatore (nodo B, cioe' quella inviata al processo) al
setpoint desiderato. L'apertura della valvola di bypass e' fissa in
questo scheletro (TODO: un secondo anello di controllo potrebbe
modularla per affinare la temperatura di ritorno).

Uso:
    python3 examples/example_bypass_circuit.py
(oppure incollare il contenuto in una cella di Google Colab dopo aver
clonato/installato il pacchetto ``tcu_model``.)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # backend non interattivo per l'esecuzione da script
import matplotlib.pyplot as plt

from tcu_model import (PID, Circuit, Heater, HeatExchanger,
                        HydraulicResistance, Pipe, Pump, Tank, Water)


def build_circuit() -> tuple[Circuit, Tank, Heater, HeatExchanger,
                              HydraulicResistance, Pump]:
    fluid = Water()
    circuit = Circuit(fluid)

    T0 = 293.15  # 20 degC, temperatura iniziale di tutto il circuito

    tank = Tank("tank", volume=0.010, T0=T0)  # 10 L di accumulo
    circuit.add_node("tank", reference=True, p_ref=0.0, tank=tank)
    circuit.add_node("A", T0=T0)
    circuit.add_node("B", T0=T0)
    circuit.add_node("C", T0=T0)

    pump = Pump("pump", H0=8.0, a=60000.0, b=0.0, speed=1.0)
    heater = Heater("heater", volume=0.002, power=0.0, efficiency=0.98,
                     K=800.0, T0=T0)
    load_hx = HeatExchanger("load_hx", volume=0.001, epsilon=0.35,
                             T_sink=300.15, K=3000.0, T0=T0)
    bypass_valve = HydraulicResistance("bypass_valve", K=1500.0, opening=0.4)
    return_pipe = Pipe("return_pipe", length=3.0, diameter=0.02,
                        roughness=1.5e-5, U_amb=8.0, T_amb=293.15, T0=T0)

    circuit.add_branch("pump", pump, "tank", "A")
    circuit.add_branch("heater", heater, "A", "B")
    circuit.add_branch("load_hx", load_hx, "B", "C")
    circuit.add_branch("bypass_valve", bypass_valve, "B", "C")
    circuit.add_branch("return_pipe", return_pipe, "C", "tank")

    return circuit, tank, heater, load_hx, bypass_valve, pump


def run_simulation():
    circuit, tank, heater, load_hx, bypass_valve, pump = build_circuit()

    setpoint_T = 318.15  # 45 degC
    P_max = 6000.0  # W, potenza massima del riscaldatore
    pid = PID(kp=3000.0, ki=80.0, kd=0.0, out_min=0.0, out_max=P_max)

    dt = 5.0  # s, passo di controllo (idraulica quasi-stazionaria per ogni passo)
    n_steps = 240  # 240*5s = 1200s = 20 min

    log = {"t": [], "T_tank": [], "T_A": [], "T_B": [], "T_C": [],
           "mdot_pump": [], "mdot_load": [], "mdot_bypass": [], "power": []}

    t = 0.0
    for _ in range(n_steps):
        mdots = circuit.solve_hydraulics()
        node_T = circuit.thermal_step(dt, mdots)

        measurement = node_T["B"]
        power = pid.update(setpoint_T, measurement, dt)
        heater.set_power(power)

        t += dt
        log["t"].append(t)
        log["T_tank"].append(node_T["tank"] - 273.15)
        log["T_A"].append(node_T["A"] - 273.15)
        log["T_B"].append(node_T["B"] - 273.15)
        log["T_C"].append(node_T["C"] - 273.15)
        log["mdot_pump"].append(mdots["pump"])
        log["mdot_load"].append(mdots["load_hx"])
        log["mdot_bypass"].append(mdots["bypass_valve"])
        log["power"].append(power)

    return log, setpoint_T - 273.15


def plot_results(log, setpoint_C, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(log["t"], log["T_A"], label="A (uscita pompa)")
    axes[0].plot(log["t"], log["T_B"], label="B (uscita riscaldatore)")
    axes[0].plot(log["t"], log["T_C"], label="C (ritorno, dopo merge)")
    axes[0].plot(log["t"], log["T_tank"], label="serbatoio", linestyle="--")
    axes[0].axhline(setpoint_C, color="k", linestyle=":", label="setpoint")
    axes[0].set_ylabel("Temperatura [degC]")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log["t"], log["mdot_pump"], label="pompa (totale)")
    axes[1].plot(log["t"], log["mdot_load"], label="ramo load")
    axes[1].plot(log["t"], log["mdot_bypass"], label="ramo bypass")
    axes[1].set_ylabel("Portata [kg/s]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(log["t"], log["power"])
    axes[2].set_ylabel("Potenza riscaldatore [W]")
    axes[2].set_xlabel("Tempo [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Grafico salvato in {out_path}")


if __name__ == "__main__":
    log, setpoint_C = run_simulation()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_results(log, setpoint_C, os.path.join(out_dir, "example_bypass_circuit.png"))

    print(f"T_B finale: {log['T_B'][-1]:.2f} degC (setpoint {setpoint_C:.2f} degC)")
    print(f"mdot pompa finale: {log['mdot_pump'][-1]:.4f} kg/s")
    print(f"mdot load/bypass finali: {log['mdot_load'][-1]:.4f} / "
          f"{log['mdot_bypass'][-1]:.4f} kg/s")
