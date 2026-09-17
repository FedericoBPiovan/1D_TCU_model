"""Controllore PID discreto, per pilotare gli attuatori del circuito
(potenza riscaldatore, apertura valvola di bypass, velocita' pompa...)."""

from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = 0.0, out_max: float = 1.0,
                 anti_windup: bool = True):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.anti_windup = anti_windup
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        integral_candidate = self._integral + error * dt
        output_unclipped = (self.kp * error + self.ki * integral_candidate
                             + self.kd * derivative)
        output = min(max(output_unclipped, self.out_min), self.out_max)

        # anti-windup: aggiorna l'integrale solo se non si e' in saturazione
        # (o se l'errore spingerebbe l'uscita a rientrare dalla saturazione)
        if not self.anti_windup or output == output_unclipped:
            self._integral = integral_candidate

        self._prev_error = error
        return output
