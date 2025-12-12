# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
from pymodaq import Q_, Unit

from pulseblaster import _spinapi


class Sequence_Rabi:
    """
    Rabi Pulse Sequence class with geometrically sapced delays,
    to be used in the PyMoDAQ Pulsed Measurement extension.

    TODO: Complete the attributes of this docstring with the corresponding parameters of your sequence.

    Attributes:
    -----------
    _final_inst: instruction to be set at the end of the sequence (see _spinapi.SpinAPI). Default to STOP.
    _final_inst_data: data for the final instruction (see _spinapi.SpinAPI). Default to 0.
    channel1 (int): channel for an instrument1.
    param1 (type): sequence parameter 1 (the time unit should be ns).
    ...

    """

    gui_start = Q_(50, Unit("ns"))
    gui_stop = Q_(1, Unit("us"))
    gui_N_points = 5
    gui_laser_length = Q_(5, Unit("us"))
    gui_margin = Q_(1, Unit("us"))

    def __init__(self, final_inst=_spinapi.SpinAPI.STOP, final_inst_data=0, **kwargs):
        self._final_inst = final_inst
        self._final_inst_data = final_inst_data
        self.gui_start = kwargs["start"]
        self.gui_stop = kwargs["stop"]
        self.gui_N_points = kwargs["N_points"]
        self.gui_laser_length = kwargs["laser_length"]
        self.gui_margin = kwargs["margin"]
        self.final_wait = kwargs["Final_wait"]
        self.aom_delay = kwargs["AOM_delay"]
        self.laser_channel = kwargs["Laser"]
        self.mw_channel = kwargs["MW"]
        self.counter_channel = kwargs["Counter"]

    def build(self):
        """
        Build and return the sequence as a list containing the instructions to be sent to the pulseblaster queue.

        The list must have the following structure: [(channel, channel_instructions), ...]
        and channel_instructions = [(0, t1), (1, t2), ...] where 0 is for low and 1 for high and where t1 and t2
        are the durations of each instruction in ns.
        """
        margin = round(self.gui_margin.to("ns").magnitude)
        laser_length = round(self.gui_laser_length.to("ns").magnitude)
        delay = round(self.aom_delay.to("ns").magnitude)
        wait = round(self.final_wait.to("ns").magnitude)
        MW_delays = self.delays.magnitude
        laser_inst = [(0, margin)]
        laser_inst.append((1, laser_length))
        for j in range(self.gui_N_points):
            laser_inst.append((0, delay))
            laser_inst.append((0, MW_delays[j]))
            laser_inst.append((1, laser_length))
        laser_inst.append((0, margin + wait))

        MW_inst = [(0, margin + laser_length)]
        for j in range(self.gui_N_points):
            MW_inst.append((0, delay))
            MW_inst.append((1, MW_delays[j]))
            MW_inst.append((0, laser_length))
        MW_inst.append((0, margin + wait))

        counter_inst = [(0, 50), (1, 50), (0, 50)]

        return [
            (self.laser_channel, laser_inst),
            (self.mw_channel, MW_inst),
            (self.counter_channel, counter_inst),
        ]

    @property
    def length(self):
        """Calculate the duration of the sequence."""
        length_sequence = (
            (self.gui_N_points + 1) * self.gui_laser_length
            + self.delays.sum()
            + self.gui_N_points * self.aom_delay
            + 2 * self.gui_margin
            + self.final_wait
        )
        return length_sequence

    @property
    def delays(self):
        """Return the list of free evolution times in ns as a 1D array."""
        start = round(self.gui_start.to("ns").magnitude)
        stop = round(self.gui_stop.to("ns").magnitude)
        delays = Q_(np.linspace(start, stop, self.gui_N_points), Unit("ns"))
        return delays


class Sequence_T1:
    """
    Pulse Sequence class to be used in the PyMoDAQ Pulsed Measurement extension.

    TODO: Complete the attributes of this docstring with the corresponding parameters of your sequence.


    Attributes:
    -----------
    _final_inst: instruction to be set at the end of the sequence (see _spinapi.SpinAPI). Default to STOP.
    _final_inst_data: data for the final instruction (see _spinapi.SpinAPI). Default to 0.
    gui_start_tau (Quantity): parameter that sets the lowest tau time in the sequence.
    gui_stop_tau (Quantity): parameter that sets the highest tau time in the sequence.
    gui_N_points (int): parameter that sets the number of points in the T1 sequence.
    gui_laser_length (Quantity): parameter that sets the length of the laser pulse.
    gui_margin (Quantity): parameter that sets the waiting time at the beginning and at the end of the sequence before the final wait.
    Laser (int): Laser channel.
    Counter (int): FastComTec channel.
    Final_wait (Quantity): parameter that sets the final wainting time at the end of the sequence.

    """

    gui_laser_length = Q_(5, Unit("us"))
    gui_margin = Q_(1, Unit("us"))
    gui_start = Q_(50, Unit("ns"))
    gui_stop = Q_(30, Unit("us"))
    gui_N_points = 20

    def __init__(self, final_inst=_spinapi.SpinAPI.STOP, final_inst_data=0, **kwargs):
        self._final_inst = final_inst
        self._final_inst_data = final_inst_data
        self.gui_laser_length = kwargs["laser_length"]
        self.gui_margin = kwargs["margin"]
        self.gui_start = kwargs["start"]
        self.gui_stop = kwargs["stop"]
        self.gui_N_points = kwargs["N_points"]

        self.wait: Q_ = kwargs["Final_wait"]
        self.laser_channel = kwargs["Laser"]
        self.counter_channel = kwargs["Counter"]

    def build(self):
        """
        Build and return the sequence as a list containing the instructions to be sent to the pulseblaster queue.

        The list must have the following structure: [(channel, channel_instructions), ...]
        and channel_instructions = [(0, t1), (1, t2), ...] where 0 is for low and 1 for high and where t1 and t2
        are the durations of each instruction in ns.
        """
        margin = round(self.gui_margin.to("ns").magnitude)
        laser_length = round(self.gui_laser_length.to("ns").magnitude)
        wait = round(self.wait.to("ns").magnitude)
        laser_inst = [(0, margin)]
        laser_inst.append((1, laser_length))
        tau = self.delays.magnitude
        for i in range(self.gui_N_points):
            laser_inst.append((0, tau[i]))
            laser_inst.append((1, laser_length))
        laser_inst.append((0, margin + wait))
        counter_inst = [(0, 50), (1, 50), (0, 50)]
        return [
            (self.laser_channel, laser_inst),
            (self.counter_channel, counter_inst),
        ]

    @property
    def length(self):
        """Calculate the duration of the sequence."""
        length = (
            self.gui_margin * 2
            + self.gui_laser_length * (self.gui_N_points + 1)
            + self.delays.sum()
            + self.wait
        )
        return length

    @property
    def delays(self):
        """Return the list of free evolution times in ns as a 1D array."""
        start = round(self.gui_start.to("ns").magnitude)
        stop = round(self.gui_stop.to("ns").magnitude)
        delays = Q_(np.geomspace(start, stop, self.gui_N_points), Unit("ns"))
        return delays


if __name__ == "__main__":
    param_dict = {
        "start": Q_(50, Unit("ns")),
        "stop": Q_(1, Unit("us")),
        "N_points": 5,
        "laser_length": Q_(2, Unit("us")),
        "margin": Q_(1, Unit("us")),
        "Laser": 1,
        "Counter": 0,
        "MW": 2,
        "Final_wait": Q_(1, Unit("ns")),
        "AOM_delay": Q_(400, Unit("ns")),
    }
    test_seq = Sequence_Rabi(**param_dict)
    print(test_seq.length())
    print(test_seq.build())
