# -*- coding: utf-8 -*-
import numpy as np
from pulseblaster import pulseblaster, _spinapi
from pymodaq import Q_, Unit


class Sequence_Template:
    """
    Pulse Sequence class to be used in the PyMoDAQ Pulsed Measurement extension.

    TODO: Complete the attributes of this docstring with the corresponding parameters of your sequence.

    Attributes:
    -----------
    _final_inst: instruction to be set at the end of the sequence (see _spinapi.SpinAPI). Default to STOP.
    _final_inst_data: data for the final instruction (see _spinapi.SpinAPI). Default to 0.
    gui_param_1 (Quantity): parameter to be displayed in the gui as "param 1".
    param_1 (Quantity): parameter defined in the general settings section in the GUI under "param 1".
    ...

    """

    # TODO: initialize the attributes of your sequence with their default value and unit.
    # The attributes starting with gui_ will be parsed and added tothe GUI when the sequence is added.
    # The displayed names will be what follows gui_ (with spaces instead of _).
    gui_sleep = Q_(100, Unit("ns"))
    gui_laser_length = Q_(1, Unit("us"))
    gui_margin = Q_(1, Unit("us"))
    ...

    def __init__(self, final_inst=_spinapi.SpinAPI.STOP, final_inst_data=0, **kwargs):
        # TODO: set the values of the sequence parameters from keywords arguments. Each keyword argument holds
        # the value defined in the GUI for the corresponding parameter. The names of the keyword argument must
        # be the same as the name of the parameter displayed in the GUI (with _ instead of spaces).
        self._final_inst = final_inst
        self._final_inst_data = final_inst_data
        self.gui_sleep = kwargs["sleep"]
        self.gui_laser_length = kwargs["laser_length"]
        self.gui_margin = kwargs["margin"]
        ...

        self.laser_channel = kwargs["Laser"]
        self.counter_channel = kwargs["Counter"]
        ...

    def build(self):
        """
        Build and return the sequence as a list containing the instructions to be sent to the pulseblaster queue.

        The list must have the following structure: [(channel, channel_instructions), ...]
        and channel_instructions = [(0, t1), (1, t2), ...] where 0 is for low and 1 for high and where t1 and t2
        are the durations of each instruction in ns.
        """
        # TODO: when writing your own sequence, replace the following lines with your function
        margin = self.gui_laser_length.to("ns").magnitude
        laser_length = self.gui_laser_length.to("ns").magnitude
        sleep = self.gui_sleep.to("ns").magnitude
        laser_inst = [(0, margin)]
        laser_inst.append((1, laser_length))
        laser_inst.append((0, sleep))
        laser_inst.append((1, laser_length))
        laser_inst.append((0, margin))
        counter_inst = [(1, 50), (0, 50)]
        return [
            (int(self.laser_channel.magnitude), laser_inst),
            (int(self.counter_channel.magnitude), counter_inst),
        ]

    def length(self):
        """
        Calculate the duration of the sequence.
        """
        # TODO: when writing your own sequence, replace the following lines with your function
        # that computes the length of your sequence
        length = self.gui_margin * 2 + self.gui_laser_length * 2 + self.gui_sleep
        return length


if __name__ == "__main__":
    param_dict = {
        "sleep": Q_(1, Unit("us")),
        "laser_length": Q_(2, Unit("us")),
        "margin": Q_(1, Unit("us")),
        "Laser": Q_(1),
        "Counter": Q_(1),
    }
    test_seq = Sequence_Template(**param_dict)
    print(test_seq.length())
    print(test_seq.build())
