# -*- coding: utf-8 -*-
import numpy as np
from pulseblaster import pulseblaster, _spinapi
from pymodaq import Q_, Unit


class Sequence_Template:
    """
    Pulse Sequence class to be used in the PyMoDAQ Pulsed Measurement extension.

    TODO: Complete the attributes of this docstring with the corresponding parameters of your sequence.
    Also change the name of this class with Sequence_<name> and add it to the __init__.py in your pulse sequences folder.

    Attributes:
    -----------
    _final_inst: instruction to be set at the end of the sequence (see _spinapi.SpinAPI). Default to STOP.
    _final_inst_data: data for the final instruction (see _spinapi.SpinAPI). Default to 0.
    gui_param_1 (Quantity): parameter to be displayed in the gui as "param 1".
    param_1 (Quantity): parameter defined in the general settings section in the GUI under "param 1".
    ...

    """

    # TODO: initialize the attributes of your sequence with their default value and unit.
    # The attributes starting with gui_ will be parsed and added to the GUI when the sequence is added.
    # The displayed names will be what follows gui_ (with spaces instead of _).
    # WARNING! some attributes must have predefined names for the analysis to work properly:
    # - the minimum and maximum delay time values must be named start and stop respectively
    # - the length of the laser pulse must be named laser_length
    # - the number of different delay time values must be named N_points
    gui_laser_length = Q_(1, Unit("us"))
    gui_N_points = 1
    ...

    def __init__(self, final_inst=_spinapi.SpinAPI.STOP, final_inst_data=0, **kwargs):
        # TODO: set the values of the sequence parameters from keywords arguments. Each keyword argument holds
        # the value defined in the GUI for the corresponding parameter. The names of the keyword argument must
        # be the same as the name of the parameter displayed in the GUI (with _ instead of spaces).
        self._final_inst = final_inst
        self._final_inst_data = final_inst_data
        self.gui_laser_length = kwargs["laser_length"]
        self.gui_N_points = kwargs["N_points"]
        ...

        self.wait: Q_ = kwargs["Final_wait"]
        self.laser_channel = kwargs["Laser"]
        ...

    def build(self):
        """
        Build and return the sequence as a list containing the instructions to be sent to the pulseblaster queue.

        The list must have the following structure: [(channel, channel_instructions), ...]
        and channel_instructions = [(0, t1), (1, t2), ...] where 0 is for low and 1 for high and where t1 and t2
        are the durations of each instruction in ns.
        """
        # TODO: when writing your own sequence, replace the following lines with your function
        ...

    def length(self):
        """
        Calculate the duration of the sequence.
        """
        # TODO: when writing your own sequence, replace the following lines with your function
        # that computes the length of your sequence
        ...


if __name__ == "__main__":
    pass
