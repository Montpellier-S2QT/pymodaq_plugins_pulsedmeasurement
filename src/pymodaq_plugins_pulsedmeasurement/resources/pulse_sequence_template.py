# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
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
    gui_start = Q_(1, Unit("us"))
    gui_stop = Q_(1, Unit("us"))
    gui_laser_length = Q_(1, Unit("us"))
    gui_N_points = 1
    ...

    def __init__(self, final_inst=_spinapi.SpinAPI.STOP, final_inst_data=0, **kwargs):
        # TODO: set the values of the sequence parameters from keywords arguments. Each keyword argument holds
        # the value defined in the GUI for the corresponding parameter. The names of the keyword argument must
        # be the same as the name of the parameter displayed in the GUI (with _ instead of spaces).
        self._final_inst = final_inst
        self._final_inst_data = final_inst_data
        self.gui_start = kwargs["start"]
        self.gui_stop = kwargs["stop"]
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
        # TODO: when writing your own sequence, write here the function that will
        # return the list of instructions on each channels
        ...

    @property
    def length(self):
        """
        Calculate the duration of the sequence.
        """
        # TODO: when writing your own sequence, write here the function to
        # calculate the length of your sequence
        ...

    @property
    def delays(self):
        """
        Calculate the delays from the given start, stop and N_points.
        """
        start = round(self.gui_start.to("ns").magnitude)
        stop = round(self.gui_stop.to("ns").magnitude)
        # TODO: when writting your own pulse sequence choose linspace or geomspace depending
        # on which distribution of measurement points you desire
        delays = Q_(np.linspace(start, stop, self.gui_N_points), Unit("ns"))
        return delays

    class Fit:
        """
        Subclass used for fitting the data obtained with this sequence.

        Attributes:
        -----------
        fit_params (dict): a dictionnary containing the name and values of the fitting parameters
        """

        fit_params = {}

        @staticmethod
        def fit_func(x, param1, params2, etc):
            # TODO: write here the mathematical function used to fit your sequence data
            ...

        def calculate_fit(self, xdata):
            """
            Wrapper of self.fit_func returning the fit calculated with self.fit_params over xdata.
            """
            # TODO: call the fit_func with the correct argument values from fit_params
            return self.fit_func(
                xdata,
                self.fit_params["param1"],
                self.fit_params["param2"],
            )

        def guess_params(self, xdata, ydata):
            """
            Function used to make the initial gues on the fitting parameters from the experimental data.
            """
            # TODO: write the function to get the initial guess. It should return a list of the initial guesses
            # ordered in the same way as in fit_func arguments (so that curve_fit can use it correctly).
            ...

        def fit_data(self, xdata, ydata):
            """
            Fit the given data and store the fit parameters in self.fit_parms.
            """
            pguess = self.guess_params(xdata, ydata)
            popt, _ = curve_fit(self.fit_func, xdata, ydata, pguess)
            # TODO: store the fitting parameters in fit_params
            self.fit_params["param1"] = popt[0]
            self.fit_params["param2"] = popt[1]
            ...


if __name__ == "__main__":
    pass
