import numpy as np
import time
import matplotlib.pyplot as plt

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base,
    comon_parameters,
    main,
)
from pymodaq.utils.data import DataFromPlugins
from pymodaq_plugins_PulseSequences.hardware.pulsed_controller import PulsedController
from pymodaq_plugins_PulseSequences.hardware._spinapi import SpinAPI


class DAQ_1DViewer_Rabi(DAQ_Viewer_base):
    """Plugin for the rabi sequence that controls:
        - SpinCore PulseBlaster USB ESR-Pro
        - FastComTec MCS8
    This object inherits all functionality to communicate with PyMoDAQ Module
    through inheritance via DAQ_Move_base.
    It then implements the particular communication with the instrument.

    Attributes:
    -----------
    controller: PulsedController
        Instance of the class defined to communicate with both devices.

    controller.pulseblaster: PulseBlaster
        Instance of the class defined to communicate with the SpinCore device.

    controller.counter: MCS8
        Instance of the class defined to communicate with the FastComTec device.
    """

    params = comon_parameters + [
        {
            "title": "Sequence Settings",
            "name": "sequence_settings",
            "type": "group",
            "expanded": True,
            "children": [
                {
                    "title": "Sequence Margin (ns)",
                    "name": "sequence_margin",
                    "type": "float",
                    "value": 1e3,
                },
                {
                    "title": "Final Wait (ns)",
                    "name": "final_wait",
                    "type": "float",
                    "value": 1e3,
                },
                {
                    "title": "Laser Length (ns)",
                    "name": "laser_length",
                    "type": "float",
                    "value": 5e3,
                },
                {
                    "title": "Start (ns)",
                    "name": "start",
                    "type": "float",
                    "value": 50,
                },
                {
                    "title": "Stop (ns)",
                    "name": "stop",
                    "type": "float",
                    "value": 1e3,
                },
                {
                    "title": "Number of Points",
                    "name": "N",
                    "type": "int",
                    "value": 5,
                },
                {
                    "title": "AOM Delay (ns)",
                    "name": "AOM_delay",
                    "type": "list",
                    "limits": [390],
                },
            ],
        },
        {
            "title": "PulseBlaster Settings",
            "name": "pulseblaster_settings",
            "type": "group",
            "expanded": True,
            "children": [
                {
                    "title": "Laser Channel",
                    "name": "laser_channel",
                    "type": "list",
                    "limits": list(np.arange(21)),
                },
                {
                    "title": "Counter Channel",
                    "name": "counter_channel",
                    "type": "list",
                    "limits": list(np.arange(21)),
                },
                {
                    "title": "MW Channel",
                    "name": "MW_channel",
                    "type": "list",
                    "limits": list(np.arange(21)),
                },
                {
                    "title": "Board Number",
                    "name": "board_num",
                    "type": "int",
                    "value": 0,
                },
                {
                    "title": "Clock Frequency",
                    "name": "clock_freq",
                    "type": "list",
                    "limits": [500],
                },
            ],
        },
        {
            "title": "Counter Settings",
            "name": "counter_settings",
            "type": "group",
            "expanded": True,
            "children": [
                {
                    "title": "Binwidth (s)",
                    "name": "binwidth",
                    "type": "list",
                    "limits": list(200e-12 * 2 ** np.arange(21)),
                },
            ],
        },
    ]

    hardware_averaging = True

    def ini_attributes(self):
        self.controller: PulsedController = None
        self.x_axis = None
        # Defining shortcut attributes for some of the settings for convenience
        self.start_MW = self.settings.child("sequence_settings").child("start").value()
        self.stop_MW = self.settings.child("sequence_settings").child("stop").value()
        self.N = self.settings.child("sequence_settings").child("N").value()
        self.laser_length = (
            self.settings.child("sequence_settings").child("laser_length").value()
        )
        self.AOM_delay = (
            self.settings.child("sequence_settings").child("AOM_delay").value()
        )
        self.margin = (
            self.settings.child("sequence_settings").child("sequence_margin").value()
        )
        self.final_wait = (
            self.settings.child("sequence_settings").child("final_wait").value()
        )
        self.counter_channel = (
            self.settings.child("pulseblaster_settings")
            .child("counter_channel")
            .value()
        )
        self.laser_channel = (
            self.settings.child("pulseblaster_settings").child("laser_channel").value()
        )
        self.MW_channel = (
            self.settings.child("pulseblaster_settings").child("MW_channel").value()
        )
        self.iteration_count: int = None
        self.accumulated_data = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings.
        When a parameter change also changes the total length of the pulse sequence we update
        this value in the FastComTec device.
        When a parameter change also changes the timestamps of the x axis of the detector
        we update the x axis accordingly.

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "board_num":
            self.controller.pulseblaster.board_number = param.value()
        elif param.name() == "binwidth":
            self.controller.counter.set_binwidth(param.value())
        elif param.name() == "start":
            self.start_MW = param.value()
        elif param.name() == "stop":
            self.stop_MW = param.value()
        elif param.name() == "N":
            self.N = param.value()
        elif param.value() == "laser_length":
            self.laser_length == param.value()
        elif param.value() == "AOM_delay":
            self.AOM_delay == param.value()
        elif param.value() == "sequence_margin":
            self.margin == param.value()
        elif param.value() == "final_wait":
            self.final_wait == param.value()
        elif param.value() == "counter_channel":
            self.counter_channel == param.value()
        elif param.value() == "laser_channel":
            self.laser_channel == param.value()
        elif param.value() == "MW_channel":
            self.MW_channel == param.value()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        if self.is_master:
            self.controller = PulsedController()
            try:
                self.controller.init_counter()
                initialized = True
            except:
                initialized = False
                self.emit_status(
                    ThreadCommand(
                        "Update_Status",
                        ["Cannot connect to fastcomtec"],
                    )
                )
            try:
                self.controller.init_pulseblaster(
                    self.settings.child("pulseblaster_settings")
                    .child("clock_freq")
                    .value(),
                )
                initialized = True
            except:
                initialized = False
                self.emit_status(
                    ThreadCommand(
                        "Update_Status",
                        ["Cannot connect to pulseblaster"],
                    )
                )
        else:
            self.controller = controller
            initialized = True

        self.iteration_count = 0  # reinitialize the iteration counter

        length = self._sequence_length(
            self.start_MW,
            self.stop_MW,
            self.N,
            self.laser_length,
            self.margin,
            self.AOM_delay,
            self.final_wait,
        )
        print(self.controller.counter.set_length((length - self.final_wait) * 1e-9))
        data_x_axis = self.controller.counter.get_timestamps()
        self.x_axis = Axis(data=data_x_axis, label="Time", units="s", index=0)

        self.dte_signal_temp.emit(
            DataToExport(
                name="temp_rabi",
                data=[
                    DataFromPlugins(
                        name="Rabi",
                        data=[np.zeros_like(data_x_axis)],
                        dim="Data1D",
                        labels=[f"Events ({self.iteration_count} sweeps)"],
                        axes=[self.x_axis],
                    )
                ],
            )
        )

        # Program Pulse Sequence
        MW_delays = np.linspace(self.start_MW, self.stop_MW, self.N)
        laser_rabi = [(0, self.margin)]
        laser_rabi.append((1, self.laser_length))
        for j in range(self.N):
            laser_rabi.append((0, self.AOM_delay))
            laser_rabi.append((0, MW_delays[j]))
            laser_rabi.append((1, self.laser_length))
        laser_rabi.append((0, self.margin + self.final_wait))

        MW_rabi = [(0, self.margin + self.laser_length)]
        for j in range(self.N):
            MW_rabi.append((0, self.AOM_delay))
            MW_rabi.append((1, MW_delays[j]))
            MW_rabi.append((0, self.laser_length))
        MW_rabi.append((0, self.margin + self.final_wait))

        trigger_rabi = [(1, 50), (0, 50)]  # pas compris

        self.controller.pulseblaster.set_channel(
            self.laser_channel,
            laser_rabi,
        )
        self.controller.pulseblaster.set_channel(
            self.counter_channel,
            trigger_rabi,
        )
        self.controller.pulseblaster.set_channel(
            self.MW_channel,
            MW_rabi,
        )
        self.controller.pulseblaster.compile_channels()
        self.controller.pulseblaster.add_inst(
            0x000000, SpinAPI.STOP, 0, 0
        )  # the duration can be as small as the minimum intsruction length
        self.controller.pulseblaster.program()
        self.emit_status(
            ThreadCommand(
                "Update_Status",
                ["Sequence Loaded"],
            )
        )

        info = "Detector and Viewer Initialized"
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        if self.is_master:
            self.controller.counter.stop_measure()
            self.controller.pulseblaster.shutdown()
            self.iteration_count = 0

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        # Reinitialize accumulated data if this is the frst iteration
        if self.iteration_count == 0:
            # self.accumulated_data = np.zeros_like(
            #     self.controller.counter.get_timestamps()
            # )
            self.controller.counter.start_measure()

        length = self._sequence_length(
            self.start_MW,
            self.stop_MW,
            self.N,
            self.laser_length,
            self.margin,
            self.AOM_delay,
            self.final_wait,
        )

        self.controller.pulseblaster.reset()
        self.controller.pulseblaster.start()
        time.sleep(length * 1e-9)
        self.controller.pulseblaster.stop()
        # Get data from FCT
        # self.accumulated_data = self.controller.counter.get_data()
        self.iteration_count += 1
        self.dte_signal.emit(
            DataToExport(
                "Rabi",
                data=[
                    DataFromPlugins(
                        name="Rabi",
                        data=self.controller.counter.get_data(),
                        dim="Data1D",
                        labels=[f"Events ({self.iteration_count} sweeps)"],
                        axes=[self.x_axis],
                    )
                ],
            )
        )
        # If the viewer cannot support the actualisation rate add a sleep time
        # time.sleep(
        #     0.5
        # )

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        self.iteration_count = 0  # reinitilaize the number of sweeps
        self.controller.counter.stop_measure()
        self.emit_status(
            ThreadCommand("Update_Status", ["FastComTec stopped counting"])
        )
        self.controller.pulseblaster.stop()
        self.emit_status(ThreadCommand("Update_Status", ["Pulse Sequence Stopped"]))

    @staticmethod
    def _sequence_length(
        start: float,
        stop: float,
        N: int,
        laser_length: float,
        sequence_margin: float,
        AOM_delay: float,
        final_wait: float,
    ):
        """Compute the length of the sequence based on the parameters given as inputs.

        Parameters:
        start (float): the smallest interpulse delay (ns)
        stop (float): the longer interpulse delay (ns)
        N (int): the number of interpulse delays
        laser_length (float): The laser pulse duration (ns)
        sequence_margin (float): The waiting time at the beginning and at the end
        of each sequence (ns)
        AOM_delay (float): The waiting time before the MW pulse only, it corresponds to the time needed for the AOM to respond (ns)
        final_wait (float): The wainting time at the end of the sequence (ns)

        Return:
        length_sequence (float): the length of the sequence (ns)
        """
        MW_delays = np.linspace(start, stop, N)
        length_sequence = (
            (N + 1) * laser_length
            + np.sum(MW_delays)
            + N * AOM_delay
            + 2 * sequence_margin
            + final_wait
        )
        return length_sequence


if __name__ == "__main__":
    main(__file__)
