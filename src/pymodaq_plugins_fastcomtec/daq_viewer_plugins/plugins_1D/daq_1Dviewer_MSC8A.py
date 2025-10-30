import numpy as np
import time

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base,
    comon_parameters,
    main,
)
from pymodaq.utils.data import DataFromPlugins
from pymodaq_plugins_fastcomtec.hardware.msc8_wrapper import MSC8


class DAQ_1DViewer_MSC8A(DAQ_Viewer_base):
    """Instrument plugin class for a 1D viewer.

    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    TODO Complete the docstring of your plugin with:
        * The set of instruments that should be compatible with this instrument plugin.
        * With which instrument it has actually been tested.
        * The version of PyMoDAQ during the test.
        * The version of the operating system.
        * Installation instructions: what manufacturer’s drivers should be installed to make it run?

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    # TODO add your particular attributes here if any

    """

    params = comon_parameters + [
        {
            "title": "Trigger Safety (s)",
            "name": "trigger_safety",
            "type": "float",
            "value": 400e-9,
        },
        {"title": "Binwidth (s)", "name": "binwidth", "type": "float", "value": 0.2e-9},
        {
            "title": "Acquisition Time (s)",
            "name": "length",
            "type": "float",
            "value": 2e-9,
        },
        {
            "title": "Acquisition Delay (s)",
            "name": "delay",
            "type": "float",
            "value": 6.4e-9,
        },
    ]

    hardware_averaging = True

    def ini_attributes(self):
        self.controller: MSC8 = None
        self.x_axis = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "triger_safety":
            self.controller.trigger_safety = param.value()

        elif param.name() == "binwidth":
            binwidth = self.controller.set_binwidth(param.value())
            self.emit_status(
                ThreadCommand(
                    "Update_Status",
                    [f"Binwidth set to {binwidth}"],
                )
            )
            data_x_axis = self.controller.get_timestamps()
            self.x_axis = Axis(data=data_x_axis, label="Time", units="s", index=0)
        elif param.name() == "length":
            length = self.controller.set_length(param.value())
            self.emit_status(
                ThreadCommand(
                    "Update_Status",
                    [f"Acquisition time set to {length}"],
                )
            )
            data_x_axis = self.controller.get_timestamps()
            self.x_axis = Axis(data=data_x_axis, label="Time", units="s", index=0)
        elif param.name() == "delay":
            delay = self.controller.set_start_delay(param.value())
            self.emit_status(
                ThreadCommand(
                    "Update_Status",
                    [f"Acquisition delay set to {delay}"],
                )
            )
        else:
            pass

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
            self.controller = MSC8(
                trigger_safety=self.settings.child("trigger_safety").value()
            )
            initialized = (
                True  # TODO make a connection check (probably with get_status)
            )
        else:
            self.controller = controller
            initialized = True

        ## TODO for your custom plugin
        # get the x_axis (you may want to to this also in the commit settings if x_axis may have changed
        data_x_axis = self.controller.get_timestamps()
        self.x_axis = Axis(data=data_x_axis, label="Time", units="s", index=0)

        self.dte_signal_temp.emit(
            DataToExport(
                name="FastCounter",
                data=[
                    DataFromPlugins(
                        name="FastCounter",
                        data=[
                            np.zeros_like(data_x_axis),
                        ],
                        dim="Data1D",
                        labels=["Counts", "label2"],
                        axes=[self.x_axis],
                    )
                ],
            )
        )

        info = "Detector and Viewer Initialized"
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        ## TODO for your custom plugin
        raise NotImplementedError  # when writing your own plugin remove this line
        if self.is_master:
            #  self.controller.your_method_to_terminate_the_communication()  # when writing your own plugin replace this line
            ...

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
        ## TODO for your custom plugin: you should choose EITHER the synchrone or the asynchrone version following

        ##synchrone version (blocking function)

        self.controller.start_measure()
        self.emit_status(
            ThreadCommand("Update_Status", ["FastComTec started counting"])
        )
        data_tot = self.controller.get_data()
        self.dte_signal.emit(
            DataToExport(
                "FastCounter",
                data=[
                    DataFromPlugins(
                        name="FastCounter",
                        data=data_tot,
                        dim="Data1D",
                        labels=["dat0", "data1"],
                        axes=[self.x_axis],
                    )
                ],
            )
        )

        ##asynchrone version (non-blocking function with callback)
        self.controller.your_method_to_start_a_grab_snap(self.callback)
        #########################################################

    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        data_tot = self.controller.your_method_to_get_data_from_buffer()
        self.dte_signal.emit(
            DataToExport(
                "myplugin",
                data=[
                    DataFromPlugins(
                        name="Mock1",
                        data=data_tot,
                        dim="Data1D",
                        labels=["dat0", "data1"],
                    )
                ],
            )
        )

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        self.controller.stop_measure()  # when writing your own plugin replace this line
        self.emit_status(
            ThreadCommand("Update_Status", ["FastComTec stopped counting"])
        )
        ##############################
        return ""


if __name__ == "__main__":
    main(__file__)
