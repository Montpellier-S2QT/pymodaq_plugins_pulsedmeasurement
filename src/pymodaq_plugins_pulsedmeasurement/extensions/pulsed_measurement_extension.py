from qtpy import QtWidgets, QtCore

from pymodaq import Q_, Unit

from pymodaq_utils import utils
from pymodaq_gui import utils as gutils
from pymodaq_utils.config import Config, ConfigError
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_data.data import DataToExport, DataWithAxes

from pymodaq.utils.config import get_set_preset_path
from pymodaq.utils.config import Config as PyMoConfig
from pymodaq.extensions.utils import CustomExt
from pymodaq.control_modules.daq_viewer import DAQ_Viewer

from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher

from scientific_spinbox.widget import ScientificSpinBox

from typing import Union
import time
import numpy as np
from scipy import ndimage
import sys

# import all the predefined pulse sequences
sys.path.append(r"C:\Users\Aurore")
import pymodaq_pulse_sequences

from pymodaq_plugins_pulsedmeasurement.utils import Config as PluginConfig
from pymodaq_plugins_pulsedmeasurement.extensions.main_gui import Ui_MainWindow
from pymodaq_plugins_pulsedmeasurement.hardware.pulsed_controller import (
    PulsedController,
)
from pymodaq_plugins_pulsedmeasurement.hardware._spinapi import SpinAPI

logger = set_logger(get_module_name(__file__))

main_config = Config()
plugin_config = PluginConfig()
config_pymodaq = PyMoConfig()

PLOT_COLORS = [dict(color=color) for color in utils.plot_colors]

# todo: modify this as you wish
EXTENSION_NAME = (
    "Pulsed Measurement"  # the name that will be displayed in the extension list in the
)
# dashboard
CLASS_NAME = (
    "PulsedMeasurementExtension"  # this should be the name of your class defined below
)


class PulsedMeasurementExtension(CustomExt):

    # todo: if you wish to create custom Parameter and corresponding widgets. These will be
    # automatically added as children of self.settings. Morevover, the self.settings_tree will
    # render the widgets in a Qtree. If you wish to see it in your app, add is into a Dock
    params = [
        {
            "title": "Integration period:",
            "name": "period",
            "type": "int",
            "value": 100,
        },
        {
            "title": "Filter width:",
            "name": "conv_std_dev",
            "type": "float",
            "value": 20,
        },
        {"title": "Number of pulses:", "name": "N_pulses", "type": "int", "value": 5},
        {
            "title": "Pulse length (s):",
            "name": "laser_length",
            "type": "float",
            "value": 5e-6,
        },
        {
            "title": "Extraction margin (s):",
            "name": "margin",
            "type": "float",
            "value": 50e-9,
        },
    ]

    sweep_counter = 0
    extraction_done_signal = QtCore.Signal(DataToExport)
    do_integration_signal = QtCore.Signal(DataToExport)
    integration_done_signal = QtCore.Signal(DataToExport)

    def __init__(self, parent: gutils.DockArea, dashboard):
        super().__init__(parent, dashboard)

        # info: in an extension, if you want to interact with ControlModules you have to use the
        # object: self.modules_manager which is a ModulesManager instance from the dashboard

        self.setup_ui()

    def setup_docks(self):
        """Mandatory method to be subclassed to setup the docks layout

        Examples
        --------
        >>>self.docks['ADock'] = gutils.Dock('ADock name')
        >>>self.dockarea.addDock(self.docks['ADock'])
        >>>self.docks['AnotherDock'] = gutils.Dock('AnotherDock name')
        >>>self.dockarea.addDock(self.docks['AnotherDock'''], 'bottom', self.docks['ADock'])

        See Also
        --------
        pyqtgraph.dockarea.Dock
        """
        logger.debug("setting docks")

        self.docks["DetDock"] = gutils.Dock("Counter")
        self.dockarea.addDock(self.docks["DetDock"])
        self.detector = DAQ_Viewer(
            self.dockarea,
            title="Counter",
            daq_type="DAQ1D",
            dock_viewer=self.docks["DetDock"],
        )
        self.detector.detector = "PulsedCounter"
        self.controller = PulsedController()
        self.controller.init_counter()
        self.controller.init_pulseblaster()

        self.docks["program"] = gutils.Dock("Pulse Program")
        self.dockarea.addDock(self.docks["program"])
        window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(window)
        self.docks["program"].addWidget(window)

        self.docks["Settings"] = gutils.Dock("Settings")
        self.dockarea.addDock(self.docks["Settings"])
        self.docks["Settings"].addWidget(self.settings_tree, 10)

        self.docks["extracted"] = gutils.Dock("Average pulse")
        self.dockarea.addDock(self.docks["extracted"])
        self.area_computed = gutils.DockArea()
        self.docks["extracted"].addWidget(self.area_computed)
        self.extracted_viewer = ViewerDispatcher(self.area_computed)

        self.docks["integration"] = gutils.Dock("Analysis")
        self.dockarea.addDock(self.docks["integration"])
        self.area_computed = gutils.DockArea()
        self.docks["integration"].addWidget(self.area_computed)
        self.integrated_viewer = ViewerDispatcher(self.area_computed)

        # self.docks["extracted"] = gutils.Dock("Extracted Pulse")
        # self.dockarea.addDock(self.docks["extracted"])
        # self.extracted_det = DAQ_Viewer(
        #     self.dockarea,
        #     title="Extracted Pulse",
        #     daq_type="DAQ1D",
        #     dock_viewer=self.docks["extracted"],
        # )
        # self.extracted_det.detector = "Extraction"
        # self.extracted_det.controller = self

        logger.debug("docks are set")

    def setup_actions(self):
        """Method where to create actions to be subclassed. Mandatory

        Examples
        --------
        >>> self.add_action('quit', 'Quit', 'close2', "Quit program")
        >>> self.add_action('grab', 'Grab', 'camera', "Grab from camera", checkable=True)
        >>> self.add_action('load', 'Load', 'Open', "Load target file (.h5, .png, .jpg) or data from camera"
            , checkable=False)
        >>> self.add_action('save', 'Save', 'SaveAs', "Save current data", checkable=False)

        See Also
        --------
        ActionManager.add_action
        """
        self.add_action("quit", "Quit", "close2", "Quit program")

    def connect_things(self):
        """Connect actions and/or other widgets signal to methods"""
        self.ui.button_program.clicked.connect(self.program_pulseblaster)
        self.ui.param_general_Binwidth.currentTextChanged.connect(self.change_binwidth)
        self.connect_action("quit", self.quit_fun)
        self.detector.grab_done_signal.connect(self.process_extraction)
        self.extraction_done_signal.connect(self.plot_extracted_results)
        self.extraction_done_signal.connect(self.periodic_integration)
        self.do_integration_signal.connect(self.process_integration)
        self.integration_done_signal.connect(self.plot_integrated_results)

    def periodic_integration(self, dte: DataToExport):
        if self.sweep_counter < self.settings.child("period").value():
            self.sweep_counter += 1
            return
        else:
            print(self.sweep_counter)
            self.sweep_counter = 0
            self.do_integration_signal.emit(dte)

    def process_integration(self, dte: DataToExport):
        print("oui oui oui")
        # extracted_pulses = np.array(dte[0].deepcopy().data)
        # try:
        #     roi_dict = self.extracted_viewer.viewers[0].view.roi_manager.ROIs
        # except:
        #     return
        # windows = [
        #     (round(roi.pos()[0]), round(roi.pos()[1])) for roi in roi_dict.values()
        # ]
        # if windows == []:
        #     return
        # if windows[0][0] > windows[1][0]:
        #     windows = [windows[1], windows[0]]
        # ratios = np.empty(extracted_pulses.shape[0] - 2)
        # errors = np.empty(extracted_pulses.shape[0] - 2)
        # for i in range(1, extracted_pulses.shape[0] - 1):
        #     polarized = extracted_pulses[i][windows[1][0] : windows[1][1]]
        #     readout = extracted_pulses[i + 1][windows[0][0] : windows[0][1]]
        #     if np.array_equal(polarized, np.zeros_like(polarized)):
        #         ratio = 0
        #         err_ratio = 0
        #     else:
        #         ratio = np.mean(readout) / np.mean(polarized)
        #         err_ratio = ratio * np.sqrt(
        #             np.std(readout) ** 2 / np.mean(readout) ** 2
        #             + np.std(polarized) ** 2 / np.mean(polarized) ** 2
        #         )
        #     ratios[i - 1] = ratio
        #     errors[i - 1] = err_ratio

        # dte_processed = DataToExport("integrated")
        # dte_processed.append(
        #     DataWithAxes(
        #         name="Decay",
        #         source="calculated",
        #         data=[ratios],
        #         errors=[errors],
        #         labels=["Decay"],
        #     )
        # )

        # self.integration_done_signal.emit(dte_processed)

    def process_extraction(self, dte: DataToExport):
        dte_processed = DataToExport("extracted")
        dwa = dte[0].deepcopy()
        extracted_pulses = self.ungated_conv_deriv(
            count_data=dwa.data[0],
            number_of_lasers=self.settings.child("N_pulses").value(),
            laser_length=round(
                self.settings.child("laser_length").value()
                / self.controller.counter.get_binwidth()
            ),
            margin=round(
                self.settings.child("margin").value()
                / self.controller.counter.get_binwidth()
            ),
            conv_std_dev=self.settings.child("conv_std_dev").value(),
        )["laser_counts_arr"]
        dte_processed.append(
            DataWithAxes(
                name="Extracted Pulses",
                source="calculated",
                data=[pulse for pulse in extracted_pulses],
            )
        )
        self.extraction_done_signal.emit(dte_processed)

    def plot_extracted_results(self, dte: DataToExport):
        avg_pulse = np.mean(np.array(dte[0].data), axis=0)
        dte_to_plot = DataToExport("avg_pulse")
        dte_to_plot.append(
            DataWithAxes(
                name="Average pulse",
                source="calculated",
                data=[avg_pulse],
                labels=["Average pulse"],
            )
        )
        self.extracted_viewer.show_data(dte_to_plot)

    def plot_integrated_results(self, dte: DataToExport):
        self.integrated_viewer.show_data(dte)

    def snap(self):
        self.detector.snap()

    def setup_menu(self, menubar: QtWidgets.QMenuBar = None):
        """Non mandatory method to be subclassed in order to create a menubar

        create menu for actions contained into the self._actions, for instance:

        Examples
        --------
        >>>file_menu = menubar.addMenu('File')
        >>>self.affect_to('load', file_menu)
        >>>self.affect_to('save', file_menu)

        >>>file_menu.addSeparator()
        >>>self.affect_to('quit', file_menu)

        See Also
        --------
        pymodaq.utils.managers.action_manager.ActionManager
        """
        # todo create and populate menu using actions defined above in self.setup_actions
        pass

    def value_changed(self, param):
        """Actions to perform when one of the param's value in self.settings is changed from the
        user interface

        For instance:
        if param.name() == 'do_something':
            if param.value():
                print('Do something')
                self.settings.child('main_settings', 'something_done').setValue(False)

        Parameters
        ----------
        param: (Parameter) the parameter whose value just changed
        """
        pass

    def program_pulseblaster(self):
        """
        Build the selected pulse sequence based on the parameters given in the GUI, program it in the pulseblaster and plot it in the GUI.
        """
        ##############################################
        # Parse sequence parameters from GUI spinboxes
        ##############################################
        params_dict: dict[str, Union[Q_, int]] = {}
        for attr in dir(self.ui):
            # parse the parameters specific to the sequence or common to all sequences
            if (
                attr.startswith(f"param_{self.ui.selected_sequence}")
                or attr.startswith("param_general")
                or attr.startswith("param_channel")
            ):
                param_widget = getattr(self.ui, attr)
                if isinstance(param_widget, QtWidgets.QSpinBox):
                    param_value = int(param_widget.value())
                elif isinstance(param_widget, ScientificSpinBox):
                    gui_quantity = param_widget.baseQuantity
                    param_value = Q_(
                        float(gui_quantity.magnitude), Unit(str(gui_quantity.units))
                    )  # because of unit registry conflict we have to "clone" the quantity from the GUI
                params_dict["_".join(map(str, attr.split("_")[2:]))] = param_value
        self.sequence_params = params_dict
        #########################
        # Initialize the detector
        #########################
        sequence = getattr(
            pymodaq_pulse_sequences, f"Sequence_{self.ui.selected_sequence}"
        )(**params_dict)
        self.detector.settings.child("detector_settings", "name").setValue(
            self.ui.selected_sequence
        )
        self.detector.settings.child("detector_settings", "length").setValue(
            sequence.length().to("ns").magnitude
        )
        self.detector.settings.child("detector_settings", "final_wait").setValue(
            params_dict["Final_wait"].to("ns").magnitude
        )
        if not self.detector.initialized_state:
            self.detector.init_hardware()
        time.sleep(0.5)

        ##############################################################
        # Program the sequence from a defined class in sequence_folder
        ##############################################################
        instructions = sequence.build()
        for chan, inst in instructions:
            self.controller.pulseblaster.set_channel(chan, inst)
        _, _, program_data = self.controller.pulseblaster.visualize_channels()
        self.controller.pulseblaster.compile_channels()
        self.controller.pulseblaster.add_inst(
            0x000000, sequence._final_inst, sequence._final_inst_data, 0
        )
        self.controller.pulseblaster.program()
        print("Pulse sequence programmed")

        #########################################
        # Plot the programmed sequence in the GUI
        #########################################
        self.ui.plot_program.plotItem.clear()
        self.ui.plot_program.plotItem.setLabel("bottom", "Time (ns)")
        self.ui.plot_program.plotItem.addLegend()
        for i, chan_data in enumerate(program_data):
            channnel = chan_data[0]
            self.ui.plot_program.plotItem.plot(
                chan_data[1],
                2 * i + np.array(chan_data[2]),
                name=f"channel {channnel}",
                pen=PLOT_COLORS[i],
            )
        # self.ui.plot_program.plotItem.update()

    def change_binwidth(self):
        new_bin = float(self.ui.param_general_Binwidth.currentText().split(" ")[0])
        self.controller.counter.set_binwidth(new_bin)
        print(f"Changed binwidth to {self.controller.counter.get_binwidth()}")

    @staticmethod
    def ungated_conv_deriv(
        count_data, number_of_lasers, laser_length, margin, conv_std_dev=20.0
    ):
        """Detects the laser pulses in the ungated timetrace data and extracts
            them.

        @param numpy.ndarray count_data: The raw timetrace data (1D) from an ungated fast counter
        @param float conv_std_dev: The standard deviation of the gaussian used for smoothing

        @return 2D numpy.ndarray:   2D array, the extracted laser pulses of the timetrace.
                                    dimensions: 0: laser number, 1: time bin

        Procedure:
            Edge Detection:
            ---------------

            The count_data array with the laser pulses is smoothed with a
            gaussian filter (convolution), which used a defined standard
            deviation of 10 entries (bins). Then the derivation of the convolved
            time trace is taken to obtain the maxima and minima, which
            corresponds to the rising and falling edge of the pulses.

            The convolution with a gaussian removes nasty peaks due to count
            fluctuation within a laser pulse and at the same time ensures a
            clear distinction of the maxima and minima in the derived convolved
            trace.

            The maxima and minima are not found sequentially, pulse by pulse,
            but are rather globally obtained. I.e. the convolved and derived
            array is searched iteratively for a maximum and a minimum, and after
            finding those the array entries within the 4 times
            self.conv_std_dev (2*self.conv_std_dev to the left and
            2*self.conv_std_dev) are set to zero.

            The crucial part is the knowledge of the number of laser pulses and
            the choice of the appropriate std_dev for the gauss filter.

            To ensure a good performance of the edge detection, you have to
            ensure a steep rising and falling edge of the laser pulse! Be also
            careful in choosing a large conv_std_dev value and using a small
            laser pulse (rule of thumb: conv_std_dev < laser_length/10).
        """
        # print(count_data)
        # Create return dictionary
        return_dict = {
            "laser_counts_arr": np.empty(0, dtype="int64"),
            "laser_indices_rising": np.empty(0, dtype="int64"),
            "laser_indices_falling": np.empty(0, dtype="int64"),
        }
        if not isinstance(number_of_lasers, int):
            return return_dict

        # apply gaussian filter to remove noise and compute the gradient of the timetrace sum
        try:
            conv = ndimage.filters.gaussian_filter1d(
                count_data.astype(float), conv_std_dev
            )
        except:
            conv = np.zeros(count_data.size)
        try:
            conv_deriv = np.gradient(conv)
        except:
            conv_deriv = np.zeros(conv.size)

        # if gaussian smoothing or derivative failed, the returned array only contains zeros.
        # Check for that and return also only zeros to indicate a failed pulse extraction.
        if len(conv_deriv.nonzero()[0]) == 0:
            return_dict["laser_counts_arr"] = np.zeros(
                (number_of_lasers, laser_length), dtype="int64"
            )
            return return_dict

        # use a reference for array, because the exact position of the peaks or dips
        # (i.e. maxima or minima, which are the inflection points in the pulse) are distorted by
        # a large conv_std_dev value.
        try:
            conv = ndimage.filters.gaussian_filter1d(count_data.astype(float), 10)
        except:
            conv = np.zeros(count_data.size)
        try:
            conv_deriv_ref = np.gradient(conv)
        except:
            conv_deriv_ref = np.zeros(conv.size)

        # initialize arrays to contain indices for all rising and falling
        # flanks, respectively
        rising_ind = np.empty(number_of_lasers, dtype="int64")
        falling_ind = np.empty(number_of_lasers, dtype="int64")

        # Find as many rising and falling flanks as there are laser pulses in
        # the trace:
        for i in range(number_of_lasers):
            # save the index of the absolute maximum of the derived time trace
            # as rising edge position
            rising_ind[i] = np.argmax(conv_deriv)

            # refine the rising edge detection, by using a small and fixed
            # conv_std_dev parameter to find the inflection point more precise
            start_ind = int(rising_ind[i] - conv_std_dev)
            if start_ind < 0:
                start_ind = 0

            stop_ind = int(rising_ind[i] + conv_std_dev)
            if stop_ind > len(conv_deriv):
                stop_ind = len(conv_deriv)

            if start_ind == stop_ind:
                stop_ind = start_ind + 1

            rising_ind[i] = start_ind + np.argmax(conv_deriv_ref[start_ind:stop_ind])

            # set this position and the surrounding of the saved edge to 0 to
            # avoid a second detection
            if rising_ind[i] < 2 * conv_std_dev:
                del_ind_start = 0
            else:
                del_ind_start = rising_ind[i] - int(2 * conv_std_dev)
            if (conv_deriv.size - rising_ind[i]) < 2 * conv_std_dev:
                del_ind_stop = conv_deriv.size - 1
            else:
                del_ind_stop = rising_ind[i] + int(2 * conv_std_dev)
                conv_deriv[del_ind_start:del_ind_stop] = 0

            # save the index of the absolute minimum of the derived time trace
            # as falling edge position
            falling_ind[i] = np.argmin(conv_deriv)

            # refine the falling edge detection, by using a small and fixed
            # conv_std_dev parameter to find the inflection point more precise
            start_ind = int(falling_ind[i] - conv_std_dev)
            if start_ind < 0:
                start_ind = 0

            stop_ind = int(falling_ind[i] + conv_std_dev)
            if stop_ind > len(conv_deriv):
                stop_ind = len(conv_deriv)

            if start_ind == stop_ind:
                stop_ind = start_ind + 1

            falling_ind[i] = start_ind + np.argmin(conv_deriv_ref[start_ind:stop_ind])

            # set this position and the sourrounding of the saved flank to 0 to
            #  avoid a second detection
            if falling_ind[i] < 2 * conv_std_dev:
                del_ind_start = 0
            else:
                del_ind_start = falling_ind[i] - int(2 * conv_std_dev)
            if (conv_deriv.size - falling_ind[i]) < 2 * conv_std_dev:
                del_ind_stop = conv_deriv.size - 1
            else:
                del_ind_stop = falling_ind[i] + int(2 * conv_std_dev)
            conv_deriv[del_ind_start:del_ind_stop] = 0

        # sort all indices of rising and falling flanks
        rising_ind.sort()
        falling_ind.sort()

        # find the maximum laser length to use as size for the laser array
        # laser_length = np.abs(np.max(falling_ind - rising_ind))

        # initialize the empty output array
        laser_arr = np.zeros(
            (number_of_lasers, laser_length + 2 * margin), dtype="int64"
        )
        # slice the detected laser pulses of the timetrace and save them in the
        # output array according to the found rising edge
        for i in range(number_of_lasers):
            if rising_ind[i] + laser_length + margin > count_data.size:
                lenarr = count_data[rising_ind[i] - margin :].size
                laser_arr[i, 0:lenarr] = count_data[rising_ind[i] - margin :]
            elif rising_ind[i] < margin:
                laser_arr[i, margin - rising_ind[i] :] = count_data[
                    0 : rising_ind[i] + laser_length + margin
                ]
            else:
                laser_arr[i] = count_data[
                    rising_ind[i] - margin : rising_ind[i] + laser_length + margin
                ]

        return_dict["laser_counts_arr"] = laser_arr.astype("int64")
        return_dict["laser_indices_rising"] = rising_ind
        return_dict["laser_indices_falling"] = falling_ind
        return return_dict


def main():
    from pymodaq_gui.utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset
    from pymodaq.utils.messenger import messagebox

    app = mkQApp(EXTENSION_NAME)
    # try:
    #     preset_file_name = plugin_config(
    #         "presets", f"preset_for_{PulsedMeasurementExtension.lower()}"
    #     )
    #     load_dashboard_with_preset(preset_file_name, EXTENSION_NAME)
    #     app.exec()

    # except ConfigError as e:
    #     messagebox(
    #         f'No entry with name f"preset_for_{PulsedMeasurementExtension.lower()}" has been configured'
    #         f"in the plugin config file. The toml entry should be:\n"
    #         f"[presets]"
    #         f"preset_for_{PulsedMeasurementExtension.lower()} = {'a name for an existing preset'}"
    #     )
    preset_file_name = config_pymodaq("presets", "default_preset_for_datamixer")
    dashboard, extension, win = load_dashboard_with_preset(
        preset_file_name, EXTENSION_NAME
    )
    app.exec()

    return dashboard, extension, win


if __name__ == "__main__":
    main()
