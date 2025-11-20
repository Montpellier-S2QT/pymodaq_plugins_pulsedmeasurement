from qtpy import QtWidgets

from pymodaq_gui import utils as gutils
from pymodaq_utils.config import Config, ConfigError
from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq.utils.config import get_set_preset_path
from pymodaq.extensions.utils import CustomExt
from pymodaq.control_modules.daq_viewer import DAQ_Viewer

from scientific_spinbox.widget import ScientificSpinBox

import time
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
    params = []

    def __init__(self, parent: gutils.DockArea, dashboard):
        super().__init__(parent, dashboard)

        # info: in an extension, if you want to interact with ControlModules you have to use the
        # object: self.modules_manager which is a ModulesManager instance from the dashboard

        self.setup_ui()
        self.controller = self.modules_manager.get_mod_from_name(
            "PulsedCounter"
        ).controller

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
        # Make the dock for the detector
        self.docks["DetDock"] = gutils.Dock("Detector Viewer")
        self.dockarea.addDock(self.docks["DetDock"])
        self.detector = DAQ_Viewer(self.dockarea, dock_viewer=self.docks["DetDock"])
        self.detector.daq_type = "DAQ1D"
        self.detector.detector = "PulsedCounter"

        self.docks["PMDock"] = gutils.Dock("Pulsed Measurement")
        self.dockarea.addDock(self.docks["PMDock"])
        window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        print(type(self.ui))
        self.ui.setupUi(window)
        self.docks["PMDock"].addWidget(window)

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
        # TODO:
        # - change the pulseblaster wrapper so that we can get the plot data from visualize_channels
        # - update sequence plot with plot data (see pyqtgraph)

        ##############################################
        # Parse sequence parameters from GUI spinboxes
        ##############################################
        params_dict = {}
        for attr in dir(self.ui):
            # parse the parameters specific to the sequence or common to all sequences
            if (
                attr.startswith(f"param_{self.ui.selected_sequence}")
                or attr.startswith("param_general")
                or attr.startswith("param_channel")
            ):
                param_widget = getattr(self.ui, attr)
                if isinstance(param_widget, QtWidgets.QSpinBox):
                    param_value = param_widget.value()
                elif isinstance(param_widget, ScientificSpinBox):
                    param_value = float(param_widget.baseQuantity.magnitude)
                params_dict["_".join(map(str, attr.split("_")[2:]))] = param_value

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
            sequence.length()
        )
        self.detector.settings.child("detector_settings", "final_wait").setValue(
            params_dict["Final_wait"]
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
        # plot_data = self.detector.detector.controller.pulseblaster.visualize_channels
        self.controller.pulseblaster.add_inst(
            0x000000, sequence._final_inst, sequence._final_inst_data, 0
        )
        self.controller.pulseblaster.program()
        print("Pulse sequence programmed")

        #########################################
        # Plot the programmed sequence in the GUI
        #########################################
        pass

    def change_binwidth(self):
        new_bin = float(self.ui.param_general_Binwidth.currentText().split(" ")[0])
        self.controller.counter.set_binwidth(new_bin)


def main():
    from pymodaq.utils.gui_utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset
    from pymodaq.utils.messenger import messagebox

    app = mkQApp(EXTENSION_NAME)
    try:
        preset_file_name = plugin_config(
            "presets", f"preset_for_{PulsedMeasurementExtension.lower()}"
        )
        load_dashboard_with_preset(preset_file_name, EXTENSION_NAME)
        app.exec()

    except ConfigError as e:
        messagebox(
            f'No entry with name f"preset_for_{PulsedMeasurementExtension.lower()}" has been configured'
            f"in the plugin config file. The toml entry should be:\n"
            f"[presets]"
            f"preset_for_{PulsedMeasurementExtension.lower()} = {'a name for an existing preset'}"
        )


if __name__ == "__main__":
    main()
