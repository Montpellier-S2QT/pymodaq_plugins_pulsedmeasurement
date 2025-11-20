# -*- coding: utf-8 -*-

"""This file contains the Python Class wrapper for the FastComTec MSC8 card to interface it with PyMoDAQ.
Inspired from the Qudi interface code."""

import ctypes
import time
import numpy as np
from .structures import (
    AcqStatus,
    AcqSettings,
    BOARDSETTING,
)


class MCS8:
    def __init__(
        self,
        device: int = 0,
        dll_path: str = "C:\Windows\System32\DMCS8.dll",
        min_binwidth: float = 200e-12,
        max_sweep_length=6.8,  # it can go up to 8.3 days
        trigger_safety: float = 0,
    ):
        self._ndev = device
        self._dll_path = dll_path
        self.dll = ctypes.windll.LoadLibrary(self._dll_path)
        self._min_binwidth = min_binwidth
        self._max_sweep_length = max_sweep_length
        self._trigger_safety = trigger_safety
        self._stopped_or_halt = "stopped"  # needed because the fastcomtec status doesn't make the difference between paused ans stopped

    @property
    def trigger_safety(self):
        return self._trigger_safety

    def run_cmd(self, command: str) -> str:
        """Send a command string to the device and return the modified string."""
        # Create a mutable buffer with extra space for the result
        buffer_size = len(command) + 1024  # Command + space for sprintf result
        command_buffer = ctypes.create_string_buffer(
            command.encode("utf-8"), buffer_size
        )

        # Configure the DLL function signature
        self.dll.RunCmd.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.dll.RunCmd.restype = None

        # Call the function - it will modify command_buffer in-place
        self.dll.RunCmd(self._ndev, command_buffer)

        # Return the modified string
        return command_buffer.value.decode("utf-8")

    def get_status(self):
        """
        Receives the current status of the Fast Counter and outputs it as return value.
        0 = unconfigured
        1 = idle
        2 = running
        3 = paused
        -1 = error state
        """
        status = AcqStatus()
        self.dll.GetStatusData(ctypes.byref(status), self._ndev)
        # status.started = 3 means that fastcomtec is about to stop
        while status.started == 3:
            time.sleep(0.1)
            self.dll.GetStatusData(ctypes.byref(status), self._ndev)
        if status.started == 1:
            return 2
        elif status.started == 0:
            if self._stopped_or_halt == "stopped":
                return 1
            elif self._stopped_or_halt == "halt":
                return 3
            else:
                print(
                    "There is an unknown status from FastComtec. The status message was %s"
                    % (str(status.started))
                )
                return -1
        else:
            print(
                "There is an unknown status from FastComtec. The status message was %s"
                % (str(status.started))
            )
            return -1

    def start_measure(self):
        """Start the measurement."""
        status = self.dll.Start(self._ndev)
        while self.get_status() != 2:
            time.sleep(0.05)
        return status

    def stop_measure(self):
        """Stop the measurement."""
        self._stopped_or_halt = "stopped"
        status = self.dll.Halt(self._ndev)
        while self.get_status() != 1:
            time.sleep(0.05)
        return status

    def pause_measure(self):
        """Make a pause in the measurement, which can be continued."""
        self._stopped_or_halt = "halt"
        status = self.dll.Halt(self._ndev)
        while self.get_status() != 3:
            time.sleep(0.05)
        return status

    def continue_measure(self):
        """Continue a paused measurement."""
        status = self.dll.Continue(self._ndev)
        while self.get_status() != 2:
            time.sleep(0.05)
        return status

    def get_binwidth(self):
        """Returns the width of a single timebin in the timetrace in seconds.
        The read out bitshift will be converted to binwidth. The binwidth is
        defined as 2**bitshift*minimal_binwidth.
        """
        return self._min_binwidth * (2 ** int(self._get_bitshift()))

    def set_binwidth(self, binwidth):
        """Set the binwidth of the FastComTec. Returns the actual updated binwidth.
        The binwidth is converted into to an appropiate bitshift defined as 2**bitshift*minimal_binwidth.
        """
        bitshift = int(np.log2(binwidth / self._min_binwidth))
        new_bitshift = self._set_bitshift(bitshift)
        return self._min_binwidth * (2**new_bitshift)

    def get_length(self):
        """Return the length of the current measurement in number of bins (int)."""
        setting = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(setting), self._ndev)
        length = int(setting.range)
        return length * self.get_binwidth()

    def set_length(self, length):
        """
        Set the measurement length.

        Parameters:
        length (float): length of the measurement in s

        Return:
        Updated measurement length (float)
        """
        if length < self._max_sweep_length:
            N_bins = int((length - self._trigger_safety) / self.get_binwidth())
            # Smallest increment is 64 bins. Since it is better if the range is too short than too long, round down
            N_bins = int(64 * int(N_bins / 64))
            cmd = f"range={int(N_bins)}"
            self.run_cmd(cmd)
            # insert sleep time, otherwise fast counter crashed sometimes!
            time.sleep(0.5)
            return N_bins * self.get_binwidth()
        else:
            raise ValueError(f"Dimensions {length} are too large for fast counter1!")

    def get_data(self):
        """
        Returns the timetrace data from the FastComTec as a 1D numpy array (dtype = int64): array[bin_index].
        """
        setting = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(setting), self._ndev)
        N = setting.range
        data = np.empty((N,), dtype=np.uint32)

        p_type_ulong = ctypes.POINTER(ctypes.c_uint32)
        ptr = data.ctypes.data_as(p_type_ulong)
        self.dll.LVGetDat(ptr, self._ndev)
        time_trace = np.int64(data)

        return time_trace

    def get_timestamps(self):
        """Return a 1D numpy array with the data time stamps."""
        setting = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(setting), self._ndev)
        N = setting.range
        timestamps = self.get_binwidth() * np.arange(1, N + 1)
        return timestamps

    def get_start_delay(self):
        """Return the starting delay which is the waiting time after the trigger before the FastComTec starts counting."""
        bsetting = BOARDSETTING()
        self.dll.GetMCSSetting(ctypes.byref(bsetting), self._ndev)
        delay_s = (
            bsetting.fstchan * 6.4e-9
        )  # the delay is internally normalized by 6.4ns (minimum acquisition delay)
        return delay_s

    def set_start_delay(self, delay_s):
        """Sets the record delay length.

        Parameters:
        delay_s (float): Record delay after receiving a start trigger

        Return:
        Actual updated delay in s (float)
        """
        # A delay can only be adjusted in steps of 6.4ns
        delay = np.rint(delay_s / 6.4e-9)
        cmd = f"fstchan={int(delay)}"
        self.run_cmd(cmd)
        return self.get_start_delay

    def get_boardsettings(self):
        bsetting = BOARDSETTING()
        self.dll.GetMCSSetting(ctypes.byref(bsetting), self._ndev)
        return bsetting

    def get_acquisition_settings(self):
        settings = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(settings), self._ndev)
        return settings

    def _get_bitshift(self):
        """Get the bitshift from the FastComTec (used to define the binwidth)."""
        settings = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(settings), self._ndev)
        return int(settings.bitshift)

    def _set_bitshift(self, bitshift):
        """Set the bitshift of the FastComTec. Returns the actual updated bitshift."""
        cmd = f"bitshift={bitshift}"
        self.run_cmd(cmd)
        return self._get_bitshift()
