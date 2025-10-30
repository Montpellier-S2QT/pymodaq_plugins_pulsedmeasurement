# -*- coding: utf-8 -*-

"""This file contains the Python Class wrapper for the FastComTec MSC8 card to interface it with PyMoDAQ.
Inspired from the Qudi interface code."""

import ctypes
import time
import numpy as np
from pymodaq_plugins_fastcomtec.hardware.structures import (
    AcqStatus,
    AcqSettings,
    BOARDSETTING,
)


class MSC8:
    def __init__(
        self,
        device: int = 0,
        dll_path: str = "C:\Windows\System32\DMCS8.dll",
        min_binwidth: float = 0.2e-9,
        max_sweep_length=6.8,
        trigger_safety: float = 400e-9,
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
        # status.started = 3 means that fct is about to stop
        while status.started == 3:
            time.sleep(0.1)
            self.dll.GetStatusData(ctypes.byref(status), self._ndev)
        if status.started == 1:
            return 2
        elif status.started == 0:
            if self.stopped_or_halt == "stopped":
                return 1
            elif self.stopped_or_halt == "halt":
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
        self.stopped_or_halt = "stopped"
        status = self.dll.Halt(self._ndev)
        while self.get_status() != 1:
            time.sleep(0.05)
        return status

    def pause_measure(self):
        """Make a pause in the measurement, which can be continued."""
        self.stopped_or_halt = "halt"
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
        return self._min_binwidth * (2 ** int(self.get_bitshift()))

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
        self.dll.GetSettingData(ctypes.byref(setting), 0)
        length = int(setting.ranges)
        return length

    def set_length(self, length):
        """
        Set the measurement length.

        Parameters:
        length (float): length of the measurement in s

        Return:
        Updated measurement length (float)
        """
        if length < self._max_sweep_length:
            # Convert the length in number of bins
            # NOTE: maybe it should be given in number of bins directly
            if length % self.get_binwidth() != 0:
                raise ValueError(
                    f"Measurement length should be a multiple of the binwidth. Current binwidth: {self.get_binwidth()}"
                )
            else:
                length_bins = length // self.get_binwidth()
                # Smallest increment is 64 bins. Since it is better if the range is too short than too long, round down
                length_bins = int(64 * int(length_bins / 64))
                cmd = "RANGE={0}".format(int(length_bins))
                self.dll.RunCmd(0, bytes(cmd, "ascii"))
                # insert sleep time, otherwise fast counter crashed sometimes!
                time.sleep(0.5)
                return length_bins
        else:
            raise ValueError(
                "Dimensions {0} are too large for fast counter1!".format(length)
            )

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

    def _get_bitshift(self):
        """Get the bitshift from the FastComTec (used to define the binwidth)."""
        settings = AcqSettings()
        self.dll.GetSettingData(ctypes.byref(settings), self._ndev)
        return int(settings.bitshift)

    def _set_bitshift(self, bitshift):
        """Set the bitshift of the FastComTec. Returns the actual updated bitshift."""
        cmd = "BITSHIFT={0}".format(bitshift)
        self.dll.RunCmd(self._ndev, bytes(cmd, "ascii"))
        return self.get_bitshift()

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
        cmd = "fstchan={0}".format(int(delay))
        self.dll.RunCmd(self._ndev, bytes(cmd, "ascii"))
        return self.get_start_delay
