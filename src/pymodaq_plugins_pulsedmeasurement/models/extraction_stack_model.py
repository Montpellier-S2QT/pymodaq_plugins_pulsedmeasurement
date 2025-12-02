import numpy as np
from scipy import ndimage
from pymodaq.extensions.data_mixer.model import (
    DataMixerModel,
    np,
)  # np will be used in method eval of the formula

from pymodaq_utils.math_utils import gauss1D, my_moment

from pymodaq_data.data import DataToExport, DataWithAxes
from pymodaq_gui.parameter import Parameter

from pymodaq.extensions.data_mixer.parser import (
    extract_data_names,
    split_formulae,
    replace_names_in_formula,
)


def ungated_conv_deriv(count_data, number_of_lasers, conv_std_dev=20.0):
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
        conv = ndimage.filters.gaussian_filter1d(count_data.astype(float), conv_std_dev)
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
            (number_of_lasers, 10), dtype="int64"
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
    laser_length = np.max(falling_ind - rising_ind)

    # initialize the empty output array
    laser_arr = np.zeros((number_of_lasers, laser_length), dtype="int64")
    # slice the detected laser pulses of the timetrace and save them in the
    # output array according to the found rising edge
    for i in range(number_of_lasers):
        if rising_ind[i] + laser_length > count_data.size:
            lenarr = count_data[rising_ind[i] :].size
            laser_arr[i, 0:lenarr] = count_data[rising_ind[i] :]
        else:
            laser_arr[i] = count_data[rising_ind[i] : rising_ind[i] + laser_length]

    return_dict["laser_counts_arr"] = laser_arr.astype("int64")
    return_dict["laser_indices_rising"] = rising_ind
    return_dict["laser_indices_falling"] = falling_ind
    return return_dict


class DataMixerModelFit(DataMixerModel):
    params = []

    def ini_model(self):
        pass

    def update_settings(self, param: Parameter):
        pass

    def process_dte(self, dte: DataToExport):
        dte_processed = DataToExport("computed")
        dwa = dte.get_data_from_full_name(
            "Spectrum - ROI_00/Hlineout_ROI_00"
        ).deepcopy()

        dte_processed.append(dwa)
        dte_processed.append(dwa.fit(gaussian_fit, self.get_guess(dwa)))

        return dte_processed

    @staticmethod
    def get_guess(dwa):
        offset = np.min(dwa).value()
        moments = my_moment(dwa.axes[0].get_data(), dwa.data[0])
        amp = (np.max(dwa) - np.min(dwa)).value()
        x0 = float(moments[0])
        dx = float(moments[1])

        return amp, x0, dx, offset
