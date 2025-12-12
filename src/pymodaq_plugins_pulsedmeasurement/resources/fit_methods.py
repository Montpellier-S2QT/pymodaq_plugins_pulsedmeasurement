# -*- coding: utf-8 -*-
"""
This file contains all the methods used to fit the data from pulsed measurements in the extension.
It is heavily inspired on the fitting methods of qudi.
"""

from lmfit import Model
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import filters
from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))


###################
# Utility Functions
###################


def compute_fft(x_data, y_data, zeropad=0):
    """
    Compute the fast Fourier transform with zero padding and return the spectrum and its frequency axis.

    :param x_data: x axis of the data (1D array).
    :param y_data: y axis of the data (1D array). Same size as x_data.
    :param zeropad: add zeros at the end of y_data for more precise spectrum.
                    zeropad=1 outputs a spectrum which is the same length as the input data.
    """
    y_data_corr = y_data - np.mean(y_data)
    zeropad_arr = np.zeros(len(y_data_corr) * (zeropad + 1))
    zeropad_arr[: len(y_data_corr)] = y_data_corr
    fft_y = np.abs(np.fft.fft(zeropad_arr))
    middle = int((len(zeropad_arr) + 1) // 2)
    stepsize = x_data[1] - x_data[0]
    fft_x = np.fft.fftfreq(len(zeropad_arr), d=stepsize)
    return fft_x[:middle], fft_y[:middle]


###############
# Custom Models
###############


def _constant_model(prefix=None):
    """
    Return a constant Model and its associated Parameter object.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    def constant_function(x, offset):
        return offset

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(constant_function)
    else:
        model = Model(constant_function, prefix=prefix)

    params = model.make_params()

    return model, params


def _amplitude_model(prefix=None):
    """
    Return a constant scaling Model and its associated Parameter object.
    This is the same model as _constant_model but making another one helps
    reducing the prefix needed in the composite models.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    def amplitude_function(x, amplitude):
        return amplitude

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(amplitude_function)
    else:
        model = Model(amplitude_function, prefix=prefix)

    params = model.make_params()

    return model, params


def _bare_sine_model(prefix=None):
    """
    Return a bare sine Model without amplitude or offset and its associated Parameter object.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    def bare_sine_function(x, frequency, phase):
        return np.sin(2 * np.pi * frequency * x + phase)

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(bare_sine_function)
    else:
        model = Model(bare_sine_function, prefix=prefix)

    params = model.make_params()

    return model, params


def _bare_stretched_decay_model(prefix=None):
    """
    Make a stretched exponential decay Model and its associated Parameter object.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    def bare_stretched_decay_func(x, lifetime, beta):
        return np.exp(-((x / lifetime) ** beta))

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(bare_stretched_decay_func)
    else:
        model = Model(bare_stretched_decay_func, prefix=prefix)

    params = model.make_params()

    return model, params


def _sine_model(prefix=None):
    """
    Return a sine Model with amplitude as well as its associated Parameter object.
    Composite model of _bare_sine_model and _amplitude_model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    bare_sine, _ = _bare_sine_model(prefix=prefix)
    amplitude, _ = _amplitude_model(prefix=prefix)

    sine = bare_sine * amplitude
    params = sine.make_params()

    return sine, params


def _multi_sine_model(N, prefix=None):
    """
    Return a superpositon of N sine Models as well as its associated Parameter object.
    Composite model of _sine_model and _constant_model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    if prefix is None:
        add_text = ""
    else:
        add_text = prefix

    multi_sine = _sine_model(prefix=f"s{0}_" + add_text)[0]

    for i in range(1, N):
        sine, _ = _sine_model(prefix=f"s{i}_" + add_text)
        multi_sine += sine

    params = multi_sine.make_params()

    return multi_sine, params


def sine_decay_model(prefix=None):
    """
    Return a decaying sine Model with an offset as well as its associated Parameter object.
    Composite model of _sine_model, _bare_stretched_decay_model and _constant_model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    sine, _ = _sine_model(prefix=prefix)
    decay, _ = _bare_stretched_decay_model(prefix=prefix)
    offset, _ = _constant_model(prefix=prefix)

    sine_decay = sine * decay + offset
    params = sine_decay.make_params()

    return sine_decay, params


def multi_sine_decay_model(N, prefix=None):
    """
    Return a decaying multi sine Model with an offset as well as its associated Parameter object.
    Composite model of _multi_sine_model, _bare_stretched_decay_model and _constant_model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    multi_sine, _ = _multi_sine_model(N=N, prefix=prefix)
    decay, _ = _bare_stretched_decay_model(prefix=prefix)
    offset, _ = _constant_model(prefix=prefix)

    multi_sine_decay = multi_sine * decay + offset
    params = multi_sine_decay.make_params()

    return multi_sine_decay, params


def stretched_decay_model(prefix=None):
    """
    Return a stretched exponential decay Model with an amplitude and an offset as well as its associated Parameter object.
    Composite model of _bare_stretched_decay_model, _amplitude_model and _constant_model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    decay, _ = _bare_stretched_decay_model(prefix=prefix)
    amplitude, _ = _amplitude_model(prefix=prefix)
    offset, _ = _constant_model(prefix=prefix)

    stretched_decay = amplitude * decay + offset
    params = stretched_decay.make_params()

    return stretched_decay, params


####################################################
# Methods to Guess the Initial Values for Parameters
####################################################


def _sine_guess(x_axis: NDArray, data: NDArray, params):
    fft_x, fft_y = compute_fft(x_axis, data, zeropad=1)
    fft_x_red = fft_x[np.where(fft_y > 0)]
    fft_y_red = fft_y[np.where(fft_y > 0)]
    frequency = fft_x_red[np.argmax(np.log(fft_y_red))]
    # find minimal distance to the next meas point in the corresponding time value>
    min_x_diff = np.ediff1d(x_axis).min()
    # How many points are used to sample the estimated frequency with min_x_diff:
    iter_steps = int(1 / (frequency * min_x_diff))
    if iter_steps < 1:
        iter_steps = 1
    sum_res = np.zeros(iter_steps)
    # Procedure: Create sin waves with different phases and perform a summation.
    #            The sum shows how well the sine was fitting to the actual data.
    #            The best fitting sine should be a maximum of the summed time
    #            trace.
    for iter_s in range(iter_steps):
        func_val = np.sin(
            2 * np.pi * frequency * x_axis + iter_s / iter_steps * 2 * np.pi
        )
        sum_res[iter_s] = np.abs(data - func_val).sum()
    # The minimum indicates where the sine function was fittng the worst,
    # therefore subtract pi. This will also ensure that the estimated phase will
    # be in the interval [-pi,pi].
    phase = sum_res.argmax() / iter_steps * 2 * np.pi - np.pi

    amplitude = (np.max(data) - np.min(data)) / 2

    params["amplitude"].set(value=amplitude, min=0.0)
    params["frequency"].set(
        value=frequency, min=0.0, max=1 / (x_axis[1] - x_axis[1]) * 3
    )
    params["phase"].set(value=phase, min=-np.pi, max=np.pi)

    return params


def stretched_decay_guess(x_axis: NDArray, data: NDArray, params):
    # Roughly smooth the data
    filter_std_dev = 10
    data_smoothed = filters.gaussian_filter1d(data, filter_std_dev)

    # The offset is estimated from the last 10% of the data
    offset = np.mean(data_smoothed[-max(1, int(len(x_axis) / 10)) :])

    # Substraction of the offset and correction of the decay behaviour
    # (decay to a bigger value or decay to a smaller value)
    if data_smoothed[0] < data_smoothed[-1]:
        data_smoothed = offset - data_smoothed
        ampl_sign = -1
    else:
        data_smoothed = data_smoothed - offset
        ampl_sign = 1

    if data_smoothed.min() <= 0:
        data_smoothed = data_smoothed - data_smoothed.min()

    # Take all values up to the standard deviation, the remaining values are
    # more disturbing the estimation then helping:
    for stop_index in range(0, len(x_axis)):
        if data_smoothed[stop_index] <= data_smoothed.std():
            break

    data_level_log = np.log(data_smoothed[0:stop_index])

    # Polynomial fit with a second order polynom on the remaining data
    poly_coef = np.polyfit(x_axis[0:stop_index], data_level_log, deg=2)
    lifetime = 1 / np.sqrt(abs(poly_coef[0]))
    amplitude = np.exp(poly_coef[2])

    # Include all the estimated fit parameter:
    params["amplitude"].set(value=amplitude * ampl_sign)
    params["offset"].set(value=offset)

    min_lifetime = 2 * (x_axis[1] - x_axis[0])
    params["lifetime"].set(value=lifetime, min=min_lifetime)

    # Arbitrary starting point for the stretch factor !
    params["beta"].set(value=2, min=0)

    return params


def sine_decay_guess(x_axis: NDArray, data: NDArray, params):
    offset = np.mean(data)  # The offset is estimated as the averge data
    data_level = data - offset
    ampl_val = (np.max(data) - np.min(data)) / 2

    fft_x, fft_y = compute_fft(x_axis, data_level, zeropad=1)
    stepsize = x_axis[1] - x_axis[0]  # for frequency axis
    fft_x_red = fft_x[np.where(fft_y > 0)]
    fft_y_red = fft_y[np.where(fft_y > 0)]
    frequency = fft_x_red[np.argmax(np.log(fft_y_red))]  # log acts as noise filter

    # Remove noise
    a = np.std(fft_y)
    for i in range(0, len(fft_x)):
        if fft_y[i] <= a:
            fft_y[i] = 0
    # Calculating the width of the FT peak for the estimation of lifetime
    s = 0
    for i in range(0, len(fft_x)):
        s += fft_y[i] * abs(fft_x[1] - fft_x[0]) / max(fft_y)
    lifetime_val = 0.5 / s

    # Find minimal distance to the next meas point in the corresponding x value
    min_x_diff = np.ediff1d(x_axis).min()
    # How many points are used to sample the estimated frequency with min_x_diff:
    iter_steps = int(1 / (frequency * min_x_diff))
    if iter_steps < 1:
        iter_steps = 1
    sum_res = np.zeros(iter_steps)
    # Procedure: Create sin waves with different phases and perform a summation.
    #            The sum shows how well the sine was fitting to the actual data.
    #            The best fitting sine should be a maximum of the summed time
    #            trace.
    for iter_s in range(iter_steps):
        func_val = ampl_val * np.sin(
            2 * np.pi * frequency * x_axis + iter_s / iter_steps * 2 * np.pi
        )
        sum_res[iter_s] = np.abs(data_level - func_val).sum()
    # The minimum indicates where the sine function was fittng the worst,
    # therefore subtract pi. This will also ensure that the estimated phase will
    # be in the interval [-pi,pi].
    phase = (sum_res.argmax() / iter_steps * 2 * np.pi - np.pi) % (2 * np.pi)

    # Set values and bounds of initial parameters
    params["frequency"].set(
        value=frequency,
        min=min(0.1 / (x_axis[-1] - x_axis[0]), fft_x[3]),
        max=min(0.5 / stepsize, fft_x.max() - abs(fft_x[2] - fft_x[0])),
    )
    params["phase"].set(value=phase, min=-2 * np.pi, max=2 * np.pi)
    params["amplitude"].set(value=ampl_val, min=0)
    params["offset"].set(value=offset)

    params["lifetime"].set(
        value=lifetime_val,
        min=2 * (x_axis[1] - x_axis[0]),
        max=1 / (abs(fft_x[1] - fft_x[0]) * 0.5),
    )

    return params


def multi_sine_decay_guess(x_axis: NDArray, data: NDArray, params, N: int):
    # Procedure: make successive single sine_decay fits substracting each result before going to the next fit.
    data_sub = data
    lifetimes = np.empty(N)
    for i in range(N):
        result = sine_decay_fit(x_axis=x_axis, data=data_sub, guesser=_sine_guess)
        if result == -1:
            logger.warning("The multi sine decay guess did not work.")
            return params
        lifetimes[i] = result.params["lifetime"].value
        data_sub -= result.best_fit

        # Fill the parameter dict:
        params[f"s{i}_amplitude"].set(value=result.params["amplitude"].value)
        params[f"s{i}_frequency"].set(value=result.params["frequency"].value)
        params[f"s{i}_phase"].set(value=result.params["phase"].value)

    params["offset"].set(value=data.mean())
    params["lifetime"].set(value=np.mean(lifetimes), min=2 * (x_axis[1] - x_axis[0]))

    return params


#############
# Fit Methods
#############


def stretched_decay_fit(
    x_axis: NDArray, data: NDArray, guesser=stretched_decay_guess, **kwargs
):
    decay, params = stretched_decay_model()
    params = guesser(x_axis, data, params)
    try:
        result = decay.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = -1
        logger.warning("The stretched exponential decay fit did not work.")
    return result


def sine_decay_fit(x_axis: NDArray, data: NDArray, guesser=sine_decay_guess, **kwargs):
    sine_decay, params = sine_decay_model()
    params = guesser(x_axis, data, params)
    try:
        result = sine_decay.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = -1
        logger.warning("The sine decay fit did not work.")
    return result


def multi_sine_decay_fit(
    x_axis: NDArray, data: NDArray, guesser=multi_sine_decay_guess, N: int = 1, **kwargs
):
    multi_sine_decay, params = multi_sine_decay_model(N=N)
    params = guesser(x_axis, data, params, N=N)
    try:
        result = multi_sine_decay.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = -1
        logger.warning("The multi sine decay fit did not work.")
    return result
