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


###############
# Custom Models
###############


def _custom_model(prefix=None):
    """
    Custom Model template.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    # TODO: Define your custom mathematical function
    def custom_function(x, param1, param2): ...

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(custom_function)
    else:
        model = Model(custom_function, prefix=prefix)

    params = model.make_params()

    return model, params


def _another_custom_model(prefix=None):
    """
    TODO: you can remove this model, it is only here for illustration purposes.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """

    def another_custom_function(x, param1, param2): ...

    if not isinstance(prefix, str) and prefix is not None:
        logger.warning(
            "The passed prefix <{0}> of type {1} is not a string and"
            "cannot be used as a prefix and will be ignored for now."
            "Correct that!".format(prefix, type(prefix))
        )
        model = Model(another_custom_function)
    else:
        model = Model(another_custom_function, prefix=prefix)

    params = model.make_params()

    return model, params


def composite_model(prefix=None):
    """
    Template for a composite Model.

    :param prefix (str): optional, if multiple models should be used in a
                       composite way and the parameters of each model should be
                       distinguished from each other to prevent name collisions.
    """
    if prefix == None:
        add_text = ""
    else:
        add_text = prefix

    # If you call the same model multiple times when building the composite model,
    # you need to add a prefix to the variables to avoid name conflict.
    model1, _ = _custom_model(prefix="m1_" + add_text)
    model2, _ = _custom_model(prefix="m2_" + add_text)
    # The other models can simply inherit from the prefix passed as argument.
    another_model, _ = _another_custom_model(prefix=prefix)

    # Build you composite model using mathematical operations on the base models.
    composite_model = (model1 + model2) * another_model
    params = composite_model.make_params()

    return composite_model, params


####################################################
# Methods to Guess the Initial Values for Parameters
####################################################


def _custom_model_guess(x_axis: NDArray, data: NDArray, params):
    """
    Template for the function that estimates the initial vaues of the fitting parameters.

    :param x_axis: x data.
    :type x_axis: NDArray
    :param data: y data.
    :type data: NDArray
    :param params: Parameters object containing the parameters of the model.
    """
    # TODO: write a function that guesses the initial parameters from the input data.
    ...

    # TODO: set the Paramaters values with the guessed values.
    # For intsance: params["param1"].set(value=guessed_param1)
    ...

    return params


#############
# Fit Methods
#############


def custom_model_fit(
    x_axis: NDArray, data: NDArray, guesser=_custom_model_guess, **kwargs
):
    """
    Template for the actual fitting function based on _custom_model.

    :param x_axis: x data
    :type x_axis: NDArray
    :param data: y data
    :type data: NDArray
    :param guesser: method to guess the initial values.
    :param kwargs: extra keywords arguments to be passed in the lmfit fit method.
    """
    model, params = _custom_model()
    params = guesser(x_axis, data, params)
    try:
        result = model.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = -1
        logger.warning(
            "The custom model fit did not work."
        )  # TODO: change the name of the fit
    return result
