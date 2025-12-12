pymodaq_plugins_template
########################

.. the following must be adapted to your developed package, links to pypi, github  description...

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_template.svg
   :target: https://pypi.org/project/pymodaq_plugins_template/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/Montpellier-S2QT/pymodaq_plugins_pulsesequences/workflows/Upload%20Python%20Package/badge.svg
   :target: https://github.com/Montpellier-S2QT/pymodaq_plugins_pulsesequences
   :alt: Publication Status

.. image:: https://github.com/Montpellier-S2QT/pymodaq_plugins_pulsesequences/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/Montpellier-S2QT/pymodaq_plugins_pulsesequences/actions/workflows/Test.yml


This plugin controls a spincore PulseBlaster and a fastcomtec fast counter to perform pulsed measurements.


Authors
=======

* Lucas Moreau--Lalaux  (lucas.moreau-lalaux@ens-lyon.fr)
* Jessica Tournaud (jessica.tournaud@umontpellier.fr)

.. if needed use this field

    Contributors
    ============


.. if needed use this field



Instruments
===========

Below is the list of instruments included in this plugin

Actuators
+++++++++


Viewer0D
++++++++


Viewer1D
++++++++

* **PulsedCounter**: custom 1D detector that launches the programmed sequence and starts the measurement. It is meant to be used within the extension.


Viewer2D
++++++++


PID Models
==========


Extensions
==========
* **pulsed_measurement_extension**: core of this plugin. This extension allows for programming the pulse sequences, piloting the measurement and analysing the data.


Installation instructions
=========================

* PyMoDAQ version 5.
* Developed under windows 10.
* Drivers: 
   - spincore drivers https://www.spincore.com/support/spinapi/
   - fastcomtec software and drivers https://www.fastcomtec.com/ufm/mcs8a (the software must be opened to run the plugin)

* Default pulse sequences and fit methods are used from the default_pulse_sequences.py and fit_methods.py files in the ressources folder. If you want to use your own you can specify a path to the new sequences/methods in the plugin config file. When writing your own sequences or fit methods note that it is crucial to follow the structure used in the default files. A template for defining new pulse sequences is therefore available in the resources folder.