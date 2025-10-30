# -*- coding: utf-8 -*-

"""Reconstruct the structures of the variables compatible with the dll"""

import ctypes


"""
Remark to the usage of ctypes:
All Python types except integers (int), strings (str), and bytes (byte) objects
have to be wrapped in their corresponding ctypes type, so that they can be
converted to the required C data type.

ctypes type     C type                  Python type
----------------------------------------------------------------
c_bool          _Bool                   bool (1)
c_char          char                    1-character bytes object
c_wchar         wchar_t                 1-character string
c_byte          char                    int
c_ubyte         unsigned char           int
c_short         short                   int
c_ushort        unsigned short          int
c_int           int                     int
c_uint          unsigned int            int
c_long          long                    int
c_ulong         unsigned long           int
c_longlong      __int64 or
                long long               int
c_ulonglong     unsigned __int64 or
                unsigned long long      int
c_size_t        size_t                  int
c_ssize_t       ssize_t or
                Py_ssize_t              int
c_float         float                   float
c_double        double                  float
c_longdouble    long double             float
c_char_p        char *
                (NUL terminated)        bytes object or None
c_wchar_p       wchar_t *
                (NUL terminated)        string or None
c_void_p        void *                  int or None

"""


# --- Constants (from C# implementation) ---
class Constants:
    ST_RUNTIME = 0
    ST_OFLS = 1
    ST_TOTALSUM = 2
    ST_ROISUM = 3
    ST_ROIRATE = 4
    ST_SWEEPS = 5
    ST_STARTS = 6
    ST_ZEROEVTS = 7


# --- Structure definitions ---
class AcqStatus(ctypes.Structure):
    """Create a structured Data type with ctypes where the dll can write into.

    This object handles and retrieves the acquisition status data from the
    Fastcomtec.

    int started;                // acquisition status: 1 if running, 0 else
    double runtime;             // running time in seconds
    double totalsum;            // total events
    double roisum;              // events within ROI
    double roirate;             // acquired ROI-events per second
    double nettosum;            // ROI sum with background subtracted
    double sweeps;              // Number of sweeps
    double stevents;            // Start Events
    unsigned long maxval;       // Maximum value in spectrum
    """

    _fields_ = [
        ("started", ctypes.c_int),
        ("runtime", ctypes.c_double),
        ("totalsum", ctypes.c_double),
        ("roisum", ctypes.c_double),
        ("roirate", ctypes.c_double),
        ("ofls", ctypes.c_double),
        ("sweeps", ctypes.c_double),
        ("stevents", ctypes.c_double),
        ("maxval", ctypes.c_ulong),
    ]


class AcqSettings(ctypes.Structure):
    _fields_ = [
        ("range", ctypes.c_long),
        ("cftfak", ctypes.c_long),
        ("roimin", ctypes.c_long),
        ("roimax", ctypes.c_long),
        ("nregions", ctypes.c_long),
        ("caluse", ctypes.c_long),
        ("calpoints", ctypes.c_long),
        ("param", ctypes.c_long),
        ("offset", ctypes.c_long),
        ("xdim", ctypes.c_long),
        ("bitshift", ctypes.c_ulong),
        ("active", ctypes.c_long),
        ("eventpreset", ctypes.c_double),
        ("dummy1", ctypes.c_double),
        ("dummy2", ctypes.c_double),
        ("dummy3", ctypes.c_double),
    ]


class ACQDATA(ctypes.Structure):
    """Create a structured Data type with ctypes where the dll can write into.

    This object handles and retrieves the acquisition data of the Fastcomtec.
    """

    _fields_ = [
        ("s0", ctypes.POINTER(ctypes.c_ulong)),
        ("region", ctypes.POINTER(ctypes.c_ulong)),
        ("comment", ctypes.c_char_p),
        ("cnt", ctypes.POINTER(ctypes.c_double)),
        ("hs0", ctypes.c_int),
        ("hrg", ctypes.c_int),
        ("hcm", ctypes.c_int),
        ("hct", ctypes.c_int),
    ]


class BOARDSETTING(ctypes.Structure):
    _fields_ = [
        ("sweepmode", ctypes.c_long),
        ("prena", ctypes.c_long),
        ("cycles", ctypes.c_long),
        ("sequences", ctypes.c_long),
        ("syncout", ctypes.c_long),
        ("digio", ctypes.c_long),
        ("digval", ctypes.c_long),
        ("dac0", ctypes.c_long),
        ("dac1", ctypes.c_long),
        ("dac2", ctypes.c_long),
        ("dac3", ctypes.c_long),
        ("dac4", ctypes.c_long),
        ("dac5", ctypes.c_long),
        ("fdac", ctypes.c_int),
        ("tagbits", ctypes.c_int),
        ("extclk", ctypes.c_int),
        ("maxchan", ctypes.c_long),
        ("serno", ctypes.c_long),
        ("ddruse", ctypes.c_long),
        ("active", ctypes.c_long),
        ("holdafter", ctypes.c_double),
        ("swpreset", ctypes.c_double),
        ("fstchan", ctypes.c_double),
        ("timepreset", ctypes.c_double),
    ]


class DATSETTING(ctypes.Structure):
    """Data settings structure"""

    _fields_ = [
        ("savedata", ctypes.c_int),  # bit 0: auto save after stop
        # bit 1: write listfile
        # bit 2: listfile only, no evaluation
        # bit 5: drop zero events
        ("autoinc", ctypes.c_int),  # 1 if auto increment filename
        ("fmt", ctypes.c_int),  # format type (separate spectra):
        # 0 == ASCII, 1 == binary,
        # 2 == GANAAS, 3 == EMSA, 4 == CSV
        ("mpafmt", ctypes.c_int),  # format used in mpa datafiles
        ("sephead", ctypes.c_int),  # separate Header
        ("smpts", ctypes.c_int),  # Missing field from C# version
        ("caluse", ctypes.c_int),  # Calibration use
        ("filename", ctypes.c_char * 256),  # Filename
        ("specfile", ctypes.c_char * 256),  # Spectrum file
        ("command", ctypes.c_char * 256),  # Command
    ]
