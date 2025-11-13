from ._spinapi import SpinAPI as spinapi


class Instruction:
    """
    A class to represent PulseBlaster instructions

    ...

    Attributes
    ----------
    flags : int
        determines state of each TTL output bit.
    inst : int
        determines which type of instruction is to be executed
    inst_data : int
        data to be used with the previous 'inst' field
    length : int
        duration of the pulse program instruction, specified in ns

    Methods
    -------
    program():
        Programs the instruction onto the currently selected board
    """

    def __init__(self, flags, inst, inst_data, length):
        """
        Constructs alll the necessary attributes for the instruction object

        Parameters
        ----------
        flags : int
            determines state of each TTL output bit.
        inst : int
            determines which type of instruction is to be executed
        inst_data : int
            data to be used with the previous 'inst' field
        length : int
            duration of the pulse program instruction, specified in ns
        """

        self.flags = flags
        self.inst = inst
        self.inst_data = inst_data
        self.length = length

    def __str__(self):
        """
        Convert instruction into a string representation for readability

        Parameters
        ----------
        None

        Returns
        -------
        str
            formatted string that represents the contents of the instruction
        """

        opcodes = [
            "CONTINUE",
            "STOP",
            "END LOOP",
            "JSR",
            "RTS",
            "BRANCH",
            "LONG DELAY",
            "WAIT",
        ]

        return f"{format(self.flags, '#026b')} | {format(opcodes[self.inst], '^10s')} | {self.inst_data} | {self.length} ns"

    def program(self):
        """
        Programs the instruction onto the currently selected board

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Will need to change this if we support PulseBlasterDDS and RadioProcessor boards.
        # Or, we could have a different class for RF instructions
        spinapi.pb_inst_pbonly(self.flags, self.inst, self.inst_data, self.length)
