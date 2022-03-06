from xmlrpc.client import Boolean
import numpy as np
import os

class stat_recorder:
    """
    Simple class to log to file.
    """

    def __init__(self, file_name: str, reset: Boolean = True):
        self.logfile = file_name

        # clear file
        if reset and os.path.isfile(self.logfile):
            open(self.logfile, 'w')


    def write(self, data):
        with open(self.logfile, 'a') as file:
            file.write(data)
            file.write('\n')