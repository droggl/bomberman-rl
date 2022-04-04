from xmlrpc.client import Boolean
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

    def write_list(self, data):
        with open(self.logfile, 'a') as file:
            for x in data[:-1]:
                file.write(str(x) + ", ")
            file.write(str(data[-1]) + '\n')