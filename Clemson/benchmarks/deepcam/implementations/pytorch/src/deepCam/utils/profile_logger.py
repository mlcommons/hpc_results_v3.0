# The MIT License (MIT)
#
# Modifications Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

import os
import time
import itertools
import json

def _current_time_ms():
    """Returns current milliseconds since epoch."""
    return int(time.time() * 1e3)


class PLogger(object):

    __instance = None

    # Constant values - log event type
    INTERVAL_END = "INTERVAL_END"
    INTERVAL_START = "INTERVAL_START"
    POINT_IN_TIME = "POINT_IN_TIME"
    LOG_FORMAT = ":::PLOG {json_dump}"

    @staticmethod
    def getInstance(**kwargs):
        """ Static access method. """
        if PLogger.__instance == None:
            PLogger(**kwargs)
        return PLogger.__instance
        
    def __init__(self, **kwargs):
        """ Virtually private constructor. """
        if PLogger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PLogger.__instance = self

        # import stuff
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD.Dup()
        self.comm_rank = self.comm.Get_rank()

        # enabled
        self.enabled = False
        if "enabled" in kwargs:
            self.enabled = kwargs["enabled"]

        if not self.enabled:
            return
        
        # logfile
        if "logfile" not in kwargs:
            raise KeyError("Please specify a valid logfile")

        self.logfilename = kwargs["logfile"]
        self.logfile = open(self.logfilename, "a")

        # write frequency
        self.write_frequency = 1
        if "write_frequency" in kwargs:
            self.write_frequency = kwargs["write_frequency"]
            
        self.loglines = []
    
    def log_event(self, event_type, key, value=None, metadata=None):
        if not self.enabled:
            return
            
        # this we want to write
        outdict = {"time_ms": _current_time_ms(), "event_type": event_type, "key": key, "value": value, "rank": self.comm_rank, "metadata": metadata}
        
        # append to loglines
        if self.write_frequency > 1:
            self.loglines.append(outdict)
            if len(self.loglines) >= self.write_frequency:
                self.logfile.write("\n".join([PLogger.LOG_FORMAT.format(json_dump=json.dumps(x)) for x in self.loglines]))
                self.loglines = []
        else:
            self.logfile.write(PLogger.LOG_FORMAT.format(json_dump=json.dumps(outdict)) + "\n")

    def __del__(self):
        if not self.enabled:
            return

        # close logfile
        self.logfile.close()
        

        

    
