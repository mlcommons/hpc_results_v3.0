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
import threading

# NVML stuff
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetClock, \
                   nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage, nvmlDeviceGetEnforcedPowerLimit, \
                   nvmlDeviceGetMemoryInfo
from pynvml import NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, NVML_CLOCK_ID_APP_CLOCK_TARGET

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
        
    def __init__(self, device, **kwargs):
        """ Virtually private constructor. """
        if PLogger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PLogger.__instance = self

        # import stuff
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD.Dup()
        self.comm_rank = self.comm.Get_rank()

        # some default variables
        self.nvml_handle = None
        self.sampling_thread = None
        self.sampling_results = {}

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

        # init loglines
        self.loglines = []
            
        # NVML stuff
        if ("enable_nvml_logging" in kwargs) and (kwargs["enable_nvml_logging"]):
            nvmlInit()
            self.nvml_handle = nvmlDeviceGetHandleByIndex(device.index)
            self.sampling_info = {"stop": False, "sample_frequency": 0.01}
            # collect some initial device stats
            effective_power_limit = nvmlDeviceGetEnforcedPowerLimit(self.nvml_handle)
            target_clock = nvmlDeviceGetClock(self.nvml_handle, NVML_CLOCK_SM, NVML_CLOCK_ID_APP_CLOCK_TARGET)
            self.event(self.POINT_IN_TIME, "target_gpu_clock_sm", value=target_clock)
            self.event(self.POINT_IN_TIME, "effective_power_limit", value=effective_power_limit)
            

    def _nvml_worker(self, info, results):

        # initilize variables
        ticks = 1
        clock = nvmlDeviceGetClock(self.nvml_handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
        util = nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
        power = nvmlDeviceGetPowerUsage(self.nvml_handle)
        memory = nvmlDeviceGetMemoryInfo(self.nvml_handle).used

        # enter update loop
        while not info["stop"]:
            ticks += 1
            clock += nvmlDeviceGetClock(self.nvml_handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
            util += nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            power += nvmlDeviceGetPowerUsage(self.nvml_handle)
            memory += nvmlDeviceGetMemoryInfo(self.nvml_handle).used
            time.sleep(info["sample_frequency"])

        results["average_gpu_sm_utilization"] = util / float(ticks)
        results["average_gpu_clock_sm"] = clock / float(ticks)
        results["average_gpu_power"] = power / float(ticks)
        results["average_gpu_memory_utilization"] = memory / float(ticks)
        return
        
    def nvml_log_start(self):
        if (not self.enabled) or (self.nvml_handle is None):
            return

        if self.sampling_thread is not None:
            self.sampling_info["stop"] = True
            self.sampling_thread.join()

        # start new sampling thread
        self.sampling_info["stop"] = False
        self.sampling_thread = threading.Thread(target=self._nvml_worker, args=(self.sampling_info, self.sampling_results))
        self.sampling_thread.start()

    def nvml_log_stop(self, metadata=None):
        if (not self.enabled) or (self.sampling_thread is None):
            return
        
        self.sampling_info["stop"] = True
        self.sampling_thread.join()
        self.sampling_thread = None

        # log everything
        for key,value in self.sampling_results.items():
            self.event(self.POINT_IN_TIME, key, value=value, metadata=metadata)
        
    def event(self, event_type, key, value=None, metadata=None):
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
        

        

    
