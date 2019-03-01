import numpy as np
import utils

class TSEntry:
    def __init__(self, entry, time):
        # entry should be np.array
        self.entry = entry
        self.time = time
        self.next = None
        self.prev = None


class TimeSeries:
    # if file_prefix provided, loads data from file file_prefixT upon request for time series at time T
    def __init__(self, init_entry, file_prefix=None, max_time=1000):
        self.file_prefix = file_prefix
        if file_prefix is not None:
            self.current = TSEntry(init_entry, 0)
            self.max_time = max_time
            # first/last references not supported when loading from file
            self.first = None
            self.last = None
        else:
            self.current = TSEntry(init_entry, 0)
            self.first = self.current
            self.last = self.first
        self.shape = init_entry.shape
        self.time_elapsed = 0


    def add_entry_delta(self, delta):
        new_entry = TSEntry(utils.cap_profiles(self.last.entry+delta), self.time_elapsed+1)
        new_entry.prev = self.last
        self.last.next = new_entry

        self.last = new_entry
        self.time_elapsed += 1

    def add_entry(self, entry):
        self.last.next = entry
        self.last = entry
        self.time_elapsed += 1

    def load_entry(self, delta=None, time=None):
        if delta is not None:
            # could cache next delta file here
            time = self.time_elapsed + delta
        if time is None:
            raise Exception('Specify time or delta!')
        if time > self.max_time:
            self.current = None
        else:
            with open(self.file_prefix + str(time), 'rb') as data_file:
                ts_entry = TSEntry(np.reshape(np.fromstring(data_file.read()), (-1, 3)), time)
                self.current = ts_entry
                self.time_elapsed = time
                self.shape = ts_entry.entry.shape

    def get_next(self, delta=1):
        if self.file_prefix is None:
            count = 0
            while count < delta:
                self.current = self.current.next
            return self.current
        else:
            self.load_entry(delta=delta)
            return self.current
