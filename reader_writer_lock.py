import threading

class ReaderWriterLock:
    def __init__(self):
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader_count = 0

    def reader_acquire(self):
        with self._read_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                self._write_lock.acquire()

    def reader_release(self):
        with self._read_lock:
            self._reader_count -= 1
            if self._reader_count == 0:
                self._write_lock.release()

    def writer_acquire(self):
        self._write_lock.acquire()

    def writer_release(self):
        self._write_lock.release()
