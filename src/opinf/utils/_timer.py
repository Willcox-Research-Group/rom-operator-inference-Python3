# utils/_timer.py
"""Context manager for timing blocks of code."""


import time
import signal
import logging


class timed_block:
    """Context manager for timing a block of code and reporting the timing.

    **WARNING**: this context manager may only function on Linux/Unix machines
    (Windows is not supported).

    Parameters
    ----------
    message : str
        Message to log / print.
    timelimit : float
        Number of seconds to wait before raising an error.

    Examples
    --------
    >>> with timed_block("This is a test"):
    ...     # Code to be timed
    ...     time.sleep(2)
    ...
    This is a test...done in 2.00 s.

    >>> with timed_block("Another test", timelimit=3):
    ...     # Code to be timed and halted within the specified time limit.
    ...     i = 0
    ...     while True:
    ...         i += 1
    Another test...TIMED OUT after 3.00 s.
    """

    verbose = True

    @staticmethod
    def _signal_handler(signum, frame):
        raise TimeoutError("timed out!")

    @property
    def timelimit(self):
        """Time limit (in seconds) for the block to complete."""
        return self._timelimit

    def __init__(self, message, timelimit=None):
        """Store print/log message."""
        self._frontend = "\n" if message.endswith("\n") else ""
        self.message = message.rstrip()
        self._backend = "\n" if "\r" not in message else ""
        self._timelimit = timelimit

    def __enter__(self):
        """Print the message and record the current time."""
        if self.verbose:
            print(f"{self.message}...", end=self._frontend, flush=True)
        self._tic = time.time()
        if self._timelimit is not None:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self._timelimit)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Calculate and report the elapsed time."""
        self._toc = time.time()
        if self._timelimit is not None:
            signal.alarm(0)
        elapsed = self._toc - self._tic
        if exc_type:  # Report an exception if present.
            if self._timelimit is not None and exc_type is TimeoutError:
                print(
                    f"TIMED OUT after {elapsed:.2f} s.",
                    flush=True,
                    end=self._backend,
                )
                logging.info(f"TIMED OUT after {elapsed:.2f} s.")
                raise
            print(f"{exc_type.__name__}: {exc_value}")
            logging.info(self.message)
            logging.error(
                f"({exc_type.__name__}) {exc_value} "
                f"(raised after {elapsed:.6f} s)"
            )
            raise
        else:  # If no exception, report execution time.
            if self.verbose:
                print(
                    f"done in {elapsed:.2f} s.",
                    flush=True,
                    end=self._backend,
                )
            logging.info(f"{self.message}...done in {elapsed:.6f} s.")
        self.elapsed = elapsed
        return
