# utils/_timer.py
"""Context manager for timing blocks of code."""

__all__ = [
    "TimedBlock",
]

import io
import os
import sys
import time
import signal
import logging


class TimedBlock:
    r"""Context manager for timing a block of code and reporting the timing.

    Parameters
    ----------
    message : str
        Message to log / print.
    timelimit : int
        Number of seconds to wait before raising an error.
        Floats are rounded down to an integer.

    Warnings
    --------
    This context manager may only function on Linux/Unix machines
    (Windows is not currently supported).

    Examples
    --------
    >>> import time
    >>> import opinf

    Without a time limit.

    >>> with opinf.utils.TimedBlock():
    ...     # Code to be timed
    ...     time.sleep(2)
    Running code block...done in 2.00 s.

    With a custom message.

    >>> with opinf.utils.TimedBlock("This is a test"):
    ...     time.sleep(3)
    This is a test...done in 3.00 s.

    With a time limit.

    >>> with opinf.utils.TimedBlock("Another test", timelimit=3):
    ...     # Code to be timed and halted within the specified time limit.
    ...     i = 0
    ...     while True:
    ...         i += 1
    Another test...
    TimeoutError: TIMED OUT after 3.00s.

    Set up a logfile to record messages to.

    >>> opinf.utils.TimedBlock.setup_logfile("log.log")
    Logging to '/path/to/current/folder/log.log'

    ``TimedBlock()`` will now write to the log file as well as print to screen.

    >>> with opinf.utils.TimedBlock("logfile test"):
    ...     time.sleep(1)
    logfile test...done in 1.00 s.
    >>> with open("log.log", "r") as infile:
    ...     print(infile.read().strip())
    INFO:   logfile test...done in 1.001150 s.

    Turn off print statements (but keep logging).

    >>> opinf.utils.TimedBlock.verbose = False
    >>> with opinf.utils.TimedBlock("not printed to the screen"):
    ...     time.sleep(1)
    >>> with open("log.log", "r") as infile:
    ...     print(infile.read().strip())
    INFO:   logfile test...done in 1.001150 s.
    INFO:   not printed to the screen...done in 1.002232 s.

    Capture the time elapsed for later use.

    >>> with opinf.utils.TimedBlock("how long?") as timer:
    ...     time.sleep(2)
    >>> timer.elapsed
    2.002866268157959
    """

    verbose = True
    rebuffer = False
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def __init__(
        self,
        message: str = "Running code block",
        timelimit: int = None,
    ):
        """Store print/log message."""
        self.__original_stdout = sys.stdout
        self.__new_buffer = None
        self.__front = "\n" if message.endswith("\n") else ""
        self.message = message.rstrip()
        self.__back = "\n" if "\r" not in message else ""
        if timelimit is not None:
            timelimit = max(int(timelimit), 1)
        self.__timelimit = timelimit
        self.__elapsed = None

    @property
    def timelimit(self):
        """Time limit (in seconds) for the block to complete."""
        return self.__timelimit

    @property
    def elapsed(self):
        """Actual time (in seconds) the block took to complete."""
        return self.__elapsed

    @staticmethod
    def _signal_handler(signum, frame):
        raise TimeoutError("timed out!")

    def _reset_stdout(self):
        text = self.__new_buffer.getvalue()
        sys.stdout = self.__original_stdout
        print(text, end="", flush=True)

    def __enter__(self):
        """Print the message and record the current time."""
        if self.rebuffer:
            sys.stdout = self.__new_buffer = io.StringIO()
        if self.verbose:
            print(f"{self.message}...", end=self.__front, flush=True)
        self._tic = time.time()
        if self.timelimit is not None:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self.timelimit)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Calculate and report the elapsed time."""
        self._toc = time.time()
        if self.timelimit is not None:
            signal.alarm(0)
        elapsed = self._toc - self._tic
        if exc_type:  # Report an exception if present.
            if self.timelimit is not None and exc_type is TimeoutError:
                print(flush=True)
                report = f"TIMED OUT after {elapsed:.2f} s."
                logging.info(f"{self.message}...{report}")
                if self.rebuffer:
                    self._reset_stdout()
                raise TimeoutError(report)
            print(f"{exc_type.__name__}: {exc_value}")
            logging.info(self.message)
            logging.error(
                f"({exc_type.__name__}) {exc_value} "
                f"(raised after {elapsed:.6f} s)"
            )
            if self.rebuffer:
                self._reset_stdout()
            raise
        else:  # If no exception, report execution time.
            if self.verbose:
                print(f"done in {elapsed:.2f} s.", flush=True, end=self.__back)
            logging.info(f"{self.message}...done in {elapsed:.6f} s.")
        self.__elapsed = elapsed
        if self.rebuffer:
            self._reset_stdout()
        return

    @classmethod
    def add_logfile(cls, logfile: str = "log.log") -> None:
        """Instruct :class:`TimedBlock` to log messages to the ``logfile``.

        Parameters
        ----------
        logfile : str
            File to log to.
        """
        logger = logging.getLogger()
        logpath = os.path.abspath(logfile)

        # Check that we aren't already logging to this file.
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and os.path.abspath(handler.baseFilename) == logpath
            ):
                if cls.verbose:
                    print(f"Already logging to {logpath}")
                return

        # Add a new handler for this file.
        newhandler = logging.FileHandler(logpath, "a")
        newhandler.setFormatter(cls.formatter)
        newhandler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(newhandler)
        if cls.verbose:
            print(f"Logging to '{os.path.abspath(logfile)}'")
