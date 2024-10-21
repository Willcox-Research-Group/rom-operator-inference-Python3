# utils/test_timer.py
"""Tests for utils._timer."""

import os
import time
import pytest
import platform

import opinf


SYSTEM = platform.system()


def skipwindows(func):

    def skip(self, *args, **kwargs):
        pass

    return skip if SYSTEM == "Windows" else func


class MyException(Exception):
    pass


class TestTimedBlock:
    """Test utils.TimedBlock."""

    Timer = opinf.utils.TimedBlock

    @skipwindows
    def test_standard(self, message="TimedBlock test, no timelimit"):
        # No time limit.
        with self.Timer() as obj:
            pass
        assert obj.timelimit is None
        assert isinstance(obj.elapsed, float)

        # Time limit that does not expire.
        with self.Timer(message, timelimit=100) as obj:
            pass
        assert obj.message == message

    @skipwindows
    def test_timeout(self, message="TimedBlock test with problems"):
        # Time limit expires.
        with pytest.raises(TimeoutError) as ex:
            with self.Timer(message, timelimit=1):
                time.sleep(10)
        assert ex.value.args[0].startswith("TIMED OUT after ")

        # Exception occurs in the block.
        with pytest.raises(MyException) as ex:
            with self.Timer(message):
                raise MyException("failure in the block")
        assert ex.value.args[0] == "failure in the block"

    @skipwindows
    def test_log(
        self,
        message: str = "TimedBlock test with log",
        target: str = "_timedblocktest.log",
    ):
        if os.path.isfile(target):
            os.remove(target)

        # Set up a log file.
        self.Timer.add_logfile(target)

        # See if we write to the log file.
        with self.Timer(message, timelimit=100):
            pass

        assert os.path.isfile(target)
        with open(target, "r") as infile:
            text = infile.read().strip()
        assert text.count(message) == 1

        with pytest.raises(TimeoutError) as ex:
            with self.Timer(message, timelimit=1):
                time.sleep(10)
        assert ex.value.args[0].startswith("TIMED OUT after ")

        with open(target, "r") as infile:
            text = infile.read().strip()
        assert text.count(message) == 2
        assert text.count("TIMED OUT after ") == 1

        with pytest.raises(MyException) as ex:
            with self.Timer(message):
                raise MyException("failure in the block")
        assert ex.value.args[0] == "failure in the block"

        # Log to the same file.
        newmessage = f"{message} AGAIN!"
        self.Timer.add_logfile(target)

        # Log to another file.
        newtarget = f"_{target}"
        if os.path.isfile(newtarget):
            os.remove(newtarget)

        self.Timer.add_logfile(newtarget)
        with self.Timer(newmessage):
            pass
        for tfile in target, newtarget:
            with open(tfile, "r") as infile:
                text = infile.read().strip()
            assert text.count(newmessage) == 1
            os.remove(tfile)


if __name__ == "__main__":
    pytest.main([__file__])
