# utils/test_timer.py
"""Tests for utils._timer."""

import os
import time
import pytest

import opinf


def test_timed_block(message="timed_block test", target="_timedblocktest.log"):
    """Test timed_block context manager."""
    Timer = opinf.utils.timed_block
    if os.path.isfile(target):
        os.remove(target)

    with Timer(message) as obj:
        pass
    assert obj.message == message
    assert obj.timelimit is None
    assert isinstance(obj.elapsed, float)

    with Timer(message, timelimit=100) as obj:
        pass
    assert obj.message == message

    with pytest.raises(TimeoutError) as ex:
        with Timer(message, timelimit=1):
            time.sleep(10)
    assert ex.value.args[0].startswith("TIMED OUT after ")

    class MyException(Exception):
        pass

    with pytest.raises(MyException) as ex:
        with Timer(message):
            raise MyException("failure in the block")
    assert ex.value.args[0] == "failure in the block"

    # Set up a log file.
    Timer.add_logfile(target)

    # See if we write to the log file.
    with Timer(message, timelimit=100) as obj:
        pass

    assert os.path.isfile(target)
    with open(target, "r") as infile:
        text = infile.read().strip()
    assert text.count(message) == 1

    with pytest.raises(TimeoutError) as ex:
        with Timer(message, timelimit=1):
            time.sleep(10)
    assert ex.value.args[0].startswith("TIMED OUT after ")

    with open(target, "r") as infile:
        text = infile.read().strip()
    assert text.count(message) == 2
    assert text.count("TIMED OUT after ") == 1

    with pytest.raises(MyException) as ex:
        with Timer(message):
            raise MyException("failure in the block")
    assert ex.value.args[0] == "failure in the block"

    # Log to the same file.
    newmessage = f"{message} AGAIN!"
    Timer.add_logfile(target)

    # Log to another file.
    newtarget = f"_{target}"
    if os.path.isfile(newtarget):
        os.remove(newtarget)

    Timer.add_logfile(newtarget)
    with Timer(newmessage) as obj:
        pass
    for tfile in target, newtarget:
        with open(tfile, "r") as infile:
            text = infile.read().strip()
        assert text.count(newmessage) == 1
        os.remove(tfile)
