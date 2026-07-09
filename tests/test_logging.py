from jaxpi.logging import Logger


def test_logger_info_emits(capsys):
    logger = Logger(name="test-logger-info")
    logger.info("hello world")
    assert "hello world" in capsys.readouterr().err


def test_log_iter_tabulates_losses_and_errors(capsys):
    logger = Logger(name="test-logger-iter")
    logger.log_iter(
        step=10,
        start_time=0.0,
        end_time=1.5,
        log_dict={"loss/ics": 1e-3, "loss/res": 2e-4, "error/u": 0.05, "lr": 1e-3},
    )
    err = capsys.readouterr().err
    assert "loss/ics" in err
    assert "loss/res" in err
    assert "error/u" in err
    # keys outside the loss/ and error/ sections are not tabulated
    assert "lr" not in err.replace("loss/ics", "").replace("loss/res", "").replace("error/u", "")
