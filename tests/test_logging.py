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
        log_dict={"ics_loss": 1e-3, "res_loss": 2e-4, "u_error": 0.05, "lr": 1e-3},
    )
    err = capsys.readouterr().err
    assert "ics_loss" in err
    assert "res_loss" in err
    assert "u_error" in err
    # keys without _loss/_error suffix are not tabulated
    assert "lr" not in err.replace("ics_loss", "").replace("res_loss", "").replace("u_error", "")
