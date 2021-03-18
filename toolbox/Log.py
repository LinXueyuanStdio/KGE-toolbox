# 日志
import logging


def Log(filename: str, name_scope="0"):
    """Return instance of logger 统一的日志样式

        Examples:
           >>> from toolbox.Log import Log
           >>> log = Log("./train.log")
           >>> log.info("abc")
    """
    logger = logging.getLogger('logger-%s' % name_scope)
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))

    logging.getLogger().addHandler(handler)

    return logger


def log_result(logger, result):
    """
    :param logger: from toolbox.Log.Log()
    :param result: from toolbox.Evaluate.evaluate()
    """
    from toolbox.Evaluate import pretty_print
    pretty_print(result, logger.info)
