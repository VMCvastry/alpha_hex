import logging, logging.config

import coloredlogs

config = {
    "version": 1,
    "formatters": {
        "light": {
            "format": u"%(asctime)s %(levelname)-8s: \n\t %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "detailed": {
            "format": u"%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d —\n\t %(message)s",
            "datefmt": "%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "light",
            "level": "INFO",
        },
        "log": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "logs.log",
            "mode": "a",
            "maxBytes": 1000000000,
            "backupCount": 1,
            "level": "DEBUG",
        },
    },
    "root": {"handlers": ["console", "log"], "level": "DEBUG"},
}


coloredlogs.install(level="INFO")
logging.config.dictConfig(config)
