# import logging

# class CustomFormatter(logging.Formatter):

#     grey = "\x1b[38;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = "%(asctime)s - %(name)s - %(message)s"

#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: grey + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }

#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)
    
# # create logger with 'spam_application'
# logger = logging.getLogger("AeroGA")
# logger.setLevel(logging.DEBUG)

# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# ch.setFormatter(CustomFormatter())

# logger.addHandler(ch)


import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger('pythonConfig')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)

# log.debug("A quirky message only developers care about")
# log.info("Curious users might want to know this")
# log.warn("Something is wrong and any user should be informed")
# log.error("Serious stuff, this is red for a reason")
# log.critical("OH NO everything is on fire")