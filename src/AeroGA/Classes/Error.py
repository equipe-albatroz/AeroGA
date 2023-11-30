from loguru import logger

class ErrorType:
    type: str
    message: str
    source: str

    def __init__(self, type, message, source):
        self.type = type
        self.message = message
        self.source = source
    
    def __str__(self) -> str:
        return f'Message: {self.message}\nType: {self.type}\nSource: {self.source}'

class Log(object):
    def __init__(self, log_file, source):
        self.log_file = log_file
        logger.add(log_file)
        self.source = source
        
    def info(self, message):
        logger.info(f"[{self.source}] {message}")
    
    def error(self, message):
        logger.error(f"[{self.source}] {message}")