import os
import traceback
from datetime import datetime

class ErrorType:
    type: str
    message: str
    stack_trace: list

    def __init__(self, type, message, source):
        self.type = type
        self.message = message
        self.source = source
        self.stack_trace = traceback.extract_stack()[:-1]  # Exclude this function call
        self._register_error()
    
    def __str__(self) -> str:
        return f'Message: {self.message}\nType: {self.type}'

    def _register_error(self):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # Get the last frame in the stack trace (where the error originated)
        error_frame = self.stack_trace[-1]
        filename, lineno, func_name, _ = error_frame

        with open(os.path.join("./Logs", "error.log"), 'a') as file:
            file.write(f"{current_time} - {str(self)}\n")
            file.write(f"Occurred in file: {filename}, function: {func_name} at line: {lineno}\n")
            file.write("Stack trace:\n")
            for frame in reversed(self.stack_trace):
                file.write(f"  File '{frame.filename}', line {frame.lineno}, in {frame.name}\n")
                if frame.line:
                    file.write(f"    {frame.line.strip()}\n")
            file.write("\n")


# from loguru import logger

# class ErrorType:
#     type: str
#     message: str
#     source: str

#     def __init__(self, type, message, source):
#         self.type = type
#         self.message = message
#         self.source = source
    
#     def __str__(self) -> str:
#         return f'Message: {self.message}\nType: {self.type}\nSource: {self.source}'

# class Log(object):
#     def __init__(self, log_file, source):
#         self.log_file = log_file
#         logger.add(log_file)
#         self.source = source
        
#     def info(self, message):
#         logger.info(f"[{self.source}] {message}")
    
#     def error(self, message):
#         logger.error(f"[{self.source}] {message}")