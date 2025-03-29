import traceback
import os
from datetime import datetime

class ErrorType:
    def __init__(self, type, message, source, traceback_info):
        self.type = type
        self.message = message
        self.source = source
        self.stack_trace = traceback_info  # Stack trace real do erro
        self._register_error()
    
    def __str__(self) -> str:
        return f'Message: {self.message}\nType: {self.type}'

    def _register_error(self):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        log_path = os.path.join("./Logs", "error.log")
        with open(log_path, 'a') as file:
            file.write(f"{current_time} - {str(self)}\n")
            file.write("Stack trace:\n")
            for frame in self.stack_trace:
                file.write(f"  File '{frame.filename}', line {frame.lineno}, in {frame.name}\n")
                if frame.line:
                    file.write(f"    {frame.line.strip()}\n")
            file.write("\n")
