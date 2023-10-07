import os
import datetime

class LogWriter:
    def __init__(self):
        self.success_path = 'log/log-success.txt'
        self.error_path = 'log/log-error.txt'

    def writeSuccess(self, info):
        with open(self.success_path, 'a+') as file:
            file.write(
                f"{datetime.datetime.now()} {info} \n")
            
    def writeError(self, info):
        with open(self.error_path, 'a+') as file:
            file.write(
                f"{datetime.datetime.now()} {info} \n")