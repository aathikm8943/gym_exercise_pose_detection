import os
import logging
from time import asctime
from datetime import datetime

logFileExtPath = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_logs"
logFile = "logs"
os.makedirs(logFile, exist_ok=True)

logFilePath = os.path.join(f"{logFile}/{logFileExtPath}.log")

logging.basicConfig(
    level=logging.INFO,
    filename=logFilePath,
    format=f"[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s" 
)