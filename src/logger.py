import logging
import os
from datetime import datetime

# Every run creates a new log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Logs folder will be created inside project root   
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, 
)
# Also print logs to terminal so developer can see them live
console = logging.StreamHandler()   
console.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console.setFormatter(formatter) 
logging.getLogger().addHandler(console)
logger = logging.getLogger("dropout_prediction")
