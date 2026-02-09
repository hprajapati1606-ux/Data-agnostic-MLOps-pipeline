import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_latest_file(directory: str) -> str:
    """
    Scans the given directory and returns the path of the latest file.
    """
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        list_of_files = glob.glob(os.path.join(directory, '*'))
        list_of_files = [f for f in list_of_files if os.path.isfile(f)]
        
        if not list_of_files:
            raise FileNotFoundError(f"Directory '{directory}' is empty.")

        latest_file = max(list_of_files, key=os.path.getctime)
        
        logger.info(f"📂 Automatically detected file: {latest_file}")
        return latest_file

    except Exception as e:
        logger.error(f"❌ Error finding file: {e}")
        raise e