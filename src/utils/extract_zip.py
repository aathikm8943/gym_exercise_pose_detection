import zipfile
import os
from src.loggingInfo.loggingFile import logging

class ZipExtractor:
    def __init__(self, zip_dir: str, extract_to: str):
        self.zip_dir = zip_dir
        self.extract_to = extract_to
        os.makedirs(extract_to, exist_ok=True)

    def extract_all(self):
        logging.info(f"Extracting zip files from {self.zip_dir} to {self.extract_to}")
        zip_files = [f for f in os.listdir(self.zip_dir) if f.endswith('.zip')]

        for zip_file in zip_files:
            zip_path = os.path.join(self.zip_dir, zip_file)

            # Create subfolder based on ZIP filename (without .zip)
            folder_name = os.path.splitext(zip_file)[0]
            target_folder = os.path.join(self.extract_to, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            logging.info(f"Extracting {zip_file} to {target_folder}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder)

            logging.info(f"Successfully extracted {zip_file} to {target_folder}")