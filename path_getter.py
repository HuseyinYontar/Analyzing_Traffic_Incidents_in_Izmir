from dotenv import load_dotenv
import os

load_dotenv()
file_path = os.getenv("CLEANED_DATA_FILE")
def get_path_for_plotting() -> str:
    return "../"+file_path

def get_path_for_one_directory_in() -> str:
    return file_path

def get_path_for_binned_directory_in() -> str:
    return "../../"+os.getenv("BINNED_DATA_FILE")

def get_path_for_binned_one_directory_in() -> str:
    return "../"+os.getenv("BINNED_DATA_FILE")