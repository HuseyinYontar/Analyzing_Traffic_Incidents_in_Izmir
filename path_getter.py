from dotenv import load_dotenv
import os

load_dotenv()
file_path = os.getenv("CLEANED_DATA_FILE")
def get_path_for_plotting() -> str:
    return file_path