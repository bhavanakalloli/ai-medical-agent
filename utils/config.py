from dotenv import load_dotenv

import os
load_dotenv()

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error occurred while fetching GOOGLE_API_KEY: {e}")