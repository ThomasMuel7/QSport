import os
from dotenv import load_dotenv

load_dotenv()

BSD_API_TOKEN = os.getenv("BSD_API_TOKEN")
GOOGLE_TIMEZONE_KEY = os.getenv("GOOGLE_TIMEZONE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
