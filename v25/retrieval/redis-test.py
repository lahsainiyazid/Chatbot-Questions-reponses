from dotenv import load_dotenv
import os
from urllib.parse import urlparse

load_dotenv()

url = os.getenv("UPSTASH_REDIS_URL")
print(url)

u = urlparse(url)
print("scheme:", u.scheme)
print("host:", u.hostname)
print("port:", u.port)
