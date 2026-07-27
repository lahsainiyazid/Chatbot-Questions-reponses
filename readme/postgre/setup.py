import psycopg2
from psycopg2 import Binary 
import os 
conn=psycopg2.connect(db_name="documents_db",
                      user="yazid",
                      password=os.environ.get("password"),
                      host="localhost")
cur=conn.cursor()

