import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve database connection parameters from environment variables
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "enterprise_kb")

# Construct SQLAlchemy database connection URL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL)

# Retrieve all CSV files from the raw data directory
csv_files = glob.glob("data/raw/*.csv")

if not csv_files:
    print("No CSV files found in data/raw/ directory!")
else:
    for file_path in csv_files:
        # Derive sanitized table name from filename
        file_name = os.path.basename(file_path)
        table_name = os.path.splitext(file_name)[0].lower().replace("-", "").replace(".", "")
        
        print(f"Processing {file_name} -> Loading to table '{table_name}'...")
        df = pd.read_csv(file_path)
        
        # Ingest dataframe into MySQL table
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        print(f"Successfully loaded {len(df)} rows into '{table_name}'.")

print("Data ingestion to MySQL complete.")