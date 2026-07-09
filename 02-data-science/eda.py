from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "data_clean.csv") #reading csv file

df.head(5)
print(df.head(5))