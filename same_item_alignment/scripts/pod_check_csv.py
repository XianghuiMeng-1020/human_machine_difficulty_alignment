import pandas as pd
import sys
p = sys.argv[1]
df = pd.read_csv(p)
print(df.columns.tolist())
print(len(df))
print(df.question_id.nunique())
print(df.head(2))
print(df.parse_success.mean())
print(df.machine_correct.mean())
