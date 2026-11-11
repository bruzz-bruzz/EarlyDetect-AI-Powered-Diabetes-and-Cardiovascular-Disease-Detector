import pandas as pd
data = pd.read_csv("data2.csv",index_col=False,sep=';')
print(data['cardio'].value_counts())
data.to_csv("data3.csv",index=False)