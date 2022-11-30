import numpy as np
import pandas as pd

f = pd.read_csv("price.csv")

Dic = {}
for i in range(600000,600500):
    Dic[i] = []
Dates = []
for i in range(f.shape[0]):
    Dic[f.loc[i]['ticker']].append(f.loc[i]['preClosePrice'])
    if (f.loc[i]['ticker']==600000):
        Dates.append(f.loc[i]['tradeDate'])
#print(Dates)
for i in range(600000,600500):
    if (len(Dic[i])<len(Dates)):
        Dic.pop(i)

df = pd.DataFrame(Dic, index=Dates)
df.to_csv("Agu.csv")