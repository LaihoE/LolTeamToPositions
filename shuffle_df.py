import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
from bs4 import BeautifulSoup, SoupStrainer
import urllib.request
from sqlalchemy import create_engine
import numpy as np
sqlEngine = create_engine('')
dbConnection = sqlEngine.connect()

df=pd.read_sql('games',dbConnection)

shuffled=[]

for x in range(len(df)):
    row=df.iloc[x].tolist()
    random.shuffle(row)
    shuffled.append(row)

bplayer0=[]
bplayer1=[]
bplayer2=[]
bplayer3=[]
bplayer4=[]
rplayer0=[]
rplayer1=[]
rplayer2=[]
rplayer3=[]
rplayer4=[]

for x in range(len(shuffled)):
    bplayer0.append(shuffled[x][0])
    bplayer1.append(shuffled[x][1])
    bplayer2.append(shuffled[x][2])
    bplayer3.append(shuffled[x][3])
    bplayer4.append(shuffled[x][4])
    rplayer0.append(shuffled[x][5])
    rplayer1.append(shuffled[x][6])
    rplayer2.append(shuffled[x][7])
    rplayer3.append(shuffled[x][8])
    rplayer4.append(shuffled[x][9])

finaldict={
    "bplayer0":bplayer0,
    "bplayer1":bplayer1,
    "bplayer2":bplayer2,
    "bplayer3":bplayer3,
    "bplayer4":bplayer4,
    "rplayer0":rplayer0,
    "rplayer1":rplayer1,
    "rplayer2":rplayer2,
    "rplayer3":rplayer3,
    "rplayer4":rplayer4,
}

finaldict

df=pd.DataFrame.from_dict(finaldict)

df

df
