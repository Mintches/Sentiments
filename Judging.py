from mltext import classifyText, storeText
from mlmodel import trainModel, checkModel
import pandas as pd
import time
import numpy as np

#setup 
API_KEY="81ca7a50-e57b-11ea-84b2-415661022c632a07694e-1ecb-4d99-a520-6bb99cee5d7e"
status = checkModel (API_KEY)
csv = pd.read_csv("contestant_judgment.csv")

#Judging
x = 0
while x < len(csv.index):
    test_text = csv.iat[x, 2]
    demo = classifyText(API_KEY, test_text)
    label = demo["class_name"]
    print("'%s'"%(label))
    predictions.append(label)
    x += 1

#Creating new column
df.insert(3, "Prediction", predictions, True)

#Download
df.to_csv(r'C:\Users\mincy\Desktop\Ignition\test.csv', index=False)
