from mltext import classifyText, storeText
from mlmodel import trainModel, checkModel
import pandas as pd
import time
import numpy as np

#setup and reading ect
API_KEY="81ca7a50-e57b-11ea-84b2-415661022c632a07694e-1ecb-4d99-a520-6bb99cee5d7e"
status = checkModel (API_KEY)
df = pd.read_csv("training_data.csv")
csv = pd.read_csv("contestant_judgment.csv")
inputs = []
results = []
predictions = []

#convert int to string
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1,np.size(var)))[0]))[1:-1]

#Training and printing to make sure it works
i = 0
while i < 500:
    inputs.append(df.iat[i, 2])
    results.append(df.iat[i, 3])
    if to_str(results[i]) == "0":
        print(inputs[i] + " is: 0")
        storeText(API_KEY, inputs[i], "0")
    elif to_str(results[i]) == "1":
        print(inputs[i] + " is: 1")
        storeText(API_KEY, inputs[i], "1")
    i += 1

#Completing training
trainModel(API_KEY)
print("Training AI model")

while status != "ready to use":
    status = checkModel(API_KEY)
    if status == "problem":
        print("Problem")
        break
    time.sleep(3)
    print("Training...")
