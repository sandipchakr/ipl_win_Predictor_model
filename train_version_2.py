import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score

import pickle

#load dataset:-
df = pd.read_csv("ipl_dataset.csv",encoding="latin1",low_memory=False)
df = df.dropna(subset=["winner", "team_a", "team_b"])

df["team_a_won"] = (df["winner"] == df["team_a"]).astype(int)    # team_a_won is the output

# head to head feature:- win rate of team_a vs team_b

h2h = (
    df.groupby(["team_a","team_b"])["team_a_won"]
    .expanding().mean().shift(1).reset_index(level=[0,1],drop=True)
)
df["h2h_win_pct"] = h2h.fillna(0.5)

# home ground advantage
# print(df["city"])
HOME_CITIES = {
    "Mumbai Indians": "Mumbai",
    "Chennai Super Kings": "Chennai",
    "Royal Challengers Bangalore": "Bengaluru",
    "Kolkata Knight Riders": "Kolkata",
    "Delhi Capitals": "Delhi",
    "Rajasthan Royals": "Jaipur",
    "Punjab Kings": "Mohali",
    "Sunrisers Hyderabad": "Hyderabad",
}

df["team_a_home"] = df.apply(
    lambda r: 1 if HOME_CITIES.get(r["team_a"])==r["city"] else 0, axis=1
)
df["team_b_home"] = df.apply(
    lambda r: 1 if HOME_CITIES.get(r["team_b"])==r["city"] else 0, axis=1
)
df["team_a_home"] = df["team_a_home"].fillna(0)
df["team_b_home"] = df["team_b_home"].fillna(0)

# print(df[["team_a","team_b","city","team_a_home","team_b_home"]].tail(5))

# toss advantage feature:-
df["team_a_toss_win"] = (df["toss_winner"]==df["team_a"]).astype(int)

# encoding______________________________________
dataset = {}
# labelencode:-
dataset_columns = [
    "team_a",
    "team_b",
    "venue",
    "city",
    # "toss_winner",
    "toss_decision",
    # "winner"
]

encoders = {}
for col in dataset_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# print(encoders)
# final_dataset = pd.DataFrame(dataset)
# # print(final_dataset.head(5))

# example feature:-
x = df[[ 
    "team_a",
    "team_b",
    "venue",
    "city",
    # "toss_winner",
    # "toss_decision",
    # "team_a_toss_win",
    "team_a_NRR",
    "team_b_NRR",
    "team_a_last5_win_pct",
    "team_b_last5_win_pct",
    "h2h_win_pct", # new add
    # "team_a_home", # new add
    # "team_b_home" # new add
    ]]
y = df["team_a_won"] # binary terget

# train test split:-
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# model selection:-

model = RandomForestClassifier( n_estimators=300,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42)

# # train model:-

model.fit(x_train,y_train)

# # predict :-
train_pred = model.predict(x_train)
value = model.predict(x_test)
# print(value)


team_a_names = encoders["team_a"].inverse_transform(x_test["team_a"])
team_b_names = encoders["team_b"].inverse_transform(x_test["team_b"])

# for i in range(len(value)):
#     if value[i] == 1:
#         print(team_a_names[i])
#     else:
#         print(team_b_names[i])

# decode_winner = encoders["winner"].inverse_transform(value)
# print(decode_winner[0])

# using matplotlib_____________________________________________________________________

# team_data = x_test[["team_a","team_b"]]
# plt.plot(team_data,y_test,color="red",marker="^")
# plt.plot(team_data,value,color="blue",marker="o")
# plt.show()

print(f"accuracy score:- {round(accuracy_score(y_test,value)*100,2)} %")
print(f"precision score:- {round(precision_score(y_test,value,average="weighted")*100, 2)}%")
print(f"recall score:- {round(recall_score(y_test,value,average="weighted")*100, 2)}%")

train_acc = round(accuracy_score(y_train, train_pred)*100,2)
test_acc = round(accuracy_score(y_test, value)*100,2)

scores = cross_val_score(model, x, y, cv=5)
print(f"cross validation score:- {round(scores.mean()*100,2)}")

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

importance = model.feature_importances_

for name, score in zip(x.columns, importance):
    print(f"{name}: {round(score*100,2)}")
# save model:-
with open("ipl_model.pkl","wb") as f:
    pickle.dump({
        "model": model,
        "encoders": encoders
        },f)