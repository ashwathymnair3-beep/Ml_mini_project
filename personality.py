import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

st.title("🧠 Personality Prediction System")
st.write("Predict whether a person is an **Extrovert, Introvert or Ambivert**.")
st.info("🎚️ Move the sliders according to your traits and click **Predict Personality Type**.")
st.subheader("📝 Enter Your Details to Predict Personality")
st.markdown("---")

data=pd.read_csv('personality_synthetic_dataset.csv')
df=pd.DataFrame(data)
df=df.drop(columns=
           ['reading_habit',
            'curiosity',
            'group_comfort',
            'adventurousness',
            'gadget_usage',
           'sports_interest'],axis=1)

le=LabelEncoder()
df['personality_type']=le.fit_transform(df['personality_type'])
x = df.drop(columns=['personality_type'])
y= df['personality_type']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=5)
model=knn.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_pred,y_test)



cm=confusion_matrix(y_test,y_pred)

user_input = {}
for i in x.columns:
    user_input[i] = st.slider(
        f"{i}",
        float(df[i].min()),
        float(df[i].max()),
        float(df[i].mean()))
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
st.markdown("---")



if st.button("🔍 Predict Personality Type"):
    pred = model.predict(input_scaled)[0]
    personality = le.inverse_transform([pred])[0]
    st.success(f"Predicted Personality Type: **{personality}**")
else:
    st.warning("👉 Click the button above to get your personality prediction.")










