import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


raw_mail_data = pd.read_csv('spam.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']


if Y.isnull().any():
    raise ValueError("Target variable contains NaN values.")
if not set(Y.unique()).issubset({0, 1}):
    raise ValueError("Target variable contains values other than 0 and 1.")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


model = LogisticRegression()
model.fit(X_train_features, Y_train)


st.title("Spam Email Classifier")


email_text = st.text_area("Enter the email text:", height=300)

if st.button("Classify"):
    if email_text:
        input_mail = [email_text]
        input_data_features = feature_extraction.transform(input_mail)
        prediction = model.predict(input_data_features)

        if prediction[0] == 0:
            st.write("**This is a spam email.**")
        else:
            st.write("**This is not a spam email.**")
    else:
        st.write("Please enter an email text to classify.")
