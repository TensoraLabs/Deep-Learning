import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="No-Code DL Platform", layout="wide")
st.title("No-Code Deep Learning Platform")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.write(df.head())

    all_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox(" Select Target Column", all_columns)

    model_type = st.sidebar.selectbox(" Select Model Type", ["ANN", "CNN", "RNN", "LSTM", "GRU"])
    test_size = st.sidebar.slider(" Test Size (in %)", 10, 50, 20) / 100
    epochs = st.sidebar.slider(" Epochs", 1, 100, 10)

    if st.sidebar.button("🚀 Train Model"):
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = st.session_state.scaler.fit_transform(X_train)
        X_test = st.session_state.scaler.transform(X_test)

        input_shape = (X_train.shape[1],)
        model = Sequential()

        if model_type == "ANN":
            model.add(Dense(64, activation='relu', input_shape=input_shape))
            model.add(Dense(32, activation='relu'))
        elif model_type == "CNN":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(Flatten())
        elif model_type == "RNN":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model.add(SimpleRNN(32, input_shape=(X_train.shape[1], 1)))
        elif model_type == "LSTM":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
        elif model_type == "GRU":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model.add(GRU(32, input_shape=(X_train.shape[1], 1)))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        with st.spinner("Training..."):
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            loss = model.evaluate(X_test, y_test)
        st.success(f"Model Trained. Test Loss: {loss:.4f}")

        model.save("trained_model.h5")
        st.session_state.model = model

    if st.session_state.model:
        st.subheader(" Predict with Trained Model")
        with st.form("prediction_form"):
            user_input = []
            for col in df.drop(target_column, axis=1).columns:
                val = st.number_input(f"{col}")
                user_input.append(val)
            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            input_array = np.array(user_input).reshape(1, -1)
            input_array = st.session_state.scaler.transform(input_array)
            if model_type in ["CNN", "RNN", "LSTM", "GRU"]:
                input_array = input_array.reshape(1, input_array.shape[1], 1)
            prediction = st.session_state.model.predict(input_array)
            st.success(f" Prediction: {prediction[0][0]:.4f}")

        with open("trained_model.h5", "rb") as f:
            st.download_button(" Download Trained Model", f, file_name="trained_model.h5")
