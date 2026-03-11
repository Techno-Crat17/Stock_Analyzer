import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import os

start = '2010-01-01'
end = '2025-12-31'

st.title('📈 Stock Trend Prediction App')

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Stock Search")

user_input = st.sidebar.text_input('Enter Stock Ticker', 'HDB')

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1 Month","6 Months","1 Year","5 Years","Max"]
)

# -------------------------
# Load Stock Data
# -------------------------

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start=start, end=end)

df = load_data(user_input)

if df.empty:
    st.error("No stock data found. Please check ticker symbol.")
    st.stop()

# Fix multi-index columns
df.columns = df.columns.get_level_values(0)

# -------------------------
# Timeframe Filter
# -------------------------

if timeframe == "1 Month":
    df_chart = df.tail(22)

elif timeframe == "6 Months":
    df_chart = df.tail(126)

elif timeframe == "1 Year":
    df_chart = df.tail(252)

elif timeframe == "5 Years":
    df_chart = df.tail(1260)

else:
    df_chart = df


# -------------------------
# Candlestick Chart
# -------------------------

st.subheader("📊 Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=df_chart.index,
    open=df_chart['Open'],
    high=df_chart['High'],
    low=df_chart['Low'],
    close=df_chart['Close']
)])

fig_candle.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig_candle)

# -------------------------
# Dataset Description
# -------------------------

st.subheader('Data Summary')
st.write(df.describe())

# -------------------------
# Closing Price Chart
# -------------------------

st.subheader('Closing Price vs Time')

fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Closing Price")

st.pyplot(fig)

# -------------------------
# Moving Average Chart
# -------------------------

st.subheader('Closing Price with 100MA & 200MA')

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(df.Close,label="Close Price")
plt.plot(ma100,'r',label="100 MA")
plt.plot(ma200,'g',label="200 MA")

plt.legend()

st.pyplot(fig)

# -------------------------
# Train Test Split
# -------------------------

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# -------------------------
# Prepare Training Data
# -------------------------

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# -------------------------
# Load Model
# -------------------------

model_path = "keras_model.keras"

if not os.path.exists(model_path):
    st.error("Model file 'keras_model.keras' not found!")
    st.stop()

model = load_model(model_path, compile=False)

# -------------------------
# Testing Data
# -------------------------

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

# -------------------------
# Inverse Scaling
# -------------------------

y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1))
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# -------------------------
# Prediction vs Original
# -------------------------

st.subheader('Prediction vs Original Price')

fig2 = plt.figure(figsize=(10,6))

plt.plot(y_test, label="Original Price")
plt.plot(y_predicted, label="Predicted Price")

plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

st.pyplot(fig2)

# ===============================
# Next Day Prediction
# ===============================

st.subheader("🔮 Next Day Predicted Price")

next_price = y_predicted[-1][0]

st.write(f"Predicted Next Price: ${next_price:.2f}")

# ===============================
# Prediction Accuracy
# ===============================

st.subheader("Model Prediction Accuracy")

error = np.mean(np.abs(y_test - y_predicted))

accuracy = max(0, 100 - error)

st.write(f"Average Prediction Error: {error:.2f}")
st.write(f"Prediction Accuracy Score: {accuracy:.2f}%")

# ===============================
# Stock Risk Indicator
# ===============================

st.subheader("Stock Risk Indicator")

df['Daily Return'] = df['Close'].pct_change()

volatility = df['Daily Return'].std() * np.sqrt(252)

st.write(f"Annual Volatility: {volatility:.2f}")

if volatility < 0.2:
    st.success("Low Risk Stock")

elif volatility < 0.4:
    st.warning("Medium Risk Stock")

else:
    st.error("High Risk Stock")

# ===============================
# Latest Data
# ===============================

st.subheader("Latest Market Data")

st.write(df.tail())


