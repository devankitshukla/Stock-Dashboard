import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly

st.title('Stock Dashboard')

# User inputs
defaultCompany = "AAPL"
companyTicker = st.text_input('Stock Ticker:', defaultCompany)
defaultDate = date(2022, 10, 7)
startDate = st.date_input('Start Date:', defaultDate)
endDate = st.date_input('End Date:')

# Download data
data = yf.download(companyTicker, start=startDate, end=endDate)

# Check if data was downloaded successfully
if data.empty:
    st.error(f"No data found for ticker '{companyTicker}'. Please check the ticker symbol and date range.")
    st.stop()

data = data.reset_index()

# Flatten MultiIndex columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Ensure 'Adj Close' column exists (some stocks might not have it)
if 'Adj Close' not in data.columns:
    data['Adj Close'] = data['Close']

try:
    # Select box to choose an indicator
    indicator = st.selectbox("Select Indicator",
                             ["None", "SMA (Simple Moving Average)", "EMA (Exponential Moving Average)"])

    if indicator != "None":
        window = st.slider("Select Window Size for Indicator", min_value=2, max_value=len(data) // 2, value=10)

        if indicator == "SMA (Simple Moving Average)":
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
        elif indicator == "EMA (Exponential Moving Average)":
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

    # Create tabs
    line, candle, area, forecast = st.tabs(['Line Chart', 'Candle Chart', 'Area Chart', 'Forecast Data'])

    # Line Chart Tab
    with line:
        open_tab, close_tab, adjClose_tab = st.tabs(['Open', 'Close', 'Adj Close'])
        
        with open_tab:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data['Date'], y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
            chart.update_layout(title_text='Open Price - Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
        
        with close_tab:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data['Date'], y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
            chart.update_layout(title_text='Close Price - Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
        
        with adjClose_tab:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data['Date'], y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name='Adj Close Price'))
            chart.update_layout(title_text='Adj Close Price - Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
    
    # Candle Chart Tab
    with candle:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data['Date'], 
            open=data['Open'], 
            high=data['High'], 
            low=data['Low'], 
            close=data['Close'],
            increasing_line_color='green', 
            decreasing_line_color='red'
        ))
        fig.update_layout(
            title_text='Candlestick Chart',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig)
    
    # Area Chart Tab
    with area:
        st.subheader('Close Price - Area Chart')
        areaChart = st.area_chart(data, x='Date', y='Close')
    
    # Forecast Tab
    with forecast:
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365
        
        # Prepare data for Prophet
        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        
        # Fit Prophet model
        m = Prophet()
        m.fit(df_train)
        
        # Make future predictions
        future = m.make_future_dataframe(periods=period, freq='D')
        forecast_data = m.predict(future)
        
        # Display forecast
        st.subheader('Forecast Data')
        st.write(f'Forecast plot for {n_years} years')
        
        fig = plot_plotly(m, forecast_data)
        fig.update_layout(
            title='Prophet Forecast with Actual Data',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig)
        
        # Show forecast components
        st.subheader('Forecast Components')
        fig2 = m.plot_components(forecast_data)
        st.pyplot(fig2)
    
    # Display raw data
    st.header(f"Price Data from {startDate} to {endDate}")
    st.dataframe(data, use_container_width=True)
    
except ValueError as e:
    st.error(f"Error: {str(e)}")
    st.error("Please enter a valid date range or company ticker symbol")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your inputs and try again")