

import streamlit as st
# # %%capture
# # !pip install -q quantstats PyPortfolioOpt ta

# # Data handling and statistical analysis
# import pandas as pd
# import numpy as np
# from scipy import stats
# # from pandas_datareader import data as pdr

# # Data visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# #Max-min scaling
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# #Machine Learning
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import Huber
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Optimization and allocation
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt import black_litterman, BlackLittermanModel

# # Financial data
# import quantstats as qs
# import ta
# import yfinance as yf

# # For time stamps
# from datetime import datetime

# # Linear Regression Model
# from sklearn.linear_model import LinearRegression

# # Enabling Plotly offline
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)

# # Datetime and hiding warnings
# import datetime as dt
# from datetime import datetime, timedelta
# import warnings
# import pytz
# import os

# # Other
# from tabulate import tabulate

# warnings.filterwarnings("ignore")

# """# 2.  Getting Started

# We will use Yahoo Finance to analyze the stock market performance of four major technology companies: Apple (AAPL), Amazon (AMZN), Tesla (TSLA), and Microsoft (MSFT). Our dataset contains daily returns for these companies from January 1, 2020, to January 1, 2025.
# """

# import yfinance as yf
# import pandas as pd
# from datetime import datetime, timedelta

# df=pd.DataFrame()

# # List of tickers
# tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT']

# # Get yesterday's date
# yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

# # Download data
# df = yf.download(tickers, start="2020-01-01", end=yesterday,group_by=tickers, auto_adjust=False)

# # ✅ Reset index to move 'Date' from index to a column
# df = df.stack(level=0).reset_index()

# # ✅ Rename columns for clarity
# df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# # Display the cleaned DataFrame
# df.tail(10)


# df=df.set_index('Date')


# df[df['Ticker']=='AAPL'].describe()


# df[df['Ticker']=='AAPL'].info()


# df.nunique()



# # Set the figure size
# sns.set(rc={'figure.figsize':(11.7, 8.27)}, palette="pastel")  # Soft color palette (pastel)

# # List of tickers you want to plot
# tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
# companies=['Apple','Tesla','Microsoft','Amazon']
# # Create a 2x2 grid of subplots
# plt.figure(figsize=(11.7, 8.27))

# # Loop through each ticker and plot the distribution of 'Open' prices
# for i, (ticker,company) in enumerate(zip(tickers,companies),1):
#     # Filter data for the current ticker
#     ticker_data = df[df['Ticker'] == ticker]

#     # Create a subplot for each ticker
#     plt.subplot(2, 2, i)

#     # Plot histogram with KDE
#     sns.histplot(data=ticker_data, x='Open', kde=True, color=sns.color_palette()[i-1])

#     # Add title and labels
#     plt.title(f"Histogram of 'Open' Prices for {company}")
#     plt.xlabel("Open Price")
#     plt.ylabel("Frequency")

# # Adjust layout
# plt.tight_layout()
# plt.show()

# """This figure aims to show the distribution of the 'open' prices of the stock. By visualizing the distribution, we can see if it is skewed or symmetric and if there are any outliers. This information can help us understand the range of prices and how the prices are distributed.

# """

# # Measuring skewness with quantstats
# print('Measuring skewness of each company:')
# print("Apple's skewness: ", qs.stats.skew(df[df['Ticker'] == 'AAPL']['Open']).round(2))
# print("Tesla's skewness: ", qs.stats.skew(df[df['Ticker'] == 'TSLA']['Open']).round(2))
# print("Microsoft's skewness: ", qs.stats.skew(df[df['Ticker'] == 'MSFT']['Open']).round(3))
# print("Amazon's skewness: ", qs.stats.skew(df[df['Ticker'] == 'AMZN']['Open']).round(3))



# sns.set(rc={'figure.figsize':(11.7, 8.27)}, palette="Reds")
# # List of tickers you want to plot
# tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
# companies=['Apple','Tesla','Microsoft','Amazon']
# # Create a 2x2 grid of subplots
# plt.figure(figsize=(11.7, 8.27))

# # Loop through each ticker and plot the distribution of 'Open' prices
# for i, (ticker,company) in enumerate(zip(tickers,companies), 1):
#     # Filter data for the current ticker
#     ticker_data = df[df['Ticker'] == ticker]

#     # Create a subplot for each ticker
#     plt.subplot(2, 2, i)

#     # Plot histogram with KDE
#     sns.histplot(data=ticker_data, x='Close', kde=True, color=sns.color_palette()[i-1])

#     # Add title and labels
#     plt.title(f"Histogram of 'Close' Prices for {company}")
#     plt.xlabel("Close Price")
#     plt.ylabel("Frequency")

# # Adjust layout
# plt.tight_layout()
# plt.show()

# # Measuring skewness with quantstats
# print('Measuring skewness of each company:')
# print("Apple's skewness: ", qs.stats.skew(df[df['Ticker'] == 'AAPL']['Close']).round(2))
# print("Tesla's skewness: ", qs.stats.skew(df[df['Ticker'] == 'TSLA']['Close']).round(2))
# print("Microsoft's skewness: ", qs.stats.skew(df[df['Ticker'] == 'MSFT']['Close']).round(3))
# print("Amazon's skewness: ", qs.stats.skew(df[df['Ticker'] == 'AMZN']['Close']).round(3))


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Set figure size and use a pastel color palette
# sns.set(rc={'figure.figsize':(11.7, 8.27)}, palette="pastel")

# # List of tickers and corresponding company names
# tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
# companies = ['Apple', 'Tesla', 'Microsoft', 'Amazon']

# # Create a 2x2 grid of subplots
# plt.figure(figsize=(11.7, 8.27))

# # Loop through each ticker and corresponding company name
# for i, (ticker, company) in enumerate(zip(tickers, companies), 1):
#     # Filter data for the current ticker
#     ticker_data = df[df['Ticker'] == ticker]

#     # Create a subplot for each ticker
#     plt.subplot(2, 2, i)

#     # Plot scatterplot for Open vs Close prices
#     sns.scatterplot(x='Open', y='Close', data=ticker_data, color=sns.color_palette()[i-1])

#     # Add title and labels
#     plt.title(f"{company} - Open vs Close Price")
#     plt.xlabel("Open Price")
#     plt.ylabel("Close Price")

# # Adjust layout to avoid overlap
# plt.tight_layout()
# plt.show()



# df['Daily Return'] =df.groupby('Ticker')['Adj Close'].pct_change()
# # Then we'll plot the daily return percentage


# # Filter data for apple, tesla, microsoft and amazon
# aapl_returns = df[df['Ticker'] == 'AAPL']['Daily Return'].dropna()
# tsla_returns = df[df['Ticker'] == 'TSLA']['Daily Return'].dropna()
# msft_returns= df[df['Ticker'] == 'MSFT']['Daily Return'].dropna()
# amzn_returns=df[df['Ticker'] == 'AMZN']['Daily Return'].dropna()


# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_figheight(10)
# fig.set_figwidth(15)
# aapl_returns.plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
# axes[0,0].set_title('APPLE')
# tsla_returns.plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
# axes[0,1].set_title('TESLA')
# msft_returns.plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
# axes[1,0].set_title('MICROSOFT')
# amzn_returns.plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
# axes[1,1].set_title('AMAZON')
# fig.tight_layout()



# print('\n')
# print('\nApple Daily Returns Histogram')
# qs.plots.histogram(aapl_returns, resample = 'D')
# print('\n')

# print('\nTesla  Daily Returns Histogram')
# qs.plots.histogram(tsla_returns, resample = 'D')
# print('\n')
# print('\nMicrosoft Returns Histogram')
# qs.plots.histogram(msft_returns, resample = 'D')
# print('\n')
# print('\nAmazon Returns Histogram')
# qs.plots.histogram(amzn_returns, resample = 'D')



# # Create a dictionary for the company DataFrames and their corresponding tickers
# companies = {
#     'AAPL': df[df['Ticker'] == 'AAPL'],
#     'TSLA': df[df['Ticker'] == 'TSLA'],
#     'MSFT': df[df['Ticker'] == 'MSFT'],
#     'AMZN': df[df['Ticker'] == 'AMZN']
# }

# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=0.9, bottom=0.1)

# # Iterate over the companies dictionary to plot the adjusted closing prices
# for i, (ticker, company_data) in enumerate(companies.items(), 1):
#     plt.subplot(2, 2, i)
#     company_data['Adj Close'].plot(color='red', legend=False)
#     plt.ylabel('Adj Close')
#     plt.xlabel(None)
#     plt.title(f"Adjusted Closing Price of {ticker}")

# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=0.9, bottom=0.1)

# # Loop through company_list and tech_list to plot the volume
# for i, (ticker,company) in enumerate(companies.items(), 1):
#     plt.subplot(2, 2, i)
#     company['Volume'].plot(color='red', legend=False)
#     plt.ylabel('Volume')
#     plt.xlabel(None)
#     plt.title(f"Sales Volume for {ticker}")

# plt.tight_layout()
# plt.show()


# # Define the moving average days
# ma_day = [10, 20, 50]

# # Compute moving averages for each company
# for ma in ma_day:
#     for ticker, company in companies.items():
#         column_name = f"MA for {ma} days"
#         company[column_name] = company['Adj Close'].rolling(ma).mean()

# # Create subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# # Plot for each company
# companies['AAPL'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 0])
# axes[0, 0].set_title('APPLE')

# companies['TSLA'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 1])
# axes[0, 1].set_title('TESLA')

# companies['MSFT'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1, 0])
# axes[1, 0].set_title('MICROSOFT')

# companies['AMZN'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1, 1])
# axes[1, 1].set_title('AMAZON')

# # Adjust layout
# fig.tight_layout()
# plt.show()



# # Pivot the DataFrame to get 'Adj Close' values for each company as columns
# adj_close_df = df.pivot_table(values='Daily Return', index='Date', columns='Ticker')
# adj_close_df

# sns.jointplot(x='AAPL', y='MSFT', data=adj_close_df, kind='scatter', color='seagreen')



# # Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
# returns_fig = sns.PairGrid(adj_close_df)

# # Using map_upper we can specify what the upper triangle will look like.
# returns_fig.map_upper(plt.scatter,color='red')

# # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
# returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# # Finally we'll define the diagonal as a series of histogram plots of the daily return
# returns_fig.map_diag(plt.hist,bins=30)



# sns.set_style("white")

# plt.figure(figsize=(12, 12))

# # First heatmap with a soft "PuBu" colormap
# plt.subplot(2, 2, 1)
# sns.heatmap(adj_close_df.corr(), annot=True, cmap='PuBu', linewidths=0.3, square=True, cbar_kws={"shrink": 0.8})
# plt.title('Correlation of stock return', fontsize=14)

# # Second heatmap with a soft "BuGn" colormap
# plt.subplot(2, 2, 2)
# sns.heatmap(adj_close_df.corr(), annot=True, cmap='BuGn', linewidths=0.3, square=True, cbar_kws={"shrink": 0.8})
# plt.title('Correlation of stock closing price', fontsize=14)

# plt.show()


# rets = adj_close_df.dropna()
# area = np.pi * 20
# plt.figure(figsize=(10, 8))
# plt.scatter(rets.mean(), rets.std(), s=area)
# plt.xlabel('Expected return')
# plt.ylabel('Risk')
# for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#     plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
#                  arrowprops=dict(arrowstyle='-', color='red', connectionstyle='arc3,rad=-0.3'))


# df = yf.download('AAPL', start='2020-01-01', end=datetime.now(),auto_adjust=False)
# # Reshape the DataFrame so tickers appear in rows
# df.columns =df.columns.droplevel(1)

# #df=df.set_index('Date')
# # Step 2: Filter only the relevant columns (Price, Adj Close, Close, High, Low, Open, Volume)
# df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]


# # Display the reshaped DataFrame
# df.tail(10)

# def extracting_features(df):
#     """Extracts key financial indicators from a Yahoo Finance dataset."""

#     df = df.copy()

#     #Price Change (Close - Open)
#     df['price_change'] = df['Close'] - df['Open']

#     #Returns (Daily Percentage Change)
#     df['returns'] = df['Close'].pct_change()

#     #Create a new feature 'average_price'
#     df['average_price'] = (df['Close'] + df['Open']) / 2

#     #Price Range (High - Low)
#     df['price_range'] = df['High'] - df['Low']

#     #Volume Change
#     df['volume_change'] = df['Volume'].diff()

#     #Moving Average
#     df['moving_average_10'] = df['Close'].rolling(window=10).mean()


#     #Relative Strength Index (RSI)
#     window = 14
#     delta = df['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     df['RSI'] = 100 - (100 / (1 + rs))

#     # MACD (Moving Average Convergence Divergence)
#     short_ema = df['Close'].ewm(span=12, adjust=False).mean()
#     long_ema = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = short_ema - long_ema
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

#     # Bollinger Bands (20-Day Moving Average ± 2 Standard Deviations)
#     df['20_SMA'] = df['Close'].rolling(window=20).mean()
#     df['BB_Upper'] = df['20_SMA'] + (df['Close'].rolling(window=20).std() * 2)
#     df['BB_Lower'] = df['20_SMA'] - (df['Close'].rolling(window=20).std() * 2)

#     return df

# df = extracting_features(df)


# df.tail(5)

# df=df.reset_index()
# df.index.name = "Index"

# df=df.reset_index(drop=True)



# # Function to normalize data using MinMax Scaling for all features
# def min_max_scaling(data):
#     """Applies Min-Max scaling to normalize all features (Open, High, Low, Close, Adj Close, Volume)."""
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)  # Normalize all features
#     print("Data normalization using Min-Max Scaling completed.")
#     return scaler, scaled_data

# columns_to_scale = df.columns.difference(["Date"])  # Exclude Date column
# scaler, normalized_data = min_max_scaling(df[columns_to_scale])
# new_df = pd.DataFrame(normalized_data, columns=columns_to_scale, index=df.index)  # Keep original index2
# new_df["Date"] = df.index
# new_df["Date"] = df["Date"].values
# new_df=new_df.set_index('Date')
# new_df=new_df.fillna(0)



# def denormalize_predictions(y_pred, df):
#     """
#     Denormalize the predicted values using the MinMaxScaler fitted on the original data.

#     Args:
#         y_pred (np.array): The normalized predicted stock values.
#         df (pd.DataFrame): The original dataframe containing the stock prices.

#     Returns:
#         np.array: The denormalized predicted values.
#     """
#     # Extract 'Close' column from the dataframe, which was used for normalization
#     close_prices = df['Close'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)

#     # Create a MinMaxScaler and fit it using the 'Close' prices
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler.fit(close_prices)  # Fit the scaler on the 'Close' price column

#     # Apply the inverse_transform to denormalize the predicted values
#     y_pred_denormalized = scaler.inverse_transform(y_pred.reshape(-1, 1))  # Denormalize predictions

#     return y_pred_denormalized.flatten()  # Flatten to return a 1D array



# def split_and_reshape_data(dataframe, pred_days, company):
#     """
#     Splits the dataset into training and testing sets, then reshapes it for LSTM models.

#     Parameters:
#         dataframe (pandas DataFrame): Scaled dataset.
#         pred_days (int): Number of previous days used for prediction.
#         company (str): Company name.

#     Returns:
#         X_train, y_train, X_test, y_test: Reshaped datasets for model training and testing.
#     """
#     prediction_days = pred_days

#     train_size = int(np.ceil(len(dataframe) * 0.70))  # 70% for training data
#     test_size = len(dataframe) - train_size  # Remaining 30% for testing data
#     print(f'The training size for {company} is {train_size} rows')
#     print(f'The testing size for {company.title()} is {test_size} rows')

#     # Use .iloc[] for proper slicing of pandas DataFrame
#     train_data = dataframe.iloc[0: train_size, :]  # Use iloc for slicing DataFrame
#     test_data = dataframe.iloc[train_size - prediction_days:, :]  # Use iloc for slicing DataFrame

#     X_train, y_train, X_test, y_test = [], [], [], []

#     # Loop to create X_train and y_train for training data
#     for i in range(prediction_days, len(train_data)):
#         X_train.append(train_data.iloc[i - prediction_days: i, :].values)  # Features: previous 'pred_days' values for all columns
#         y_train.append(train_data.iloc[i, 3])  # Target: next day's 'Close' value (index 3 corresponds to 'Close')

#     # Loop to create X_test and y_test for testing data
#     for i in range(prediction_days, len(test_data)):
#         X_test.append(test_data.iloc[i - prediction_days: i, :].values)  # Features: previous 'pred_days' values for all columns
#         y_test.append(test_data.iloc[i, 3])  # Target: next day's 'Close' value (index 3 corresponds to 'Close')

#     # Convert the lists to numpy arrays
#     X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

#     # Reshape the data to be suitable for LSTM model (3D array: samples, time steps, features)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))  # Number of features (columns) will be dynamic
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # Same for test data

#     print(f'Data for {company.title()} split successfully')

#     return X_train, y_train, X_test, y_test


# stock_name = 'Apple'
# X_train, y_train, X_test, y_test = split_and_reshape_data(new_df, 30, stock_name)


# df=df.set_index('Date')

# def linear_prediction(df):
#     # Define features and target
#     X = df[['Open', 'High', 'Low', 'Volume',
#        'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper',
#        'BB_Lower']]
#     y = df['Close']
#     X=X.fillna(0)
#     # Normalize features using Min-Max scaling
#     #scaler = MinMaxScaler()
#     #X_scaled = scaler.fit_transform(X)

#     # Perform train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     print("Shapes of X_train, X_test, y_train, y_test:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#     # Fit the linear regression model
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)

#     # Make predictions
#     y_pred = regressor.predict(X_test)
#     #y_pred_demor=denormalize_predictions(y_pred, df)

#     # Calculate metrics using imported functions directly
#     mae = mean_absolute_error(y_test,y_pred )
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)

#     # Store intercept and coefficients
#     intercept = regressor.intercept_
#     coefficients = regressor.coef_

#     # Extract the dates corresponding to the test set
#     test_dates = y_test.index

#     # Create DataFrame to compare actual and predicted values
#     compare = pd.DataFrame({
#         'Date': test_dates,
#         'Actual': y_test.values,
#         'Predicted': y_pred
#     })

#     # Set date as index
#     compare.set_index('Date', inplace=True)

#     # Sort DataFrame by date
#     compare = compare.sort_index()

#     # Display metrics using tabulate
#     metrics_data = [
#         ['Mean Absolute Error', mae],
#         ['Mean Squared Error', mse],
#         ['Root Mean Squared Error', rmse],
#         ['R^2 Score', r2]
#     ]
#     print("Metrics:")
#     print(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='psql'))



#     # Display comparison DataFrame
#     #print("\nComparison DataFrame:")
#     #print(compare)

#     # Plot predicted vs. actual values
#     plt.figure(figsize=(10, 6))
#     plt.plot(compare.index, compare['Actual'], label='Actual')
#     plt.plot(compare.index, compare['Predicted'], label='Predicted')
#     plt.xlabel('Date')
#     plt.ylabel('Close Price')
#     plt.title('Actual vs. Predicted Close Prices')
#     plt.legend()
#     plt.show()

#     # Plot regression plot
#     sns.regplot(x=y_pred.flatten(), y=y_test.values.flatten(), scatter_kws={"color": "blue"}, line_kws={"color": "red"})
#     plt.xlabel('Predicted Price')
#     plt.ylabel('Actual Price')
#     plt.title('Actual vs. Predicted Price')
#     plt.show()

#     return regressor, mae, mse, rmse, r2, compare

# # Call the function and get the regressor object, scaler, metrics, and comparison DataFrame
# regressor, mae, mse, rmse, r2, comparison_df = linear_prediction(df)



# def linear_forecasting(df, future_days=30):
#     # Define features and target
#     X = df[['Open', 'High', 'Low', 'Volume',
#        'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper',
#        'BB_Lower']]
#     y = df['Close']
#     X=X.fillna(0)
#     # Normalize features using Min-Max scaling
#     #scaler = MinMaxScaler()
#     #X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Fit the linear regression model
#     regressor = LinearRegression()
#     regressor.fit(X, y)
#     # Predicting for the future dates
#     # Tarih indeksini datetime formatına çevir (eğer değilse)
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)


#     # Predicting for the future dates
#     future_dates = pd.date_range(df.index[-1] + pd.DateOffset(1), periods=future_days, freq='B')
#     future_features = df.iloc[-1*future_days:][['Open', 'High', 'Low', 'Volume',
#        'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper',
#        'BB_Lower']]
#     #future_features_scaled = scaler.transform(future_features)
#     future_predictions = regressor.predict(future_features)

#     # Creating a DataFrame for future predictions
#     future_prediction_df = pd.DataFrame({
#         'Date': future_dates,
#         'Predicted': future_predictions
#     })
#     future_prediction_df.set_index('Date', inplace=True)

#     # Create figure with plotly
#     fig = go.Figure()

#     # Historical data trace
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Historical Close'))

#     # Future predictions trace
#     fig.add_trace(go.Scatter(x=future_prediction_df.index, y=future_prediction_df['Predicted'], mode='lines+markers', name='Predicted Close'))

#     # Update layout for better interactive controls
#     fig.update_layout(
#         title='Historical vs Predicted Close Prices',
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1, label="1m", step="month", stepmode="backward"),
#                     dict(count=6, label="6m", step="month", stepmode="backward"),
#                     dict(count=1, label="YTD", step="year", stepmode="todate"),
#                     dict(count=1, label="1y", step="year", stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         ),
#         yaxis=dict(
#             title="Close Price",
#             autorange=True,
#             type="linear"
#         )
#     )

#     fig.show()

#     return regressor, scaler, future_prediction_df

# regressor, scaler, future_predictions = linear_forecasting(df,future_days=30)

# # Print  future_predictions DataFrame
# print("Future Predictions:")
# print(future_predictions)



# def build_lstm_model():
#     """
#     Build and return a BiLSTM model for time-series prediction.
#     """
#     model = Sequential([
#         Conv1D(32, 3, strides=1, activation='relu', input_shape=[30, 18]),
#         Bidirectional(LSTM(64, return_sequences=True)),
#         Bidirectional(LSTM(64, return_sequences=True)),
#         Bidirectional(LSTM(64)),
#         Dense(32, activation='relu'),
#         Dense(1)  # Output layer, predicting the 'Close' price
#     ])

#     model.compile(optimizer=Adam(), loss=Huber(), metrics=['mse', 'mae'])
#     return model

# def train_lstm_model(X_train, y_train, X_test, y_test, company):
#     """
#     Train an LSTM model on the given data.

#     Args:
#         X_train: Training features
#         y_train: Training labels
#         X_test: Testing features
#         y_test: Testing labels
#         company: Name of the stock/company

#     Returns:
#         model: Trained LSTM model
#         history: Training history
#         y_pred: Predictions on test set
#     """

#     print(f'========= Training LSTM Model for {company} =========')

#     # Build model
#     model = build_lstm_model()

#     # Train model
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

#     # Predict on test data
#     y_pred = model.predict(X_test)

#     return model, history, y_pred



# model, predictor, y_pred = train_lstm_model(X_train, y_train, X_test, y_test, stock_name)

# """After training the model, let's visualize if there is any reduction in loss."""

# fig, axes = plt.subplots()
# plt.suptitle('Model Loss')
# axes.plot(predictor.epoch, predictor.history['loss'], label = 'loss')
# axes.plot(predictor.epoch, predictor.history['val_loss'], label = 'val_loss')
# axes.set_title(stock_name)
# axes.set_xlabel('Epochs')
# axes.set_ylabel('Loss')
# axes.xaxis.set_tick_params()
# axes.yaxis.set_tick_params()
# axes.legend(loc = 'upper left')



# y_pred_denormalized = denormalize_predictions(y_pred, df)
# y_test_denormalized = denormalize_predictions(y_test, df)
# # Calculate RMSE
# rmse = np.sqrt(np.mean((y_pred_denormalized - y_test_denormalized) ** 2))

#     # Calculate MAE (Mean Absolute Error)
# mae = mean_absolute_error(y_test, y_pred)

#     # Calculate R-squared (R²)
# r2 = r2_score(y_test, y_pred)

#     # Print all metrics
# print(f'The RMSE for Apple is {rmse}')
# print(f'The MAE for Apple is {mae}')
# print(f'The R-squared (R²) for Apple is {r2}')



# def create_dataframes_for_plots(dataframe, y_pred):
#     training_data_len = int(np.ceil(len(dataframe) * 0.70))

#     # Split the data
#     plot_train = dataframe.iloc[:training_data_len].copy()
#     plot_test = dataframe.iloc[training_data_len:].copy()

#     # Align Predictions with the same index
#     plot_test.loc[:, 'Predictions'] = y_pred.flatten()  # Ensure predictions are 1D

#     return plot_train, plot_test

# # Create train and test dataframes
# plot_train, plot_test = create_dataframes_for_plots(df, y_pred_denormalized)

# # Ensure train and validation sets maintain the original index
# training_data_len = int(np.ceil(len(df) * 0.70))
# train = df.iloc[:training_data_len].copy()
# valid = df.iloc[training_data_len:].copy()

# # Correctly align Predictions in `valid`
# valid['Predictions'] = plot_test['Predictions']

# # 🛠️ Ensure `valid` and `Predictions` align in length
# print(f"Validation set size: {valid.shape}, Predictions size: {plot_test['Predictions'].shape}")

# # Plot the data
# plt.figure(figsize=(16,6))
# plt.title('Stock Price Prediction Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)

# # Fix: Use original index for proper alignment
# plt.plot(train.index, train['Close'], label='Train')


# plt.plot(valid.index, valid['Close'], label='Validation')

# plt.plot(valid.index, valid['Predictions'], label='Predictions', linestyle='dashed')

# # Ensure the x-axis (date) is correctly aligned
# plt.legend(loc='lower right')
# plt.show()


# def forecast_lstm(model, df, normalized_df, future_days=30, n_steps=50):
#     """
#     Forecast the next `future_days` using a trained LSTM model.

#     Args:
#         model: Trained LSTM model.
#         df (DataFrame): Original DataFrame with stock prices (should have a DateTime index).
#         normalized_df (np.array): Normalized dataset used as LSTM input.
#         future_days (int): Number of days to predict (default: 30).
#         n_steps (int): Number of time steps for LSTM input (default: 50).

#     Returns:
#         forecast_df (DataFrame): DataFrame with forecasted dates and denormalized prices.
#     """

#     print("\n========= Forecasting Next 30 Days =========")

#     # Ensure normalized_df is a NumPy array
#     if isinstance(normalized_df, pd.DataFrame):
#         normalized_df = normalized_df.values  # Convert to array

#     # Validate input shape
#     if len(normalized_df.shape) != 2 or normalized_df.shape[1] != 18:
#         raise ValueError(f"Expected normalized_df shape (X, 18), but got {normalized_df.shape}")

#     # Get test data (last 50 days)
#     x_input = normalized_df[-n_steps:].reshape((1, n_steps, 18))  # Reshape for LSTM input
#     temp_input = x_input[0].tolist()  # Convert to a properly structured list of lists

#     lstm_op = []

#     # Ensure df index is datetime format
#     df.index = pd.to_datetime(df['Date'])
#     last_date = df.index[-1]

#     # Generate forecast dates (business days only)
#     forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

#     # **Fit MinMaxScaler on the original DataFrame**
#     scaler = MinMaxScaler()
#     df_close = df[['Close']]  # Select the 'Close' price column
#     df_scaled = scaler.fit_transform(df_close)  # Fit and transform only the Close price column

#     # Generate predictions iteratively
#     for i in range(future_days):
#         # Convert temp_input back to a NumPy array before reshaping
#         x_input = np.array(temp_input[-n_steps:])  # Take last 50 time steps
#         x_input = x_input.reshape((1, n_steps, 18))  # Reshape for LSTM input

#         # Predict next step (normalized output)
#         predicted = model.predict(x_input, verbose=0).flatten()  # Ensure 1D shape

#         # Adjust shape handling based on model output
#         if predicted.shape[0] == 1:
#             predicted_features = np.zeros(18)  # Create an empty array with 18 features
#             predicted_features[0] = predicted[0]  # Store the predicted Close price
#         else:
#             predicted_features = predicted

#         # Append only the first feature (Close price) to output
#         lstm_op.append(predicted_features[0])

#         # Append full predicted feature set & maintain `n_steps` length
#         temp_input.append(predicted_features.tolist())  # Append adjusted feature array
#         temp_input = temp_input[1:]  # Keep only last `n_steps` elements

#     # Convert lstm_op to NumPy array for inverse transformation
#     lstm_op = np.array(lstm_op).reshape(-1, 1)  # Reshape to (30, 1) for inverse scaling

#     # Prepare for denormalization: Create an array with the correct number of features (11)
#     n_features_in = df_scaled.shape[1]  # The number of features the scaler was originally fitted on
#     lstm_op_with_features = np.zeros((lstm_op.shape[0], n_features_in))  # (30, 1)
#     lstm_op_with_features[:, 0] = lstm_op.flatten()  # Place predicted Close price in the first column

#     # Manually denormalize the predicted values (using the min and max of Close)
#     X_min = scaler.data_min_[0]  # Minimum value of Close Price
#     X_max = scaler.data_max_[0]  # Maximum value of Close Price

#     # Manually denormalize the predictions (flattened lstm_op)
#     denormalized_predictions = lstm_op.flatten() * (X_max - X_min) + X_min

#     # Debugging: Check the first few denormalized values
#     #print(f"Manually Denormalized Predictions: {denormalized_predictions[:5]}")

#     # Create DataFrame with predicted dates & prices
#     forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': denormalized_predictions})

#     # Ensure that forecast_df has the same length as forecast_dates
#     if len(forecast_df) != len(forecast_dates):
#         raise ValueError(f"The length of forecast_df ({len(forecast_df)}) does not match the length of forecast_dates ({len(forecast_dates)}).")

#     forecast_df.set_index('Date', inplace=True)

#     print("\nFinal Forecasted Prices:")
#     print(forecast_df)

#     return forecast_df

# df=df.reset_index()

# forecast_df=forecast_lstm(model, df, new_df, future_days=30, n_steps=50)



# last_date = df.index[-1]  # This will be the last date in your actual data

# # Generate the actual dates for the previous 50 business days
# dates_actual = pd.date_range(end=last_date - pd.Timedelta(days=1), periods=30, freq='B')

# # Generate forecast dates for the next 30 business days after the last available date
# forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')


# # Plot the actual data (y_test)
# plt.plot(dates_actual,df['Close'][-30:].values, label='Actual Price')

# # Plot the predicted data (lstm_op)
# plt.plot(forecast_dates, forecast_df, label='Predicted Price', linestyle='--')

# # Set labels and title
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title(f'Price Forecasting for Apple')

# # Rotate date labels for better readability
# plt.xticks(rotation=45)

# # Show legend
# plt.legend()

# # Display the plot
# plt.tight_layout()
# plt.show()


st.markdown("""<span style='color: #63589F;'> This is a red font color</span>""", unsafe_allow_html=True)
with open('./materilas/text/practice.md', 'r') as f:
    markdown_string = f.read()

st.markdown(markdown_string, unsafe_allow_html=True)