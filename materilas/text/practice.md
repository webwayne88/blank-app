# 2.  Getting Started

We will use Yahoo Finance to analyze the stock market performance of four major technology companies: Apple (AAPL), Amazon (AMZN), Tesla (TSLA), and Microsoft (MSFT). Our dataset contains daily returns for these companies from January 1, 2020, to January 1, 2025.

Dataset explanation :

**Open**: Price at market opening.

**High**: The highest price during the day.

**Low**: The lowest price during the day.

**Close**: Price at market close.

**Adj Close**: The closing price adjusted for dividends, stock splits, etc., making it useful for analysis over time.

**Volume**: The total number of shares traded during the day.


Examining our dataset, we observe that the data consists of numerical values, with the date serving as the index.

## 2.1 Descriptive Statistics

`.describe()` provides a summary of descriptive statistics for your dataset, including measures of central tendency (mean, median), dispersion (standard deviation, min, max), and distribution (percentiles). It excludes NaN values and works with both numeric and object data types. The output adapts based on the data type of the columns analyzed.



We have only 1,259 records over the five-year period because the data excludes weekends and public holidays when the stock market is closed.

## 2.2 Information About the Data

The `.info()` method displays details about a DataFrame, such as the index type, column data types, the number of non-null values in each column, and the memory usage.



## 2.3 Unique Values

The `df.nunique()` method shows the number of unique values in each column of the DataFrame:

- **Ticker**: 4 unique tickers (AAPL, AMZN, TSLA, MSFT).  
- **Open, High, Low, Close, Adj Close**: Thousands of unique price values due to daily fluctuations.  
- **Volume**: 5017 unique trading volumes, reflecting the variability in daily stock activity.



# 3. Exploratory Data Analysis (EDA)

## 3.1 Distribution of Open Prices


# Skewness of Each Company's Opening Prices

- **Apple's skewness: -0.12**  
Apple's opening price distribution is slightly negatively skewed, meaning the left tail of the distribution is a bit longer than the right. This indicates that Apple has experienced occasional lower-than-usual opening prices, which pulls the distribution slightly to the left.

- **Tesla's skewness: 0.39**  
Tesla's opening price distribution shows a mild positive skew, meaning the right tail is longer than the left. This suggests that Tesla has had occasional higher-than-usual opening prices, which skews the distribution to the right.

- **Microsoft's skewness: -0.134**  
Microsoft's opening price distribution is slightly negatively skewed. Similar to Apple, this indicates that Microsoft has experienced occasional lower-than-usual opening prices, shifting the distribution slightly to the left.

- **Amazon's skewness: -0.037**  
Amazon's opening price distribution is very close to symmetrical, with a skewness value near zero. This suggests that Amazon’s opening prices are fairly evenly distributed, without significant skew toward higher or lower values.


- Apple, Microsoft, and Amazon all show slight negative skew, indicating occasional lower-than-usual opening prices.
- Tesla has a mild positive skew, suggesting it has experienced occasional higher-than-usual opening prices.
- Overall, the skewness is relatively mild for all companies, with Tesla exhibiting the most notable positive skew, and the others showing mild negative skew.

## 3.2 Distribution of Close Prices


### Skewness Interpretation for Each Company's Closing Prices

- **Apple's skewness: 0.11**  
Apple's closing price distribution is slightly positively skewed, meaning the right tail of the distribution is a bit longer than the left. This suggests that there are occasional higher closing prices, pulling the distribution slightly to the right, but the effect is mild.

- **Tesla's skewness: 0.21**  
Tesla's closing price distribution is also mildly positively skewed. The right tail is slightly longer, indicating that there are occasional higher closing prices that are skewing the distribution to the right, though this effect is not very pronounced.

- **Microsoft's skewness: -0.014**  
Microsoft's closing price distribution is very close to being symmetric, with a skewness value near zero. This suggests that the distribution of Microsoft’s closing prices is almost normal, without any significant skew to either the left or right.

- **Amazon's skewness: 0.071**  
Amazon's closing price distribution is slightly positively skewed. Similar to Apple and Tesla, it has a mild tendency for higher closing prices, but the effect is relatively small and does not significantly alter the overall shape of the distribution.

In this case, all four companies show either a mild positive or near-zero skew. Tesla has the most noticeable positive skew, while Microsoft shows almost no skew at all. Apple and Amazon show slight positive skew, with Amazon's being the least pronounced. None of the companies have significant negative skew, meaning there are no notable patterns of unusually low closing prices for any of them.

## 3.3 Open vs Close Price Scatter Plot


- In this figure, we examine the relationship between the stock's opening and closing prices to understand how closely they align throughout the day. By observing whether the prices are correlated or diverge, we can gain insights into how similar or different the stock's value is at the start and end of the trading session. This information helps in assessing the volatility of the stock.

From the plot, we can see that for each company, the open and close prices are close, indicating that there isn't significant fluctuation between the beginning and end of the trading day. This suggests a level of stability in the stock's daily performance for these companies.

## 3.4 What was the daily return of the stock on average?


### **Interpretation of the Daily Return Plots**  

The four histograms show **daily return plots** for **Apple (AAPL), Tesla (TSLA), Microsoft (MSFT), and Amazon (AMZN)** from 2020 to early 2025. Each plot represents the daily percentage change in adjusted closing prices, showing the **volatility** and **return distribution** over time.  

 **Most returns cluster around zero:**  
   - Across all four stocks, the majority of daily returns remain within a small range (±5%), suggesting **relative stability** most of the time.  
   - However, there are noticeable **outliers** indicating high volatility on certain days.  

 **Tesla (TSLA) shows the highest volatility:**  
   - Tesla’s plot has extreme daily return spikes, sometimes exceeding **+30%** and **-20%**, highlighting **higher price fluctuations**.  
   - This suggests that Tesla is a **high-risk, high-reward** stock, likely driven by market sentiment, earnings reports, or industry trends (e.g., EV sector growth).  

 **Microsoft (MSFT) has a more balanced and stable return distribution:**  
   - Most of its daily returns stay within **±10%**, showing less volatility compared to Tesla.  
   - This aligns with Microsoft’s position as a **stable, blue-chip tech company**, less affected by market shocks.  

 **Apple (AAPL) and Amazon (AMZN) show moderate volatility:**  
   - Both exhibit **occasional spikes**, but their daily returns remain mostly between **±7.5%**.  
   - Apple, being a major consumer tech company, experiences some volatility around **product launches and earnings reports**.  
   - Amazon, a major e-commerce and cloud computing player, shows **relatively lower volatility** than Tesla but still has some notable outliers.

Great! Now, let's visualize the average daily return using a histogram. We'll leverage Seaborn to generate both a histogram and a KDE plot within the same figure.

Histogram visuals provide a clearer representation of daily returns compared to line plots. The analysis highlights the following key points:

- Tesla's returns exhibit high volatility, with frequent extreme values. Positive outliers reach approximately 30%, while negative returns are mostly capped at around -20%, indicating significant price swings.

- Microsoft's returns show a more stable pattern, with values predominantly ranging between -10% and 10%. The majority of daily returns are concentrated near zero, suggesting a more balanced and predictable performance compared to Tesla.

## 3.5 Adjusted Closing Price

- The **Adjusted Closing Price** is the closing price of a stock adjusted for factors like **dividends**, **stock splits**, and **new stock offerings**. Unlike the regular closing price, it reflects these corporate actions to give a more accurate view of a stock’s performance over time. This helps investors assess the true value and growth of a stock, accounting for events that affect the stock price but don't reflect the company’s actual market performance.

- Upon analyzing the charts, we can observe that the closing prices for all four companies have generally risen over the years. Microsoft, in particular, reached its highest closing price of approximately $450. However, in 2023, each of the companies experienced a noticeable decline in their closing prices. Among them, Amazon appears to have been the most significantly impacted by this downturn, showing the sharpest decrease.

## 3.6 Volume of Sales

- The **Volume of Sales** refers to the total number of shares of a stock that are traded during a specific period. It indicates the level of activity or liquidity in the market for that stock. Higher trading volume often suggests increased investor interest, while lower volume can indicate less market activity.

- Initially, Microsoft and Tesla had the highest sales volumes among the companies. However, as we move towards the present day, it's clear that the trading volumes for these companies have declined. This shift may reflect changing market conditions, investor sentiment, or other external factors that have influenced the level of trading activity over time.

## 3.7 What was the moving average of the various stocks?

- The **Moving Average (MA)** is a commonly used tool in technical analysis that smooths price data by calculating a continuously updated average. This average is computed over a defined time period, such as 10 days, 20 minutes, 30 weeks, or any other period selected by the trader.


- We have a daily dataset, so we applied moving averages for 10, 20, and 50 days to the data. The graph demonstrates that the 10-day and 20-day moving averages are most effective for capturing trends, as they reflect the underlying patterns while minimizing noise in the data.

## 3.8 What was the correlation between different stocks closing prices?

- The joint plot illustrates the relationship between **Apple (AAPL)** and **Microsoft (MSFT)** daily returns.
- The scatter plot at the center indicates a positive correlation, suggesting that when Apple’s stock returns increase, Microsoft’s stock returns tend to rise as well.
- The histograms on the top and right display the individual return distributions for Apple and Microsoft, which appear approximately normal but may exhibit some skewness.
- The spread of points suggests some degree of volatility in returns, but the positive trend implies these two stocks generally move in sync within the market.

This visualization is a **Seaborn PairPlot (PairGrid)**, commonly used in **exploratory data analysis (EDA)** to examine relationships between multiple numerical variables. The plot  show relationships between stock returns of **AAPL (Apple), AMZN (Amazon), MSFT (Microsoft), and TSLA (Tesla)**.

- **AAPL vs AMZN, AAPL vs MSFT, AMZN vs MSFT:** These stocks have strong correlations, as seen in the tight clustering along a diagonal trend.
- **TSLA vs Other Stocks:** Tesla's return distribution has a wider spread, suggesting higher volatility compared to other stocks.
- **Density plots suggest dependencies:** The tighter the density contours, the stronger the correlation.
- **Outliers and extreme returns:** Some scatter plots show individual points far from the main clusters, indicating potential outliers or extreme market movements.

- Two graphics display heatmaps of stock returns and closing prices, respectively. These heatmaps illustrate the relationships between different features. Notably, Apple and Microsoft exhibit a strong positive correlation, as do Amazon and Microsoft. Interestingly, all technology companies show a positive correlation with one another.

## 3.9  How much value do we put at risk by investing in a particular stock?

- The graph visualizes the risk (likely standard deviation) associated with different companies.
- **Tesla (TSLA)** exhibits the highest risk, positioned at the top, indicating significant price volatility.
- **Microsoft (MSFT)**, **Apple (AAPL)**, and **Amazon (AMZN)** have relatively lower risk levels, clustered closely together.
- The upward position of **Tesla** suggests it experiences the most substantial price fluctuations, making it the most volatile stock in the group.

# 4. Feature Engineering

- In this section, we will concentrate on predicting the future closing price of **Apple Inc. (AAPL)**. To begin, we will collect historical stock data from **Yahoo Finance**, covering the period from 2010 to the present. After gathering the data, we will proceed with feature extraction, identifying key variables that can enhance the accuracy of our predictions.

- By analyzing historical trends, market patterns, and statistical indicators, we aim to develop a robust forecasting model that provides insights into Apple's potential future performance.


Below is a detailed explanation of each feature:

- **`price_change`**: Measures the difference between the closing and opening prices for a given day. This helps determine whether the stock gained or lost value during the trading session.  
- **`returns`**: Represents the percentage change in the closing price from one day to the next, giving insight into the stock’s overall performance over time.  
- **`average_price`**: Computes the average of the opening and closing prices for a given day, serving as a general indicator of that day’s stock value.  
- **`price_range`**: Captures the difference between the highest and lowest prices of the day, reflecting the stock’s intraday volatility.  
- **`volume_change`**: Measures the change in trading volume compared to the previous day, helping to analyze market interest and liquidity trends.  
- **`moving_average_10`**: Calculates the 10-day moving average of the closing price, smoothing short-term fluctuations and indicating the overall trend.  
- **`RSI` (Relative Strength Index)**: Measures the momentum of price movements on a scale from 0 to 100. It is calculated using the following formula:  

 $$ RSI = 100 - \frac{100}{1 + RS} $$

  where **RS** (Relative Strength) is calculated as:  

  $$
  RS = \frac{\text{Average Gain over } n \text{ periods}}{\text{Average Loss over } n \text{ periods}}
  $$

  A high RSI suggests overbought conditions, while a low RSI indicates oversold conditions.  

- **`MACD` (Moving Average Convergence Divergence)**: Computes the difference between the 12-day and 26-day exponential moving averages (EMAs) of the closing price. It is used to identify trend direction, strength, and potential reversals.
- The **MACD Signal Line** is an essential technical indicator used in stock analysis. It is calculated as the **9-day moving average of the MACD (Moving Average Convergence Divergence)**. This feature helps smooth out fluctuations in the MACD line and provides clearer signals for potential buy or sell opportunities. Traders often use the MACD Signal Line to identify trend changes and confirm market momentum. The formula for the MACD Signal Line is:

$$
\text{MACD Signal} = \text{9-day EMA of MACD}
$$


- **`20_SMA` (20-Day Simple Moving Average)**: Represents the simple moving average over 20 days, providing a longer-term trend indication.  
- **`BB_Upper` (Upper Bollinger Band)**: The upper bound of Bollinger Bands, calculated using the 20-day SMA plus two standard deviations. It helps identify potential overbought conditions.
  The formula for the **Upper Bollinger Band** is:
  

$$
\text{Upper Band} = \text{20-day SMA} + (2 \times \sigma)
$$
- **`BB_Lower` (Lower Bollinger Band)**: The lower bound of Bollinger Bands, computed as the 20-day SMA minus two standard deviations. It helps identify potential oversold conditions. The formula for the **Lower Bollinger Band** is:

$$
\text{Lower Band} = \text{20-day SMA} - (2 \times \sigma)
$$

# 5. Preprocessing

## 5.1 Normalizing

- In this section, we will perform data normalization using the max-min normalization method. The goal of normalization is to standardize the numerical features in the dataset, ensuring that they all fall within a consistent range or scale.

## 5.2 Denormalization

After making predictions with the model, it is necessary to convert the normalized values back to their original scale in order to accurately interpret the results. This step is crucial because the model typically works with normalized data, but to understand the actual predictions, we need them in their original, unscaled form. Therefore, we perform this denormalization to ensure the predicted values reflect the true price or value.

## 5.3 Splitting Data

- The below function splits a time series dataset into training and testing sets (80% train, 20% test), creates input features and target labels for a specified number of prediction days (pred_days), and reshapes the data for use in machine learning models such as LSTM. The function returns the reshaped training and testing datasets.

# 6. Modeling

- In this section, we will utilize two different forecasting models to evaluate their stock price prediction performance: LSTM and linear regression. We will begin by implementing the Linear Regression.

## 6.1 Linear Regression

The `linear_prediction` function predicts stock closing prices using linear regression by processing and normalizing historical stock data, then evaluating the model's performance with metrics like MAE, MSE, RMSE, and R². It also visualizes the actual vs. predicted values and returns the model, scaler, and evaluation results.

The provided performance metrics reflect how well the model predicted the stock prices:

- **Mean Absolute Error (0.713)**: On average, the model's predictions are off by **\$0.71** from the actual values, indicating relatively small errors.

- **Root Mean Squared Error (0.947)**: The model's error, on average, is approximately **$0.95**, indicating a slightly higher degree of error than the MAE.
- **R² Score (0.9995)**: The model explains **99.95%** of the variance in the data, which indicates an **exceptionally good fit** and suggests that the model's predictions are highly accurate.

the **R² score** being close to **1** demonstrates that the model is highly effective, with minimal error in its predictions.

## 6.1.2 Future Forecasting for 30 days
- The forecasting result shows the **predicted stock closing prices** for the next 30 days (from February 27, 2025, to April 9, 2025). These predictions are based on a model that forecasts future values based on historical data and other relevant features. The values represent the estimated closing price of the stock for each corresponding date.

For instance:
- On **April 9, 2025**, the predicted closing price is **241.17**.
- On **February 27, 2025**, the predicted closing price is **230.48**.
- As the dates move forward, the predicted prices fluctuate based on the model's interpretation of trends, seasonal effects, or other factors.

## 6.2 Long Term Short Memory (LSTM)

LSTM is a type of recurrent neural network (RNN) designed to handle sequential data and time series forecasting. It overcomes the vanishing gradient problem in standard RNNs by using memory cells to retain long-term dependencies.

The `build_model` function creates and trains a Bidirectional LSTM (BiLSTM) model to predict stock prices. It begins by adding a Conv1D layer to capture short-term patterns in the stock data, followed by three Bidirectional LSTM layers to capture both past and future dependencies in the time-series data. A Dense layer is used for further feature extraction, and the output layer predicts the stock price. The model is compiled with the Adam optimizer and Huber loss, and trained for 50 epochs with performance metrics MSE and MAE. After training, the function generates stock price predictions on the test data. The function returns the trained model, training history, and predicted prices.

Now that the model is created, we will proceed to train it using our dataset.

**X-Axis (Epochs)** : Represents the number of training iterations (from 0 to 50).

**Y-Axis (Loss)** : Measures how well the model is performing (lower values indicate better performance).

**Blue Line (Loss)** : Represents the training loss, which shows how well the model is learning from the training dataset.

**Orange Line (Validation Loss)**: Represents the validation loss, which shows how well the model is performing on unseen validation data.


Initially, both loss and validation loss are high. Over time, both losses decrease, indicating that the model is learning and improving.
The training loss (blue line) is consistently lower than the validation loss (orange line), which is expected.
Some fluctuations in validation loss suggest some level of overfitting, but overall, the model seems to generalize well.

## 6.2.1 Prediction and Metrics

We will evaluate the performance of our model by comparing it with the test data and the training data, scaled accordingly, and analyze the RMSE, MAE, and R-squared values.
- **RMSE (6.60)**: On average, the model's predictions are **\$6.60 off** from the actual Apple stock prices.  
- **MAE (0.023)**: The model's predictions deviate from the actual prices by about **\$0.023 on average**.  
- **R² (0.940)**: The model explains **94.0% of the variance** in Apple’s stock prices, meaning it fits the data very well.  

 these results show that the model performs well, with **small errors relative to stock price fluctuations** and a **high degree of accuracy**.

Next, we will visualize the predicted values.

- The graphic above illustrates the model's performance across training, validation, and prediction stages. As shown, the predictions closely align with the validation values, indicating the model's strong ability to generalize. Additionally, the test values are accurately predicted, demonstrating the model's effectiveness in forecasting future data.

## 6.2.2 Predicting & Forecasting for next 30 Days

- In this section, we use the trained model to predict and forecast stock prices for the next 30 days. The model leverages historical data to generate accurate predictions, helping to estimate future trends and price movements.

- The below code prepares data for time-series prediction by sorting the dataset, splitting it into training and test sets, and extracting the last 50 days of the test data as input (`x_input`) for making future predictions.

- We use the previous 50 days of data because recent trends provide the most relevant information for accurately predicting future values in time-series forecasting.


# 7. Comparing Model Performances

\
\begin{array}{|c|c|c|}
\hline
\textbf{Model} & \textbf{MAE} & \textbf{RMSE} \\
\hline
\textbf{Linear Regression} & 0.7366 & 0.9899  \\
\hline
\textbf{LSTM} & 0.023 & 6.60  \\
\hline
\end{array}

When analyzing the results closely, we can see that **Linear Regression** outperforms **LSTM** in this particular scenario. Let's break it down metric by metric:

#### **📌 Mean Absolute Error (MAE)**
-MAE measures the average absolute difference between predicted and actual values.
-A lower MAE indicates better accuracy.
-LSTM (0.023) significantly outperforms Linear Regression (0.7366) in terms of MAE, meaning it produces predictions that are much closer to actual values on average.

#### **📌 Root Mean Squared Error (RMSE)**
-RMSE penalizes larger errors more than MAE.
-Linear Regression has a much lower RMSE (0.9899) compared to LSTM (6.60).
-The higher RMSE for LSTM suggests that, although its average error (MAE) is low, it sometimes makes large mistakes that increase the overall error.

---
### **📈 Key Observations**
✔️ LSTM has a much lower MAE (0.023), meaning it provides more precise predictions on average.

✔️ LSTM has a much higher RMSE (6.60), indicating occasional large errors, which can be problematic in financial forecasting.

✔️ Linear Regression is more stable, with a higher MAE but lower RMSE, meaning its errors are more consistent and predictable.

#### Which Model is Better?
-If consistent, stable predictions are preferred. Linear Regression is the better choice.

-If minimizing small errors is the main goal (and occasional large mistakes are acceptable).LSTM might be preferable.

However, LSTM’s high RMSE suggests that it may struggle with outliers or sudden market shifts.

# 8. Conclusion

In this notebook, we conducted a comprehensive stock market analysis using the Yahoo Finance library. Our approach involved the following key steps:  

- **Data Retrieval**: We collected historical stock data from Yahoo Finance for analysis.  
- **Exploratory Data Analysis (EDA)**: We visualized trends and patterns using various techniques to gain deeper insights.  
- **Data Preprocessing**: We applied essential preprocessing steps, including normalization and data reshaping, to prepare the dataset for modeling.  
- **Feature Engineering**: We extracted meaningful indicators such as **Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands,** and **Moving Averages** to enhance predictive performance.  
- **Train-Test Split**: We divided the dataset into **70% training and 30% testing** to evaluate model performance effectively.  
- **Modeling**: We implemented and trained two different machine learning models: **Long Short-Term Memory (LSTM) networks** and **Linear Regression** for stock price prediction.  
- **Performance Evaluation**: We compared the results of both models using key metrics, assessing their predictive accuracy and overall effectiveness.  

This analysis provides valuable insights into stock price movements and helps in identifying the most suitable model for future predictions.

# 9. Future Work & Improvements

- **Hyperparameter Optimization:**  We will fine-tune model parameters to enhance predictive accuracy and reduce errors.
- **Model Comparison:** In addition to **LSTM**, we will evaluate and compare performance with **ARIMA**, **XGBoost**, and **CNN** models to identify the most effective approach.
- **Feature Expansion:** We plan to incorporate additional financial indicators and external factors (e.g., market sentiment, economic trends) to improve model robustness.
- **Hybrid Model Approach:** Combining **LSTM** with **ARIMA**, **XGBoost**, and **CNN** to leverage both deep learning and traditional statistical methods for improved performance.