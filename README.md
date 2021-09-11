# nifty-50-prediction-a-multi-factor-approach

## Abstract

The stock market is highly volatile and highly non-linear and predicting the stock price accurately is very hard. The price of a stock goes up or down based on demand and supply but is influenced by various factors like corporate actions, economic policies, inflation, interest rates, global economic conditions, crude oil prices, war, natural calamities, etc. As India is an emerging market, Indian stock markets are also influenced by global events. NIFTY 50 price-to-earnings ratio (PE ratio), features from Global Indices like S&P 500, NASDAQ Composite, Euronext 100, and other factors like US 10 year treasury yields, Gold prices, Crude oil prices, USD-INR exchange rate are used to predict NIFTY 50 Index value. Different Machine learning and Deep learning algorithms are used to predict the NIFTY 50 Index value and evaluated using the Root Mean Squared Error (RMSE) Metric.

## [Link](https://binginagesh.medium.com/nifty-50-index-prediction-a-multi-factor-based-approach-224d60a43d23) to medium blog.

## Results

| Model        | test RMSE           |
| ------------- |:-------------:|
|  Baseline Model    | 1863.719 |
| Simple Moving Average      | 150.375      |
| Exponential Moving Average Model | 149.114      |
|  ARIMA(2, 1, 2) Model    | 1875.698 |
| Linear Regression Model      | 149.809      |
| **Linear Regression Model with L1 Regularization** | **144.816**      |
|  Linear Regression Model with L2 Regularization    | 149.603 |
| Linear Regression Model with L1 & L2 Regularization      | 144.851      |
| Support Vector Regressor (linear) | 186.138      |
| LSTM      | 168.314      |
| LSTM with other features | 166.954      |

## [Link](https://nifty-50-prediction.herokuapp.com/) to Heroku web-app.

## Sample predictions
![Post - Feed](https://user-images.githubusercontent.com/46963154/132938670-a633a353-a800-4326-91aa-9233d02ccb79.gif)



https://user-images.githubusercontent.com/46963154/132727840-85ac71ff-237b-4d57-b845-368e1a7cdce2.mp4


## Conclusion

Stock prices are extremely volatile and highly non-linear. Accurate prediction of stock prediction is an extremely difficult task. As India is an emerging economy, global events impact the Indian stock market. In this Case Study, we tried to incorporate global event factors to predict NIFTY 50 closing value. Experiments are done with different algorithms. Linear Regression with L1 regularization gave the lowest RMSE. One important thing to note is in the Simple Moving Average Model, we did the prediction that yesterday’s closing value of NIFTY 50 is today’s closing value. With this, we got an RMSE of **150.375** but using Linear Regression with L1 regularization we got an RMSE of **144.816**. This proves that predicting stock prices is an extremely difficult task. We did not consider all the variables that impact NIFTY 50, we did consider only a few and we could improve from the SMA model. With the multi-factor approach, we could slightly decrease the RMSE of the LSTM model from **168.314** to **166.954**. This also proves that other indices affect Indian indices.

