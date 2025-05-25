# Japanese future real estate prices

## Introduction
The goal of this project is to apply machine learning to predict and uncover patterns in the Japanese real estate economy and in doing so brush up on some general data analysis techniques.

## Data Sets
The 2005-2013 data set ultimately come from the MLIT (Ministry of Land, Infrastructure, Transport and Tourism of Japan) and taken from this [Kaggle page](https://www.kaggle.com/datasets/nishiodens/japan-real-estate-transaction-prices?resource=download)

The more updated data set from 2005 to 2024 also come from the [MLIT](https://www.reinfolib.mlit.go.jp/realEstatePrices/)

The internal migration data from 2008 to 2024 is taken from [e-stat](https://www.e-stat.go.jp/en/stat-search/database?page=1&toukei=00200523&bunya_l=02&tstat=000000070001), the official site for Japanese government statistics.

## Regression Analysis
A number of approaches were taken including but not limited to over sampling data on the tails of the price distribution, creating custom loss functions, quantile regression and feature engineering.
Ultimately, the approach(es) with the best results so far are shown on this repo. 

 - Model 1 shows a simple dense neural network targeting log transformed transaction values of properties based mainly on the physical traits of the property as well as the time of construction and transaction

 - Model 2 is similar but a new feature is added, namely the total internal migrants to the area in which the property is located in an attempt to simulate overall demand to live in that area.

## Conclusions
One of the major takeaways from this project is the difficulty in price prediction for cheap properties. While expensive properties tend to correlate strongly with physical features allowing a mean relative error of < 25%, cheap properties do not share this property.
Perhaps there are yet unknown features (particular laws, selling under unique personal circumstances to name only a few) that may improve the models ability to predict values accurately in this lower range.

Duplicating / particular sampling methods to fairly represent cheaper properties seem to reduce the models performance overall. Similar results for custom loss functions.
The best results have come from feature engineering. This can be seen in the difference between the two models.
