This project trains several machine learning models in Python to predict nightly price of Airbnb listings in the Northern Beaches council area of Sydney based on features of listings

•	Exploratory Data Analysis
Performed statistical imputation to handle significant amount of missing values, analysed the distribution of continuous features, and the correlation between features and rental price

•	Feature Engineering
Mapped GPS coordinates into postcodes using geo-location API to generate a new feature representing district area, calculated distance from each listing to two popular tourist attractions - Westfield Shopping Center and Manly Beach.

•	Model Selection
Trained gradient boosting trees, extremely randomised trees, Lasso, OLS and Ridge with additive boosting. Combined models using stacking to improve their predictive accuracy
