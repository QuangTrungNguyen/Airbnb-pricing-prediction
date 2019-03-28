This project analysed a dataset containing Airbnb listings in the Northern Beaches council area of Sydney, with 28 featurs including the number of beds, baths, people, cleaning & deposite fees, reviews, GPS coordinates. I then used these features to train different machine learning models in Python to predict nightly price of Airbnb listings.

•	Exploratory Data Analysis

- Performed statistical imputation to handle significant amount of missing values
- Annalysed the distribution of continuous features, and the correlation between features and rental price.
- Performed log-normal transformationa and z-score standardization of the features to handle non-linear relationships and highly-skewed distributions

•	Feature Engineering

- Mapped GPS coordinates into postcodes using geo-location API to generate a new feature representing district area, 
- Calculated distance from each Airbnb listing to two popular tourist attractions - Westfield Shopping Center and Manly Beach.

•	Model Selection

- Trained gradient boosting trees, extremely randomised trees, Lasso, OLS and Ridge with additive boosting. 
- Combined models using stacking to improve their predictive accuracy.

