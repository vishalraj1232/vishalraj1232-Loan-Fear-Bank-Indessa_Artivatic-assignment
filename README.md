# vishalraj1232-Loan-Fear-Bank-Indessa_Artivatic-assignment
## Prediction of loan defaulter based on training set of more than 5L records using Python, Numpy, Pandas and XGBoost.
The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.

After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.

This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a loan of $20000, along with bank, the investors perform a due diligence on the requested loan application. Keep this in mind while understanding data.

In this challenge, you will help this bank by predicting the probability that a member will default.
## Author
Vishal Raj
## Solution Approach
1) Read data, include relevant columns
2) Transform data (data cleansing)
3) Missing value imputation
4) Categorize/Dummify
5) Move target variable to different dataframe
6) Split - train and cross validation sets
7) Predict
