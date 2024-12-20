## House Price Prediction

This project aims to predict house prices based on various features such as the year built, neighborhood, exterior materials, and more. We use machine learning models, including Support Vector Machines (SVM), Random Forest Regressor, and Linear Regression, to predict the target variable, SalePrice.
Project Overview

In this project, we perform:

    Exploratory Data Analysis (EDA): We analyze the dataset to understand the distribution and correlations of different features.
    Data Preprocessing: Clean missing values, drop irrelevant columns, and apply One-Hot Encoding to convert categorical features into numerical values.
    Model Training & Evaluation: Train multiple regression models and evaluate their performance using metrics such as Mean Absolute Percentage Error (MAPE) and R-squared.

Key Steps in the Project
1. Dataset Overview

The dataset includes information about houses and various features, such as the size of the house, year built, exterior materials, and more. The target variable is SalePrice, which represents the sale price of the house.
2. Exploratory Data Analysis (EDA)

    We analyze the dataset’s shape, missing values, and perform correlation analysis between numerical features.
    Visualizations like bar plots and heatmaps are created to understand feature distributions and relationships.

3. Data Preprocessing

    Irrelevant columns (e.g., Id) are dropped.
    Missing values in the SalePrice column are filled with the mean value.
    Categorical variables are one-hot encoded to convert them into numerical format.

4. Model Training

    We split the data into training and validation sets.
    We train the following regression models:
        Support Vector Machine (SVM)
        Random Forest Regressor
        Linear Regression

5. Model Evaluation

    The models are evaluated using the Mean Absolute Percentage Error (MAPE) and R-squared values.
    The model with the lowest error and highest R-squared score is selected.

6. Conclusion

The SVM model provides the best results, with the lowest mean absolute error among all models.
Installation

To run this project, you need to install the following dependencies:

pip install pandas matplotlib seaborn scikit-learn

Project Dependencies

    pandas: Data manipulation and analysis.
    matplotlib: Data visualization.
    seaborn: Advanced data visualization.
    scikit-learn: Machine learning library for model building and evaluation.

Files

    HousePricePrediction.xlsx: The dataset used in this project.
    house_price_prediction.py: Python script for data analysis, preprocessing, model training, and evaluation.

Usage

    Clone the repository:

git clone https://github.com/adityak957/HousePricePrediction.git

Navigate to the project directory:

cd HousePricePrediction

Run the script:

    python house_price_prediction.py

Future Improvements

    Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV for optimal model performance.
    Ensemble Methods: Try ensemble learning techniques like boosting (e.g., XGBoost, LightGBM) to further improve accuracy.
    Feature Engineering: Create additional features or interactions that could improve the model's predictive power.

License

This project is licensed under the MIT License - see the LICENSE file for details.
