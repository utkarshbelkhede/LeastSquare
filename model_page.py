import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

lr_loaded = data["lr"]
ridge_loaded = data["ridge"]
lasso_loaded = data["lasso"]

def format_float(num):
    return np.format_float_positional(num, trim='-')


def ploy_fit(degree, X_train, X_test, y_train):
    poly = PolynomialFeatures(degree)
    x_poly_train = poly.fit_transform(X_train)
    x_poly_test = poly.transform(X_test)

    # fitting model
    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    # predicting values
    y_pred = model.predict(x_poly_test)

    return y_pred

def check_accuracy(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    dict_ = {
        "Mean Absolute Error": [format_float(mae)],
        "Mean Squared Error": [(mse)],
        "Root Mean Squred Error": [(rmse)],
        "R-Squred": [(r2)]
    }

    results = pd.DataFrame(dict_)
    results.index = ["Values"]

    return results


def compare_model_page():
    cars = pd.read_csv("/home/utkarsh/Documents/MF/Datasets/Car details v3.csv")
    cars_final_num = cars.select_dtypes(['number'])

    # We compute age of car and store it in the age columns
    cars["age"] = 2022 - cars.year

    # We encode the owner categories in the order : 'Test Drive Car' > 'First Owner' > 'Second Owner' > 'Third Owner' > 'Fourth & Above Owner'
    cars["owners"] = cars.owner.map(
        {'Test Drive Car': 5, 'First Owner': 4, 'Second Owner': 3, 'Third Owner': 2, 'Fourth & Above Owner': 1})

    # Encoding Transmission values
    cars["transmission_manual"] = cars.transmission.map({'Manual': 1, 'Automatic': 0})

    # Encoding Fuel values
    cars["fuel_petrol"] = cars.fuel.map({'Diesel': 0, 'Petrol': 1, 'LPG': 0, 'CNG': 0})
    cars["fuel_diesel"] = cars.fuel.map({'Diesel': 1, 'Petrol': 0, 'LPG': 0, 'CNG': 0})
    cars["fuel_lpg"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 1, 'CNG': 0})
    cars["fuel_cng"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 0, 'CNG': 1})

    # Encoding Seller Information in the order: 'Individual' < 'Dealer' < 'Trustmark Dealer'
    cars["seller"] = cars.seller_type.map({'Trustmark Dealer': 3, 'Dealer': 2, 'Individual': 1})

    # Extracting Mileage information from the column
    cars["mileage"] = cars.mileage.str.extract(r'(^[0-9]*.[0-9]*)').astype("float64")

    # Dropping missing values as they are not significant and are very few. They are due to a data collection error.
    cars = cars.dropna()

    # Remove extreme values
    q_low = cars["selling_price"].quantile(0.05)
    q_hi = cars["selling_price"].quantile(0.95)
    cars = cars[(cars["selling_price"] < q_hi) & (cars["selling_price"] > q_low)]

    # We drop columns 'name', 'engine', 'max_power', 'torque' as their information cannot be captured by the regression model directly. They could be grouped and added to the analysis but that is out of the scope of this project.

    # We wish to predict the values of selling_price based on all the other values
    cars_final = cars[
        ['age', 'km_driven', 'owners', 'seats', 'transmission_manual', 'mileage', 'fuel_petrol', 'fuel_diesel',
         'fuel_lpg', 'fuel_cng', 'seller', 'selling_price']]

    X = cars_final.drop(['selling_price', 'seats'], axis=1)
    y = cars_final['selling_price']

    st.write("""
    ### Let's Compare All Models
    """)

    test_data = st.slider('Test Data Percentage', 10, 90)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data/100, random_state=42)

    st.write("""
    ### Linear Regression
    """)

    lr = LinearRegression()

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.dataframe(check_accuracy(y_test,y_pred))

    st.write("""
    ### Ridge Regression
    """)

    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

    ridgeCV = RidgeCV(alphas=alphas,
                      cv=4).fit(X_train, y_train)

    y_pred = ridgeCV.predict(X_test)

    st.write("Ridge Alpha",ridgeCV.alpha_)

    st.dataframe(check_accuracy(y_test, y_pred))

    st.write("""
    ### Lasso Regression
    """)

    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

    lassoCV = LassoCV(alphas=alphas,
                      cv=4).fit(X_train, y_train)

    y_pred = lassoCV.predict(X_test)

    st.write("Lasso Alpha",lassoCV.alpha_)

    st.dataframe(check_accuracy(y_test, y_pred))

    st.write("""
        ### Ploynomial Regression
        """)

    how_many_poly = st.slider('Enter How many degree ploynomial you want to fit?', 1, 10)

    degree = []

    for i in range(how_many_poly):
        degree.append(ploy_fit(i + 1, X_train, X_test, y_train))

    for i in range(how_many_poly):
        st.write("Degree ", i+1)
        st.dataframe(check_accuracy(y_test, degree[i]))










