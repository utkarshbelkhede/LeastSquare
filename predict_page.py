import streamlit as st
import pickle
import pandas as pd


def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

lr_loaded = data["lr"]
ridge_loaded = data["ridge"]
lasso_loaded = data["lasso"]


# code_loaded = data["code"]

def code_(cars):
    cars["age"] = 2022 - cars.year

    cars["owners"] = cars.owner.map(
        {'Test Drive Car': 5, 'First Owner': 4, 'Second Owner': 3, 'Third Owner': 2, 'Fourth & Above Owner': 1})

    cars["transmission_manual"] = cars.transmission.map({'Manual': 1, 'Automatic': 0})
    cars["transmission_automatic"] = cars.transmission.map({'Manual': 0, 'Automatic': 1})

    cars["fuel_petrol"] = cars.fuel.map({'Diesel': 0, 'Petrol': 1, 'LPG': 0, 'CNG': 0})
    cars["fuel_diesel"] = cars.fuel.map({'Diesel': 1, 'Petrol': 0, 'LPG': 0, 'CNG': 0})
    cars["fuel_lpg"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 1, 'CNG': 0})
    cars["fuel_cng"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 0, 'CNG': 1})

    cars["seller"] = cars.seller_type.map({'Trustmark Dealer': 3, 'Dealer': 2, 'Individual': 1})

    cars_final = cars[['age', 'km_driven', 'owners', 'transmission_manual',
                       'transmission_automatic', 'fuel_petrol', 'fuel_diesel', 'fuel_lpg',
                       'fuel_cng', 'seller']]

    return cars_final


def show_predict_page():
    st.title("Car Price Prediction")

    st.write("""### We need some information to predict the Price""")

    own = (
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    )
    trans = (
        "Manual",
        "Automatic"
    )

    fu = (
        "Diesel",
        "Petrol",
        "LPG",
        "CNG"
    )

    se_type = (
        "Individual",
        "Dealer",
        "Trustmark Dealer"
    )

    mo = (
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    )

    owner = st.selectbox("Owner Type", own)

    year = st.number_input('Purchased In', 0, 2030)

    km_driven = st.number_input('Kilometer Driven', 0, 10000000)

    seller_type = st.selectbox("Seller Type", se_type)

    fuel = st.selectbox("Fuel Type", fu)

    transmission = st.radio("Transmission Type", trans)

    algo = st.selectbox("Select Model", mo)

    ok = st.button("Calculate Price")
    if ok:
        dict_ = {
            "year": [year],
            "km_driven": [km_driven],
            "owner": [owner],
            # "seats":[seats],
            "transmission": [transmission],
            "fuel": [fuel],
            "seller_type": [seller_type]
        }

        results = pd.DataFrame(dict_)
        X = code_(results)

        if algo == "Linear Regression":
            price = lr_loaded.predict(X)
        elif algo == "Ridge Regression":
            price = ridge_loaded.predict(X)
        elif algo == "Lasso Regression":
            price = lasso_loaded.predict(X)

        st.subheader(f"The estimated price of the care is Rupees {price[0]:.2f}")
