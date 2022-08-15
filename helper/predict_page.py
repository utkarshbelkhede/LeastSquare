from helper.libraries import *
from helper.functions import feature_eng, load_model


def show_predict_page(name):
    st.title("Car Price Prediction")

    st.write("""### We need some information to predict the Price XXX""")

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

    name = st.selectbox("Car Model", name)

    year = st.number_input('Purchased In', 2000, 2030)

    km_driven = st.number_input('Kilometer Driven', 1000, 10000000)

    fuel = st.selectbox("Fuel Type", fu)

    seller_type = st.selectbox("Seller Type", se_type)

    transmission = st.radio("Transmission Type", trans)

    owner = st.selectbox("Owner Type", own)

    mileage = st.number_input('Mileage in kmpl', 5, 50)

    engine = st.number_input('Engine in CC', 800, 5000)

    max_power = st.number_input('Power in bmph', 500, 5000)

    torque = ""

    seats = st.number_input('Seats', 2, 10)

    model = st.selectbox("Select Model", mo)

    ok = st.button("Calculate Price")

    if ok:

        dict_ = {
            "name": [name],
            "year": [year],
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner],
            "mileage": [str(mileage) + " kmpl"],
            "engine": [str(engine) + " CC"],
            "max_power": [str(max_power) + " bmph"],
            "torque": [torque],
            "seats": [seats]
        }

        results = pd.DataFrame(dict_)
        results = feature_eng(results)

        data = load_model()
        lr_loaded = data["lr"]
        ridge_loaded = data["ridge"]
        lasso_loaded = data["lasso"]

        if model == "Linear Regression":
            price = lr_loaded.predict(results)
        elif model == "Ridge Regression":
            price = ridge_loaded.predict(results)
        elif model == "Lasso Regression":
            price = lasso_loaded.predict(results)

        st.subheader(f"The estimated price of your Car is Rupees {price[0]:,.2f}")
