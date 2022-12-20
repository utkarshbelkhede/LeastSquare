from utils.libraries import *
from utils.functions import feature_engineering, load_model


def show_predict_page(name, variant):
    st.title("Car Price Prediction")

    st.write("""### We need some information to predict the Price""")
    
    trans = (
        "Manual",
        "Automatic"
    )

    own = (
        "1st Owner",
        "2nd Owner",
        "3rd Owner"
    )

    fu = (
        "Diesel",
        "Petrol",
        "Petrol + CNG"
    )

    mo = (
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    )

    Name = st.selectbox("Car Model", name)

    Variant = st.selectbox("Car Variant", variant)

    Transmission = st.radio("Transmission Type", trans)

    km_driven = st.number_input('Kilometer Driven', 1000, 10000000)

    Owner_Type = st.selectbox("Owner Type", own)

    Fuel = st.selectbox("Fuel Type", fu)
    
    Year = st.number_input('Purchased In', 2000, 2030)
    Age = date.today().year - Year

    model = st.selectbox("Select Model", mo)

    ok = st.button("Calculate Price")

    if ok:

        dict_ = {
            "Name": [Name],
            "Variant": [Variant],
            "Transmission": [Transmission],
            "km_driven":[km_driven],
            "Owner_Type": [Owner_Type],
            "Fuel": [Fuel],
            "Age": [Age]
        }

        results = pd.DataFrame(dict_)

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
