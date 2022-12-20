from utils.libraries import *
from utils.functions import *


def prepare_for_modeling(cars):
    cars['zscore'] = (cars['Price'] - cars['Price'].mean()) / cars['Price'].std()

    cars = cars[(cars['zscore'] > -3) & (cars['zscore'] < 3)]

    del cars["zscore"]

    return cars


def compare_model_page(cars):
    st.write("""
        ### Let's Compare All Models
        """)

    cars = prepare_for_modeling(cars)

    # Splitting Data into X and y
    X = cars.drop(columns=['Price'])
    y = cars['Price']

    ohe = OneHotEncoder()
    ohe.fit(X[['Name', 'Variant', 'Transmission', 'Owner_Type', 'Fuel']])

    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['Name', 'Variant', 'Transmission', 'Owner_Type', 'Fuel']),
        remainder='passthrough'
    )

    scaler = StandardScaler(with_mean=False)

    # Linear Regression
    lr = LinearRegression()

    st.write("""
        #### Linear Regression
    """)

    metrics_df, lr = train_model(X, y, column_trans, scaler, lr)
    st.dataframe(metrics_df)

    st.write("""
            #### Ridge Regression
        """)

    # Alpha values
    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

    alpha_r = st.selectbox("Select Alpha for Ridge", alphas)

    ridge = Ridge(alpha=int(alpha_r))

    metrics_df, ridge = train_model(X, y, column_trans, scaler, ridge)
    st.dataframe(metrics_df)

    st.write("""
            #### Lasso Regression
        """)

    alpha_l = st.selectbox("Select Alpha for Lasso", alphas)

    lasso = Lasso(alpha=alpha_l)

    metrics_df, lasso = train_model(X, y, column_trans, scaler, lasso)
    st.dataframe(metrics_df)

    ok = st.button("Save Regression Model")

    if ok:
        data = {"lr": lr, "ridge": ridge, "lasso": lasso}

        with open('./pickle/saved_models.pkl', 'wb') as file:
            pickle.dump(data, file)
