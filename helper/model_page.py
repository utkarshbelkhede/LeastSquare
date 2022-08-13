from helper.libraries import *
from helper.functions import *


def compare_model_page(cars):
    st.write("""
        ### Let's Compare All Models
        """)

    # Splitting Data into X and y
    X = cars.drop(columns=['selling_price'])
    y = cars['selling_price']

    # One Hot Encoding
    ohe = OneHotEncoder()
    ohe.fit(X[['name', 'fuel']])

    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'fuel']),
        remainder='passthrough')

    scaler = StandardScaler(with_mean=False)

    # Linear Regression
    lr = LinearRegression()

    # Alpha values
    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

    # Ridge and Lasso
    ridgecv = RidgeCV(alphas=alphas, cv=4)
    lassocv = LassoCV(alphas=alphas, cv=4, normalize=True)

    st.write("""
        #### Linear Regression
    """)

    metrics, lr = train_model(X, y, column_trans, scaler, lr)
    st.dataframe(metrics)

    st.write("""
            #### Ridge Regression
        """)

    metrics, ridgecv = train_model(X, y, column_trans, scaler, ridgecv)
    st.dataframe(metrics)

    st.write("""
                #### Lasso Regression
            """)

    metrics, lassocv = train_model(X, y, column_trans, scaler, lassocv)
    st.dataframe(metrics)

    ok = st.button("Save Regression Model")

    if ok:
        data = {"lr": lr, "ridge": ridgecv, "lasso": lassocv}

        with open('./pickle/saved_models.pkl', 'wb') as file:
            pickle.dump(data, file)
