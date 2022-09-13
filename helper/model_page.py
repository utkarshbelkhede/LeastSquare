from helper.libraries import *
from helper.functions import *

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
    X = cars.drop(columns =['Price'])
    y = cars['Price']

    ohe = OneHotEncoder()
    ohe.fit(X[['Name','Variant','Transmission','Owner_Type','Fuel']])
    
    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_),['Name','Variant','Transmission','Owner_Type','Fuel']),
        remainder='passthrough'
    )  

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
            #### Ridge Regression
        """)

    metrics, lassocv = train_model(X, y, column_trans, scaler, lassocv)
    st.dataframe(metrics)

    ok = st.button("Save Regression Model")

    if ok:
        data = {"lr": lr, "ridge": ridgecv, "lasso": lassocv}

        with open('./pickle/saved_models.pkl', 'wb') as file:
            pickle.dump(data, file)
