from helper.libraries import *


# For Feature Engineering
def feature_eng(cars):
    # extracts the company and model name from name column
    cars["name"] = cars["name"].str.split().str.slice(start=0, stop=2).str.join(' ')

    # We compute age of car and store it in the age columns
    cars["age"] = date.today().year - cars.year

    # extracting numbers from mileage and converting into float
    cars["mileage_kmpl"] = cars.mileage.str.extract(r'(^[0-9]*.[0-9]*)').astype("float64")

    # extracting numbers from max_power and converting into float
    cars["max_power_bhp"] = cars.max_power.str.extract(r'(^[0-9]*.[0-9]*)').astype("float64")

    # extracting numbers from engine and converting into int
    cars["engine_cc"] = cars.engine.str.extract(r'(^[0-9]*.[0-9]*)').astype(int)

    # We encode the owner categories in the order :
    # 'Test Drive Car' > 'First Owner' > 'Second Owner' > 'Third Owner' > 'Fourth & Above Owner'
    cars["owner"] = cars.owner.map(
        {'Test Drive Car': 5, 'First Owner': 4, 'Second Owner': 3, 'Third Owner': 2, 'Fourth & Above Owner': 1})

    # Encoding Transmission values
    cars["transmission_manual"] = cars.transmission.map({'Manual': 1, 'Automatic': 0})

    # Encoding Seller Information in the order:
    # 'Individual' < 'Dealer' < 'Trustmark Dealer'
    cars["seller_type"] = cars.seller_type.map({'Trustmark Dealer': 3, 'Dealer': 2, 'Individual': 1})

    # converting column seats into int
    cars["seats"] = cars.seats.astype(int)

    # columns to remove
    remove_cols = ["year", "mileage", "engine", "torque", "max_power", "transmission"]

    # removing the columns
    cars.drop(columns=remove_cols, inplace=True)

    return cars


# For converting big values into readable form
def format_float(num):
    return np.format_float_positional(round(num, 2), trim='-')


# Returns Dataframe consisting all errors
def metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    dict_ = {
        "MAE": [format_float(mae)],
        "MSE": [format_float(mse)],
        "RMSE": [format_float(rmse)],
        "R2": [(r2)]
    }

    results = pd.DataFrame(dict_)
    results.index = ["Values"]

    return results


# For Training model
def train_model(X, y, transformer, scaler, model):
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    pipe = make_pipeline(transformer, scaler, model)
    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(x_test)

    return metrics(y_test, y_pred), pipe


# For Loading the Pickle File
def load_model():
    with open('./pickle/saved_models.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
