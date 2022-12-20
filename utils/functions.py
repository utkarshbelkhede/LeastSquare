from utils.libraries import *

# To Perform Regex Operation
import re 

# Will return string containing numbers
def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return "".join(num)

# For Feature Engineering
def feature_engineering(cars):
    # First seven columns are relevant
    cars = cars.iloc[:,:7]

    # Giving Proper names to features
    cars.rename(columns = {'Title':'Name', 'cvakb':'Variant', 'cvakb1':'Transmission', 'bvr0c':'km_driven', 'bvr0c2':'Owner_Type', 'bvr0c3':'Fuel', '_7udzz':'Price'}, inplace = True)

    # Extracting only numbers
    cars["Price"] = cars["Price"].apply(lambda x: find_number(x))

    # Extracting year of purchase from Name
    cars["Year_Purchased"] = cars["Name"].str.split().str.slice(start=0,stop=1).str.join(' ')

    # Extracting name excluding year of purchase
    cars["Name"] = cars["Name"].str.split().str.slice(start=1,stop=3).str.join(' ')

    # Removing "km"
    cars["km_driven"] = cars["km_driven"].str.split().str.slice(start=0,stop=1).str.join(' ')

    # Extracting only numbers
    cars["km_driven"] = cars["km_driven"].apply(lambda x: find_number(x))

    # Removing Transmission type from the end of Variant
    cars["Variant"] = cars["Variant"].str.rsplit(' ',1).str[0]

    # Converting features to int
    cars = cars.astype({"km_driven":"int","Price":"int", "Year_Purchased":"int"})

    # Deriving Age of Vehical from Year of Purchase
    cars["Age"] = date.today().year - cars["Year_Purchased"]
    cars.drop(['Year_Purchased'], axis=1, inplace=True)

    return cars


# For converting big values into readable form
def format_float(num):
    return np.format_float_positional(round(num, 2), trim='-')


# Returns Dataframe consisting all errors
def metrics(y_test, y_pred, X_train):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Number of rows
    n = len(X_train)

    # Number of Independent Features
    k = len(X_train.columns)

    adj_r2 = 1- ((1-r2) * (n-1)/(n-k-1))

    dict_ = {
        "MAE": [format_float(mae)],
        "MSE": [format_float(mse)],
        "RMSE": [format_float(rmse)],
        "R2": [(r2)],
        "Adjusted-R2": [(adj_r2)]
    }

    results = pd.DataFrame(dict_)
    results.index = ["Values"]

    return results


# For Training model
def train_model(X, y, transformer, scaler, model):
    pipe = make_pipeline(transformer, scaler, model)
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(x_test)
    
    return metrics(y_test,y_pred, X_train), pipe


# For Loading the Pickle File
def load_model():
    with open('./pickle/saved_models.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
