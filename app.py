from utils.functions import feature_engineering
from utils.predict_page import show_predict_page
from utils.explore_page import show_explore_page
from utils.model_page import compare_model_page
from utils import config
from utils.libraries import *


def side_menu():
    try:
        cars = pd.read_csv(config.main_data)
        cars = feature_engineering(cars)
    except NameError:
        print("Some problem with file...")

    page = st.sidebar.selectbox("Explore Or Predict Or Else", ("Understanding the Data", "Compare Models", "Predict"))

    if page == "Understanding the Data":
        show_explore_page(cars)
    elif page == "Compare Models":
        cars.dropna(inplace=True)
        compare_model_page(cars)
    elif page == "Predict":
        name = list(set(cars["Name"]))
        variant = list(set(cars["Variant"]))        
        show_predict_page(name, variant)


if __name__ == '__main__':
    side_menu()
