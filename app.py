from helper.functions import feature_engineering
from helper.predict_page import show_predict_page
from helper.explore_page import show_explore_page
from helper.model_page import compare_model_page
from helper.libraries import *


def side_menu():
    try:
        cars = pd.read_csv("./datasets/Cars24.csv")
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
