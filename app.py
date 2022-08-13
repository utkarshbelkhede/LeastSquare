from helper.functions import feature_eng
from helper.predict_page import show_predict_page
from helper.explore_page import show_explore_page
from helper.model_page import compare_model_page
from helper.libraries import *


def side_menu():
    try:
        cars = pd.read_csv("./datasets/Car_details_v3.csv")
    except NameError:
        print("Some problem with file...")

    page = st.sidebar.selectbox("Explore Or Predict Or Else", ("Understanding the Data", "Compare Models", "Predict"))

    if page == "Understanding the Data":
        show_explore_page(cars)
    elif page == "Compare Models":
        cars.dropna(inplace=True)
        cars = feature_eng(cars)
        compare_model_page(cars)
    elif page == "Predict":
        name = list(set(cars["name"].str.split().str.slice(start=0, stop=2).str.join(' ')))
        show_predict_page(name)
        pass



if __name__ == '__main__':
    side_menu()
