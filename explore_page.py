import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt


def show_explore_page():
    cars = pd.read_csv("/home/utkarsh/PycharmProjects/LeastSquare/Car details v3.csv")
    cars_final_num = cars.select_dtypes(['number'])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Explore Cars Dataset")

    st.write(
        """
    ### Cars
    """
    )
    st.dataframe(cars.head())

    shape = cars.shape
    st.write("There are", shape[0], "rows and ", shape[1], "columns in the Dataset.")

    # We compute age of car and store it in the age columns
    cars["age"] = 2022 - cars.year

    # We encode the owner categories in the order : 'Test Drive Car' > 'First Owner' > 'Second Owner' > 'Third Owner' > 'Fourth & Above Owner'
    cars["owners"] = cars.owner.map(
        {'Test Drive Car': 5, 'First Owner': 4, 'Second Owner': 3, 'Third Owner': 2, 'Fourth & Above Owner': 1})

    # Encoding Transmission values
    cars["transmission_manual"] = cars.transmission.map({'Manual': 1, 'Automatic': 0})
    cars["transmission_automatic"] = cars.transmission.map({'Manual': 0, 'Automatic': 1})

    # Encoding Fuel values
    cars["fuel_petrol"] = cars.fuel.map({'Diesel': 0, 'Petrol': 1, 'LPG': 0, 'CNG': 0})
    cars["fuel_diesel"] = cars.fuel.map({'Diesel': 1, 'Petrol': 0, 'LPG': 0, 'CNG': 0})
    cars["fuel_lpg"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 1, 'CNG': 0})
    cars["fuel_cng"] = cars.fuel.map({'Diesel': 0, 'Petrol': 0, 'LPG': 0, 'CNG': 1})

    # Encoding Seller Information in the order: 'Individual' < 'Dealer' < 'Trustmark Dealer'
    cars["seller"] = cars.seller_type.map({'Trustmark Dealer': 3, 'Dealer': 2, 'Individual': 1})

    # Dropping missing values as they are not significant and are very few. They are due to a data collection error.
    cars = cars.dropna()

    # We drop columns 'name', 'mileage', 'engine', 'max_power', 'torque' as their information cannot be captured by the regression model directly. They could be grouped and added to the analysis but that is out of the scope of this project.

    # We wish to predict the values of selling_price based on all the other values
    cars_final = cars[['age', 'km_driven', 'owners', 'seats', 'transmission_manual',
                       'transmission_automatic', 'fuel_petrol', 'fuel_diesel', 'fuel_lpg',
                       'fuel_cng', 'seller', 'selling_price']]

    #cars_final_num = cars_final[['age', 'km_driven', 'owners', 'seats', 'mileage', 'selling_price']]


    st.write("""
    ### Countplot
    #### Owner Type Vs Number of Cars
    """)

    sns.countplot(x='owner', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    fig = sns.set(rc={'figure.figsize': (14, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation
    **First Owned Cars** are **highest among all**.
    """)

    st.write("""
    ### Pie Chart
    #### Type of Owner Vs Number of cars.
    """)

    #plt.pie(cars['owner'].value_counts(), labels=cars['owner'].unique(), autopct='%.2f')
    #fig = plt.set(rc={'figure.figsize': (14, 10)})\
    #fig = plt.figure(figsize=(2, 2))
    #st.pyplot(fig)

    st.write("""
    ### Observation
    1. **65.96 %** of cars are **First Owned**.
    2. **25.50 %** of cars are **Second Owned**.
    3. **6.45 %** of cars are **Third Owned**.
    4. **2.02 %** of cars are **Fourth and Above Owned**.
    5. **0.06 %** of cars are **Test Drive Cars**.
    """)

    st.write("""
    ### Barplot
    #### Owner Vs Selling Price
    """)

    sns.barplot(x='owner', y='selling_price', data=cars, palette='spring')
    fig = sns.set(rc={'figure.figsize': (10, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation
    **Test Drive cars** have **high average selling price**. 
    
    As **number of owners** increases the **selling price** of car **decreases**.
    """)

    trans_cars = cars['transmission'].value_counts()
    st.write("There are ", trans_cars[0], " Manual and ", trans_cars[1], " Automatic Cars.")

    st.write("""
    ### Countplot
    #### Transmission Vs Number of Cars
    """)

    sns.countplot(x='transmission', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    fig = sns.set(rc={'figure.figsize': (10, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation

    Most of the cars are Manual.
    """)

    st.write("""
    ### Barplot
    #### Transmission Vs Selling Price
    """)

    sns.barplot(x='transmission', y='selling_price', data=cars, palette='spring')
    fig = sns.set(rc={'figure.figsize': (10, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation
    Cars having **Automatic Transmission have high selling price**.
    """)

    st.write("""
    ### Countplot
    #### Fuel Vs Number of Cars
    """)

    sns.countplot(x='fuel', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    fig = sns.set(rc={'figure.figsize': (10, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation

    Most of the cars are **Diesel**.
    """)

    st.write("""
    ### Barplot
    #### Fuel Vs Selling Price
    """)

    sns.barplot(x='fuel', y='selling_price', data=cars, palette='spring')
    fig = sns.set(rc={'figure.figsize': (10, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation
    **Diesel cars** have **high average selling price**.
    """)

    st.write("""
    ### Countplot

    #### Number of cars sold per year.
    """)

    sns.countplot(x='year', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    fig = sns.set(rc={'figure.figsize': (20, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation

    Most of the cars were sold in the year **2017**.
    """)

    st.write("""
    ### Scatter plot
    #### Plotting Selling Price vs Km Driven
    """)

    cars = cars[cars['selling_price'] < 500000]
    plt.figure(figsize=(10, 10))
    sns.regplot(x='km_driven', y='selling_price', data=cars)
    fig = sns.set(rc={'figure.figsize': (20, 10)})
    st.pyplot(fig)

    st.write("""
    ### Observation

    Regression line shows us the trend that,
    As Kilometers driven increases Selling price of the vehicle drops sharply.
    """)

    st.write("""
    ### Regression plot
    #### Owner Vs Selling Price
    """)

    fig = plt.figure(figsize=(10, 10))
    sns.regplot(x='owners', y='selling_price', data=cars)
    st.pyplot(fig)

    st.write("""
    The regression line shows that, **Selling price decreases as number of owner increase**.
    Test drive cars have higest selling price.
    """)

    st.write("""
    ### Regression plot
    #### Seats Vs Selling Price
    """)

    fig = plt.figure(figsize=(10, 10))
    sns.regplot(x='seats', y='selling_price', data=cars_final)
    st.pyplot(fig)

    st.write("""
    ### Correlation Matrix
    """)

    correlations = cars_final_num.corr()
    correlations

    indx = correlations.index
    fig = plt.figure(figsize=(14, 10))
    sns.heatmap(cars_final_num[indx].corr(), annot=True, cmap="YlGnBu")
    st.pyplot(fig)




