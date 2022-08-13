from helper.libraries import *


def show_explore_page(cars):

    st.title("Explore Cars Dataset")

    st.write(
        """
    ### Cars
    """
    )
    st.dataframe(cars.head())

    shape = cars.shape
    st.write("There are", shape[0], "rows and ", shape[1], "columns in the Dataset.")

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (8, 5)})
    plt.title("Heat Map for Missing Values")
    sns.heatmap(cars.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    st.pyplot(fig)

    cars.dropna(inplace=True)

    st.write("""
        #### Exploratory Data Analysis
        """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10,8)})
    plt.title("Countplot Owner Type Vs Number of Cars")
    sns.countplot(x='owner', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    st.pyplot(fig)

    st.write("""
        #### Observation
        **First Owned Cars** are **highest among all**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10, 8)})
    plt.title("Owner Vs Selling Price")
    sns.barplot(x='owner', y='selling_price', data=cars, palette='spring')
    st.pyplot(fig)

    st.write("""
        #### Observation
        **Test Drive cars** have **high average selling price**. 
        As **number of owners** increases the **selling price** of car **decreases**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10, 8)})
    plt.title("Transmission Vs Number of Cars")
    sns.countplot(x='transmission', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    st.pyplot(fig)

    st.write("""
        #### Observation
        Most of the cars are **Manual**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10, 8)})
    plt.title("Transmission Vs Selling Price")
    sns.barplot(x='transmission', y='selling_price', data=cars, palette='spring')
    st.pyplot(fig)

    st.write("""
        #### Observation
        Cars having **Automatic Transmission have high selling price**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10, 8)})
    plt.title("Fuel Vs Number of Cars")
    sns.countplot(x='fuel', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    st.pyplot(fig)

    st.write("""
        #### Observation
        Most of the cars are **Diesel**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (10, 8)})
    plt.title("Fuel Vs Selling Price")
    sns.barplot(x='fuel', y='selling_price', data=cars, palette='spring')
    st.pyplot(fig)

    st.write("""
        #### Observation
        **Diesel cars** have **high average selling price**.
    """)

    fig, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize': (15, 10)})
    plt.title("Fuel Vs Selling Price")
    sns.countplot(x='year', data=cars).set_ylabel("No. Of Cars", fontsize=10)
    st.pyplot(fig)

    st.write("""
        #### Observation
        In the year 2017, Most of the cars were sold.
    """)



