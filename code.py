# Import the pandas library for data manipulation and analysis
import pandas as pd

# Import the Matplotlib library for creating plots and visualizations
import matplotlib.pyplot as plt

# Import the NumPy library for numerical computations
import numpy as np

# Load the oil consumption data from an Excel file into a pandas DataFrame
oil_consumption = pd.read_excel("/content/oilconsumption.xlsx")

# Display the first five rows of the oil consumption DataFrame
print(oil_consumption.head(5))

# Display the last five rows of the oil consumption DataFrame
print(oil_consumption.tail(5))

import pandas as pd

def clean_dataframe(df):
    """
    Cleans the DataFrame by removing specified rows and columns
    and rounding numeric entries for simplified analysis.

    Parameters:
    df (DataFrame): The DataFrame to clean.

    Returns:
    DataFrame: A cleaned version of the DataFrame.
    """
    # Trim the DataFrame to relevant data and round numeric values
    cleaned_df = df.iloc[1:-4, :-3].round()
    return cleaned_df

def set_custom_index(df):
    """
    Sets a new index for the DataFrame based on its first column,
    assigns a label to the index, and removes the original column.

    Parameters:
    df (DataFrame): The DataFrame to reindex.

    Returns:
    DataFrame: The DataFrame with a custom index applied.
    """
    index = df.iloc[:, 0]
    index_copy = index.copy()
    index_copy.iloc[0] = "Name"
    df = df.set_index(index_copy)
    df.index.name = "Name"
    df = df.drop("Unnamed: 0", axis=1)
    return df

def rename_columns(df):
    """
    Updates the column names in the DataFrame using values from
    the row labeled 'Name' and removes that row afterward.

    Parameters:
    df (DataFrame): The DataFrame to update column names.

    Returns:
    DataFrame: The DataFrame with new column names.
    """
    column_name = df.loc["Name"]
    column_name_copy = column_name.copy()
    column_name_copy = column_name_copy.astype(int)
    df = df.rename(columns=column_name_copy)
    df = df.drop("Name", axis=0)
    return df

def get_consumption_data(oil_data, countries, year):
    """
    Retrieves oil consumption data for specific countries and a selected year.

    Parameters:
    - oil_data (DataFrame): DataFrame containing oil consumption data with
       country names as indices and years as columns.
    - countries (list): List of country names to extract data for.
    - year (int): The year for which to retrieve oil consumption data.

    Returns:
    - dict: Dictionary where each key is a country name and each value is
       the oil consumption for the specified year.
    """
    # Verify the specified year exists in the DataFrame columns
    if year not in oil_data.columns:
        raise ValueError(f"Year {year} not found in the DataFrame.")

    # Attempt to filter data for the specified countries and year
    try:
        consumption_data = oil_data.loc[countries, year]
    except KeyError as e:
        raise ValueError(f"One or more countries in {countries} are not found in the DataFrame.") from e

    # Convert the selected data to a dictionary format
    return consumption_data.to_dict()


def generate_pie_chart(data, year):
    """
    Generates a pie chart for oil consumption data for specified countries in a given year.

    Parameters:
    - data (dict): Dictionary with countries as keys and consumption values as values.
    - year (int): The year of the data to display in the chart title.
    """
    # Extract country labels and consumption values
    labels = list(data.keys())
    sizes = list(data.values())

    # Explode the first slice (India) for emphasis
    explode = (0.1, 0, 0, 0, 0)

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90
    )

    # Ensure the pie chart is circular
    ax.axis('equal')

    # Add title and display the chart
    plt.title(f'Oil Consumption by Country in {year}')
    plt.show()

# Apply the cleaning, indexing, and renaming functions to the oil_consumption DataFrame
oil_consumption = clean_dataframe(oil_consumption)
oil_consumption = set_custom_index(oil_consumption)
oil_consumption = rename_columns(oil_consumption)

# Display information about the cleaned and updated DataFrame, including data types and memory usage
print(oil_consumption.info())

# Display summary statistics for the numerical columns in the DataFrame
print(oil_consumption.describe())

# Display the correlation matrix for the numerical columns in the DataFrame
print(oil_consumption.corr())

# Define the list of countries for which to retrieve oil consumption data
countries = ['India', 'China', 'Russia', 'Brazil', 'South Africa']

# Get oil consumption data for the specified countries in 1990
data_1990 = get_consumption_data(oil_consumption, countries, 1990)

# Get oil consumption data for the specified countries in 2020
data_2020 = get_consumption_data(oil_consumption, countries, 2020)

# Generate pie charts for oil consumption in 1990 and 2020
generate_pie_chart(data_1990, 1990)
generate_pie_chart(data_2020, 2020)

# Create a new figure with specified size and resolution
plt.figure(figsize=(12, 6), dpi=310)

# Plot oil consumption data for different countries over time
# India: blue solid line
plt.plot(
    oil_consumption.columns,
    oil_consumption.loc["India"],
    color="blue",
    label="India"
)

# China: black dashed line
plt.plot(
    oil_consumption.columns,
    oil_consumption.loc["China"],
    color="black",
    linestyle="--",
    label="China"
)

# Russia: red dotted line
plt.plot(
    oil_consumption.columns,
    oil_consumption.loc["Russia"],
    color="red",
    linestyle=":",
    label="Russia"
)

# Brazil: yellow dash-dot line
plt.plot(
    oil_consumption.columns,
    oil_consumption.loc["Brazil"],
    color="yellow",
    linestyle="-.",
    label="Brazil"
)

# South Africa: green dash-dot line
plt.plot(
    oil_consumption.columns,
    oil_consumption.loc["South Africa"],
    color="green",
    linestyle="-.",
    label="South Africa"
)

# Add labels and title
plt.xlabel("Time Interval (1990-2020)")
plt.ylabel("Oil Consumption (Mt)")
plt.title("Total Oil Consumption Over Time")

# Customize the plot
plt.xlim(1989, 2021)

# Add a legend to the graph
plt.legend()

# Save the graph as a PNG file and display it
plt.savefig("oil_consumption_line.png", dpi=310)
plt.show()

import seaborn as sns

# Define the countries for which to visualize oil consumption data
countries = ['India', 'China', 'Russia', 'Brazil', 'South Africa']

# Extract and transpose data for the selected countries for plotting
data_for_violin = oil_consumption.loc[countries].transpose()

# Create a violin plot to visualize oil consumption distribution over time for each country
plt.figure(figsize=(10, 6))
sns.violinplot(data=data_for_violin)
plt.title('Oil Consumption Violin Plot ')
plt.xlabel('Countries')
plt.ylabel('Oil Consumption (Mt)')

# Save and display the plot
plt.savefig("oil_consumption_violin_plot.png", dpi=310)
plt.show()

