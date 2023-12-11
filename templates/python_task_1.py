import pandas as pd
import numpy as np
def generate_car_matrix(csv_url):
    # Use the raw content URL of the CSV file
    df = pd.read_csv(csv_url)

    # Create a pivot table with 'id_1' as index, 'id_2' as columns, and 'car' as values
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0 and set diagonal values to 0
    car_matrix = car_matrix.fillna(0).astype(int)
    car_matrix.values[[range(len(car_matrix))]*2] = 0

    return car_matrix

# Replace 'https://raw.githubusercontent.com/username/repository/branch/path/to/your/file.csv'
# with the actual raw content URL of your CSV file
csv_url = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-1.csv'

# Call the function to generate the car matrix
result_matrix = generate_car_matrix(csv_url)

# Display the result
print("Generated Car Matrix:")
print(result_matrix)



import pandas as pd
import numpy as np


def get_type_count(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Add a new column 'car_type' based on the conditions specified
    conditions = [
        (df['car'] <= 15),
        ((df['car'] > 15) & (df['car'] <= 25)),
        (df['car'] > 25)
    ]

    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices, default='Unknown'))

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = {key: type_counts[key] for key in sorted(type_counts)}

    return sorted_type_counts


file_path = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-1.csv'
result = get_type_count(file_path)
print(result)



def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    import pandas as pd

def get_bus_indexes(dataframe):
    # Calculate the mean value of the 'bus' column
    bus_mean = dataframe['bus'].mean()

    # Identify indices where the 'bus' values are greater than twice the mean
    bus_indexes = dataframe[dataframe['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage:
if __name__ == "__main__":

    file_path = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-1.csv'

    # Read the CSV file into a DataFrame
    dataset = pd.read_csv(file_path)

    # Call the function to get the desired indices
    result = get_bus_indexes(dataset)

    # Display the result
    print("Indices where 'bus' values are greater than twice the mean:", result)


    return list()


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    import pandas as pd

def filter_routes(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter routes based on the average of the truck column
    filtered_routes = df.groupby('route')['truck'].mean().reset_index()
    filtered_routes = filtered_routes[filtered_routes['truck'] > 7]

    # Sort the routes
    sorted_routes = filtered_routes.sort_values(by='route')['route'].tolist()

    return sorted_routes

# Example usage:
csv_path = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-1.csv'
result = filter_routes(csv_path)
print(result)


    return list()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    import pandas as pd

def multiply_matrix(result_matrix):
    # Create a deep copy of the input DataFrame to avoid modifying the original
    modified_dataframe = result_matrix.copy()

    # Apply the specified logics to modify the values in the DataFrame
    modified_dataframe = modified_dataframe.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_dataframe = modified_dataframe.round(1)

    return modified_dataframe

# Assuming 'result_matrix' is the DataFrame generated from the previous code
# You may need to adjust the variable name if it's different
result_matrix = generate_car_matrix('https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-1.csv')

# Call the function to get the modified DataFrame
modified_df = multiply_matrix(result_matrix)

# Print the modified DataFrame
print("Modified Car Matrix:")
print(modified_df)


    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    import pandas as pd

def verify_time_completeness(df):
    # Convert timestamp columns to datetime format with explicit format
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')

    # Print the first few rows of the DataFrame to check if the data is loaded correctly
    print(df.head())

    # Print the column names of the DataFrame
    print("Column names:", df.columns)

    try:
        # Check if 'id' and 'id_2' columns exist in your DataFrame
        result = (df['end_timestamp'] - df['start_timestamp']) == pd.Timedelta(days=1)
        result_series = result.groupby(['id', 'id_2']).all()
        return result_series
    except KeyError as e:
        # Print an error message if KeyError occurs
        print(f"KeyError: {e}")
        return None

# Read the dataset directly from the URL
url = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-2.csv'
df = pd.read_csv(url)

# Call the function and get the result
result_series = verify_time_completeness(df)

# Print the result if it's not None
if result_series is not None:
    print(result_series)


    return pd.Series()
