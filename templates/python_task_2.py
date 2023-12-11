import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    import pandas as pd

def calculate_distance_matrix(csv_url):
    # Read the dataset
    df = pd.read_csv(csv_url)

    # Check the actual column names in the dataset
    print("Column Names:", df.columns)

    # Assuming your column names are case-sensitive, adjust them accordingly
    start_col = 'Start'
    end_col = 'End'
    distance_col = 'Distance'

    try:
        # Use the original column names if they exist
        df[start_col]
        df[end_col]
        df[distance_col]
    except KeyError:
        # Use alternative column names if the original ones are not found
        start_col = 'id_start' if 'id_start' in df.columns else 'ID_START'
        end_col = 'id_end' if 'id_end' in df.columns else 'ID_END'
        distance_col = 'distance' if 'distance' in df.columns else 'DISTANCE'

    # Create a dictionary to store known distances between toll locations
    distance_dict = {}

    # Populate the distance dictionary with known distances
    for index, row in df.iterrows():
        start_location = row[start_col]
        end_location = row[end_col]
        distance = row[distance_col]

        # Add bidirectional distances to the dictionary
        distance_dict[(start_location, end_location)] = distance
        distance_dict[(end_location, start_location)] = distance

    # Create a list of unique toll locations
    toll_locations = sorted(list(set(df[start_col].unique()) | set(df[end_col].unique())))

    # Initialize an empty distance matrix
    distance_matrix = pd.DataFrame(0, index=toll_locations, columns=toll_locations)

    # Populate the distance matrix with cumulative distances along known routes
    for i in toll_locations:
        for j in toll_locations:
            if i != j:
                # Check if the distance is known in the dictionary
                if (i, j) in distance_dict:
                    distance_matrix.at[i, j] = distance_dict[(i, j)]
                else:
                    # Calculate cumulative distance along known routes
                    known_routes = [(i, k) for k in toll_locations if (i, k) in distance_dict]
                    distances = [distance_dict[(start, end)] + distance_matrix.at[end, j] for start, end in known_routes]
                    if distances:
                        distance_matrix.at[i, j] = min(distances)

    return distance_matrix


csv_url = 'https://raw.githubusercontent.com/mapup/MapUp-Data-Assessment-F/main/datasets/dataset-3.csv'
resulting_matrix = calculate_distance_matrix(csv_url)
print(resulting_matrix)


    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    import pandas as pd

def unroll_distance_matrix(resulting_matrix):
    # Create an empty DataFrame to store the unrolled data
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through each row in the resulting_matrix DataFrame
    for index, row in resulting_matrix.iterrows():
        # Check if the required columns exist in the row
        if 'id_start' in row.index and 'id_end' in row.index and 'distance' in row.index:
            id_start = row['id_start']
            id_end = row['id_end']
            distance = row['distance']
        else:
            print(f"Missing columns in row {index}: {row}. Skipping this row.")
            continue

        # Iterate through all other rows to add combinations (excluding the same id_start to id_end)
        for _, other_row in resulting_matrix.iterrows():
            # Check if the required columns exist in the other_row
            if 'id_start' in other_row.index and 'id_end' in other_row.index:
                other_id_start = other_row['id_start']
                other_id_end = other_row['id_end']
            else:
                print(f"Missing columns in other_row: {other_row}. Skipping this row.")
                continue

            # Check if the combination is not the same id_start to id_end
            if id_start != other_id_start or id_end != other_id_end:
                # Add the combination to the unrolled DataFrame
                unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': other_id_end, 'distance': distance}, ignore_index=True)

    return unrolled_df

# Example usage:
try:
    unrolled_df = unroll_distance_matrix(resulting_matrix).copy()
except Exception as e:
    print(f"An error occurred: {e}")


    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    import pandas as pd


def find_ids_within_ten_percentage_threshold(input_df, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = input_df[input_df['id_start'] == reference_value]['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold = 0.1 * reference_avg_distance

    # Filter rows with id_start values within the threshold range
    filtered_ids = input_df[(input_df['distance'] >= (reference_avg_distance - threshold)) & (input_df['distance'] <= (reference_avg_distance + threshold))]['id_start']

    # Return a sorted list of unique values
    return sorted(filtered_ids.unique())

# Example usage:
# Assuming resulting_matrix is the DataFrame from Question 2
resulting_matrix = pd.DataFrame({'id_start': [1, 2, 3], 'id_end': [4, 5, 6], 'distance': [10, 20, 30]})
unrolled_df = unroll_distance_matrix(resulting_matrix).copy()
#  Assuming result_df is the DataFrame from Question 2
result_df = pd.DataFrame({'id_start': [1, 2, 3], 'distance': [10, 20, 30]})

# Choose a reference value from id_start column
reference_value = 2

# Find ids within ten percentage threshold of the reference value
ids_within_threshold = find_ids_within_ten_percentage_threshold(result_df, reference_value)

# Display the resulting DataFrames and the list of ids within the threshold
print("\nDataFrame after unrolling distances:")
print(unrolled_df)

print("\nDataFrame for Question 2:")
print(result_df)
    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
import pandas as pd

def calculate_toll_rate(input_df):
    # Copy the input DataFrame to avoid modifying the original DataFrame
    result_df = input_df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        result_df[vehicle_type] = result_df['distance'] * rate_coefficient

    return result_df


# Assuming resulting_matrix is the DataFrame from Question 1
resulting_matrix = pd.DataFrame({'id_start': [1, 2, 3], 'id_end': [4, 5, 6], 'distance': [10, 20, 30]})
unrolled_df = unroll_distance_matrix(resulting_matrix).copy()

# Display the resulting DataFrame after unrolling distances
print("Unrolled DataFrame:")
# print(unrolled_df)

# Calculate toll rates for each vehicle type
result_df_with_toll = calculate_toll_rate(unrolled_df).copy()

# Display the resulting DataFrame with toll rates
print("\nDataFrame with Toll Rates:")
print(result_df_with_toll)

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
