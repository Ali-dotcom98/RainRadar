import pandas as pd

# Load the CSV file
file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\train_data.csv'

# Load the weather data
weather_data = pd.read_csv(file_path)

# Display the entire dataset
print(weather_data)

# Display the first 5 rows
print(weather_data.head())  # Shows the first 5 rows

# Display specific rows and columns (replace 'Column1', 'Column2' with actual column names)
print(weather_data[['Column1', 'Column2']].head())  # Replace Column1, Column2 with actual column names
