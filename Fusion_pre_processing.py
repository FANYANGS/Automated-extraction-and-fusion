# Code by Anbo Li, Fan Yang, Xinan Zhang, Xicai Pan, Xianli Xie
import pandas as pd
import re
import os
import openpyxl

# Define the directory path containing the Excel files to be processed
dir_path = r'C:\\'

# Define a list of keywords to identify specific elements within rows (elements without units)
keyword_list = ['ph', 'cn', 'tc', 'fc', 'temp', 'depth', 'eh']

def transpose(df):
    """
    Transpose the given DataFrame.

    Parameters:
    df (pd.DataFrame): The Pandas DataFrame to be transposed.

    Returns:
    pd.DataFrame: The transposed DataFrame.
    """
    df_transposed = df.transpose()
    return df_transposed

# Iterate over all files in the directory
for filename in os.listdir(dir_path):
    # Check if the file is an Excel file, filter out non-.xlsx files
    if not filename.endswith('.xlsx'):
        continue

    # Generate the full path of the file
    file_path = os.path.join(dir_path, filename)

    # Load the workbook of the Excel file
    book = openpyxl.load_workbook(file_path)

    # Create an empty dictionary to store data from each sheet
    sheets_data = {}

    # Iterate over all sheets in the workbook
    for sheet in book.sheetnames:
        # Read the content of the current sheet into a Pandas DataFrame, header=None means not using the first row as column names
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)

        # Initialize identifier to detect the row where the header is located
        header_row = None

        # Iterate over each row in the DataFrame to identify the header row
        for i, row in df.iterrows():
            # Use regular expressions to match specific units (such as /, ppm, ppb) and the keyword list to identify the header
            metal_count = sum([1 for cell in row if re.search(r'\(.*(/|ppm|ppb).*\)', str(cell)) or any(
                keyword in str(cell).lower() for keyword in keyword_list)])

            # If the row meets the header criteria, record the index of the row and break the loop
            if metal_count >= 2:
                header_row = i
                break

        # If no header is found, transpose the sheet and save it with a "_t" suffix
        if header_row is None:
            df = transpose(df)
            sheets_data[sheet + "_t"] = df  # Store the transposed data in the dictionary
        else:
            # If a header is found, retain the original data
            sheets_data[sheet] = df  # Store the original data in the dictionary

    # Write the processed data back to the original Excel file, retaining the sheet names
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Iterate over the stored data dictionary and write each sheet to the Excel file
        for sheet, df in sheets_data.items():
            # Write to Excel without retaining index and header
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)

