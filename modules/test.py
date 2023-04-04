import pandas as pd

# Create example dataframe
df = pd.DataFrame({'key': ['A', 'B', 'C'],
                   'value': [1, 2, 3]})

# Define the key and value to check and add/update
key_to_check = 'B'
value_to_add = 3

# Check if the key is in the dataframe
if key_to_check in df['key'].values:
    # If the key is in the dataframe, update its value to the running average
    key_rows = df.loc[df['key'] == key_to_check]
    df.loc[df['key'] == key_to_check, 'value'] = key_rows['value'].expanding().mean()

else:
    # If the key is not in the dataframe, add a new row with the key and value
    new_row = pd.DataFrame({'key': [key_to_check], 'value': [value_to_add]})
    df = pd.concat([df, new_row], ignore_index=True)

print(df)