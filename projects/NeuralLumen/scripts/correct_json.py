import os.path
import re

# File path
filename = 'transforms_val.json'
input_file_path = os.path.join('/ghome/yyang/PycharmProjects/neuralangelo/dataset_syn/Furball_intrinsic/wrong/', filename)
output_file_path = os.path.join('/ghome/yyang/PycharmProjects/neuralangelo/dataset_syn/Furball_intrinsic/', filename)

# Regular expression for matching
pattern = r'/r_(\d+)'

# Function to format the matched object
def replacement_func(match):
    # Extracting the number from the match
    num = match.group(1)
    # Formatting the number with leading zeros
    formatted_num = str(num).zfill(3)
    # Returning the replacement string
    return f'/{formatted_num}_'

# Reading the file content
with open(input_file_path, 'r', encoding='utf-8') as file:
    file_contents = file.read()

# Finding and replacing with the function
new_contents = re.sub(pattern, replacement_func, file_contents)

# Writing the modified content back to the file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_contents)

print('Replacement complete.')
