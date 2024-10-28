import os
import re

# Set the path where the log files are located and the output file name
log_dir = "./logs_kibo"  # Update with the actual path to your log files
output_file = "error_file_list.txt"

# Error messages to look for
errors_to_find = [
    "numpy.linalg.LinAlgError: Matrix is not positive definite",
    "ValueError: All-NaN slice encountered",
]

# Regex pattern to extract hpx_start, hpx_end, and job ID from the filename
filename_pattern = re.compile(r"error_kibo_(\d+)-(\d+)_\d+_\d+\.log")

# Initialize lists to store problematic hpx ranges
error_ranges = []

# Iterate through files in the log directory
for filename in os.listdir(log_dir):
    # Check if the file matches the expected naming pattern
    match = filename_pattern.match(filename)
    if match:
        hpx_start, hpx_end = match.groups()

        # Open the file and search for the error messages
        with open(os.path.join(log_dir, filename), "r") as file:
            file_content = file.read()
            # Check for each error message
            if any(error in file_content for error in errors_to_find):
                error_ranges.append((hpx_start, hpx_end))

# Write the identified ranges to the output file
with open(output_file, "w") as f_out:
    for hpx_start, hpx_end in error_ranges:
        f_out.write(f"{hpx_start}-{hpx_end}\n")

print(f"Error ranges saved to {output_file}")
