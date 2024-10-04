import re
import numpy as np

def compute_statistics(file_path, regex_pattern):
    values = []

    # Compile the regular expression pattern
    pattern = re.compile(regex_pattern)

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into string and float parts
            try:
                string_part, float_part = line.strip().split(':')
                float_value = float(float_part)

                # Check if the string part matches the regex pattern
                if pattern.match(string_part):
                    values.append(float_value)
            except ValueError:
                print(f"Skipping line due to ValueError: {line.strip()}")

    # Calculate statistics if values were found
    if values:
        mean_value = np.mean(values)
        std_dev_value = np.std(values)
        max_value = np.max(values)
        min_value = np.min(values)

        print (f"Mean: {mean_value} + std dev: {std_dev_value}")
        print (f"min: {min_value} /  max: {max_value}")

def main ():    
    file = "measurements.log"
    #pattern = r'set_vect_goal_\d+'
    pattern = r'find_best_cost_waypoint_with_heading_\d+'
    
    
    compute_statistics(file, pattern)


if __name__ == "__main__":
    main()