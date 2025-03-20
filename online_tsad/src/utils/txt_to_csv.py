import csv
import re
import ast

def convert_txt_to_csv(input_file, output_file):
    # Initialize variables
    data_list = []
    current_data = None
    
    # Process file line by line
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Skip empty lines
                continue
            
            # Check if this is the start of a new iteration
            iter_match = re.search(r'iter: (\d+)', line)
            if iter_match:
                # Save previous iteration data if it exists
                if current_data is not None:
                    data_list.append(current_data)
                
                # Start a new iteration
                current_iter = int(iter_match.group(1))
                current_data = {'iter': current_iter}
                
                # Extract other values from this line
                wd_match = re.search(r'wd: ([\d.-]+)', line)
                if wd_match:
                    current_data['wd'] = float(wd_match.group(1))
                
                f1_match = re.search(r'f1-score: ([\d.-]+)', line)
                if f1_match:
                    current_data['f1-score'] = float(f1_match.group(1))
            
            # Check if this line contains next_point
            elif 'next_point:' in line and current_data is not None:
                next_point_match = re.search(r'next_point: ({[^}]+})', line)
                if next_point_match:
                    try:
                        next_point_str = next_point_match.group(1)
                        next_point = ast.literal_eval(next_point_str)
                        
                        current_data['platform_level'] = next_point.get('platform_level')
                        current_data['platform_length'] = next_point.get('platform_length')
                        current_data['mean_level'] = next_point.get('mean_level')
                        current_data['mean_length'] = next_point.get('mean_length')
                    except Exception as e:
                        print(f"Error parsing next_point: {next_point_str}")
                        print(f"Error details: {e}")
            
            # Skip lines with valid_point (as instructed)
            
        # Don't forget to add the last iteration
        if current_data is not None:
            data_list.append(current_data)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f_out:
        fieldnames = ['iter', 'wd', 'f1-score', 'platform_level', 'platform_length', 'mean_level', 'mean_length']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for data in data_list:
            if all(key in data for key in fieldnames):
                writer.writerow(data)
            else:
                missing = set(fieldnames) - set(data.keys())
                print(f"Skipping record for iter {data.get('iter', 'unknown')} due to missing fields: {missing}")

# Usage
input_file = 'logs/training/hpo_both/bayes_wd_f1score_both_0.5_0.3_logs.txt'
output_file = 'logs/training/hpo_both/bayes_wd_f1score_both_0.5_0.3_logs.csv'
convert_txt_to_csv(input_file, output_file)
print(f'Conversion completed. CSV file saved to {output_file}')