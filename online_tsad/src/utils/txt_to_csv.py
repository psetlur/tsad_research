import os
import csv
import re
import ast

def parse_txt_to_csv(input_file, output_file):
    try:
        # Read the text file
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Extract wd list
        wd_start = content.find('wd: [') + 5
        wd_end = content.find(']', wd_start)
        wd_str = content[wd_start:wd_end]
        wd_list = [float(x.strip()) for x in wd_str.split(',')]
        
        # Extract f1score list
        f1score_start = content.find('f1score: [') + 10
        f1score_end = content.find(']', f1score_start)
        f1score_str = content[f1score_start:f1score_end]
        f1score_list = [float(x.strip()) for x in f1score_str.split(',')]
        
        # Extract points data more robustly
        points_start = content.find('points: [') + 9
        points_end = content.rfind('}]')
        if points_end == -1:  # If there's no closing bracket
            points_end = len(content)
        else:
            points_end += 1  # Include the closing brace
        
        points_str = content[points_start:points_end]
        
        # Use regex to find all dictionaries
        dict_pattern = r'\{[^{}]*\}'
        dict_matches = re.findall(dict_pattern, points_str)
        
        points_list = []
        for match in dict_matches:
            try:
                # Replace single quotes with double quotes for proper JSON parsing
                json_str = match.replace("'", '"')
                # Use ast.literal_eval which is safer for parsing Python literals
                point_dict = ast.literal_eval(match)
                
                # Ensure all required keys are present
                points_dict = {
                    'platform_level': point_dict.get('platform_level', ''),
                    'platform_length': point_dict.get('platform_length', ''),
                    'mean_level': point_dict.get('mean_level', ''),
                    'mean_length': point_dict.get('mean_length', ''),
                    'spike_level': point_dict.get('spike_level', ''),
                    'spike_p': point_dict.get('spike_p', '')
                }
                points_list.append(points_dict)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing dictionary: {match}")
                print(f"Error details: {e}")
        
        # Determine the number of iterations dynamically
        num_iterations = max(len(wd_list), len(f1score_list), len(points_list))
        
        # Create CSV rows
        rows = []
        
        # Use the dynamic number of iterations
        for i in range(num_iterations):
            row = {
                'iter': i,
                'wd': wd_list[i] if i < len(wd_list) else '',
                'f1score': f1score_list[i] if i < len(f1score_list) else ''
            }
            
            # Add points data if available
            if i < len(points_list):
                row.update({
                    'platform_level': points_list[i]['platform_level'],
                    'platform_length': points_list[i]['platform_length'],
                    'mean_level': points_list[i]['mean_level'],
                    'mean_length': points_list[i]['mean_length'],
                    'spike_level': points_list[i]['spike_level'],
                    'spike_p': points_list[i]['spike_p']
                })
            else:
                row.update({
                    'platform_level': '',
                    'platform_length': '',
                    'mean_level': '',
                    'mean_length': '',
                    'spike_level': '',
                    'spike_p': ''
                })
            
            rows.append(row)
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['iter', 'wd', 'f1score', 'platform_level', 'platform_length', 
                         'mean_level', 'mean_length', 'spike_level', 'spike_p']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Total iterations processed: {num_iterations}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "logs/training/hpo_three/bayes_wd_f1score_all_0.5_0.3.txt"
    output_file = "logs/csv/hpo_three/bayes_wd_f1score_all_0.5_0.3.csv"
    os.makedirs("logs/csv/hpo_three/", exist_ok=True)
    parse_txt_to_csv(input_file, output_file)