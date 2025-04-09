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
        if points_end == -1:
            points_end = len(content)
        else:
            points_end += 1
        
        points_str = content[points_start:points_end]
        
        dict_pattern = r'\{[^{}]*\}'
        dict_matches = re.findall(dict_pattern, points_str)
        
        points_list = []
        for match in dict_matches:
            try:
                point_dict = ast.literal_eval(match)
                points_dict = {
                    'platform_level': point_dict.get('platform_level', ''),
                    'platform_length': point_dict.get('platform_length', ''),
                    'mean_level': point_dict.get('mean_level', ''),
                    'mean_length': point_dict.get('mean_length', ''),
                    'spike_level': point_dict.get('spike_level', ''),
                    'spike_p': point_dict.get('spike_p', ''),
                    'amplitude_level': point_dict.get('amplitude_level', ''),
                    'amplitude_length': point_dict.get('amplitude_length', ''),
                    'trend_slope': point_dict.get('trend_slope', ''),
                    'trend_length': point_dict.get('trend_length', ''),
                    'variance_level': point_dict.get('variance_level', ''),
                    'variance_length': point_dict.get('variance_length', '')
                }
                points_list.append(points_dict)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing dictionary: {match}")
                print(f"Error details: {e}")
        
        num_iterations = max(len(wd_list), len(f1score_list), len(points_list))
        
        rows = []
        for i in range(num_iterations):
            row = {
                'iter': i,
                'wd': wd_list[i] if i < len(wd_list) else '',
                'f1score': f1score_list[i] if i < len(f1score_list) else ''
            }
            if i < len(points_list):
                row.update(points_list[i])
            else:
                row.update({
                    'platform_level': '',
                    'platform_length': '',
                    'mean_level': '',
                    'mean_length': '',
                    'spike_level': '',
                    'spike_p': '',
                    'amplitude_level': '',
                    'amplitude_length': '',
                    'trend_slope': '',
                    'trend_length': '',
                    'variance_level': '',
                    'variance_length': ''
                })
            rows.append(row)
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['iter', 'wd', 'f1score', 'platform_level', 'platform_length',
                          'mean_level', 'mean_length', 'spike_level', 'spike_p',
                          'amplitude_level', 'amplitude_length',
                          'trend_slope', 'trend_length',
                          'variance_level', 'variance_length']
            
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
    input_file = "logs/training/six_anomalies/bayes_wd_f1score_p_m_s.txt"
    output_file = "logs/csv/six_anomalies/bayes_wd_f1score_p_m_s.csv"
    os.makedirs("logs/csv/six_anomalies/", exist_ok=True)
    parse_txt_to_csv(input_file, output_file)
