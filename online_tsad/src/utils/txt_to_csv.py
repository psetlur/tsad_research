import csv
import re

def parse_line(line):
    data = {}

    try:
        # Extract iter
        iter_match = re.search(r'iter: (\d+)', line)
        if iter_match:
            data['iter'] = int(iter_match.group(1))

        # Extract level and length
        next_point_match = re.search(r'next_point\.level\.length: ([^,]+)', line)
        if next_point_match:
            level_length_str = next_point_match.group(1)
            parts = level_length_str.split('.')

            if len(parts) >= 4:
                # Combine parts to get level and length
                data['level'] = float(parts[0] + '.' + parts[1])
                data['length'] = float(parts[2] + '.' + parts[3])

        # Extract wd
        wd_match = re.search(r'wd: ([\d.]+)', line)
        if wd_match:
            data['wd'] = float(wd_match.group(1))

        # Extract f1-score (rename to f1 in the output)
        f1_match = re.search(r'f1-score: ([\d.]+)', line)
        if f1_match:
            data['f1'] = float(f1_match.group(1))

    except Exception as e:
        print(f"Error parsing line: {line}")
        print(f"Error details: {e}")

    return data

def convert_txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        # Set up CSV writer
        fieldnames = ['iter', 'level', 'length', 'wd', 'f1']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # Process each line in the input file
        for line in f_in:
            line = line.strip()
            if line:  # Skip empty lines
                data = parse_line(line)
                if all(key in data for key in fieldnames):  # Make sure all fields are present
                    writer.writerow(data)
                else:
                    print(f"Skipping line due to missing fields: {line}")

# Usage
input_file = 'logs/training/inject_spike/bayes_spike_0.5_logs.txt'  # Replace with your input file path
output_file = 'logs/training/inject_spike/bayes_spike_0.5_logs.csv'  # Replace with your desired output file path
convert_txt_to_csv(input_file, output_file)
print(f'Conversion completed. CSV file saved to {output_file}')