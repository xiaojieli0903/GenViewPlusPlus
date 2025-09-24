import argparse
import numpy as np
import os
import csv

def map_fg_ratio_to_range(value, ranges, mapping_values):
    """
    Maps a fg_ratio value to a specific range and returns the corresponding mapping value.
    """
    index = np.digitize(value, ranges, right=True) - 1
    index = min(max(index, 0), len(mapping_values) - 1)  # Ensure index is valid
    return mapping_values[index]

def distribute_fg_ratios(input_file, output_dir, ranges, mapping_values, append=False):
    """
    Reads fg_ratio values from a CSV file, maps them to specified ranges,
    and writes ONE CSV file with image, prompt, noise_level.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'noise_level.csv')

    rows_out = []

    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if not {'image', 'ratio', 'prompt'}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain columns: image, ratio, prompt")

        for row in reader:
            try:
                fg_ratio = float(row['ratio'])
                mapped_value = map_fg_ratio_to_range(fg_ratio, ranges, mapping_values)
                rows_out.append([row['image'], row['prompt'], mapped_value])
            except ValueError:
                print(f"Skipping invalid ratio: {row['ratio']}")

    # 保存结果
    write_header = True
    if append and os.path.exists(output_file_path):
        write_header = False  # 已存在文件，追加模式时不写 header

    with open(output_file_path, 'a' if append else 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        if write_header:
            writer.writerow(['image', 'prompt', 'noise_level'])
        writer.writerows(rows_out)

    mode = "appended to" if append else "saved as"
    print(f"File {mode}: {output_file_path}")
    print(f"Total records written: {len(rows_out)}")

def main():
    parser = argparse.ArgumentParser(description='Distribute FG Ratios into ONE CSV (with noise_level)')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the fg_ratios.csv file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the noise_level.csv file')
    parser.add_argument('--append', action='store_true', help='Append to existing noise_level.csv instead of overwriting')
    args = parser.parse_args()

    # Validate input file existence
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Define ranges and corresponding mapping values
    ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mapping_values = [0, 100, 200, 300, 400, 500]

    # Run distribution
    distribute_fg_ratios(args.input_file, args.output_dir, ranges, mapping_values, append=args.append)

if __name__ == '__main__':
    main()
