#!/usr/bin/env python3
import argparse
from pathlib import Path

def convert_fcsv_to_tag(fcsv_file, tag_file):
    try:
        with open(fcsv_file, 'r') as f:
            tag_lines = []
            tag_lines.append("MNI Tag Point File\nVolumes = 1;\n\nPoints =")

            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split(',')
                if len(parts) < 14:
                    continue
                
                x, y, z = map(float, parts[1:4])
                label = parts[11].split('-')[-1].strip()

                tag_line = f" {x} {y} {z} 1 1 {label} \"Marker\""
                tag_lines.append(tag_line)
            tag_lines[-1] += ';'

        with open(tag_file, 'w') as f:
            f.write("\n".join(tag_lines) + "\n")

        print(f"Conversion complete. {len(tag_lines) - 2} points written to {tag_file}")

    except FileNotFoundError:
        print(f"Error: File '{fcsv_file}' not found.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def batch_convert_fcsv_to_tag(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    if not output_path.is_dir():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return

    fcsv_files = input_path.glob("*.fcsv")
    for fcsv_file in fcsv_files:
        output_file = output_path / (fcsv_file.stem + ".tag")
        convert_fcsv_to_tag(fcsv_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .fcsv files to .tag files.")
    parser.add_argument("input_dir", help="Input directory containing .fcsv files.")
    parser.add_argument("output_dir", help="Output directory where .tag files will be saved.")
    args = parser.parse_args()

    batch_convert_fcsv_to_tag(args.input_dir, args.output_dir)
