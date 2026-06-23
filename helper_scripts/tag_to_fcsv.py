#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

def parse_tag_points(content):
    """Parse points from tag file content."""
    # Find the Points section
    points_match = re.search(r'Points\s*=\s*([^;]+);', content, re.DOTALL)
    if not points_match:
        raise ValueError("No Points section found in tag file")
    
    points_text = points_match.group(1).strip()
    points = []
    
    # Parse each point line
    for line in points_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Extract values using regex to handle scientific notation
        values = re.findall(r'[-+]?(?:\d*\.*\d+(?:[eE][-+]?\d+)?)|"[^"]*"', line)
        if len(values) < 7:
            continue
            
        x, y, z = map(float, values[0:3])
        label = values[6].strip('"')
        points.append((x, y, z, label))
    
    return points

def convert_tag_to_fcsv(tag_file, fcsv_file):
    """Convert a .tag file to .fcsv format."""
    try:
        with open(tag_file, 'r') as f:
            content = f.read()
        
        points = parse_tag_points(content)
        
        # Write FCSV file
        with open(fcsv_file, 'w') as f:
            # Write header
            f.write("# Markups fiducial file version = 5.6\n")
            f.write("# CoordinateSystem = LPS\n")
            f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
            
            # Write each point
            for i, (x, y, z, label) in enumerate(points, 1):
                
                lps_x = x
                lps_y = y
                lps_z = z
                
                # Format point line with all required fields
                point_line = (f"{i},{lps_x},{lps_y},{lps_z},"
                            f"0,0,0,1,1,1,0,F-{label},,,2,0\n")
                f.write(point_line)
        
        print(f"Conversion complete. {len(points)} points written to {fcsv_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{tag_file}' not found.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def batch_convert_tag_to_fcsv(input_dir, output_dir):
    """Batch convert all .tag files in input_dir to .fcsv files in output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    if not output_path.is_dir():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return
    
    tag_files = input_path.glob("*.tag")
    for tag_file in tag_files:
        output_file = output_path / (tag_file.stem + ".fcsv")
        convert_tag_to_fcsv(tag_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tag files to .fcsv files.")
    parser.add_argument("input_dir", help="Input directory containing .tag files.")
    parser.add_argument("output_dir", help="Output directory where .fcsv files will be saved.")
    args = parser.parse_args()
    
    batch_convert_tag_to_fcsv(args.input_dir, args.output_dir)
