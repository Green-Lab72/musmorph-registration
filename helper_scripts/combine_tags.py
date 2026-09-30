#!/usr/bin/env python3
import os
import glob
import argparse

def combo_tag(base_spec, spec_name, mnc_dir):
    # More flexible file matching - look for any tag file starting with the base_spec
    base_spec_pattern = os.path.join(mnc_dir, f"{base_spec}*.tag")
    spec_name_pattern = os.path.join(mnc_dir, f"{spec_name}*.tag")
    
    base_spec_files = glob.glob(base_spec_pattern)
    spec_name_files = glob.glob(spec_name_pattern)
    
    if base_spec_files and spec_name_files:
        base_spec_file = base_spec_files[0]  # Take the first matching file
        spec_name_file = spec_name_files[0]
    else:
        if not base_spec_files:
            raise FileNotFoundError(f"BaseSpec file starting with {base_spec} does not exist.")
        if not spec_name_files:
            raise FileNotFoundError(f"SpecName file starting with {spec_name} does not exist.")

    # Rest of the function remains the same
    with open(base_spec_file, 'r') as base_file, open(spec_name_file, 'r') as spec_file:
        base_data = [line.split() for line in base_file.readlines()[4:]]
        spec_data = [line.split() for line in spec_file.readlines()[4:]]

    output_file = f"Tag_{base_spec}_to_{spec_name}.tag"
    
    with open(output_file, 'w') as out_file:
        out_file.write("MNI Tag Point File\n")
        out_file.write("Volumes = 2;\n")
        out_file.write(f"%VIO_Volume: {base_spec}.mnc\n")
        out_file.write(f"%VIO_Volume: {spec_name}.mnc\n")
        out_file.write("\nPoints = \n")
        
        for base_lm, spec_lm in zip(base_data, spec_data):
            if len(base_lm) >= 3 and len(spec_lm) >= 3:
                base_coords = " ".join(base_lm[:3])
                spec_coords = " ".join(spec_lm[:3])
                base_label = " ".join(base_lm[3:]) if len(base_lm) > 3 else ""
                spec_label = " ".join(spec_lm[3:]) if len(spec_lm) > 3 else ""
                out_file.write(f"{base_coords} {spec_coords} {base_label}\n")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Combine .tag files")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Directory containing .tag files")
    parser.add_argument("-s", "--spec_list", type=str, required=True,
                        help="Path to spec_list.txt file")
    parser.add_argument("-r", "--reference", type=str, required=True,
                        help="Reference specimen name")
    args = parser.parse_args()
   
    # Assign arguments to variables
    mnc_dir = args.directory
    spec_list_file = args.spec_list
    base_spec = args.reference
   
    # Read in the specimen list
    with open(spec_list_file, 'r') as f:
        spec_list = [line.strip() for line in f.readlines()]
   
    # Set working directory to the directory containing .tag files
    # os.chdir(mnc_dir)
   
    # Combine .tag files
    for spec_name in spec_list:
        try:
            combo_tag(base_spec, spec_name, mnc_dir)
            print(f"Combined {base_spec} and {spec_name} successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
