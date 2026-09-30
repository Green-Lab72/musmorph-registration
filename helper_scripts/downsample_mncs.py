#!/usr/bin/env python3

import subprocess
import os
import glob
import math

def downsample_minc_file(input_file, output_file, scale_factor):
    # Extract image information using mincinfo
    def get_minc_info(attribute, which_attribute, file):
        result = subprocess.run(['mincinfo', attribute, which_attribute, file], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    
    # Get original voxel steps
    x_step = float(get_minc_info('-attvalue', 'xspace:step', input_file))
    y_step = float(get_minc_info('-attvalue', 'yspace:step', input_file))
    z_step = float(get_minc_info('-attvalue', 'zspace:step', input_file))
    
    # Get original voxel counts
    x_length = int(get_minc_info('-dimlength', 'xspace', input_file))
    y_length = int(get_minc_info('-dimlength', 'yspace', input_file))
    z_length = int(get_minc_info('-dimlength', 'zspace', input_file))
    
    # Get start positions
    x_start = float(get_minc_info('-attvalue', 'xspace:start', input_file))
    y_start = float(get_minc_info('-attvalue', 'yspace:start', input_file))
    z_start = float(get_minc_info('-attvalue', 'zspace:start', input_file))
    
    # Calculate new dimensions and steps
    new_x_length = math.ceil(x_length / scale_factor)
    new_y_length = math.ceil(y_length / scale_factor) 
    new_z_length = math.ceil(z_length / scale_factor)
    
    new_x_step = x_step * scale_factor
    new_y_step = y_step * scale_factor  
    new_z_step = z_step * scale_factor
    
    print(f"Original dimensions: {z_length}x{y_length}x{x_length}")
    print(f"New dimensions: {new_z_length}x{new_y_length}x{new_x_length}")
    print(f"Original steps: {z_step}, {y_step}, {x_step}")
    print(f"New steps: {new_z_step}, {new_y_step}, {new_x_step}")
    
    # Build the mincresample command
    command = [
        'mincresample',
        '-xstart', str(x_start), '-xstep', str(new_x_step), '-xnelements', str(new_x_length),
        '-ystart', str(y_start), '-ystep', str(new_y_step), '-ynelements', str(new_y_length),
        '-zstart', str(z_start), '-zstep', str(new_z_step), '-znelements', str(new_z_length),
        input_file, output_file
    ]
    
    # Execute the command
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)

def process_directory(input_dir, output_dir, scale_factor):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all MINC files in the input directory
    minc_files = glob.glob(os.path.join(input_dir, '*.mnc'))
    print(f"Found {len(minc_files)} MINC files")
    
    for input_file in minc_files:
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, file_name)
        
        print(f'Processing {input_file}...')
        downsample_minc_file(input_file, output_file, scale_factor)
        print(f'Downsampled file saved to {output_file}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Downsample MINC files in a directory.')
    parser.add_argument('input_dir', help='Directory containing input MINC files')
    parser.add_argument('output_dir', help='Directory to save downsampled MINC files')
    parser.add_argument('scale_factor', type=float, help='Scaling factor for downsampling (e.g., 3 for reducing size by factor of 3)')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.scale_factor)
