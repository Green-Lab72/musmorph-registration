#!/usr/bin/env python3
### RUN WITH DEFAULTS BEFORE DOWNSAMPLING FOR NOW ###
import os
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
import argparse
import uuid
from pathlib import Path

def resize_and_load_images(image_dir, resize_factor=1.0, quality=95):
    """Load and resize BMP files with consistent dimensions."""
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
    
    if not files:
        print(f"No BMP files found in {image_dir}")
        return None, None
    
    print(f"Found {len(files)} BMP files to process")
    print(f"Resize factor: {resize_factor}")
    
    img_stack = []
    valid_files = []
    std_shape = None
    original_size = None
    resized_size = None
    
    for i, filename in enumerate(files):
        try:
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            
            # Store original size from first image
            if original_size is None:
                original_size = img.size
                print(f"Original image size: {original_size[0]}x{original_size[1]}")
            
            # Check if current image matches expected original size
            if img.size != original_size:
                print(f"Warning: Skipping '{filename}' (size {img.size}, expected {original_size})")
                continue
            
            # Resize image if factor is not 1.0
            if resize_factor != 1.0:
                new_width = int(img.width * resize_factor)
                new_height = int(img.height * resize_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                if resized_size is None:
                    resized_size = (new_width, new_height)
                    print(f"Resized image size: {resized_size[0]}x{resized_size[1]}")
            else:
                resized_size = original_size
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Set expected shape from the first processed image
            if std_shape is None:
                std_shape = img_array.shape
                print(f"Standard array dimensions: {std_shape}")
            
            # Check dimensions and skip if inconsistent
            if img_array.shape != std_shape:
                print(f"Warning: Skipping '{filename}' after resize (dimensions {img_array.shape}, expected {std_shape})")
                continue
            
            img_stack.append(img_array)
            valid_files.append(filename)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(files)} images")
                
        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")
            continue
    
    if not img_stack:
        print("No valid images found after filtering and resizing")
        return None, None
    
    volume = np.stack(img_stack, axis=0)
    
    print(f"\n=== Processing Summary ===")
    print(f"Original files found: {len(files)}")
    print(f"Valid files processed: {len(img_stack)}")
    print(f"Skipped files: {len(files) - len(img_stack)}")
    print(f"Final volume shape: {volume.shape}")
    
    if resize_factor != 1.0:
        original_voxels = original_size[0] * original_size[1] * len(files)
        new_voxels = volume.shape[0] * volume.shape[1] * volume.shape[2]
        if len(img_stack) < len(files):
            # Adjust for skipped files
            expected_voxels = resized_size[0] * resized_size[1] * len(img_stack)
            actual_reduction = (original_size[0] * original_size[1] * len(img_stack)) / expected_voxels
        else:
            actual_reduction = original_voxels / new_voxels
        print(f"Volume size reduced by factor of {actual_reduction:.1f}x")
    
    return volume, valid_files

def calculate_spacing(original_spacing, resize_factor):
    """Calculate adjusted spacing based on resize factor."""
    # CORRECTED: Use same spacing for all dimensions
    adjusted_xy = original_spacing[0] / resize_factor
    spacing = (adjusted_xy, adjusted_xy, adjusted_xy)  # Isotropic
    
    print(f"\n=== Spacing Calculation ===")
    print(f"Resize factor: {resize_factor}")
    print(f"Original spacing: {original_spacing[0]} mm isotropic")
    print(f"Adjusted spacing: {spacing[0]} mm isotropic")
    
    return spacing

def save_as_minc(volume, output_path, spacing=(0.008, 0.008, 0.008)):
    """Save as MINC with isotropic spacing."""
    print(f"\n=== MINC Conversion ===")
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel spacing: {spacing}")
    
    # Calculate physical dimensions
    phys_z = volume.shape[0] * spacing[2]
    phys_y = volume.shape[1] * abs(spacing[1])
    phys_x = volume.shape[2] * spacing[0]
    print(f"Physical size: {phys_x:.2f} × {phys_y:.2f} × {phys_z:.2f} mm")
    
    try:
        # Create SimpleITK image
        itk_image = sitk.GetImageFromArray(volume)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin((0.0, 0.0, 0.0))
        
        # Create temporary NIfTI file
        temp_nifti = f'temp_{uuid.uuid4().hex[:8]}.nii'
        sitk.WriteImage(itk_image, temp_nifti)
        print(f"Temporary NIfTI created: {temp_nifti}")
        
        # Convert to MINC
        ret = os.system(f'nii2mnc -clobber {temp_nifti} {output_path}')
        if ret != 0:
            raise RuntimeError(f"nii2mnc failed with exit code {ret}")
        
        print(f"Successfully created MINC file: {output_path}")
        
        # Verify the output
        verify_minc_file(output_path, spacing)
        
        return True
    
    except Exception as e:
        print(f"Error during MINC conversion: {str(e)}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_nifti):
            os.remove(temp_nifti)
            print(f"Cleaned up temporary file: {temp_nifti}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert BMP stack to isotropic MINC volume",
        epilog="Example: python bmp_to_minc.py /data/bmps output.mnc --resize-factor 0.5"
    )
    parser.add_argument("input_dir", help="Directory with BMP files")
    parser.add_argument("output_mnc", help="Output MINC file")
    parser.add_argument("--resize-factor", type=float, default=1.0,
                       help="XY downsampling factor (default: 1.0)")
    parser.add_argument("--spacing", type=float, default=0.008,
                       help="Isotropic voxel size in mm (default: 0.008)")
    parser.add_argument("--quality", type=int, default=95,
                       help="Resampling quality (1-100)")
    
    args = parser.parse_args()
    
    # Load images
    volume, _ = resize_and_load_images(
        args.input_dir, 
        args.resize_factor,
        args.quality
    )
    
    if volume is None:
        sys.exit(1)
    
    # Calculate spacing (isotropic)
    spacing = calculate_spacing(
        (args.spacing, args.spacing, args.spacing),  # Isotropic tuple
        args.resize_factor
    )
    
    # Save as MINC
    save_as_minc(volume, args.output_mnc, spacing)

if __name__ == "__main__":
    main()
