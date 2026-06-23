#!/usr/bin/env python3
"""
Enhanced Mesh Distance Comparison Tool - Signed Distance Version

This script compares meshes from NIfTI images using SIGNED distances:
- Negative = atlas vertex inside specimen (shrinkage)
- Positive = atlas vertex outside specimen (growth/expansion)
"""

import numpy as np
import nibabel as nib
from skimage import measure
from skimage.filters import threshold_otsu
import open3d as o3d
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fast_simplification
import os
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Default parameters
DEFAULT_SIMPLIFICATION_THRESHOLD = 400000


class MeshProcessor:
    """Class to handle mesh processing operations."""
    
    def __init__(self, simplification_threshold=DEFAULT_SIMPLIFICATION_THRESHOLD):
        self.simplification_threshold = simplification_threshold
    
    def load_and_preprocess_nifti(self, filepath: str, threshold: Optional[float] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Load NIfTI file and preprocess for mesh generation."""
        print(f"Loading {os.path.basename(filepath)}...")
        img = nib.load(filepath)
        
        # Get canonical orientation for consistent processing
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
        
        # Get voxel sizes from affine
        voxel_sizes = nib.affines.voxel_sizes(img.affine)
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = self.calculate_auto_threshold(data)
            print(f"  Auto-calculated threshold: {threshold:.3f}")
        
        # Print data statistics
        print(f"  Shape: {data.shape}, Range: [{np.min(data):.1f}, {np.max(data):.1f}]")
        print(f"  Voxel sizes: {voxel_sizes}")
        
        return data, threshold, voxel_sizes, img.affine
    
    @staticmethod
    def calculate_auto_threshold(data: np.ndarray) -> float:
        """Calculate threshold using Otsu's method."""
        flat_data = data.flatten()
        flat_data = flat_data[flat_data > 0]
        return threshold_otsu(flat_data) if len(flat_data) > 0 else 0.075
    
    def generate_mesh_marching_cubes(self, data: np.ndarray, threshold: float, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"  Generating mesh (threshold={threshold:.3f})...")
        # Extract in IJK index space
        verts_ijk, faces, _, _ = measure.marching_cubes(
            data,
            level=threshold,
            spacing=(1.0, 1.0, 1.0)
        )
        # Map to world coordinates with affine
        verts_h = np.c_[verts_ijk, np.ones(len(verts_ijk))]
        verts_world = (affine @ verts_h.T).T[:, :3]
        print(f"  Generated: {len(verts_world):,} vertices, {len(faces):,} faces")
        return verts_world, faces
    
    def simplify_mesh_if_needed(self, vertices: np.ndarray, faces: np.ndarray, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Simplify mesh if it exceeds threshold."""
        if force or len(vertices) > self.simplification_threshold:
            factor = max(min(1, 1 - (self.simplification_threshold / len(vertices))), 0)
            print(f"  Simplifying mesh (factor={factor:.3f})...")
            vertices, faces = fast_simplification.simplify(vertices, faces, factor)
            print(f"  Simplified: {len(vertices):,} vertices, {len(faces):,} faces")
        return vertices, faces
    
    @staticmethod
    def calculate_signed_distances_from_atlas_to_specimen(
        atlas_vertices_mm: np.ndarray,
        specimen_vertices_mm: np.ndarray,
        specimen_faces: np.ndarray,
        atlas_affine: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Signed distance via Open3D RaycastingScene.
        Negative = atlas vertex inside specimen, Positive = outside.
        """
        print("  Calculating SIGNED distances with Open3D RaycastingScene (atlas -> specimen)...")

        if specimen_vertices_mm.size == 0 or specimen_faces.size == 0:
            raise ValueError("Empty specimen mesh")

        # Build legacy mesh
        mesh_legacy = o3d.geometry.TriangleMesh()
        mesh_legacy.vertices = o3d.utility.Vector3dVector(specimen_vertices_mm.astype(np.float64))
        mesh_legacy.triangles = o3d.utility.Vector3iVector(specimen_faces.astype(np.int32))

        # Optional: remove degenerates & orient
        mesh_legacy.remove_duplicated_vertices()
        mesh_legacy.remove_duplicated_triangles()
        mesh_legacy.remove_degenerate_triangles()
        mesh_legacy.remove_non_manifold_edges()
        mesh_legacy.compute_vertex_normals()
        mesh_legacy.orient_triangles()  # helps sign consistency

        # To tensor mesh for raycasting
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)

        # Vectorized signed distance
        pts = o3d.core.Tensor(atlas_vertices_mm.astype(np.float32))
        # Open3D returns signed distances in same units as input (mm here)
        sd = scene.compute_signed_distance(pts).numpy().astype(np.float64)

        # (Optional) convert to atlas "vox" using atlas affine metric if you want
        signed_vox = None

        return sd, signed_vox


class BatchProcessor:
    """Class to handle batch processing of multiple volumes."""
    
    def __init__(self, mesh_processor: MeshProcessor):
        self.mesh_processor = mesh_processor
    
    def process_volume(self, volume_path: str, threshold: Optional[float] = None, 
                    simplify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"\nProcessing: {os.path.basename(volume_path)}")
        data, used_threshold, voxel_sizes, affine = self.mesh_processor.load_and_preprocess_nifti(volume_path, threshold)
        vertices, faces = self.mesh_processor.generate_mesh_marching_cubes(data, used_threshold, affine)

        if simplify:
            vertices, faces = self.mesh_processor.simplify_mesh_if_needed(vertices, faces)

        return vertices, faces, affine
    
    def process_directory(self, directory: str, pattern: str = "*.nii.gz", 
                         threshold: Optional[float] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Process all volumes in a directory."""
        volume_paths = sorted(glob.glob(os.path.join(directory, pattern)))
        
        if not volume_paths:
            raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")
        
        print(f"Found {len(volume_paths)} volumes to process")
        
        results = []
        for path in volume_paths:
            try:
                vertices, faces, _ = self.process_volume(path, threshold)
                results.append((path, vertices, faces))
            except Exception as e:
                print(f"  Error processing {os.path.basename(path)}: {e}")
                continue
        
        return results
    
    def calculate_average_distances_on_atlas(
        self,
        atlas_vertices: np.ndarray,
        atlas_faces: np.ndarray,
        atlas_affine: np.ndarray,
        specimen_results: List[Tuple[str, np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]], dict]:
        """
        Calculate average signed distances sampled on atlas vertices.
        Returns: (avg_mm, avg_vox, all_mm, all_vox, specimen_stats)
        """
        print("\n### Computing Signed Distances from Atlas to Each Specimen ###")

        dist_mm_list: List[np.ndarray] = []
        dist_vox_list: List[Optional[np.ndarray]] = []
        specimen_stats: dict = {}

        for path, s_verts, s_faces in specimen_results:
            try:
                specimen_name = os.path.basename(path)
                print(f"  Processing {specimen_name}...")

                d_mm, d_vox = self.mesh_processor.calculate_signed_distances_from_atlas_to_specimen(
                    atlas_vertices, s_verts, s_faces, atlas_affine
                )

                dist_mm_list.append(d_mm)
                dist_vox_list.append(d_vox)

                specimen_stats[specimen_name] = {
                    'mean_mm': float(np.mean(d_mm)),
                    'std_mm':  float(np.std(d_mm)),
                    'max_mm':  float(np.max(d_mm)),
                    'min_mm':  float(np.min(d_mm)),
                    'p95_mm':  float(np.percentile(d_mm, 95)),
                    'p05_mm':  float(np.percentile(d_mm, 5)),
                    'median_mm': float(np.median(d_mm)),
                }
                
                # Add voxel stats only if available
                if d_vox is not None:
                    specimen_stats[specimen_name].update({
                        'mean_vox': float(np.mean(d_vox)),
                        'std_vox':  float(np.std(d_vox)),
                        'max_vox':  float(np.max(d_vox)),
                        'min_vox':  float(np.min(d_vox)),
                    })
                
                print(f"    Mean: {specimen_stats[specimen_name]['mean_mm']:+.3f} mm "
                    f"(range: {specimen_stats[specimen_name]['min_mm']:+.3f} to {specimen_stats[specimen_name]['max_mm']:+.3f} mm)")
            except Exception as e:
                print(f"    ERROR processing {specimen_name}: {e}")
                print(f"    Skipping this specimen and continuing...")
                continue

        if dist_mm_list:
            avg_mm  = np.mean(np.vstack(dist_mm_list), axis=0)
            
            # Only compute voxel average if at least one is not None
            if any(d is not None for d in dist_vox_list):
                valid_vox = [d for d in dist_vox_list if d is not None]
                avg_vox = np.mean(np.vstack(valid_vox), axis=0) if valid_vox else None
            else:
                avg_vox = None
            
            print(f"\n  Overall Average Distance: {np.mean(avg_mm):+.3f} ± {np.std(avg_mm):.3f} mm")
            print(f"  Negative (shrinkage): {np.sum(avg_mm < 0)} vertices ({100*np.sum(avg_mm < 0)/len(avg_mm):.1f}%)")
            print(f"  Positive (growth):    {np.sum(avg_mm > 0)} vertices ({100*np.sum(avg_mm > 0)/len(avg_mm):.1f}%)")
            
            return avg_mm, avg_vox, dist_mm_list, dist_vox_list, specimen_stats

        return np.array([]), None, [], [], {}


class Visualizer:
    """Interactive mesh visualizer with signed distance support."""

    def __init__(self):
        self.range_modes = [
            (3.0, "3 STD"),
            (2.0, "2 STD"),
            (1.0, "1 STD"),
            (10.0, "Fixed ±10"),
            (5.0,  "Fixed ±5"),
            (3.0,  "Fixed ±3"),
            (1.0,  "Fixed ±1"),
            (0.5,  "Fixed ±0.5"),
        ]
        self.current_mode_idx = 0

        self.unit_mode = "mm"
        self.data_mm = None
        self.data_vox = None
        self.clip_to_percentile = None

        self.per_specimen_mm = None
        self.specimen_labels = None

    def visualize_with_stats(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        distances_mm: np.ndarray,
        *,
        distances_vox: Optional[np.ndarray] = None,
        title: str = "Mesh Distance Heatmap",
        specimen_count: int = 1,
        specimen_stats: Optional[dict] = None,
        clip_percentile: Optional[float] = None,
        per_specimen_mm: Optional[List[np.ndarray]] = None,
        specimen_labels: Optional[List[str]] = None,
    ):
        """Render mesh with signed distances and interactive controls."""
        print(f"\nCreating visualization: {title}")
        print("  Blue = Shrinkage (inside) | White = No change | Red = Growth (outside)")

        # Store data for live toggling
        self.data_mm = distances_mm
        self.data_vox = distances_vox
        self.clip_to_percentile = clip_percentile
        self.per_specimen_mm = per_specimen_mm
        self.specimen_labels = specimen_labels

        def current_data():
            if self.unit_mode == "vox" and self.data_vox is not None:
                return self.data_vox
            return self.data_mm

        # Initial stats
        data0 = current_data()
        mean_dist = float(np.mean(data0)) if data0.size else 0.0
        std_dist  = float(np.std(data0))  if data0.size else 0.1
        max_dist  = float(np.max(data0))  if data0.size else 0.0
        min_dist  = float(np.min(data0))  if data0.size else 0.0

        # Build VTK polydata
        points = vtk.vtkPoints()
        for v in vertices:
            points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

        cells = vtk.vtkCellArray()
        for f in faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(f[0]))
            tri.GetPointIds().SetId(1, int(f[1]))
            tri.GetPointIds().SetId(2, int(f[2]))
            cells.InsertNextCell(tri)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Build a point locator on the atlas vertices for nearest-vertex lookup
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(polydata)   # polydata has your atlas points
        locator.BuildLocator()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName(title)
        render_window.SetSize(1200, 800)

        # Overlays & color bar
        self._create_text_actors(renderer, mean_dist, std_dist, max_dist, min_dist, specimen_count)
        scalar_bar = self._create_scalar_bar(renderer)

        def compute_display_range(arr: np.ndarray) -> Tuple[float, float]:
            """Get symmetric range for signed distances."""
            base_std = np.std(arr) if arr.size else 0.1
            multiplier, desc = self.range_modes[self.current_mode_idx]
            
            if "Fixed" in desc:
                max_val = multiplier
            else:
                max_val = multiplier * base_std

            # Optional percentile clamp
            if self.clip_to_percentile is not None:
                p_high = np.percentile(arr, self.clip_to_percentile)
                p_low = np.percentile(arr, 100 - self.clip_to_percentile)
                max_val = min(max_val, max(abs(p_high), abs(p_low)))
            
            # Return symmetric range
            max_val = max(max_val, 1e-6)
            return -max_val, max_val

        def unit_label():
            return "voxels" if (self.unit_mode == "vox" and self.data_vox is not None) else "mm"

        def update_visualization():
            arr = current_data()
            if arr is None or arr.size == 0:
                return

            # Update stats
            mean_val = float(np.mean(arr))
            std_val  = float(np.std(arr))
            max_val  = float(np.max(arr))
            min_val  = float(np.min(arr))

            # Set scalars
            vtk_dist = numpy_support.numpy_to_vtk(arr)
            vtk_dist.SetName("SignedDistances")
            polydata.GetPointData().SetScalars(vtk_dist)

            # Configure mapper with symmetric range
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()

            vmin, vmax = compute_display_range(arr)
            mapper.SetScalarRange(vmin, vmax)
            lut = self._create_diverging_lookup_table(vmin, vmax)
            mapper.SetLookupTable(lut)
            mapper.InterpolateScalarsBeforeMappingOn()

            scalar_bar.SetLookupTable(lut)
            scalar_bar.SetTitle(f"Signed Distance ({unit_label()})")
            
            mode_name = self.range_modes[self.current_mode_idx][1]
            self.range_text.SetInput(
                f"Range: {vmin:.3f} to {vmax:+.3f} {unit_label()} ({mode_name}"
                + (f"; p{int(self.clip_to_percentile)} clip" if self.clip_to_percentile else "")
                + ")"
            )

            # Update stats text
            neg_pct = 100 * np.sum(arr < 0) / len(arr)
            pos_pct = 100 * np.sum(arr > 0) / len(arr)
            self.stats_text.SetInput(
                f"Mean: {mean_val:+.3f} {unit_label()} | STD: {std_val:.3f} | "
                f"Range: [{min_val:+.3f}, {max_val:+.3f}] | "
                f"Shrink: {neg_pct:.1f}% | Growth: {pos_pct:.1f}% | Specimens: {specimen_count}"
            )

            render_window.Render()

        # Key bindings
        def key_press_callback(obj, event):
            key = obj.GetKeySym()
            if key == "Tab":
                self.current_mode_idx = (self.current_mode_idx + 1) % len(self.range_modes)
                update_visualization()
                print(f"Switched to: {self.range_modes[self.current_mode_idx][1]}")
            elif key in ("u", "U"):
                if self.unit_mode == "mm" and self.data_vox is not None:
                    self.unit_mode = "vox"
                else:
                    self.unit_mode = "mm"
                print(f"Switched units to: {self.unit_mode}")
                update_visualization()
            elif key in ("p", "P"):
                if self.clip_to_percentile is None:
                    self.clip_to_percentile = 95.0
                else:
                    self.clip_to_percentile = None
                print(f"Percentile clipping: {self.clip_to_percentile if self.clip_to_percentile else 'OFF'}")
                update_visualization()
            elif key in ("s", "S"):
                self._save_statistics(
                    current_data(),
                    specimen_count,
                    specimen_stats,
                    unit=unit_label(),
                )

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # Use a CELL picker (hits triangles), not a point picker
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)  # tweak if needed (in world coords fraction of bbox)
        
        interactor.SetPicker(picker)
        interactor.AddObserver("KeyPressEvent", key_press_callback)

        def left_button_press_callback(obj, event):
            x, y = interactor.GetEventPosition()
            # Try to pick against the renderer's actors
            hit = picker.Pick(x, y, 0, renderer)
            if hit:
                # World-space position on the picked triangle
                px, py, pz = picker.GetPickPosition()
                # Map to nearest atlas vertex
                pid = locator.FindClosestPoint((px, py, pz))
                if pid >= 0:
                    print(f"Picked world ({px:.3f}, {py:.3f}, {pz:.3f}) -> vertex id {pid}")
                    self._plot_per_specimen_for_vertex(pid)
            # still allow default rotate behavior
            obj.OnLeftButtonDown()

        interactor.AddObserver("LeftButtonPressEvent", left_button_press_callback)

        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.2)

        update_visualization()
        render_window.Render()
        interactor.Initialize()    # <-- add this
        interactor.Start()

    def _create_text_actors(self, renderer, mean_dist, std_dist, max_dist, min_dist, specimen_count):
        # Range text
        self.range_text = vtk.vtkTextActor()
        self.range_text.SetPosition(10, 750)
        self.range_text.GetTextProperty().SetFontSize(16)
        self.range_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        renderer.AddActor2D(self.range_text)

        # Stats text
        self.stats_text = vtk.vtkTextActor()
        self.stats_text.SetInput(
            f"Mean: {mean_dist:+.3f} mm | STD: {std_dist:.3f} mm | "
            f"Range: [{min_dist:+.3f}, {max_dist:+.3f}] | Specimens: {specimen_count}"
        )
        self.stats_text.SetPosition(10, 720)
        self.stats_text.GetTextProperty().SetFontSize(14)
        self.stats_text.GetTextProperty().SetColor(0.9, 0.9, 0.9)
        renderer.AddActor2D(self.stats_text)

        # Instructions
        instruction_text = vtk.vtkTextActor()
        instruction_text.SetInput(
            "TAB: Cycle ranges | U: Toggle units | P: Percentile clip | S: Save stats | Click: Pick vertex"
        )
        instruction_text.SetPosition(10, 690)
        instruction_text.GetTextProperty().SetFontSize(12)
        instruction_text.GetTextProperty().SetColor(0.7, 0.7, 0.7)
        renderer.AddActor2D(instruction_text)
        
        # Color legend
        legend_text = vtk.vtkTextActor()
        legend_text.SetInput("Blue = Shrinkage (inside) | White = No change | Red = Growth (outside)")
        legend_text.SetPosition(10, 30)
        legend_text.GetTextProperty().SetFontSize(14)
        legend_text.GetTextProperty().SetColor(1.0, 1.0, 0.8)
        renderer.AddActor2D(legend_text)

    def _create_scalar_bar(self, renderer):
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle("Signed Distance (mm)")
        scalar_bar.SetNumberOfLabels(7)
        scalar_bar.SetPosition(0.85, 0.1)
        scalar_bar.SetWidth(0.1)
        scalar_bar.SetHeight(0.8)
        scalar_bar.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetLabelTextProperty().SetColor(1.0, 1.0, 1.0)
        renderer.AddActor2D(scalar_bar)
        return scalar_bar

    def _create_diverging_lookup_table(self, vmin, vmax):
        """Create diverging colormap for signed distances (blue-white-red)."""
        lut = vtk.vtkLookupTable()
        n = 256
        lut.SetNumberOfTableValues(n)
        
        # Use coolwarm colormap: blue (negative) -> white (zero) -> red (positive)
        colormap = cm.get_cmap('coolwarm')
        for i in range(n):
            r, g, b, a = colormap(i / (n-1))
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
        
        lut.SetRange(float(vmin), float(vmax))
        lut.Build()
        return lut

    def _save_statistics(self, distances, specimen_count, specimen_stats=None, unit="mm"):
        stats = {
            'unit': unit,
            'specimen_count': specimen_count,
            'overall_statistics': {
                'mean': float(np.mean(distances)) if distances is not None else None,
                'std':  float(np.std(distances))  if distances is not None else None,
                'max':  float(np.max(distances))  if distances is not None else None,
                'min':  float(np.min(distances))  if distances is not None else None,
                'median': float(np.median(distances)) if distances is not None else None,
                'negative_count': int(np.sum(distances < 0)) if distances is not None else None,
                'positive_count': int(np.sum(distances > 0)) if distances is not None else None,
                'negative_pct': float(100 * np.sum(distances < 0) / len(distances)) if distances is not None else None,
                'positive_pct': float(100 * np.sum(distances > 0) / len(distances)) if distances is not None else None,
                'percentiles': {
                    '5':  float(np.percentile(distances, 5))  if distances is not None else None,
                    '25': float(np.percentile(distances, 25)) if distances is not None else None,
                    '50': float(np.percentile(distances, 50)) if distances is not None else None,
                    '75': float(np.percentile(distances, 75)) if distances is not None else None,
                    '95': float(np.percentile(distances, 95)) if distances is not None else None,
                }
            }
        }
        if specimen_stats:
            stats['per_specimen'] = specimen_stats

        filename = f"signed_distance_stats_{specimen_count}_specimens_{unit}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filename}")

    def plot_distance_histogram(self, distances_list: List[np.ndarray], labels: List[str] = None,
                                title_suffix: str = ""):
        """Plot histogram of signed distances."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Individual histograms
        for i, distances in enumerate(distances_list):
            label = labels[i] if labels else f"Specimen {i+1}"
            ax1.hist(distances, bins=50, alpha=0.5, label=label[:30])
        
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero (no change)')
        ax1.set_xlabel("Signed Distance (mm)")
        ax1.set_ylabel("Vertex Count")
        ax1.set_title("Signed Distance Distribution by Specimen")
        if len(distances_list) <= 10:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Combined histogram
        all_distances = np.concatenate(distances_list)
        ax2.hist(all_distances, bins=50, color='purple', alpha=0.7)
        ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero')
        ax2.axvline(np.mean(all_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_distances):+.3f}mm')
        ax2.axvline(np.median(all_distances), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_distances):+.3f}mm')
        
        ax2.set_xlabel("Signed Distance (mm)")
        ax2.set_ylabel("Vertex Count")
        ax2.set_title(f"Combined Signed Distance Distribution{title_suffix}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("signed_distance_distribution.png", dpi=150)
        plt.show()
        
    def _plot_per_specimen_for_vertex(self, vidx: int):
        """
        Show a bar chart of signed distance at this vertex for each specimen.
        Requires self.per_specimen_mm to be a list of arrays, all len = #vertices.
        """
        if self.per_specimen_mm is None or len(self.per_specimen_mm) == 0:
            print("No per-specimen data available for plotting.")
            return

        values = []
        labels = []
        for i, arr in enumerate(self.per_specimen_mm):
            if arr is None or vidx >= len(arr):
                continue
            values.append(arr[vidx])
            if self.specimen_labels and i < len(self.specimen_labels):
                labels.append(self.specimen_labels[i])
            else:
                labels.append(f"Specimen {i+1}")

        if not values:
            print("No values for this vertex.")
            return

        # make a quick bar plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(values))
        ax.bar(x, values)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Signed distance (mm)")
        ax.set_title(f"Vertex {vidx}: specimen offsets")
        fig.tight_layout()
        plt.show(block=False)
        try:
            plt.pause(0.001)  # helps macOS backends show non-blocking windows
        except Exception:
            pass


def create_dummy_data():
    """Create dummy NIfTI files for testing signed distances."""
    os.makedirs("./test_data", exist_ok=True)
    
    size = 200
    
    # Create atlas (sphere)
    atlas_data = np.zeros((size, size, size))
    center = size // 2
    radius = 70
    x, y, z = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    atlas_data[mask] = 1000
    
    atlas_nifti = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_nifti, "./test_data/atlas.nii.gz")
    
    # Create specimens with varying sizes (shrinkage and growth)
    for i in range(3):
        specimen_data = np.zeros((size, size, size))
        
        # Vary radius to create shrinkage and growth
        radius_var = radius + (i - 1) * 10  # -10, 0, +10
        offset = np.random.randint(-5, 5, 3)
        
        x, y, z = np.ogrid[:size, :size, :size]
        mask = ((x - center - offset[0])**2 + 
                (y - center - offset[1])**2 + 
                (z - center - offset[2])**2 <= radius_var**2)
        specimen_data[mask] = 1000
        
        specimen_nifti = nib.Nifti1Image(specimen_data, np.eye(4))
        nib.save(specimen_nifti, f"./test_data/specimen_{i+1}.nii.gz")
    
    print("Created test data in ./test_data/")
    return "./test_data/atlas.nii.gz", "./test_data/"


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Mesh Distance Comparison Tool - Signed Distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single specimen to atlas
  %(prog)s atlas.nii.gz specimen.nii.gz
  
  # Process directory of specimens
  %(prog)s atlas.nii.gz --directory /path/to/specimens/
  
  # Use specific thresholds
  %(prog)s atlas.nii.gz specimen.nii.gz --threshold-atlas 0.5 --threshold-specimen 0.6
  
  # Test with dummy data
  %(prog)s --test

Signed Distance Interpretation:
  - Negative values: Atlas vertex is INSIDE specimen surface (shrinkage)
  - Positive values: Atlas vertex is OUTSIDE specimen surface (growth/expansion)
  - Zero: Perfect alignment
        """
    )
    
    parser.add_argument("atlas", nargs="?", help="Path to atlas NIfTI file")
    parser.add_argument("specimen", nargs="?", help="Path to specimen NIfTI file (or use --directory)")
    
    parser.add_argument("-d", "--directory", help="Directory containing specimen NIfTI files")
    parser.add_argument("-p", "--pattern", default="*.nii.gz", help="File pattern for directory processing")
    
    parser.add_argument("--threshold-atlas", type=float, help="Threshold for atlas")
    parser.add_argument("--threshold-specimen", type=float, help="Threshold for specimens")
    
    parser.add_argument("--no-simplification", action="store_true", help="Disable mesh simplification")
    parser.add_argument("--simplification-threshold", type=int, default=DEFAULT_SIMPLIFICATION_THRESHOLD,
                       help=f"Vertex count threshold for simplification")
    
    parser.add_argument("--save-meshes", action="store_true", help="Save processed meshes to files")
    parser.add_argument("--output-dir", default="./output", help="Output directory for saved files")
    
    parser.add_argument("--test", action="store_true", help="Run with test data")
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        print("Running in test mode with signed distances...")
        atlas_path, specimen_dir = create_dummy_data()
        args.atlas = atlas_path
        args.directory = specimen_dir
        args.threshold_atlas = 500
        args.threshold_specimen = 500
    
    # Validate arguments
    if not args.atlas:
        parser.error("Atlas file is required (or use --test)")
    
    if not args.specimen and not args.directory:
        parser.error("Either specimen file or --directory must be specified")
    
    # Create output directory if needed
    if args.save_meshes:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processors
    mesh_processor = MeshProcessor(
        simplification_threshold=args.simplification_threshold if not args.no_simplification else float('inf')
    )
    
    batch_processor = BatchProcessor(mesh_processor)
    visualizer = Visualizer()
    
    print("="*70)
    print("ENHANCED MESH DISTANCE COMPARISON TOOL - SIGNED DISTANCES")
    print("="*70)
    print("Negative = Shrinkage (inside) | Positive = Growth (outside)")
    print("="*70)
    
    # Process atlas (don't simplify to maintain vertex correspondence)
    print("\n### Processing Atlas ###")
    atlas_vertices, atlas_faces, atlas_affine = batch_processor.process_volume(
        args.atlas, 
        args.threshold_atlas,
        simplify=False
    )
    
    # Process specimens
    specimen_results = []
    
    if args.directory:
        print(f"\n### Processing Directory: {args.directory} ###")
        specimen_results = batch_processor.process_directory(
            args.directory, args.pattern, args.threshold_specimen
        )
    else:
        print("\n### Processing Single Specimen ###")
        vertices, faces, _ = batch_processor.process_volume(args.specimen, args.threshold_specimen)
        specimen_results = [(args.specimen, vertices, faces)]
    
    if not specimen_results:
        print("No specimens were successfully processed!")
        return
    
    # Calculate signed distances
    if len(specimen_results) > 1:
        avg_mm, avg_vox, all_mm, all_vox, specimen_stats = batch_processor.calculate_average_distances_on_atlas(
            atlas_vertices, atlas_faces, atlas_affine, specimen_results
        )

        specimen_labels = [os.path.basename(p) for (p, _, _) in specimen_results]

        visualizer.visualize_with_stats(
            atlas_vertices, atlas_faces,
            distances_mm=avg_mm,
            distances_vox=avg_vox,
            title=f"Atlas: Avg Signed Distance from {len(specimen_results)} Specimens",
            specimen_count=len(specimen_results),
            specimen_stats=specimen_stats,
            clip_percentile=95.0,
            per_specimen_mm=all_mm,               # list of arrays, one per specimen
            specimen_labels=specimen_labels,      # same order as specimen_results
        )
        
    else:
        # Single specimen comparison
        print("\n### Computing Signed Distances ###")
        path, specimen_vertices, specimen_faces = specimen_results[0]
        
        distances_mm, distances_vox = mesh_processor.calculate_signed_distances_from_atlas_to_specimen(
            atlas_vertices, specimen_vertices, specimen_faces, atlas_affine
        )
        
        print(f"\nSigned Distance Statistics:")
        print(f"  Mean:   {np.mean(distances_mm):+.3f} mm")
        print(f"  Median: {np.median(distances_mm):+.3f} mm")
        print(f"  STD:    {np.std(distances_mm):.3f} mm")
        print(f"  Range:  [{np.min(distances_mm):+.3f}, {np.max(distances_mm):+.3f}] mm")
        print(f"  Shrinkage (negative): {100*np.sum(distances_mm < 0)/len(distances_mm):.1f}%")
        print(f"  Growth (positive):    {100*np.sum(distances_mm > 0)/len(distances_mm):.1f}%")
        
        # Visualize on atlas mesh
        visualizer.visualize_with_stats(
            atlas_vertices, atlas_faces, 
            distances_mm,
            distances_vox=distances_vox,
            title=f"Atlas: Signed Distance to {os.path.basename(path)}",
            specimen_count=1
        )
    
    # Save meshes if requested
    if args.save_meshes:
        print(f"\n### Saving Meshes to {args.output_dir} ###")
        
        atlas_mesh = o3d.geometry.TriangleMesh()
        atlas_mesh.vertices = o3d.utility.Vector3dVector(atlas_vertices)
        atlas_mesh.triangles = o3d.utility.Vector3iVector(atlas_faces)
        atlas_filename = os.path.join(args.output_dir, "atlas_mesh.ply")
        o3d.io.write_triangle_mesh(atlas_filename, atlas_mesh)
        print(f"  Saved: {atlas_filename}")
        
        for i, (path, vertices, faces) in enumerate(specimen_results):
            specimen_mesh = o3d.geometry.TriangleMesh()
            specimen_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            specimen_mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            base_name = os.path.splitext(os.path.basename(path))[0]
            specimen_filename = os.path.join(args.output_dir, f"{base_name}_mesh.ply")
            o3d.io.write_triangle_mesh(specimen_filename, specimen_mesh)
            print(f"  Saved: {specimen_filename}")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("Visualization shows signed distances on ATLAS:")
    print("  - Blue regions: Shrinkage (atlas extends beyond specimen)")
    print("  - Red regions:  Growth (specimen extends beyond atlas)")
    print("="*70)


if __name__ == "__main__":
    main()