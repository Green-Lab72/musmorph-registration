# Atlas_Register_Subjects_016mm.py
# ===================================
# Generates SLURM .sh scripts for a SUBJECT-TO-ATLAS registration pipeline
# at 16 um (0.0159999 mm) resolution using the MINC Toolkit.
#
# Companion script to Atlas_Pipeline_016mm.py:
#   - Atlas_Pipeline_016mm.py builds an unbiased population atlas from a cohort.
#   - This script registers ADDITIONAL specimens into that fixed atlas space.
#
# Differences from the atlas-construction generator:
#   * The atlas is the FIXED target at every stage. No template drift.
#   * No pairwise LSQ12 (subject -> atlas only).
#   * No xfmavg.
#   * No mincaverage of moving templates between stages.
#   * Pipeline is linear in N (not quadratic).
#   * Per-specimen stages are SLURM array jobs (one task per specimen).
#
# Retained from the construction generator:
#   * slurm_header() with bound MINC env vars
#   * Relaxed minctracc tolerances (0.005 / 0.002 / 0.001 per affine level)
#   * Reduced NL iteration counts (NL1=20, NL2=10, NL3=8, NL4=6)
#   * NL5 stage dropped
#   * Binary -model_mask + -source_mask on every minctracc call
#   * -no_apodize on mincblur, -gradient dropped
#   * Apply-mask multiplication of every resampled output
#   * -float averaging (for the optional final aggregation only)
#
# Inputs you must already have on the cluster:
#   1. The existing atlas .mnc on the 0.0159999 mm grid.
#   2. A TIGHT binary registration mask drawn on the atlas itself
#      (the apply mask is auto-generated from this via mincmorph DD).
#   3. New specimens preprocessed onto the same 0.0159999 mm grid, with the
#      same orientation as the atlas (i.e. 'norm' volumes from preprocessing).
#   4. A specimen list file (one ID per line, no extension).

import os

# ===================================================================
# USER-EDITABLE PARAMETERS
# ===================================================================

PROJECT_NAME = input("Enter project name (e.g., Fgf8_Extension): ")
CLUSTER_PATH = input("Enter project path on cluster (e.g., /work/hallgrimsson_lab/): ")
PROJECT_PATH = CLUSTER_PATH + PROJECT_NAME + "/"

print("\nCreate directory structure on cluster. E.g.:\n")
print(f"mkdir -p {PROJECT_PATH}{{Scripts,Quality,Source/{{Atlas,MNC,Orig,Resample,Tag,XFM}},lsq6/{{Blurred,MNC,XFM}},lsq12/{{Blurred,MNC,XFM}},nl/{{Blurred,MNC,XFM}}}}\n")

LOCAL_SCRIPT_PATH = input("Enter path to local scripts directory: ")
os.chdir(LOCAL_SCRIPT_PATH)

All_Specimens = input("Enter path to specimen list (NEW specimens to register): ")

print("\nAtlas inputs (must already exist on the cluster, on the 0.0159999 mm grid):")
Atlas_Filename      = input("  Atlas .mnc filename (e.g., Fgf8_Atlas.mnc): ")
Atlas_Mask_Filename = input("  Atlas tight binary registration mask filename (e.g., Atlas_mask_reg.mnc): ")

# Cluster parameters
Module                = "minc/1.9.18"
n_nodes               = "1"
SLURM_Account         = "mmarazita"
lsq6_Time             = "07:00:00"
lsq6_Mem              = "30000M"
lsq12_Time            = "12:00:00"   # subject->atlas LSQ12 is much faster than pairwise
lsq12_Mem             = "30000M"
nl_Time               = "36:00:00"
nl_Mem                = "30000M"
atlas_blur_Time       = "02:00:00"
atlas_blur_Mem        = "16000M"
Job_Submission_Time   = "05-00:00:00"
Job_Submission_Mem    = "2000M"
Array_Throttle        = 50           # max concurrent SLURM array tasks per stage

print("\nDefault compute cluster parameters:")
print(f"  Module: {Module}")
print(f"  Account: {SLURM_Account}")
print(f"  Nodes: {n_nodes}")
print(f"  atlas blur Time/Mem: {atlas_blur_Time} / {atlas_blur_Mem}")
print(f"  lsq6  Time/Mem: {lsq6_Time} / {lsq6_Mem}")
print(f"  lsq12 Time/Mem: {lsq12_Time} / {lsq12_Mem}")
print(f"  nl    Time/Mem: {nl_Time} / {nl_Mem}")
print(f"  Job submission Time/Mem: {Job_Submission_Time} / {Job_Submission_Mem}")
print(f"  Array throttle: {Array_Throttle}\n")
print(f"Upload scripts to {PROJECT_PATH}Scripts.")
print(f"Upload new specimen .mnc files to {PROJECT_PATH}Source/MNC/.")
print(f"Atlas + registration mask must be in {PROJECT_PATH}Source/Atlas/.\n")

# ===================================================================
# PATH DEFINITIONS
# ===================================================================

Scripts_path        = PROJECT_PATH + "Scripts/"
Atlas_path          = PROJECT_PATH + "Source/Atlas/"
Source_MNC_path     = PROJECT_PATH + "Source/MNC/"
lsq6_Blurred_path   = PROJECT_PATH + "lsq6/Blurred/"
lsq6_XFM_path       = PROJECT_PATH + "lsq6/XFM/"
lsq6_MNC_path       = PROJECT_PATH + "lsq6/MNC/"
lsq12_Blurred_path  = PROJECT_PATH + "lsq12/Blurred/"
lsq12_XFM_path      = PROJECT_PATH + "lsq12/XFM/"
lsq12_MNC_path      = PROJECT_PATH + "lsq12/MNC/"
nl_Blurred_path     = PROJECT_PATH + "nl/Blurred/"
nl_XFM_path         = PROJECT_PATH + "nl/XFM/"
nl_MNC_path         = PROJECT_PATH + "nl/MNC/"

# ===================================================================
# ATLAS AND MASK FILES (the FIXED targets for everything)
# ===================================================================
# Variable names mirror the atlas-construction generator: the existing
# atlas plays the role that LM_average played there.

LM_Avg            = Atlas_path + Atlas_Filename
LM_Avg_Mask       = Atlas_path + Atlas_Mask_Filename
# Apply mask is auto-generated from the registration mask (mincmorph DD).
LM_Avg_Mask_Apply = Atlas_path + Atlas_Mask_Filename.replace(".mnc", "_apply.mnc")

# ===================================================================
# BLURRED ATLAS FILE NAMES (UNIQUE FWHMs)
# 9 unique levels: 0.077, 0.040, 0.023, 0.018, 0.016, 0.183, 0.137, 0.091, 0.046
# 0.018 is shared between LSQ6 (level 3) and LSQ12 (level 2).
# ===================================================================

LM_Avg_077 = Atlas_path + "Atlas_077"
LM_Avg_040 = Atlas_path + "Atlas_040"
LM_Avg_023 = Atlas_path + "Atlas_023"
LM_Avg_018 = Atlas_path + "Atlas_018"
LM_Avg_016 = Atlas_path + "Atlas_016"
LM_Avg_183 = Atlas_path + "Atlas_183"
LM_Avg_137 = Atlas_path + "Atlas_137"
LM_Avg_091 = Atlas_path + "Atlas_091"
LM_Avg_046 = Atlas_path + "Atlas_046"

LM_Avg_077_Blur = Atlas_path + "Atlas_077_blur.mnc"
LM_Avg_040_Blur = Atlas_path + "Atlas_040_blur.mnc"
LM_Avg_023_Blur = Atlas_path + "Atlas_023_blur.mnc"
LM_Avg_018_Blur = Atlas_path + "Atlas_018_blur.mnc"
LM_Avg_016_Blur = Atlas_path + "Atlas_016_blur.mnc"
LM_Avg_183_Blur = Atlas_path + "Atlas_183_blur.mnc"
LM_Avg_137_Blur = Atlas_path + "Atlas_137_blur.mnc"
LM_Avg_091_Blur = Atlas_path + "Atlas_091_blur.mnc"
LM_Avg_046_Blur = Atlas_path + "Atlas_046_blur.mnc"

# ===================================================================
# SPECIMEN LIST
# ===================================================================

with open(All_Specimens, 'r') as fh:
    Specimens = fh.read()
Specimen_IDs = [s for s in Specimens.split('\n') if s.strip()]
Specimen_List_Length = len(Specimen_IDs)

print(f"Specimen count: {Specimen_List_Length}\n")

# ===================================================================
# COMMAND STRINGS (preserved verbatim from the construction generator)
# ===================================================================

MNC_Blur = "mincblur -clobber -no_apodize -fwhm "

# --- LSQ6 (rigid) ---
lsq6_Register_077_Blur = ("minctracc -clobber -xcorr -lsq6 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.077 0.077 0.077 -simplex 0.178 -use_simplex -tol 0.005 ")

lsq6_Register_040_Blur = ("minctracc -clobber -xcorr -lsq6 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.040 0.040 0.040 -simplex 0.123 -use_simplex -tol 0.002 ")

lsq6_Register_018_Blur = ("minctracc -clobber -xcorr -lsq6 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.018 0.018 0.018 -simplex 0.073 -use_simplex -tol 0.001 ")

# --- LSQ12 (affine, clamped blur schedule) ---
lsq12_Register_023_Blur = ("minctracc -clobber -xcorr -lsq12 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.023 0.023 0.023 -simplex 0.224 -use_simplex -tol 0.005 ")

lsq12_Register_018_Blur = ("minctracc -clobber -xcorr -lsq12 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.018 0.018 0.018 -simplex 0.112 -use_simplex -tol 0.002 ")

lsq12_Register_016_Blur = ("minctracc -clobber -xcorr -lsq12 "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.016 0.016 0.016 -simplex 0.077 -use_simplex -tol 0.001 ")

# --- Nonlinear: NL1 (step 0.183, lattice 0.549, 20 iters) ---
nl_1_Register_Begin = ("minctracc -clobber -xcorr "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.183 0.183 0.183 -simplex 0.457 -use_simplex -tol 0.0001 ")
nl_1_Register_End = ("-iterations 20 -similarity 0.8 -weight 0.8 -stiffness 0.98 "
    "-nonlinear corrcoeff -sub_lattice 6 "
    "-lattice_diameter 0.549 0.549 0.549 -max_def_magnitude 1 -xcorr -identity ")

# --- Nonlinear: NL2 (step 0.137, lattice 0.366, 10 iters) ---
nl_2_Register_Begin = ("minctracc -clobber -xcorr "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.137 0.137 0.137 -simplex 0.457 -use_simplex -tol 0.0001 ")
nl_2_Register_End = ("-iterations 10 -similarity 0.8 -weight 0.8 -stiffness 0.98 "
    "-nonlinear corrcoeff -sub_lattice 6 "
    "-lattice_diameter 0.366 0.366 0.366 -max_def_magnitude 1 -xcorr -transform ")

# --- Nonlinear: NL3 (step 0.091, lattice 0.275, 8 iters) ---
nl_3_Register_Begin = ("minctracc -clobber -xcorr "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.091 0.091 0.091 -simplex 0.457 -use_simplex -tol 0.0001 ")
nl_3_Register_End = ("-iterations 8 -similarity 0.8 -weight 0.8 -stiffness 0.98 "
    "-nonlinear corrcoeff -sub_lattice 6 "
    "-lattice_diameter 0.275 0.275 0.275 -max_def_magnitude 1 -xcorr -transform ")

# --- Nonlinear: NL4 (step 0.046, lattice 0.137, 6 iters) ---
nl_4_Register_Begin = ("minctracc -clobber -xcorr "
    "-w_translations 0.4 0.4 0.4 -w_rotations 0.0174533 0.0174533 0.0174533 "
    "-w_scales 0.02 0.02 0.02 -w_shear 0.02 0.02 0.02 "
    "-step 0.046 0.046 0.046 -simplex 0.275 -use_simplex -tol 0.0001 ")
nl_4_Register_End = ("-iterations 6 -similarity 0.8 -weight 0.8 -stiffness 0.98 "
    "-nonlinear corrcoeff -sub_lattice 6 "
    "-lattice_diameter 0.137 0.137 0.137 -max_def_magnitude 1 -xcorr -transform ")

# --- mincaverage (used only for the optional Goal-B aggregation) ---
MNC_Avg = "mincaverage -clobber -2 -float -nonormalize "

# ===================================================================
# HELPERS
# ===================================================================

def slurm_header(mem, time, job_name=None, array=None, account=SLURM_Account):
    """Common SLURM preamble + MINC environment, identical to the
    construction generator except that account and array can be passed in."""
    h = "#!/bin/bash\n"
    if account:
        h += "#SBATCH --account=" + account + "\n"
    h += "#SBATCH --nodes=" + n_nodes + "\n"
    h += "#SBATCH --mem=" + mem + "\n"
    h += "#SBATCH --time=" + time + "\n"
    if job_name:
        h += "#SBATCH --job-name=" + job_name + "\n"
    if array:
        h += "#SBATCH --array=" + array + "\n"
    h += "\nset -euo pipefail\n"
    h += "export PATH=\"/ix1/rgreen/mdj72/miniconda3/envs/mdj_env/bin:$PATH\"\n"
    h += "export MINC_TOOLKIT=/ihome/rgreen/mdj72/minc-toolkit_v2/opt/minc/1.9.18\n"
    h += "export PATH=\"$MINC_TOOLKIT/bin:$MINC_TOOLKIT/pipeline:$PATH\"\n"
    h += "export LD_LIBRARY_PATH=\"$MINC_TOOLKIT/lib:${LD_LIBRARY_PATH:-}\"\n"
    h += "export MINC_FORCE_V2=1\n"
    h += "export PERL5LIB=\"$MINC_TOOLKIT/perl:${PERL5LIB:-}\"\n\n"
    h += "cd " + Scripts_path + "\n\n"
    h += "echo \"The job started at $(date).\"\n\n"
    return h


# Bash array literal of specimen IDs, embedded in each per-specimen array script.
spec_bash_array = " ".join('"' + s + '"' for s in Specimen_IDs)


def specimen_array_preamble():
    """Bash snippet that resolves SPEC=specimen ID for the current array task."""
    s  = "SPECS=(" + spec_bash_array + ")\n"
    s += "SPEC=${SPECS[$SLURM_ARRAY_TASK_ID]}\n"
    s += "echo \"Processing specimen: $SPEC (task $SLURM_ARRAY_TASK_ID)\"\n\n"
    return s


# ===================================================================
# STAGE 0 - Atlas blurring (one-time prep)
# ===================================================================
# Blur the atlas at every FWHM the pipeline will use, AND build the apply
# mask from the tight registration mask via mincmorph dilation.
# These outputs are written to Source/Atlas/ and reused by every later stage.

with open("atlas_blur.sh", 'w') as f:
    f.write(slurm_header(atlas_blur_Mem, atlas_blur_Time, job_name="atlas_blur"))
    f.write("echo \"Blurring the atlas at all FWHMs and building the apply mask.\"\n\n")
    # Apply mask: dilate the tight registration mask twice.
    # If you have a hand-tuned apply mask, replace this line with a cp,
    # or remove it entirely and supply LM_Avg_Mask_Apply yourself.
    f.write("mincmorph -clobber -successive DD " + LM_Avg_Mask + " " + LM_Avg_Mask_Apply + "\n\n")
    # LSQ6 levels.
    f.write(MNC_Blur + "0.077 " + LM_Avg + " " + LM_Avg_077 + "\n")
    f.write(MNC_Blur + "0.040 " + LM_Avg + " " + LM_Avg_040 + "\n")
    f.write(MNC_Blur + "0.018 " + LM_Avg + " " + LM_Avg_018 + "\n")
    # LSQ12 levels (0.018 already done above and reused).
    f.write(MNC_Blur + "0.023 " + LM_Avg + " " + LM_Avg_023 + "\n")
    f.write(MNC_Blur + "0.016 " + LM_Avg + " " + LM_Avg_016 + "\n")
    # NL levels.
    f.write(MNC_Blur + "0.183 " + LM_Avg + " " + LM_Avg_183 + "\n")
    f.write(MNC_Blur + "0.137 " + LM_Avg + " " + LM_Avg_137 + "\n")
    f.write(MNC_Blur + "0.091 " + LM_Avg + " " + LM_Avg_091 + "\n")
    f.write(MNC_Blur + "0.046 " + LM_Avg + " " + LM_Avg_046 + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# STAGE 1 - LSQ6 (subject -> atlas, rigid)
# ===================================================================
# Per-specimen array job. Each task:
#   1. Blurs the specimen at 0.077, 0.040, 0.018.
#   2. Hierarchically registers specimen blurs to atlas blurs (rigid).
#   3. Resamples the original specimen using the final LSQ6 transform.
#   4. Multiplies by the apply mask.

with open("lsq6_specimen_array.sh", 'w') as f:
    f.write(slurm_header(
        lsq6_Mem, lsq6_Time,
        job_name="lsq6_subject2atlas",
        array="0-" + str(Specimen_List_Length - 1) + "%" + str(Array_Throttle)))
    f.write(specimen_array_preamble())
    # Blur specimen at three LSQ6 levels.
    f.write(MNC_Blur + "0.077 " + Source_MNC_path + "${SPEC}.mnc " + lsq6_Blurred_path + "${SPEC}_077\n")
    f.write(MNC_Blur + "0.040 " + Source_MNC_path + "${SPEC}.mnc " + lsq6_Blurred_path + "${SPEC}_040\n")
    f.write(MNC_Blur + "0.018 " + Source_MNC_path + "${SPEC}.mnc " + lsq6_Blurred_path + "${SPEC}_018\n\n")
    # Hierarchical registration: coarse -> fine, all to the FIXED atlas.
    f.write(lsq6_Register_077_Blur
        + lsq6_Blurred_path + "${SPEC}_077_blur.mnc " + LM_Avg_077_Blur + " "
        + lsq6_XFM_path + "${SPEC}_lsq6_0.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -identity\n")
    f.write(lsq6_Register_040_Blur
        + lsq6_Blurred_path + "${SPEC}_040_blur.mnc " + LM_Avg_040_Blur + " "
        + lsq6_XFM_path + "${SPEC}_lsq6_1.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transformation " + lsq6_XFM_path + "${SPEC}_lsq6_0.xfm\n")
    f.write(lsq6_Register_018_Blur
        + lsq6_Blurred_path + "${SPEC}_018_blur.mnc " + LM_Avg_018_Blur + " "
        + lsq6_XFM_path + "${SPEC}_lsq6_2.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transformation " + lsq6_XFM_path + "${SPEC}_lsq6_1.xfm\n\n")
    # Resample original specimen into atlas-space (rigid only), mask.
    f.write("mincresample -like " + LM_Avg + " -clobber "
        + "-transformation " + lsq6_XFM_path + "${SPEC}_lsq6_2.xfm "
        + Source_MNC_path + "${SPEC}.mnc "
        + lsq6_MNC_path + "${SPEC}_lsq6_unmasked.mnc\n")
    f.write("mincmath -clobber -mult "
        + lsq6_MNC_path + "${SPEC}_lsq6_unmasked.mnc "
        + LM_Avg_Mask_Apply + " "
        + lsq6_MNC_path + "${SPEC}_lsq6.mnc\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# STAGE 2 - LSQ12 (subject -> atlas, 12-DOF affine; NO PAIRWISE)
# ===================================================================
# Per-specimen array job. Each task:
#   1. Blurs LSQ6-aligned specimen at 0.023, 0.018, 0.016.
#   2. Hierarchically registers specimen blurs to atlas blurs (12-DOF).
#   3. Concatenates LSQ6 + LSQ12 transforms into origtolsq12.xfm.
#   4. Resamples the ORIGINAL specimen with origtolsq12 and masks.
#
# Note: lsq12_2.xfm (the final affine) takes LSQ6-space directly to atlas
# space. This replaces the lsq12_AVG.xfm from the construction pipeline,
# which was the average of pairwise lsq12_2's. No xfmavg needed here.

with open("lsq12_specimen_array.sh", 'w') as f:
    f.write(slurm_header(
        lsq12_Mem, lsq12_Time,
        job_name="lsq12_subject2atlas",
        array="0-" + str(Specimen_List_Length - 1) + "%" + str(Array_Throttle)))
    f.write(specimen_array_preamble())
    # Blur LSQ6-aligned specimen at three LSQ12 levels.
    f.write(MNC_Blur + "0.023 " + lsq6_MNC_path + "${SPEC}_lsq6.mnc " + lsq12_Blurred_path + "${SPEC}_023\n")
    f.write(MNC_Blur + "0.018 " + lsq6_MNC_path + "${SPEC}_lsq6.mnc " + lsq12_Blurred_path + "${SPEC}_018\n")
    f.write(MNC_Blur + "0.016 " + lsq6_MNC_path + "${SPEC}_lsq6.mnc " + lsq12_Blurred_path + "${SPEC}_016\n\n")
    # Three-level hierarchical 12-DOF registration to the FIXED atlas.
    f.write(lsq12_Register_023_Blur
        + lsq12_Blurred_path + "${SPEC}_023_blur.mnc " + LM_Avg_023_Blur + " "
        + lsq12_XFM_path + "${SPEC}_lsq12_0.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -identity\n")
    f.write(lsq12_Register_018_Blur
        + lsq12_Blurred_path + "${SPEC}_018_blur.mnc " + LM_Avg_018_Blur + " "
        + lsq12_XFM_path + "${SPEC}_lsq12_1.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transformation " + lsq12_XFM_path + "${SPEC}_lsq12_0.xfm\n")
    f.write(lsq12_Register_016_Blur
        + lsq12_Blurred_path + "${SPEC}_016_blur.mnc " + LM_Avg_016_Blur + " "
        + lsq12_XFM_path + "${SPEC}_lsq12_2.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transformation " + lsq12_XFM_path + "${SPEC}_lsq12_1.xfm\n\n")
    # Concatenate LSQ6 + final LSQ12 into a single rigid+affine transform.
    f.write("xfmconcat -clobber "
        + lsq6_XFM_path  + "${SPEC}_lsq6_2.xfm "
        + lsq12_XFM_path + "${SPEC}_lsq12_2.xfm "
        + lsq12_XFM_path + "${SPEC}_origtolsq12.xfm\n")
    # Resample original specimen into atlas space (rigid+affine), mask.
    f.write("mincresample -like " + LM_Avg + " -clobber "
        + "-transformation " + lsq12_XFM_path + "${SPEC}_origtolsq12.xfm "
        + Source_MNC_path  + "${SPEC}.mnc "
        + lsq12_MNC_path   + "${SPEC}_lsq12_unmasked.mnc\n")
    f.write("mincmath -clobber -mult "
        + lsq12_MNC_path + "${SPEC}_lsq12_unmasked.mnc "
        + LM_Avg_Mask_Apply + " "
        + lsq12_MNC_path + "${SPEC}_lsq12.mnc\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# STAGE 3 - Nonlinear (subject -> atlas, NL1..NL4 chained)
# ===================================================================
# Per-specimen array job. Each task does ALL FOUR NL levels for one specimen.
# All NL levels register against the SAME atlas (different blurs); no template
# drift, no inter-specimen group step, no mincaverage between rounds.
#
# Each task:
#   1. Blurs LSQ12-aligned specimen at 0.183, 0.137, 0.091, 0.046.
#   2. NL1: register at 0.183 vs atlas, store nl_1.xfm (-identity init).
#   3. NL2: register at 0.137 vs atlas, init from nl_1.xfm.
#   4. NL3: register at 0.091 vs atlas, init from nl_2.xfm.
#   5. NL4: register at 0.046 vs atlas, init from nl_3.xfm.
#   6. Concatenate LSQ6 + LSQ12 + NL4 into origtonl_4.xfm.
#   7. Resample ORIGINAL specimen with origtonl_4 and apply mask.
#
# Note: minctracc with -transform <init.xfm> emits the cumulative transform,
# so nl_4.xfm already incorporates nl_1..nl_3. The final xfmconcat only
# needs lsq6_2 + lsq12_2 + nl_4 (matches the construction pipeline's pattern).

with open("nl_specimen_array.sh", 'w') as f:
    f.write(slurm_header(
        nl_Mem, nl_Time,
        job_name="nl_subject2atlas",
        array="0-" + str(Specimen_List_Length - 1) + "%" + str(Array_Throttle)))
    f.write(specimen_array_preamble())
    # Blur LSQ12-aligned specimen at all four NL levels.
    f.write(MNC_Blur + "0.183 " + lsq12_MNC_path + "${SPEC}_lsq12.mnc " + nl_Blurred_path + "${SPEC}_183\n")
    f.write(MNC_Blur + "0.137 " + lsq12_MNC_path + "${SPEC}_lsq12.mnc " + nl_Blurred_path + "${SPEC}_137\n")
    f.write(MNC_Blur + "0.091 " + lsq12_MNC_path + "${SPEC}_lsq12.mnc " + nl_Blurred_path + "${SPEC}_091\n")
    f.write(MNC_Blur + "0.046 " + lsq12_MNC_path + "${SPEC}_lsq12.mnc " + nl_Blurred_path + "${SPEC}_046\n\n")
    # NL1: -identity init.
    f.write(nl_1_Register_Begin
        + nl_Blurred_path + "${SPEC}_183_blur.mnc " + LM_Avg_183_Blur + " "
        + nl_XFM_path     + "${SPEC}_nl_1.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " "
        + nl_1_Register_End + "\n")
    # NL2: -transform from nl_1.xfm.
    f.write(nl_2_Register_Begin
        + nl_Blurred_path + "${SPEC}_137_blur.mnc " + LM_Avg_137_Blur + " "
        + nl_XFM_path     + "${SPEC}_nl_2.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " "
        + nl_2_Register_End + nl_XFM_path + "${SPEC}_nl_1.xfm\n")
    # NL3: -transform from nl_2.xfm.
    f.write(nl_3_Register_Begin
        + nl_Blurred_path + "${SPEC}_091_blur.mnc " + LM_Avg_091_Blur + " "
        + nl_XFM_path     + "${SPEC}_nl_3.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " "
        + nl_3_Register_End + nl_XFM_path + "${SPEC}_nl_2.xfm\n")
    # NL4 (final): -transform from nl_3.xfm.
    f.write(nl_4_Register_Begin
        + nl_Blurred_path + "${SPEC}_046_blur.mnc " + LM_Avg_046_Blur + " "
        + nl_XFM_path     + "${SPEC}_nl_4.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " "
        + nl_4_Register_End + nl_XFM_path + "${SPEC}_nl_3.xfm\n\n")
    # Concatenate the full chain: LSQ6 + LSQ12 + NL4.
    f.write("xfmconcat -clobber "
        + lsq6_XFM_path  + "${SPEC}_lsq6_2.xfm "
        + lsq12_XFM_path + "${SPEC}_lsq12_2.xfm "
        + nl_XFM_path    + "${SPEC}_nl_4.xfm "
        + nl_XFM_path    + "${SPEC}_origtonl_4.xfm\n")
    # Resample original specimen with the full chain, mask.
    f.write("mincresample -like " + LM_Avg + " -clobber "
        + "-transformation " + nl_XFM_path + "${SPEC}_origtonl_4.xfm "
        + Source_MNC_path  + "${SPEC}.mnc "
        + nl_MNC_path      + "${SPEC}_nl_4_unmasked.mnc\n")
    f.write("mincmath -clobber -mult "
        + nl_MNC_path + "${SPEC}_nl_4_unmasked.mnc "
        + LM_Avg_Mask_Apply + " "
        + nl_MNC_path + "${SPEC}_nl_4.mnc\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# OPTIONAL - Final expanded-atlas average (Goal B)
# ===================================================================
# This averages just the new specimens. To build a true expanded atlas you
# typically include the original cohort's nl_4.mnc files as well; edit the
# script after generation and append those paths to the mincaverage list.
# This is NOT chained from the Job_Submission scripts - run by hand only if
# you want it.

with open("optional_average_new.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, "02:00:00", job_name="optional_avg_new"))
    f.write("echo \"Averaging the NEW NL4 specimens. (Optional Goal-B step.)\"\n")
    f.write("echo \"To build an expanded atlas including the original cohort, edit\"\n")
    f.write("echo \"this script and append the original NL4 .mnc paths to the list.\"\n\n")
    avg_cmd = MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += nl_MNC_path + SpecID + "_nl_4.mnc "
    avg_cmd += PROJECT_PATH + PROJECT_NAME + "_new_average.mnc"
    f.write(avg_cmd + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# JOB SUBMISSION CHAIN
# ===================================================================
# Three stages, one chain script per stage so you can restart at any point.
# Each chain script waits (-W) on its array, then submits the next chain.

# --- Job_Submission_First: atlas blur + LSQ6 array ---
with open("Job_Submission_First.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=" + SLURM_Account + "\n")
    f.write("#SBATCH --nodes=" + n_nodes + "\n")
    f.write("#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n")
    f.write("#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("# 1) One-time atlas blurring + apply-mask generation.\n")
    f.write("sbatch -W atlas_blur.sh\n\n")
    f.write("# 2) LSQ6 subject->atlas array (one task per specimen).\n")
    f.write("sbatch -W lsq6_specimen_array.sh\n\n")
    f.write("# 3) Chain to next stage.\n")
    f.write("sbatch Job_Submission_Second.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_First.sh\n")

# --- Job_Submission_Second: LSQ12 array ---
with open("Job_Submission_Second.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=" + SLURM_Account + "\n")
    f.write("#SBATCH --nodes=" + n_nodes + "\n")
    f.write("#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n")
    f.write("#SBATCH --job-name=Job_Submission_Second.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("# LSQ12 subject->atlas array (no pairwise, no xfmavg).\n")
    f.write("sbatch -W lsq12_specimen_array.sh\n\n")
    f.write("sbatch Job_Submission_Third.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Second.sh\n")

# --- Job_Submission_Third: NL array (final stage) ---
with open("Job_Submission_Third.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=" + SLURM_Account + "\n")
    f.write("#SBATCH --nodes=" + n_nodes + "\n")
    f.write("#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n")
    f.write("#SBATCH --job-name=Job_Submission_Third.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("# Full nonlinear chain (NL1..NL4) per specimen against fixed atlas.\n")
    f.write("sbatch -W nl_specimen_array.sh\n\n")
    f.write("echo \"Subject-to-atlas registration complete.\"\n")
    f.write("echo \"Final per-specimen warped volumes: " + nl_MNC_path + "<SPEC>_nl_4.mnc\"\n")
    f.write("echo \"Concatenated transforms:           " + nl_XFM_path + "<SPEC>_origtonl_4.xfm\"\n")
    f.write("echo \"Optional next step: sbatch optional_average_new.sh (Goal B).\"\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Third.sh\n")

# ===================================================================
# DONE
# ===================================================================

print("\nScript generation complete.")
print(f"  Specimens to register: {Specimen_List_Length}")
print(f"  Atlas (fixed target):  {LM_Avg}")
print(f"  Atlas reg mask:        {LM_Avg_Mask}")
print(f"  Atlas apply mask:      {LM_Avg_Mask_Apply}  (auto-generated)")
print()
print("Generated SLURM scripts:")
print("  atlas_blur.sh                 (one-time: 9 atlas blurs + apply-mask)")
print("  lsq6_specimen_array.sh        (per-specimen rigid -> atlas)")
print("  lsq12_specimen_array.sh       (per-specimen 12-DOF -> atlas; NO pairwise)")
print("  nl_specimen_array.sh          (per-specimen NL1..NL4 -> atlas)")
print("  optional_average_new.sh       (Goal B; run only if you want an expanded average)")
print("  Job_Submission_First.sh       (atlas blur + LSQ6)")
print("  Job_Submission_Second.sh      (LSQ12)")
print("  Job_Submission_Third.sh       (NL1..NL4)")
print()
print("Pipeline is fully linear in N. No pairwise registration, no xfmavg, no")
print("inter-stage averaging. The atlas is the fixed target throughout.")
print()
print("Final per-specimen output: " + nl_MNC_path + "<SPEC>_nl_4.mnc")
print("Final per-specimen xfm:    " + nl_XFM_path + "<SPEC>_origtonl_4.xfm")