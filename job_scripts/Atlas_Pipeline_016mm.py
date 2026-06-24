# Atlas_Pipeline_016mm.py
# ========================
# Generates SLURM .sh scripts for an unbiased population average at 16 um
# (0.0159999 mm) resolution using the MINC Toolkit.
#
# Retuned from the original 24 um pipeline (scale factor 0.6666625).
# LSQ12 blur values clamped so no FWHM < 0.016 mm (the voxel size).
# File basenames renamed to reflect actual FWHM values.
#
# OPTIMISATION CHANGES (vs. earlier 016mm version):
#   - All files opened in 'w' mode (not 'a') to avoid silent duplication.
#   - Masks are used as binary volumes; no mask blurring.
#   - -source_mask added to all minctracc calls.
#   - -debug removed from all production minctracc calls.
#   - Linear tolerances relaxed: 0.005 / 0.002 / 0.001 per level.
#   - Nonlinear iterations reduced: NL1=20, NL2=10, NL3=8, NL4=6.
#   - NL5 refinement stage dropped entirely.
#   - LSQ12 pairwise stage uses a single SLURM array job (no 3-s sleeps).
#   - Self-to-self LSQ12 pairs are skipped (N*(N-1) jobs, not N*N).
#   - mincaverage uses -float instead of -filetype for higher-quality averages.
#   - -gradient removed from mincblur (outputs never referenced).
#
# IMPORTANT: LM_average.mnc and LM_average_mask.mnc must be on the
# 0.0159999 mm grid. For maximum speed, autocrop them to the tightest
# useful bounding box before running (every stage resamples -like LM_Avg).

import os
import csv

# ===================================================================
# USER-EDITABLE PARAMETERS
# ===================================================================

PROJECT_NAME = input("Enter project name (e.g., Fgf8): ")
CLUSTER_PATH = input("Enter project path on cluster (e.g., /work/hallgrimsson_lab/): ")
PROJECT_PATH = CLUSTER_PATH + PROJECT_NAME + "/"

print("\nCreate directory structure on cluster. E.g.:\n")
print(f"mkdir -p {PROJECT_PATH}{{Scripts,Quality,Source/{{Blurred,MNC,Orig,Resample,Tag,XFM}},lsq6/{{Blurred,MNC,XFM}},lsq12/{{Blurred,MNC,XFM}},nl/{{Ana_Test,Blurred,INIT,MNC,XFM}}}}\n")

LOCAL_SCRIPT_PATH = input("Enter path to local scripts directory: ")
os.chdir(LOCAL_SCRIPT_PATH)

All_Specimens = input("Enter path to specimen list: ")

# Cluster parameters
Module = "minc/1.9.18"
n_nodes = "1"
lsq6_Time = "07:00:00"
lsq6_Mem = "30000M"
lsq12_Time = "48:00:00"
lsq12_Mem = "30000M"
nl_Time = "36:00:00"
nl_Mem = "30000M"
Job_Submission_Time = "05-00:00:00"
Job_Submission_Mem = "2000M"
# Maximum concurrent SLURM array tasks for LSQ12 pairwise stage.
lsq12_Array_Throttle = 50

print("\nDefault compute cluster parameters:")
print(f"  Module: {Module}")
print(f"  Nodes: {n_nodes}")
print(f"  lsq6  Time/Mem: {lsq6_Time} / {lsq6_Mem}")
print(f"  lsq12 Time/Mem: {lsq12_Time} / {lsq12_Mem}")
print(f"  nl    Time/Mem: {nl_Time} / {nl_Mem}")
print(f"  Job Submission Time/Mem: {Job_Submission_Time} / {Job_Submission_Mem}")
print(f"  LSQ12 array throttle: {lsq12_Array_Throttle}\n")
print(f"Upload scripts to {PROJECT_PATH}Scripts and image data to {PROJECT_PATH}Source/MNC/.\n")

# ===================================================================
# PATH DEFINITIONS
# ===================================================================

Scripts_path      = PROJECT_PATH + "Scripts/"
Source_MNC_path   = PROJECT_PATH + "Source/MNC/"
lsq6_path         = PROJECT_PATH + "lsq6/"
lsq6_Blurred_path = PROJECT_PATH + "lsq6/Blurred/"
lsq6_XFM_path     = PROJECT_PATH + "lsq6/XFM/"
lsq6_MNC_path     = PROJECT_PATH + "lsq6/MNC/"
lsq12_path        = PROJECT_PATH + "lsq12/"
lsq12_Blurred_path = PROJECT_PATH + "lsq12/Blurred/"
lsq12_XFM_path    = PROJECT_PATH + "lsq12/XFM/"
lsq12_MNC_path    = PROJECT_PATH + "lsq12/MNC/"
nl_Init_path      = PROJECT_PATH + "nl/INIT/"
nl_Blurred_path   = PROJECT_PATH + "nl/Blurred/"
nl_XFM_path       = PROJECT_PATH + "nl/XFM/"
nl_MNC_path       = PROJECT_PATH + "nl/MNC/"

# ===================================================================
# AVERAGE AND MASK FILES
# ===================================================================

LM_Avg      = Source_MNC_path + "LM_average.mnc"
LM_Avg_Mask = Source_MNC_path + "LM_average_mask.mnc"
LM_Avg_Mask_Apply = Source_MNC_path + "LM_average_mask_apply.mnc"
lsq6_Avg    = lsq6_path + PROJECT_NAME + "_lsq6_average.mnc"
lsq12_Avg   = lsq12_path + PROJECT_NAME + "_lsq12_average.mnc"
nl_1_Avg    = nl_MNC_path + "NL_1_average.mnc"
nl_2_Avg    = nl_MNC_path + "NL_2_average.mnc"
nl_3_Avg    = nl_MNC_path + "NL_3_average.mnc"
nl_4_Avg    = nl_MNC_path + PROJECT_NAME + "_Atlas.mnc"   # Final atlas

# ===================================================================
# BLURRED AVERAGE FILES (images only – masks stay binary)
# Scaled from 24 um → 16 um (×0.6666625).
# LSQ12 schedule clamped so no FWHM < 0.016 mm.
# ===================================================================

# --- LSQ6 average blurs (targets for rigid registration) ---
LM_Avg_077      = Source_MNC_path + "LM_average_077"
LM_Avg_040      = Source_MNC_path + "LM_average_040"
LM_Avg_018      = Source_MNC_path + "LM_average_018"
LM_Avg_077_Blur = Source_MNC_path + "LM_average_077_blur.mnc"
LM_Avg_040_Blur = Source_MNC_path + "LM_average_040_blur.mnc"
LM_Avg_018_Blur = Source_MNC_path + "LM_average_018_blur.mnc"

# --- NL average blurs (targets for nonlinear registration) ---
lsq12_Avg_183      = nl_Init_path + PROJECT_NAME + "_lsq12_average_183"
lsq12_Avg_183_Blur = nl_Init_path + PROJECT_NAME + "_lsq12_average_183_blur.mnc"
nl_1_Avg_137       = nl_Init_path + "NL_1_average_137"
nl_1_Avg_137_Blur  = nl_Init_path + "NL_1_average_137_blur.mnc"
nl_2_Avg_091       = nl_Init_path + "NL_2_average_091"
nl_2_Avg_091_Blur  = nl_Init_path + "NL_2_average_091_blur.mnc"
nl_3_Avg_046       = nl_Init_path + "NL_3_average_046"
nl_3_Avg_046_Blur  = nl_Init_path + "NL_3_average_046_blur.mnc"

# ===================================================================
# SPECIMEN LIST
# ===================================================================

Specimen_List = open(All_Specimens, 'r')
Specimens = Specimen_List.read()
Specimen_IDs = Specimens.split('\n')
del Specimen_IDs[-1]
Specimen_List_Length = len(Specimen_IDs)
Specimen_Group = int(Specimen_List_Length / Specimen_List_Length)  # Always 1
# Number of pairwise LSQ12 jobs, excluding self-pairs.
Num_LSQ12_Pairs = Specimen_List_Length * (Specimen_List_Length - 1)
Specimen_List.close()

print(f"Specimen count: {Specimen_List_Length}")
print(f"LSQ12 pairwise jobs (excl. self): {Num_LSQ12_Pairs}\n")

# ===================================================================
# MINCBLUR STRING (no -gradient, saves disk + memory)
# ===================================================================

MNC_Blur = "mincblur -clobber -no_apodize -fwhm "

# ===================================================================
# REGISTRATION STRINGS – scaled for 16 um
# Tolerances relaxed: 0.005 / 0.002 / 0.001 per hierarchical level.
# All use binary -model_mask and -source_mask (LM_Avg_Mask).
# No -debug on any production call.
# ===================================================================

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

# --- mincaverage: -float for higher-quality intermediate averages ---
lsq6_MNC_Avg  = "mincaverage -clobber -2 -float -nonormalize "
lsq12_MNC_Avg = "mincaverage -clobber -2 -float -nonormalize "
nl_1_MNC_Avg  = "mincaverage -clobber -2 -float -nonormalize "
nl_2_MNC_Avg  = "mincaverage -clobber -2 -float -nonormalize "
nl_3_MNC_Avg  = "mincaverage -clobber -2 -float -nonormalize "
nl_4_MNC_Avg  = "mincaverage -clobber -2 -float -nonormalize "

# ===================================================================
# HELPER: SLURM header
# ===================================================================
def slurm_header(mem, time, module=Module, job_name=None):
    h = "#!/bin/bash\n"
    h += "#SBATCH --nodes=" + n_nodes + "\n"
    h += "#SBATCH --mem=" + mem + "\n"
    h += "#SBATCH --time=" + time + "\n"
    if job_name:
        h += "#SBATCH --job-name=" + job_name + "\n"
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

# ===================================================================
# LSQ6 – 6-parameter rigid body registration
# ===================================================================

# --- lsq6_First.sh: blur the LM average (target) only ---
# Masks are used as binary; no mask blurring needed.
with open("lsq6_First.sh", 'w') as f:
    f.write(slurm_header(lsq6_Mem, lsq6_Time))
    f.write("echo \"Blurring LM_average at three LSQ6 levels.\"\n\n")
    f.write(MNC_Blur + "0.077 " + LM_Avg + " " + LM_Avg_077 + "\n")
    f.write(MNC_Blur + "0.040 " + LM_Avg + " " + LM_Avg_040 + "\n")
    f.write(MNC_Blur + "0.018 " + LM_Avg + " " + LM_Avg_018 + "\n\n")
    f.write("mincmorph -clobber -successive DD " + LM_Avg_Mask + " " + LM_Avg_Mask_Apply + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# --- lsq6_Second_*.sh: blur each specimen, register, resample ---
for i, SpecID in enumerate(Specimen_IDs, start=1):
    with open("lsq6_Second_" + str(i) + ".sh", 'w') as f:
        f.write(slurm_header(lsq6_Mem, lsq6_Time))
        f.write("echo \"LSQ6 registration for " + SpecID + ".\"\n\n")
        # Blur specimen at three levels.
        f.write(MNC_Blur + "0.077 " + Source_MNC_path + SpecID + ".mnc " + lsq6_Blurred_path + SpecID + "_077\n")
        f.write(MNC_Blur + "0.040 " + Source_MNC_path + SpecID + ".mnc " + lsq6_Blurred_path + SpecID + "_040\n")
        f.write(MNC_Blur + "0.018 " + Source_MNC_path + SpecID + ".mnc " + lsq6_Blurred_path + SpecID + "_018\n\n")
        # Hierarchical registration: coarse → fine.
        f.write(lsq6_Register_077_Blur + lsq6_Blurred_path + SpecID + "_077_blur.mnc " + LM_Avg_077_Blur + " " + lsq6_XFM_path + SpecID + "_lsq6_0.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -identity\n")
        f.write(lsq6_Register_040_Blur + lsq6_Blurred_path + SpecID + "_040_blur.mnc " + LM_Avg_040_Blur + " " + lsq6_XFM_path + SpecID + "_lsq6_1.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -transformation " + lsq6_XFM_path + SpecID + "_lsq6_0.xfm\n")
        f.write(lsq6_Register_018_Blur + lsq6_Blurred_path + SpecID + "_018_blur.mnc " + LM_Avg_018_Blur + " " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -transformation " + lsq6_XFM_path + SpecID + "_lsq6_1.xfm\n\n")
        # Resample into rigid-body-aligned space.
        f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + Source_MNC_path + SpecID + ".mnc " + lsq6_MNC_path + SpecID + "_lsq6_unmasked.mnc\n")
        f.write("mincmath -clobber -mult " + lsq6_MNC_path + SpecID + "_lsq6_unmasked.mnc " + LM_Avg_Mask_Apply + " " + lsq6_MNC_path + SpecID + "_lsq6.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")

# --- lsq6_Third.sh: average all lsq6 images ---
with open("lsq6_Third.sh", 'w') as f:
    f.write(slurm_header(lsq6_Mem, lsq6_Time))
    f.write("echo \"Averaging all lsq6 files.\"\n\n")
    avg_cmd = lsq6_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += lsq6_MNC_path + SpecID + "_lsq6.mnc "
    f.write(avg_cmd + lsq6_Avg + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# LSQ12 – 12-parameter affine registration (pairwise)
# ===================================================================

# --- lsq12_First.sh: no-op (masks are binary, avg blurs unused) ---
# The LSQ12 pairwise stage registers specimens to each other, not to
# LM_average, so we only need the specimen blurs (done in lsq12_Second).
# This script is retained so the job submission chain doesn't need restructuring.
with open("lsq12_First.sh", 'w') as f:
    f.write(slurm_header(lsq12_Mem, lsq12_Time))
    f.write("echo \"LSQ12 uses pairwise specimen registration; no average blurring needed.\"\n")
    f.write("echo \"Masks are used as binary volumes; no mask blurring needed.\"\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# --- lsq12_Second_*.sh: blur each lsq6 specimen at LSQ12 levels ---
for i, SpecID in enumerate(Specimen_IDs, start=1):
    with open("lsq12_Second_" + str(i) + ".sh", 'w') as f:
        f.write(slurm_header(lsq12_Mem, lsq12_Time))
        f.write("echo \"Blurring " + SpecID + " at LSQ12 levels.\"\n\n")
        f.write(MNC_Blur + "0.023 " + lsq6_MNC_path + SpecID + "_lsq6.mnc " + lsq12_Blurred_path + SpecID + "_023\n")
        f.write(MNC_Blur + "0.018 " + lsq6_MNC_path + SpecID + "_lsq6.mnc " + lsq12_Blurred_path + SpecID + "_018\n")
        f.write(MNC_Blur + "0.016 " + lsq6_MNC_path + SpecID + "_lsq6.mnc " + lsq12_Blurred_path + SpecID + "_016\n\n")
        f.write("echo \"The job ended at $(date).\"\n")

# --- lsq12_Third_Array.sh: SLURM array for all pairwise registrations ---
# Each array task maps to one (source, target) pair, skipping self-pairs.
# Task ID ∈ [0, N*(N-1)-1].  Mapping: src = ID/(N-1), tgt_offset = ID%(N-1),
# tgt = tgt_offset if tgt_offset < src else tgt_offset+1.
spec_bash_array = " ".join('"' + s + '"' for s in Specimen_IDs)

with open("lsq12_Third_Array.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes=" + n_nodes + "\n")
    f.write("#SBATCH --mem=" + lsq12_Mem + "\n")
    f.write("#SBATCH --time=" + lsq12_Time + "\n")
    f.write("#SBATCH --array=0-" + str(Num_LSQ12_Pairs - 1) + "%" + str(lsq12_Array_Throttle) + "\n")
    f.write("#SBATCH --job-name=lsq12_pairwise\n\n")
    f.write('set -euo pipefail\n')
    f.write('export PATH="/ix1/rgreen/mdj72/miniconda3/envs/mdj_env/bin:$PATH"\n')
    f.write('export MINC_TOOLKIT=/ihome/rgreen/mdj72/minc-toolkit_v2/opt/minc/1.9.18\n')
    f.write('export PATH="$MINC_TOOLKIT/bin:$MINC_TOOLKIT/pipeline:$PATH"\n')
    f.write('export LD_LIBRARY_PATH="$MINC_TOOLKIT/lib:${LD_LIBRARY_PATH:-}"\n')
    f.write('export MINC_FORCE_V2=1\n')
    f.write('export PERL5LIB="$MINC_TOOLKIT/perl:${PERL5LIB:-}"\n\n')
    f.write("cd " + Scripts_path + "\n\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("# Specimen list embedded from the generator.\n")
    f.write("SPECS=(" + spec_bash_array + ")\n")
    f.write("N=${#SPECS[@]}\n\n")
    f.write("# Map SLURM_ARRAY_TASK_ID to (source, target) pair, skipping diagonal.\n")
    f.write("TASK=$SLURM_ARRAY_TASK_ID\n")
    f.write("SRC_IDX=$((TASK / (N - 1)))\n")
    f.write("TGT_OFFSET=$((TASK % (N - 1)))\n")
    f.write("if [ $TGT_OFFSET -ge $SRC_IDX ]; then\n")
    f.write("    TGT_IDX=$((TGT_OFFSET + 1))\n")
    f.write("else\n")
    f.write("    TGT_IDX=$TGT_OFFSET\n")
    f.write("fi\n\n")
    f.write("SRC=${SPECS[$SRC_IDX]}\n")
    f.write("TGT=${SPECS[$TGT_IDX]}\n\n")
    f.write("echo \"Registering $SRC to $TGT (task $TASK)\"\n\n")
    # Three-level hierarchical affine registration.
    f.write(lsq12_Register_023_Blur
        + lsq12_Blurred_path + "${SRC}_023_blur.mnc "
        + lsq12_Blurred_path + "${TGT}_023_blur.mnc "
        + lsq12_XFM_path + "${SRC}_to_${TGT}_lsq12_0.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " -identity\n")
    f.write(lsq12_Register_018_Blur
        + lsq12_Blurred_path + "${SRC}_018_blur.mnc "
        + lsq12_Blurred_path + "${TGT}_018_blur.mnc "
        + lsq12_XFM_path + "${SRC}_to_${TGT}_lsq12_1.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transform " + lsq12_XFM_path + "${SRC}_to_${TGT}_lsq12_0.xfm\n")
    f.write(lsq12_Register_016_Blur
        + lsq12_Blurred_path + "${SRC}_016_blur.mnc "
        + lsq12_Blurred_path + "${TGT}_016_blur.mnc "
        + lsq12_XFM_path + "${SRC}_to_${TGT}_lsq12_2.xfm "
        + "-model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask
        + " -transform " + lsq12_XFM_path + "${SRC}_to_${TGT}_lsq12_1.xfm\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# --- lsq12_Fourth_*.sh: xfmavg, concat, resample (self-pairs skipped) ---
for Element in range(Specimen_List_Length):
    with open("lsq12_Fourth_" + str(Element) + ".sh", 'w') as f:
        f.write(slurm_header(lsq12_Mem, lsq12_Time))
        for SpecID in Specimen_IDs[(Specimen_Group * Element):(Specimen_Group * (Element + 1))]:
            # Build xfmavg command, skipping self-to-self.
            XFM_Avg = "xfmavg -verbose -clobber "
            for SpecID2 in Specimen_IDs:
                if SpecID2 != SpecID:
                    XFM_Avg += lsq12_XFM_path + SpecID + "_to_" + SpecID2 + "_lsq12_2.xfm "
            f.write(XFM_Avg + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm\n")
            f.write("xfmconcat -clobber " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm " + lsq12_XFM_path + SpecID + "_origtolsq12.xfm\n")
            f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + lsq12_XFM_path + SpecID + "_origtolsq12.xfm " + Source_MNC_path + SpecID + ".mnc " + lsq12_MNC_path + SpecID + "_lsq12_unmasked.mnc\n")
            f.write("mincmath -clobber -mult " + lsq12_MNC_path + SpecID + "_lsq12_unmasked.mnc " + LM_Avg_Mask_Apply + " " + lsq12_MNC_path + SpecID + "_lsq12.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")

# --- lsq12_Fifth.sh: average all lsq12 images ---
with open("lsq12_Fifth.sh", 'w') as f:
    f.write(slurm_header(lsq12_Mem, lsq12_Time))
    f.write("echo \"Averaging all lsq12 files.\"\n\n")
    avg_cmd = lsq12_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += lsq12_MNC_path + SpecID + "_lsq12.mnc "
    f.write(avg_cmd + lsq12_Avg + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# NL1 – First nonlinear stage
# ===================================================================

# --- nl_First.sh: blur the lsq12 average (mask stays binary) ---
with open("nl_First.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, nl_Time))
    f.write(MNC_Blur + "0.183 " + lsq12_Avg + " " + lsq12_Avg_183 + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# --- nl_Second_*.sh: blur, register, concat, resample ---
for Element in range(Specimen_List_Length):
    with open("nl_Second_" + str(Element) + ".sh", 'w') as f:
        f.write(slurm_header(nl_Mem, nl_Time))
        for SpecID in Specimen_IDs[(Specimen_Group * Element):(Specimen_Group * (Element + 1))]:
            f.write(MNC_Blur + "0.183 " + lsq12_MNC_path + SpecID + "_lsq12.mnc " + nl_Blurred_path + SpecID + "_183\n")
            f.write(nl_1_Register_Begin + nl_Blurred_path + SpecID + "_183_blur.mnc " + lsq12_Avg_183_Blur + " " + nl_XFM_path + SpecID + "_nl_1.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " " + nl_1_Register_End + "\n")
            f.write("xfmconcat -clobber " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm " + nl_XFM_path + SpecID + "_nl_1.xfm " + nl_XFM_path + SpecID + "_origtonl_1.xfm\n")
            f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + nl_XFM_path + SpecID + "_origtonl_1.xfm " + Source_MNC_path + SpecID + ".mnc " + nl_MNC_path + SpecID + "_nl_1_unmasked.mnc\n")
            f.write("mincmath -clobber -mult " + nl_MNC_path + SpecID + "_nl_1_unmasked.mnc " + LM_Avg_Mask_Apply + " " + nl_MNC_path + SpecID + "_nl_1.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")

# --- nl_Third.sh: average NL1 images, blur for NL2 target ---
with open("nl_Third.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, nl_Time))
    f.write("echo \"Averaging NL1 images and blurring for NL2 target.\"\n\n")
    avg_cmd = nl_1_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += nl_MNC_path + SpecID + "_nl_1.mnc "
    f.write(avg_cmd + nl_1_Avg + "\n\n")
    f.write(MNC_Blur + "0.137 " + nl_1_Avg + " " + nl_1_Avg_137 + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# NL2 – Second nonlinear stage
# ===================================================================

for Element in range(Specimen_List_Length):
    with open("nl_Fourth_" + str(Element) + ".sh", 'w') as f:
        f.write(slurm_header(nl_Mem, nl_Time))
        for SpecID in Specimen_IDs[(Specimen_Group * Element):(Specimen_Group * (Element + 1))]:
            f.write(MNC_Blur + "0.137 " + lsq12_MNC_path + SpecID + "_lsq12.mnc " + nl_Blurred_path + SpecID + "_137\n")
            f.write(nl_2_Register_Begin + nl_Blurred_path + SpecID + "_137_blur.mnc " + nl_1_Avg_137_Blur + " " + nl_XFM_path + SpecID + "_nl_2.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " " + nl_2_Register_End + nl_XFM_path + SpecID + "_nl_1.xfm\n")
            f.write("xfmconcat -clobber " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm " + nl_XFM_path + SpecID + "_nl_2.xfm " + nl_XFM_path + SpecID + "_origtonl_2.xfm\n")
            f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + nl_XFM_path + SpecID + "_origtonl_2.xfm " + Source_MNC_path + SpecID + ".mnc " + nl_MNC_path + SpecID + "_nl_2_unmasked.mnc\n")
            f.write("mincmath -clobber -mult " + nl_MNC_path + SpecID + "_nl_2_unmasked.mnc " + LM_Avg_Mask_Apply + " " + nl_MNC_path + SpecID + "_nl_2.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")
        
# --- nl_Fifth.sh: average NL2 images, blur for NL3 target ---
with open("nl_Fifth.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, nl_Time))
    f.write("echo \"Averaging NL2 images and blurring for NL3 target.\"\n\n")
    avg_cmd = nl_2_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += nl_MNC_path + SpecID + "_nl_2.mnc "
    f.write(avg_cmd + nl_2_Avg + "\n\n")
    f.write(MNC_Blur + "0.091 " + nl_2_Avg + " " + nl_2_Avg_091 + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
 
# ===================================================================
# NL3 – Third nonlinear stage
# ===================================================================
 
for Element in range(Specimen_List_Length):
    with open("nl_Sixth_" + str(Element) + ".sh", 'w') as f:
        f.write(slurm_header(nl_Mem, nl_Time))
        for SpecID in Specimen_IDs[(Specimen_Group * Element):(Specimen_Group * (Element + 1))]:
            f.write(MNC_Blur + "0.091 " + lsq12_MNC_path + SpecID + "_lsq12.mnc " + nl_Blurred_path + SpecID + "_091\n")
            f.write(nl_3_Register_Begin + nl_Blurred_path + SpecID + "_091_blur.mnc " + nl_2_Avg_091_Blur + " " + nl_XFM_path + SpecID + "_nl_3.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " " + nl_3_Register_End + nl_XFM_path + SpecID + "_nl_2.xfm\n")
            f.write("xfmconcat -clobber " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm " + nl_XFM_path + SpecID + "_nl_3.xfm " + nl_XFM_path + SpecID + "_origtonl_3.xfm\n")
            f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + nl_XFM_path + SpecID + "_origtonl_3.xfm " + Source_MNC_path + SpecID + ".mnc " + nl_MNC_path + SpecID + "_nl_3_unmasked.mnc\n")
            f.write("mincmath -clobber -mult " + nl_MNC_path + SpecID + "_nl_3_unmasked.mnc " + LM_Avg_Mask_Apply + " " + nl_MNC_path + SpecID + "_nl_3.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")


# --- nl_Seventh.sh: average NL3 images, blur for NL4 target ---
with open("nl_Seventh.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, nl_Time))
    f.write("echo \"Averaging NL3 images and blurring for NL4 target.\"\n\n")
    avg_cmd = nl_3_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += nl_MNC_path + SpecID + "_nl_3.mnc "
    f.write(avg_cmd + nl_3_Avg + "\n\n")
    f.write(MNC_Blur + "0.046 " + nl_3_Avg + " " + nl_3_Avg_046 + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# NL4 – Fourth and FINAL nonlinear stage (NL5 dropped)
# ===================================================================

for Element in range(Specimen_List_Length):
    with open("nl_Eighth_" + str(Element) + ".sh", 'w') as f:
        f.write(slurm_header(nl_Mem, nl_Time))
        for SpecID in Specimen_IDs[(Specimen_Group * Element):(Specimen_Group * (Element + 1))]:
            f.write(MNC_Blur + "0.046 " + lsq12_MNC_path + SpecID + "_lsq12.mnc " + nl_Blurred_path + SpecID + "_046\n")
            f.write(nl_4_Register_Begin + nl_Blurred_path + SpecID + "_046_blur.mnc " + nl_3_Avg_046_Blur + " " + nl_XFM_path + SpecID + "_nl_4.xfm -model_mask " + LM_Avg_Mask + " -source_mask " + LM_Avg_Mask + " " + nl_4_Register_End + nl_XFM_path + SpecID + "_nl_3.xfm\n")
            f.write("xfmconcat -clobber " + lsq6_XFM_path + SpecID + "_lsq6_2.xfm " + lsq12_XFM_path + SpecID + "_lsq12_AVG.xfm " + nl_XFM_path + SpecID + "_nl_4.xfm " + nl_XFM_path + SpecID + "_origtonl_4.xfm\n")
            f.write("mincresample -like " + LM_Avg + " -clobber -transformation " + nl_XFM_path + SpecID + "_origtonl_4.xfm " + Source_MNC_path + SpecID + ".mnc " + nl_MNC_path + SpecID + "_nl_4_unmasked.mnc\n")
            f.write("mincmath -clobber -mult " + nl_MNC_path + SpecID + "_nl_4_unmasked.mnc " + LM_Avg_Mask_Apply + " " + nl_MNC_path + SpecID + "_nl_4.mnc\n\n")
        f.write("echo \"The job ended at $(date).\"\n")

# --- nl_Ninth.sh: average NL4 images → FINAL ATLAS ---
with open("nl_Ninth.sh", 'w') as f:
    f.write(slurm_header(nl_Mem, nl_Time))
    f.write("echo \"Averaging all NL4 warped images to produce the final atlas.\"\n\n")
    avg_cmd = nl_4_MNC_Avg
    for SpecID in Specimen_IDs:
        avg_cmd += nl_MNC_path + SpecID + "_nl_4.mnc "
    f.write(avg_cmd + nl_4_Avg + "\n\n")
    f.write("echo \"The job ended at $(date).\"\n")

# ===================================================================
# MASTER JOB SUBMISSION CHAIN
# ===================================================================

# --- Job_Submission_First: LSQ6 ---
with open("Job_Submission_First.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    # 1) Blur the average.
    f.write("sbatch -W lsq6_First.sh\n\n")
    # 2) Launch per-specimen blur+register jobs.
    f.write("NUMBERS=$(seq 1 " + str(Specimen_List_Length) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch lsq6_Second_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    # 3) Average.
    f.write("sbatch -W lsq6_Third.sh\n\n")
    # 4) Chain to next stage.
    f.write("sbatch Job_Submission_Second.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_First.sh\n")

# --- Job_Submission_Second: LSQ12 blurs ---
with open("Job_Submission_Second.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    # lsq12_First is a no-op but keeps the chain tidy.
    f.write("sbatch -W lsq12_First.sh\n\n")
    f.write("NUMBERS=$(seq 1 " + str(Specimen_List_Length) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch lsq12_Second_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("sbatch Job_Submission_Third.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Second.sh\n")

# --- Job_Submission_Third: LSQ12 pairwise (SLURM array) ---
with open("Job_Submission_Third.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    # Submit the entire pairwise stage as one array job and wait.
    f.write("# Submit LSQ12 pairwise array (" + str(Num_LSQ12_Pairs) + " tasks, self-pairs excluded).\n")
    f.write("sbatch -W lsq12_Third_Array.sh\n\n")
    f.write("sbatch Job_Submission_Fourth.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Third.sh\n")

# --- Job_Submission_Fourth: LSQ12 xfmavg + resample + average ---
with open("Job_Submission_Fourth.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("NUMBERS=$(seq 0 " + str(Specimen_List_Length - 1) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch lsq12_Fourth_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("sbatch -W lsq12_Fifth.sh\n\n")
    f.write("sbatch Job_Submission_Fifth.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Fourth.sh\n")

# --- Job_Submission_Fifth: NL1 ---
with open("Job_Submission_Fifth.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("sbatch -W nl_First.sh\n\n")
    f.write("NUMBERS=$(seq 0 " + str(Specimen_List_Length - 1) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch nl_Second_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("sbatch -W nl_Third.sh\n\n")
    f.write("sbatch Job_Submission_Sixth.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Fifth.sh\n")

# --- Job_Submission_Sixth: NL2 ---
with open("Job_Submission_Sixth.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("NUMBERS=$(seq 0 " + str(Specimen_List_Length - 1) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch nl_Fourth_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("sbatch -W nl_Fifth.sh\n\n")
    f.write("sbatch Job_Submission_Seventh.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Sixth.sh\n")

# --- Job_Submission_Seventh: NL3 ---
with open("Job_Submission_Seventh.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("NUMBERS=$(seq 0 " + str(Specimen_List_Length - 1) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch nl_Sixth_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("sbatch -W nl_Seventh.sh\n\n")
    f.write("sbatch Job_Submission_Eighth.sh\n\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Seventh.sh\n")

# --- Job_Submission_Eighth: NL4 (FINAL) ---
with open("Job_Submission_Eighth.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --account=mmarazita\n") 
    f.write("#SBATCH --nodes=" + n_nodes + "\n#SBATCH --mem=" + Job_Submission_Mem + "\n")
    f.write("#SBATCH --time=" + Job_Submission_Time + "\n#SBATCH --job-name=Job_Submission_First.sh\n\n")
    f.write("set -euo pipefail\n")
    f.write("echo \"The job started at $(date).\"\n\n")
    f.write("cd " + Scripts_path + "\n\nsleep 10\n\n")
    f.write("NUMBERS=$(seq 0 " + str(Specimen_List_Length - 1) + ")\n")
    f.write("for NUM in $NUMBERS; do\n")
    f.write("    sbatch nl_Eighth_$NUM.sh\n")
    f.write("    sleep 3\n")
    f.write("done\n\n")
    f.write("sleep 5\n")
    f.write("while [[ $(squeue -t PD -u $USER --noheader | wc -l) -gt 0 ]]; do sleep 5; done\n")
    f.write("while [[ $(squeue -t R  -u $USER --noheader | wc -l) -gt 1 ]]; do sleep 5; done\n\n")
    f.write("# Produce the final atlas.\n")
    f.write("sbatch -W nl_Ninth.sh\n\n")
    f.write("echo \"Pipeline complete. Final atlas: " + nl_4_Avg + "\"\n")
    f.write("echo \"The job ended at $(date).\"\n")
    f.write("scancel -u $USER --jobname=Job_Submission_Eighth.sh\n")

print("Script generation complete.")
print(f"Generated files for {Specimen_List_Length} specimens.")
print(f"LSQ12 pairwise: {Num_LSQ12_Pairs} array tasks (self-pairs excluded).")
print(f"Pipeline terminates after NL4. Final atlas: {PROJECT_NAME}_Atlas.mnc")