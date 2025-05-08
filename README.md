# Headspace2HRTF
**Synthetic HRTF Dataset Generation Using Mesh2HRTF and the Headspace 3D Head Model Collection**

## Aim of the Project

The goal of this project is to generate a large-scale synthetic dataset of Head-Related Transfer Functions (HRTFs) using the Mesh2HRTF tool, applied to 3D head scans from the Headspace dataset. This dataset will support research in spatial audio, personalized HRTFs, and immersive virtual environments by enabling access to high-fidelity, individualized HRTF data.

## Resources

### Mesh2HRTF

Mesh2HRTF is a modular and scriptable open-source tool for numerically calculating HRTFs via the Boundary Element Method (BEM). It consists of:

* **Mesh2Input**: Prepares and conditions 3D meshes and acoustic parameters.
* **NumCalc**: Performs the heavy BEM simulation of sound fields around the head.
* **Output2HRTFs**: Converts raw simulation output into SOFA-format HRTFs.

It supports both MATLAB and Python workflows and is designed to handle complex geometries and acoustics simulations.

### Headspace Dataset

The Headspace dataset includes high-resolution 3D scans of 1,519 human heads captured using a 3dMD imaging system. Each subject file typically includes:

* `root.obj` (geometry)
* `root.bmp` (optional texture)
* `root.mtl` (material metadata)
* `subject.txt` (demographics)
* `landmarks.txt` (facial landmarks)

The dataset provides diverse anatomical data ideal for generating individualized HRTFs.

## Challenges

1. **Data Preprocessing**:

   * Normalize and reformat models for compatibility with Mesh2HRTF.
   * Patch holes, fix mesh defects, and truncate models at the neck to remove irrelevant geometry.

2. **Large-Scale Computation**:

   * With over 1,500 high-resolution models and complex acoustic simulations, the dataset poses significant computational demands.
   * Each HRTF computation is memory-intensive and time-consuming, even with parallelization.

3. **Supercomputing Infrastructure**:

   * The project will be adapted to run on a supercomputer cluster using the SLURM job scheduling system.
   * Scripts will need to support distributed job submission, monitoring, and output aggregation.
   * Ensuring efficient use of RAM and compute time will be essential to complete the project within resource constraints.


## Expected Outcomes

1. **Public GitHub Repository**:

   * Includes a comprehensive `README.md` (in English) with setup instructions, SLURM job examples, and documentation.
   * Contains all scripts for preprocessing, simulation, and postprocessing.
   * May include template SLURM batch files for easy deployment on similar systems.

2. **Synthetic HRTF Dataset**:

   * Generated HRTFs in SOFA format for a large subset of the Headspace models.
   * Data will be organized and possibly compressed for easier distribution and use.

3. **Automated Processing Pipeline**:

   * A scalable and reproducible workflow for future HRTF generation using similar datasets and tools.
  
