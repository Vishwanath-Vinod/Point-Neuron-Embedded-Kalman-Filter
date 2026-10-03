# Point-Neuron-Embedded-Kalman-Filter


Official implementation of **“Point Neuron Embedded Kalman Filter for Narrowband Sound Source Tracking,”** published in the proceedings of the 34th European Signal Processing Conference 2026, Brussels, Belgium.

Developed by **Vishwanath Vinod** under the guidance of **Prof. Thushara Abhayapal** as part of the Future Research Talent internship at the Australian National University.

## Paper

* **Authors:** Vishwanath Vinod, Shaoheng Xu, Prasanga Samarasinghe, Amy Bastine, Thushara D. Abhayapala
* **Conference:** 34th European Signal Processing Conference, August 31st- September 4th 2026
* **Publication:** IEEE Explore
* **Paper:** [EUSIPCO 2026 Proceedings](https://eurasip.org/Proceedings/Eusipco/Eusipco2026/pdfs/0000016.pdf)

## Overview

We propose the Point Neuron Embedded Kalman Filter (PNEKF), a novel Kalman Filter variant for tracking multiple narrowband sound sources in reverberant environments. Unlike existing methods that rely on discrete, grid-based source localization in each time frame, PNEKF requires localization only at initialization and thereafter operates directly on raw microphone signals by integrating the Point Neuron framework with Kalman filtering. This enables efficient, continuous 3-D tracking. We benchmark the performance of PNEKF against conventional tracking algorithms across a range of frequencies and source trajectories for the single-source case. The results demonstrate that PNEKF consistently achieves higher tracking accuracy than the baselines under all tested conditions. Finally, we demonstrate the robustness of the framework by extending it to track multiple sources in reverberant environments.

## Algorithm

<p align="center">
  <img src="Algorithm.png" alt="Point Neuron Embedded Kalman Filter Algorithm" width="800">
</p>

*Overview of the proposed Point Neuron Embedded Kalman Filter (PNEKF).*

## Experimental Setup

The PNEKF is evaluated for **narrowband sound-source tracking in a 3D acoustic environment**. The experiments use simulated microphone measurements generated for moving sound sources under different source configurations and noise conditions.

The implementation includes experiments for:

* Single- and multiple-source localization and tracking
* Point Neuron Learning (PNL)-based sound-field reconstruction
* Extended Kalman Filter (EKF)-based tracking
* Multi-Model Kalman Filter (MMKF)-based tracking
* OMP-based source initialization and localization
* TDOA-SRP-based localization

The simulation and data-generation scripts are provided under `Data/MatLab Codes/`.

## Repository Structure

```text
.
├── Algorithm.png
├── Codes
│   ├── KF+OMP
│   │   ├── kf+omp.py
│   │   ├── kf+omp_multiple.py
│   │   └── kf.py
│   ├── Localize
│   │   ├── OMP
│   │   │   ├── error_plot.py
│   │   │   ├── initialization_omp.py
│   │   │   ├── localize_omp.py
│   │   │   ├── localize_omp_multiple.py
│   │   │   ├── omp.py
│   │   │   ├── omp_grids.py
│   │   │   └── omp_track.py
│   │   └── TDOA_SRP
│   │       ├── initialization.py
│   │       ├── localize.py
│   │       └── localizer.py
│   ├── MMKF
│   │   ├── mmkf+omp_multiple.py
│   │   ├── mmkf+omp_track.py
│   │   ├── mmkf.py
│   │   └── mmkf_track.py
│   ├── Point_Neuron_Learning
│   │   ├── helper.py
│   │   └── point_neuron.py
│   ├── plot.py
│   ├── plot_multiple.py
│   ├── pnekf.py
│   ├── track_ms+omp.py
│   ├── track_ms.py
│   └── track_ms_multiple.py
├── Data
│   └── MatLab Codes
│       ├── arrange_mics_on_boundary.m
│       ├── circular_moving_source.m
│       ├── known_sources_pn2.m
│       ├── moving_source.m
│       ├── omp.m
│       ├── plotcube.m
│       ├── sample_circular_region.m
│       └── sample_spherical_region.m
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Installation

Clone the repository and install the required Python dependencies:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
pip install -r requirements.txt
```

### Running the Code

The main implementations are located in the `Codes/` directory. Run the required training, localization, or tracking script according to the desired experiment.

For example:

```bash
python Codes/pnekf.py
```

The experimental parameters, including source configuration, microphone setup, frequency, noise level, and tracking settings, can be modified directly in the corresponding scripts.

For source localization, the implementations under:

```text
Codes/Localize/
```

provide OMP- and TDOA-SRP-based approaches.

The Kalman-filter-based tracking implementations are available under:

```text
Codes/KF+OMP/
Codes/MMKF/
```

The Point Neuron Learning implementation is available under:

```text
Codes/Point_Neuron_Learning/
```

The MATLAB scripts required for generating and visualizing the simulation data are provided under:

```text
Data/MatLab Codes/
```

Run the scripts using the configurations specified in the respective files.


## Citation

If you use this work, please cite:

```bibtex
@article{vinodpoint,
  title={Point Neuron Embedded Kalman Filter for Narrowband Sound Source Tracking},
  author={Vinod, Vishwanath and Xu, Shaoheng and Samarasinghe, Prasanga N and Bastine, Amy and Abhayapala, Thushara D}
}
