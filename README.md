# edge-ga-urban-routing
Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing — Dataset &amp; Code Repository
# Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing — Dataset & Code Repository

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.48084%2Fetasr.XXXX-blue)](https://doi.org/10.48084/etasr.XXXX)

## Overview

This repository accompanies the paper:

> **Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing**  
> Suhail Odeh, Mahmoud Obaid*, Rafik Lasari, Murad Al-Rajab, Hebatullah Khattab, Ammar Qarariyah, Djemel Ziou  
> *Engineering, Technology & Applied Science Research*, Vol. XX, No. X, 20XX  


It provides the real-world traffic dataset collected from Bethlehem City (Palestine), SUMO simulation scenarios, and the NSGA-II / Rolling-Horizon Optimization source code used in the study.

---

## Repository Structure

```
├── dataset/
│   ├── vehicle_trajectories.csv       # 200,000 vehicle trajectory records
│   ├── signal_logs.csv                # Signal log data (60,480 five-minute time slots)
│   ├── intersection_metadata.csv      # Metadata for 15 major intersections
│   └── simulation_scenarios.csv       # 60 SUMO simulation scenario configurations
├── code/
│   ├── nsga2_routing.py               # NSGA-II multi-objective GA implementation
│   ├── rolling_horizon.py             # Rolling-Horizon Optimization wrapper
│   ├── sumo_interface.py              # TraCI interface for SUMO
│   ├── objectives.py                  # Objective functions (travel time, congestion, delay)
│   └── utils.py                       # Preprocessing & normalization utilities
├── sumo_scenarios/
│   ├── bethlehem_network.net.xml      # SUMO road network – Bethlehem
│   ├── routes_baseline.rou.xml        # Vehicle routes – baseline scenario
│   └── *.sumocfg                      # SUMO configuration files per scenario
└── README.md
```

---

## Data Collection Protocol

Traffic data were captured from the Bethlehem urban network during **1–14 September 2024** using:
- **Vehicle counts** at 15 signalized intersections via inductive-loop and camera sensors.
- **Floating-car data** from instrumented probe vehicles.
- **Signal logs** recorded directly from traffic management controllers.

Raw data were converted to SUMO-compatible formats using SUMO's network conversion tools (`netconvert`, `polyconvert`) and TraCI. Link travel times were aggregated in rolling 5-minute time slices (Δt = 5 min) and dynamically updated during simulation.

---

## Requirements

### Python
```
python >= 3.9
numpy >= 1.24
pandas >= 2.0
sumolib >= 1.18
traci >= 1.18
deap >= 1.4          # NSGA-II base
matplotlib >= 3.7
scikit-learn >= 1.3
```

Install via:
```bash
pip install -r requirements.txt
```

### SUMO
Install SUMO ≥ 1.18 from [https://sumo.dlr.de](https://sumo.dlr.de).

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/mobaid12/edge-ga-urban-routing.git
cd edge-ga-urban-routing

# Install dependencies
pip install -r requirements.txt

# Run baseline SUMO + GA optimization scenario
python code/nsga2_routing.py \
    --network sumo_scenarios/bethlehem_network.net.xml \
    --routes  sumo_scenarios/routes_baseline.rou.xml \
    --config  sumo_scenarios/SCN_001_baseline.sumocfg \
    --pop_size 100 --generations 200 --horizon 15

# Results are written to results/ directory




## Contact

**Mahmoud Obaid** (corresponding author)  
Computer System Engineering Department, Arab American University, Jenin, Palestine  
✉ Mahmoud.obaid@aaup.edu
