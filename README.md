# HMIS Data Analysis Toolkit

A comprehensive toolkit for analyzing Homeless Management Information System (HMIS) data based on HUD's HMIS Data Standards.

## Overview

This toolkit provides a suite of Python scripts for generating synthetic HMIS data and performing in-depth analysis of homelessness trends, program outcomes, and service gaps. It's designed to help CoCs (Continuums of Care), homeless service providers, and data analysts gain valuable insights from HMIS data to improve service delivery and outcomes.

The toolkit helps answer critical questions such as:
- How have homelessness trends changed from 2022-2024?
- Where are clients most commonly being discharged to?
- Where are clients coming from before entering the system?
- Which interventions are most effective for different populations?
- Are there demographic disparities in outcomes?
- Where are the service gaps in the current system?

## Features

- **Synthetic Data Generation**: Create realistic HMIS test data that follows HUD's data standards
- **Data Quality Assessment**: Identify missing values, inconsistencies, and other data quality issues
- **Trends Analysis**: Track changes in homelessness patterns over time
- **Outcome Analysis**: Evaluate program success rates and discharge destinations
- **Recidivism Analysis**: Analyze patterns of returns to homelessness
- **Demographic Analysis**: Examine disparities in outcomes across demographic groups
- **Service Gap Analysis**: Identify underserved populations and resource needs
- **Comprehensive Reporting**: Generate detailed HTML reports with visualizations

## Project Structure

```
hmis-analysis-toolkit/
├── hmis_synthetic_data/        # Generated synthetic HMIS data directory
├── scripts/                        # Analysis scripts
│   ├── hmis_analysis_script.py             # Main analysis functionality
│   ├── hmis_recidivism_analysis.py         # Returns to homelessness analysis
│   ├── hmis_demographic_gaps_analysis.py   # Demographic disparities analysis
├── reports/                        # Generated reports
│   └── hmis_comprehensive_report.html      # Example of a generated report
├── plots/                          # Generated visualizations
├── hmis_synthetic_data_generator.py     # Creates synthetic test data
├── hmis_master_analysis.py             # Runs all analyses and generates report
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hmis-analysis-toolkit.git
cd hmis-analysis-toolkit
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Synthetic Data (if needed)

For testing or demonstration purposes, you can generate synthetic HMIS data:

```bash
python scripts/hmis_synthetic_data_generator.py
```

### Step 2: Run Analysis Scripts

You can run individual analysis scripts:

```bash
# Main analysis
python scripts/hmis_analysis_script.py

# Recidivism analysis
python scripts/hmis_recidivism_analysis.py

# Demographic and service gaps analysis
python scripts/hmis_demographic_gaps_analysis.py
```

### Step 3: Generate Comprehensive Report

To run all analyses and generate a complete HTML report:

```bash
python scripts/hmis_master_analysis.py
```

The comprehensive report will be saved to `reports/hmis_comprehensive_report.html` and visualizations will be saved to the `plots/` directory.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- Other dependencies listed in requirements.txt

## Acknowledgments

- This toolkit is designed to work with data formatted according to HUD's HMIS Data Standards
- The synthetic data generator creates fictional data that resembles real-world HMIS data patterns
