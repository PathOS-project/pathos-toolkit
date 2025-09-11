# Impact of Artefact Reuse in COVID-19 Publications Research Data, Code, and Analysis Results

This repository contains the complete dataset, analysis scripts, and results for the Impact of Artefact Reuse in COVID-19 Publications case study.

## Overview

This study investigates whether observable open science behaviors—specifically creating research artifacts that are subsequently reused by others—are associated with measurable downstream impact in COVID-19 research. The analysis employs a regression-based approach using a filtered sample of 115,467 COVID-19 papers that created at least one dataset or software artifact and were cited at least once, ensuring all publications had potential for visibility and reuse.

The study operationalizes reusability through reuse-artifact citances: citations where other papers explicitly reference and reuse datasets or software created by the original publication. This provides empirical evidence that artifacts were not only shared but found useful and actionable in practice.

## Repository Structure

```
covid/
├── README.md                               # This file
├── complete_collection_df_fix.parquet      # Final dataset (Parquet format)
├── complete_collection_df_fix.xlsx         # Final dataset (Excel format)
├── covid_create_collection.py              # Main collection creation script
├── covid_calculate_indicators.py           # Statistical analysis script
├── covid_find_paper_affiliations.py       # Affiliation analysis script
├── covid_find_paper_openaireids.py        # OpenAIRE ID matching script
├── covid_indicators_create_data_for_vis.py # Visualization data creation script
└── results/                                # Analysis results and outputs
    ├── *.xlsx                              # Descriptive statistics and summaries
    ├── *.txt                               # Regression model outputs
    ├── *.parquet                           # Processed datasets
    ├── tables/                             # Comprehensive analysis tables
    │   ├── 01_executive_summary.xlsx
    │   ├── 02_impact_by_artifact_type.xlsx
    │   ├── 03_statistical_significance_summary.xlsx
    │   └── ... (19 detailed analysis tables)
    ├── visualizations/                     # Statistical plots and figures
    │   ├── 01_sample_overview.png
    │   ├── 02_outcome_distributions.png
    │   └── ... (11 comprehensive visualizations)
    └── final_visualization_data_figures/   # Publication-ready figures and data
        ├── figures/                        # Final publication figures
        └── data/                           # Data exports for figures
```

## Data Description

### Main Dataset (`complete_collection_df_fix.parquet/xlsx`)

The primary dataset contains **COVID-19 research papers that created at least one research artifact** (dataset or software) and were cited at least once. Key variables include:

**Paper Identifiers:**
- `id`: Semantic Scholar paper ID
- `year`: Publication year (filtered ≤2021 to avoid citation bias)
- `citationcount`: Total citation count
- `authorcount`: Number of authors

**Artifact Creation Variables:**
- `named_datasets_created`: Count of named datasets created
- `unnamed_datasets_created`: Count of unnamed datasets created  
- `named_software_created`: Count of named software tools created
- `unnamed_software_created`: Count of unnamed software tools created
- `total_artifacts`: Total artifacts created (sum of above)

**Treatment Variable:**
- `has_reuse_artifact_citance`: Binary indicator for having ≥1 reuse-artifact citance
- `reuse_artifact_inbound`: Count of inbound reuse-artifact citances

**Outcome Variables:**
- `clinical_trial_citations`: Citations from clinical trial papers
- `clinical_guideline_citations`: Citations from clinical guideline papers
- `total_clinical_citations`: Total clinical citations (trials + guidelines)
- `clinical_trial_citations_influential`: Influential clinical trial citations
- `clinical_trial_citations_non_influential`: Non-influential clinical trial citations
- `clinical_guideline_citations_influential`: Influential clinical guideline citations
- `clinical_guideline_citations_non_influential`: Non-influential clinical guideline citations
- `patent_citations`: Citations from patent documents
- `science_industry_collaboration`: Binary indicator for science-industry collaboration

**Control Variables:**
- `fwci`: Field-Weighted Citation Impact score
- `interdisciplinarity_macro`: Macro-level interdisciplinarity score
- `interdisciplinarity_meso`: Meso-level interdisciplinarity score
- `science_industry_collaboration`: Industry collaboration indicator

**Open Access Variables:**
- `isopenaccess_oaire`: Open access status (OpenAIRE-corrected)
- `green`, `bronze`, `hybrid`, `gold`, `diamond`: Open access route indicators

## Scripts and Methodology

### Core Analysis Scripts

1. **`covid_create_collection.py`**: Main data collection and processing script
   - Integrates multiple data sources and analytical outcomes
   - Calculates derived variables and indicators
   - Produces the final collection dataset

2. **`covid_calculate_indicators.py`**: Statistical analysis and modeling
   - Regression-based analysis of treatment effects
   - Interaction effects analysis
   - Comprehensive statistical testing and visualization
   - Generates 19 detailed analysis tables and 11+ visualizations

3. **`covid_find_paper_affiliations.py`**: Institutional affiliation analysis
   - Processes OpenAIRE affiliation data
   - Identifies science-industry collaboration patterns

4. **`covid_find_paper_openaireids.py`**: OpenAIRE ID matching
   - Links papers to OpenAIRE Graph for enhanced metadata

5. **`covid_indicators_create_data_for_vis.py`**: Visualization data preparation
   - Creates publication-ready figures and data exports
   - Generates final visualization datasets

### Required External Data Sources

**Note**: Several large-scale data sources are required to fully reproduce this collection but are not included in this deposit due to size, licensing, or access restrictions:

- **Semantic Scholar Academic Graph**: Full paper metadata and citation data
- **OpenAIRE Graph**: Institutional affiliations and open access metadata  
- **PubMed**: Clinical trial and guideline classification data
- **PATSTAT**: Patent citation data
- **ROR (Research Organization Registry)**: Organization type classifications
- **CORD-19 Dataset**: Initial COVID-19 paper identification
- **SciNoBo Toolkit**: For computing interdisciplinarity, FWCI, citance analysis, and research artifact analysis

Instead, we provide the **final processed collection** with all indicators and outcomes pre-computed, enabling immediate analysis and replication of statistical results.

## Key Findings and Results

### Main Results (`results/` directory)

The analysis demonstrates that COVID-19 papers with evidence of artifact reuse achieve significantly greater downstream impact:

- **Clinical Trial Citations**: Papers with reuse evidence receive more citations from clinical trial studies
- **Clinical Guideline Citations**: Enhanced citation from clinical practice guidelines  
- **Patent Citations**: Increased innovation impact through patent citations
- **Science-Industry Collaboration**: Higher rates of collaborative arrangements

### Statistical Analysis Files

**Executive Summary**: `results/tables/01_executive_summary.xlsx`
- Sample size, treatment distribution, key impact metrics

**Regression Results**: `results/regression_output_*.txt` files
- Detailed OLS regression outputs for each outcome variable
- Coefficients, standard errors, significance tests

**Interaction Effects**: `results/tables/16-19_interaction_*.xlsx`
- Moderation analysis examining how effects vary by paper characteristics
- Marginal effects and stratified analyses

**Visualizations**: `results/visualizations/` and `results/final_visualization_data_figures/`
- Statistical plots, distribution comparisons, effect visualizations
- Publication-ready figures with accompanying data

## Path Configuration

All file paths in the scripts use placeholder variables that must be configured for your environment:

- `PATH_TO_INTERMEDIATE_RESULTS`: Intermediate processing files
- `PATH_TO_SS_COVID_PAPERS`: Semantic Scholar COVID paper data
- `PATH_TO_SS_COVID_PAPER_CITATIONS`: Citation network data
- `PATH_TO_INDICATOR_RESULTS`: Analysis output directory
- `PATH_TO_SS_PAPERS`: Full Semantic Scholar paper database
- `PATH_TO_PUBMED_PARSED_FILES`: Processed PubMed data
- `PATH_TO_TOOL_RESULTS`: SciNoBo analysis outputs
- `PATH_TO_ROR_DATA`: ROR organization data
- `PATH_TO_PATSTAT_DATA`: Patent database files
- `PATH_TO_RAA_OUTPUT`: Research artifact analysis results

## Usage Instructions

### For Statistical Analysis

1. **Load the main dataset**: Use `complete_collection_df_fix.parquet` or `.xlsx`
2. **Examine summary statistics**: Start with `results/tables/01_executive_summary.xlsx`
3. **Review regression results**: See `results/regression_results_summary_covid.xlsx`
4. **Explore interactions**: Use tables 16-19 for moderation analysis
5. **Visualize findings**: Access figures in `results/visualizations/`

### For Replication

1. **Configure paths**: Update all `PATH_TO_*` variables in scripts
2. **Install dependencies**: Ensure pandas, statsmodels, matplotlib, seaborn are available
3. **Run analysis**: Execute `covid_calculate_indicators.py` with the provided dataset
4. **Generate visualizations**: Run `covid_indicators_create_data_for_vis.py`

### For Extension

The modular structure allows researchers to:
- Apply the same methodology to other research domains
- Extend the analysis with additional outcome variables
- Incorporate different treatment definitions or time periods
- Adapt the regression framework for other bibliometric studies

## Sample Sizes and Coverage

- **Total COVID-19 papers analyzed**: 115,467 papers that created artifacts and were cited
- **Papers with reuse evidence**: Variable by outcome (see executive summary)
- **Time period**: Publications through 2021 (to avoid recent citation bias)
- **Geographic coverage**: Global (based on Semantic Scholar and OpenAIRE coverage)

## Quality Assurance

- **Multiple data validation steps**: Cross-checking across different data sources
- **Robustness testing**: Interaction effects and stratified analyses
- **Reproducible workflows**: All processing steps documented in scripts
- **Statistical best practices**: Appropriate controls, confidence intervals, effect sizes
