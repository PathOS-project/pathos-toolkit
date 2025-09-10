# Impact of Open Access Colors on Topic Persistence Case Study: Open Access Effects on Scientific Impact and Topic Persistence

This repository contains the data, scripts, and results for the Impact of Open Access Colors on Topic Persistence case study, part of the PATHOS project.

## Overview

Artificial intelligence methods are rapidly being mobilized to tackle the climate crisis, yet the knowledge base that supports this work often burns bright and fades quickly. This case study asks whether two distinct Open Access (OA) routes—**Green OA** (self-archiving in repositories) and **Published OA** (journal-mediated open access)—help AI-for-Climate research topics stay alive in the literature.

By foregrounding **"topic persistence"** and treating it as a primary dimension of impact, the study goes beyond familiar short-term metrics such as raw citation counts and examines whether openness helps research topics remain active in the literature long enough to demonstrate their potential, rather than disappearing prematurely.

### Focus Areas

The investigation concentrates on two Open Science practices:
- **Green OA**: Deposit of the full text in an openly accessible repository
- **Published OA**: Release in a journal that supplies a clear open licence

Bronze OA and dual-mode publications are excluded to preserve clean treatment definitions, with Closed Access (CA) articles providing the counterfactual.

## Repository Structure

```
├── README.md                                          # This file
├── fos_taxonomy_v0.1.2.json                         # SciNoBo Field of Science taxonomy data
├── persistent_topics_create_collection.py              # Main data collection and integration script
├── persistent_topics_find_paper_openaireids.py        # OpenAIRE ID matching for papers
├── persistent_topics_find_paper_affiliations.py       # Affiliation analysis using OpenAIRE data
├── persistent_topics_get_collection_author_gender.py  # Gender classification of authors
├── persistent_topics_calculate_indicators.py          # Main causal inference analysis
├── persistent_topics_calculate_indicators_sdg.py      # SDG-focused analysis
├── persistent_topics_indicators_create_data_for_vis.py # Visualization data preparation
└── emering_topics_collection_w_outcomes/            # Results directory
    ├── complete_collection_df.parquet                # Main dataset with all outcomes
    ├── complete_collection_df.xlsx                   # Excel version of main dataset
    ├── topic_attribution_df.parquet                  # Topic-level analysis dataset
    ├── topic_attribution_df.xlsx                     # Excel version of topic dataset
    ├── results/                                       # Main analysis results
    │   ├── analysis_conclusions.txt                   # Key findings summary
    │   ├── summary_statistics.xlsx                    # Descriptive statistics
    │   ├── psm_a_green_matched.xlsx                  # Green OA matched sample
    │   ├── psm_a_closed_matched.xlsx                 # Closed access matched sample (A)
    │   ├── psm_b_published_matched.xlsx              # Published OA matched sample
    │   ├── psm_b_closed_matched.xlsx                 # Closed access matched sample (B)
    │   ├── treatment_effects_green_oa.xlsx           # Green OA treatment effects
    │   ├── treatment_effects_published_oa.xlsx       # Published OA treatment effects
    │   ├── descriptive_effects_any_oa.xlsx           # Descriptive analysis
    │   ├── psm_a_balance_check.xlsx                  # Balance diagnostics (Green OA)
    │   ├── psm_b_balance_check.xlsx                  # Balance diagnostics (Published OA)
    │   ├── tables/                                    # Structured result tables
    │   │   ├── 01_executive_summary.xlsx             # Executive summary table
    │   │   ├── 02_treatment_group_characteristics.xlsx # Sample characteristics
    │   │   ├── 03_causal_effects_summary.xlsx        # Main causal effects
    │   │   ├── 04_topic_persistence_analysis.xlsx    # Topic persistence results
    │   │   ├── 05_gender_equity_outcomes.xlsx        # Gender analysis results
    │   │   ├── 06_economic_impact_analysis.xlsx      # Economic impact measures
    │   │   ├── 07_publication_year_analysis.xlsx     # Temporal analysis
    │   │   └── 08_robustness_analysis.xlsx           # Robustness checks
    │   ├── visualizations/                            # Generated plots
    │   │   ├── 01_sample_overview.png                # Sample composition
    │   │   ├── 02_causal_effects.png                 # Treatment effects
    │   │   ├── 03_outcome_analysis.png               # Outcome distributions
    │   │   └── 04_temporal_and_balance.png           # Time trends and balance
    │   └── final_visualization_data_figures/          # Final visualization package
    │       ├── data/                                  # Processed data for figures
    │       └── figures/                               # Publication-ready figures
    └── results_sdg_only/                             # SDG-focused analysis
        ├── sdg_analysis_conclusions.txt               # SDG-specific findings
        ├── green_matched_sdg_papers.xlsx             # SDG Green OA matched sample
        ├── published_matched_sdg_papers.xlsx         # SDG Published OA matched sample
        ├── closed_matched_a_sdg_papers.xlsx          # SDG Closed access sample (A)
        ├── closed_matched_b_sdg_papers.xlsx          # SDG Closed access sample (B)
        ├── tables/                                    # SDG analysis tables
        │   ├── 01_sdg_distribution_matched_samples.xlsx
        │   ├── 02_sdg_treatment_effects.xlsx
        │   ├── 03_sdg_vs_non_sdg_comparison.xlsx
        │   ├── 04_sdg_categories_by_impact.xlsx
        │   ├── 05_sdg_gender_industry_collaboration.xlsx
        │   ├── 06_sdg_analysis_summary.xlsx
        │   ├── 07_sdg_alignment_comparison_matched.xlsx
        │   └── 08_sdg_alignment_effects_summary.xlsx
        └── visualizations/                            # SDG-specific plots
            ├── 01_sdg_distribution_overview.png
            ├── 02_sdg_treatment_effects.png
            ├── 03_sdg_impact_analysis.png
            └── 04_sdg_alignment_comparison_matched.png
```

## Data Sources and Requirements

**NOTE**: Several data sources required to reproduce this analysis from scratch are not included in this repository due to size limitations, licensing restrictions, or access requirements. The final processed collection with metadata is provided to enable indicator calculation and analysis.

### External Data Sources Not Included:
- **Semantic Scholar Academic Graph**: Full academic publication database
- **OpenAIRE Graph**: European research infrastructure data
- **PATSTAT**: Patent database for publication-patent citation analysis
- **ROR (Research Organization Registry)**: Organization type classification
- **SciNoBo toolkit results**: Field of Science classification, Interdisciplinarity, SDG classification, FWCI scores

### Included Data:
- **Complete collection dataset**: Final processed collection with all calculated outcomes
- **Topic attribution dataset**: Paper-topic mappings with persistence scores
- **Analysis results**: All matched samples, treatment effects, and summary statistics
- **FOS taxonomy**: SciNoBo Field of Science classification system

## Scripts Description

### Core Data Processing Scripts

**`persistent_topics_create_collection.py`**
- Main data integration script that combines all outcome measures
- Loads papers from Semantic Scholar Academic Graph (PATH_TO_INTERMEDIATE_RESULTS)
- Integrates FWCI scores, gender diversity, SDG classification, FOS topics
- Performs patent citation analysis using PATSTAT data
- Conducts affiliation analysis using OpenAIRE and ROR data
- Creates final complete collection with all calculated outcomes

**`persistent_topics_find_paper_openaireids.py`**
- Maps DOIs to OpenAIRE identifiers for affiliation analysis
- Requires OpenAIRE Graph dump (PATH_TO_OAIRE_DUMP)

**`persistent_topics_find_paper_affiliations.py`**
- Extracts institutional affiliations using OpenAIRE data
- Performs science-industry collaboration analysis using ROR data

**`persistent_topics_get_collection_author_gender.py`**
- Classifies author gender using machine learning models
- Creates gender diversity indicators for publications

### Analysis Scripts

**`persistent_topics_calculate_indicators.py`**
- Main causal inference analysis using propensity score matching
- Implements two separate PSM analyses:
  - Green OA vs Closed Access
  - Published OA vs Closed Access
- Calculates treatment effects on multiple outcomes including topic persistence
- Generates matched samples and balance diagnostics

**`persistent_topics_calculate_indicators_sdg.py`**
- SDG-focused analysis using existing matched samples
- Analyzes differential effects for sustainability-related research
- Creates SDG-specific visualizations and tables

**`persistent_topics_indicators_create_data_for_vis.py`**
- Prepares final visualization data and publication-ready figures
- Creates comprehensive data exports for external visualization tools

## Key Findings

### Sample Characteristics
- **Total sample**: 132,134 papers
- **Green OA only**: 3,792 papers
- **Published OA only**: 19,045 papers
- **Closed Access**: 92,998 papers

### Novel Contributions
1. **Topic Persistence Metric**: New measure of long-term scientific relevance and knowledge sustainability
2. **Clean Treatment Definitions**: Exclusion of dual-mode and Bronze OA for causal identification
3. **Multiple OA Pathways**: Separate analysis of Green vs Published Open Access

### Main Results
- **8 significant causal effects** found across outcomes
- **Enhanced topic persistence** for Open Access publications
- **Gender equity improvements** through OA pathways
- **Economic impact** via science-industry collaboration and patent citations

### SDG Analysis
- **24,948 SDG-relevant papers** (18.9% of collection)
- **11 significant treatment effects** for sustainability research
- **Enhanced knowledge building** for achieving 2030 SDG goals

## Path Variables

The scripts use placeholder path variables that should be replaced with actual paths:

- `PATH_TO_INTERMEDIATE_RESULTS`: Intermediate processing results
- `PATH_TO_INDICATOR_RESULTS`: Final analysis outputs
- `PATH_TO_SS_PAPERS`: Semantic Scholar paper data
- `PATH_TO_TOOL_RESULTS`: SciNoBo toolkit results
- `PATH_TO_ROR_DATA`: Research Organization Registry data
- `PATH_TO_PATSTAT_DATA`: Patent database processed results
- `PATH_TO_OAIRE_DUMP`: OpenAIRE Graph dump location

## Methodology

### Causal Inference Design
- **Propensity Score Matching (PSM)** for causal identification
- **Two separate analyses** to prevent treatment contamination
- **Comprehensive covariate balancing** on publication characteristics
- **Robust outcome measurement** including novel topic persistence metric

### Treatment Definitions
- **Green OA**: Repository-based open access only
- **Published OA**: Journal-based open access (gold, hybrid, diamond)
- **Closed Access**: No open access provision
- **Excluded**: Dual-mode OA and Bronze OA for treatment purity

### Outcome Variables
1. **Citation Impact**: Traditional bibliometric measures
2. **Topic Persistence**: Novel measure of long-term scientific relevance
3. **Gender Equity**: Women's participation in authorship
4. **Economic Impact**: Patent citations and industry collaboration
5. **Field Effects**: Disciplinary and SDG-specific outcomes
