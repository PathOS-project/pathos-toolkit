# Data Access Information

## Available Data in Zenodo

The processed datasets and analysis results for this Impact of Open Access Routes on Topic Persistence case study are available in Zenodo:

**DOI**: [10.5281/zenodo.17048589](https://doi.org/10.5281/zenodo.17048589)

### What's Included in Zenodo:
- `complete_collection_df.parquet` - Main dataset with 132,134 papers and all outcomes
- `complete_collection_df.xlsx` - Excel version of the main dataset
- `topic_attribution_df.parquet` - Topic-level analysis dataset
- `topic_attribution_df.xlsx` - Excel version of topic dataset
- `fos_taxonomy_v0.1.2.json` - SciNoBo Field of Science taxonomy data
- Complete analysis results including:
  - Propensity score matched samples
  - Treatment effect estimates
  - Balance diagnostics
  - SDG-focused analysis results
  - 8 structured result tables
  - 4+ comprehensive visualizations
  - Publication-ready figures and data exports
- All analysis scripts (.py files)

### Key Variables Available:
- **Paper identifiers**: Semantic Scholar IDs, DOIs, publication metadata
- **Treatment variables**: Green OA and Published OA indicators (excluding dual-mode and Bronze OA)
- **Outcome variables**: Citation impact, topic persistence, gender equity, economic impact
- **Topic classification**: Field of Science categories and SDG alignment
- **Control variables**: FWCI scores, interdisciplinarity measures, author characteristics

### Sample Characteristics:
- **Total sample**: 132,134 papers
- **Green OA only**: 3,792 papers
- **Published OA only**: 19,045 papers
- **Closed Access**: 92,998 papers
- **SDG-relevant papers**: 24,948 papers (18.9% of collection)

## External Data Sources Required (Not Available in Repository)

The following large-scale data sources are required for full reproduction but are **not included** due to size limitations, licensing restrictions, or access requirements:

### Required External Databases:
- **Semantic Scholar Academic Graph**: Full academic publication database
  - Access: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
  - Note: Requires API access or data dumps for complete paper collection

- **OpenAIRE Graph**: European research infrastructure data
  - Access: [https://graph.openaire.eu/](https://graph.openaire.eu/)
  - Note: Requires data dump download for affiliation analysis

- **PATSTAT**: Patent database for publication-patent citation analysis
  - Access: [https://www.epo.org/searching-for-patents/business/patstat.html](https://www.epo.org/searching-for-patents/business/patstat.html)
  - Note: Commercial license required

- **ROR (Research Organization Registry)**: Organization type classification
  - Access: [https://ror.org/](https://ror.org/)
  - Note: Freely available data dumps

- **SciNoBo Toolkit Results**: Pre-computed indicators
  - Field of Science classification
  - Interdisciplinarity scores
  - SDG classification
  - FWCI scores
  - Access: Contact PathOS project team
  - Note: Proprietary analysis toolkit outputs

## Data Usage Guidelines

### For Statistical Analysis:
1. Download the processed datasets from Zenodo
2. Use the provided matched samples for immediate causal analysis
3. Run indicator calculation scripts with the final dataset
4. Access pre-computed treatment effects and visualizations

### For Replication:
1. Load the complete collection dataset from Zenodo
2. Run `persistent_topics_calculate_indicators.py` for PSM analysis
3. Run `persistent_topics_calculate_indicators_sdg.py` for SDG-focused analysis
4. Generate visualizations using `persistent_topics_indicators_create_data_for_vis.py`

### For Full Reproduction from Scratch:
1. Obtain access to all external data sources listed above
2. Configure path variables in all analysis scripts
3. Run the complete data collection pipeline starting with `persistent_topics_create_collection.py`
4. Execute affiliation and OpenAIRE ID matching scripts
5. Run gender classification analysis
6. Proceed with indicator calculation and analysis

### Path Configuration Required:
All scripts use placeholder variables that must be configured:
- `PATH_TO_INTERMEDIATE_RESULTS`
- `PATH_TO_INDICATOR_RESULTS`
- `PATH_TO_SS_PAPERS`
- `PATH_TO_TOOL_RESULTS`
- `PATH_TO_ROR_DATA`
- `PATH_TO_PATSTAT_DATA`
- `PATH_TO_OAIRE_DUMP`

## Methodology Notes

### Novel Contributions Available:
1. **Topic Persistence Metric**: New measure of long-term scientific relevance
2. **Clean Treatment Definitions**: Separate analysis of Green vs Published OA
3. **Causal Inference Design**: Propensity Score Matching with comprehensive balancing

### Analysis Framework:
- **Two separate PSM analyses** to prevent treatment contamination
- **Green OA vs Closed Access**
- **Published OA vs Closed Access**
- **Robust outcome measurement** including novel topic persistence metric

## Contact Information

For questions about data access, methodology, or reproduction issues, please contact the PathOS project team through the GitHub repository.