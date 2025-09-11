# Data Access Information

## Available Data in Zenodo

The processed datasets and analysis results for this Impact of Artefact Reuse in COVID-19 Publications case study are available in Zenodo:

**DOI**: [10.5281/zenodo.17046920](https://doi.org/10.5281/zenodo.17046920)

### What's Included in Zenodo:
- `complete_collection_df_fix.parquet` - Final processed dataset with 115,467 COVID-19 papers
- `complete_collection_df_fix.xlsx` - Excel version of the final dataset
- Complete analysis results including:
  - 19 detailed analysis tables
  - 11+ comprehensive visualizations
  - Regression model outputs
  - Publication-ready figures and data exports
- All analysis scripts (.py files)

### Key Variables Available:
- **Paper identifiers**: Semantic Scholar IDs, publication years, citation counts
- **Artifact creation variables**: Named/unnamed datasets and software created
- **Treatment variables**: Reuse-artifact citance indicators
- **Outcome variables**: Clinical citations, patent citations, science-industry collaboration
- **Control variables**: FWCI scores, interdisciplinarity measures, open access indicators

## External Data Sources Required (Not Available in Repository)

The following large-scale data sources are required for full reproduction but are **not included** due to size, licensing, or access restrictions:

### Required External Databases:
- **Semantic Scholar Academic Graph**: Full paper metadata and citation data
  - Access: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
  - Note: Requires API access or data dumps

- **OpenAIRE Graph**: Institutional affiliations and open access metadata
  - Access: [https://graph.openaire.eu/](https://graph.openaire.eu/)
  - Note: Requires data dump download

- **PubMed**: Clinical trial and guideline classification data
  - Access: [https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)
  - Note: Requires bulk download and processing

- **PATSTAT**: Patent citation data
  - Access: [https://www.epo.org/searching-for-patents/business/patstat.html](https://www.epo.org/searching-for-patents/business/patstat.html)
  - Note: Commercial license required

- **ROR (Research Organization Registry)**: Organization type classifications
  - Access: [https://ror.org/](https://ror.org/)
  - Note: Freely available data dumps

- **CORD-19 Dataset**: Initial COVID-19 paper identification
  - Access: [https://www.semanticscholar.org/cord19](https://www.semanticscholar.org/cord19)
  - Note: Historical dataset, may no longer be updated

- **SciNoBo Toolkit**: For computing interdisciplinarity, FWCI, citance analysis, and research artifact analysis
  - Access: Contact PathOS project team
  - Note: Proprietary analysis toolkit

## Data Usage Guidelines

### For Statistical Analysis:
1. Download the processed dataset from Zenodo
2. Use the provided analysis results for immediate replication
3. Run analysis scripts with the final dataset

### For Full Reproduction:
1. Obtain access to the external data sources listed above
2. Configure path variables in the analysis scripts
3. Run the complete data collection and processing pipeline

### Path Configuration Required:
All scripts use placeholder variables that must be configured:
- `PATH_TO_INTERMEDIATE_RESULTS`
- `PATH_TO_SS_COVID_PAPERS`
- `PATH_TO_SS_COVID_PAPER_CITATIONS`
- `PATH_TO_INDICATOR_RESULTS`
- `PATH_TO_SS_PAPERS`
- `PATH_TO_PUBMED_PARSED_FILES`
- `PATH_TO_TOOL_RESULTS`
- `PATH_TO_ROR_DATA`
- `PATH_TO_PATSTAT_DATA`
- `PATH_TO_RAA_OUTPUT`

## Contact Information

For questions about data access or reproduction issues, please contact the PathOS project team through the GitHub repository.