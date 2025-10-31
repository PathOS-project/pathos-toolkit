# PathOS Toolkit

A collection of code, methodologies, and case studies for analyzing the pathways from Open Science practices to scientific and societal impact.

## Overview

The PathOS (Pathways to Open Science) project investigates how Open Science behaviors translate into measurable downstream impacts across different research domains. This repository contains the computational tools, analysis scripts, and case study implementations developed as part of the PathOS research initiative.

## Repository Structure

```
pathos-toolkit/
├── README.md                    # This file
├── case-studies/               # Individual case study implementations
│   ├── covid-19/              # Impact of Artefact Reuse in COVID-19 Publications
│   ├── french-case-study/     # Open Science Access Analysis via Connection Logs
│   ├── persistent-topics/       # Impact of Open Access Routes on Topic Persistence
│   ├── repository-effect/       # Effects of Data Repositories on Data Usage
│   └── [additional-studies]/  # Future case studies
├── case-study-template/       # Template files for new case studies
│   ├── README.md             # Case study documentation template
│   ├── DATA_ACCESS.md        # Data access template
│   └── TEMPLATE_GUIDE.md     # Instructions for using templates
└── [shared-tools]/           # Common utilities and methodologies (future)
```

## Case Studies

### Impact of Artefact Reuse in COVID-19 Publications
- **Location**: `case-studies/covid-19/`
- **Focus**: Relationship between research artifact reuse and clinical impact in COVID-19 research
- **Sample**: 115,467 COVID-19 papers that created research artifacts
- **Key Finding**: Papers with evidence of artifact reuse achieve significantly greater downstream clinical impact

### Open Science Access Analysis via Connection Logs (French Case Study)
- **Location**: `case-studies/french-case-study/`
- **Focus**: Analysis of access patterns to open and closed scientific publications using HAL and OpenEdition connection logs
- **Sample**: One year of connection logs from HAL (Sept 2023-Aug 2024) and OpenEdition journals (Jan-Dec 2023)
- **Key Innovation**: [Log Explorer](https://pathos.cis.cnrs.fr/) web application for investigating Open Access Advantage across academic disciplines, socio-economic sectors, and countries
- **Methodology**: Enriched connection logs with DOI-based resource matching (OpenAlex) and IP-based user classification (IPinfo + Llama 3.3 NACE classification)

### Impact of Open Access Routes on Topic Persistence
- **Location**: `case-studies/persistent-topics/`
- **Focus**: Effects of Open Access routes on topic persistence in AI-for-Climate research
- **Sample**: 132,134 papers from emerging research topics (2000-2021)
- **Key Innovation**: Novel "topic persistence" metric for measuring long-term scientific relevance

### Effects of Data Repositories on Data Usage
- **Location**: `case-studies/repository-effect/`
- **Focus**: Impact of data repositories on subsequent data reuse and scientific impact in the Social Sciences and Humanities (SSH)
- **Sample**: Mixed-method study combining (1) quantitative data citation corpus analysis (278,922 datasets), (2) algorithmic and manual data mention extraction (162 SSH publications), and (3) qualitative interviews with SSH researchers
- **Key Finding**: Data reuse likelihood varies across repositories — specialised and curated repositories show higher reuse, while algorithmic detection of reuse in SSH is often unreliable due to low precision

## Data Philosophy

This repository follows a clear separation of concerns:

- **GitHub Repository**: Contains code, analysis scripts, documentation, and results
- **Zenodo Deposits**: Contains processed datasets with permanent DOIs for citation and reuse
- **External Sources**: Large-scale databases (Semantic Scholar, OpenAIRE, etc.) accessed separately

Each case study includes a `DATA_ACCESS.md` file with specific information about data availability and access requirements.

## Getting Started

### For Case Study Leaders
1. Copy the templates from `case-study-template/` to create your case study folder
2. Implement your analysis following the established patterns from existing case studies

### For Researchers Using the Case Studies
1. Navigate to the specific case study of interest
2. Check the `DATA_ACCESS.md` file for data availability and requirements
3. Follow the usage instructions in the case study's `README.md`

### For Method Developers
1. Examine the analysis approaches used across case studies
2. Identify common patterns and methodological innovations
3. Consider contributing shared tools and utilities

## Methodological Contributions

The PathOS toolkit develops and implements several methodological innovations:

- **Causal inference approaches** for studying Open Science impacts
- **Novel impact metrics** beyond traditional bibliometric measures
- **Clean treatment definitions** for different Open Science practices
- **Reproducible analysis pipelines** for large-scale bibliometric studies
- **Connection log analysis** for measuring real-time access patterns to scientific publications
- **Open Access Advantage metrics** for quantifying differential access to open vs. closed publications
- **AI-powered classification systems** for automated socio-economic sector identification

## Key External Dependencies

Most PathOS case studies utilize these external data sources:

- **Semantic Scholar Academic Graph**: Paper metadata and citation networks
- **OpenAIRE Graph**: European research infrastructure data
- **PATSTAT**: Patent citation data (commercial license)
- **ROR**: Research organization classifications
- **SciNoBo Toolkit**: Specialized bibliometric indicators

### French Case Study Specific Dependencies:
- **HAL & OpenEdition**: Connection log data (requires authorization)
- **OpenAlex**: Publication metadata and DOI matching
- **IPinfo Academic Research Program**: IP geolocation and organization data
- **Eurostat NACE Rev. 2.1**: Economic activity classification system

See individual case study documentation for specific requirements.

## Contributing

### Adding a New Case Study
1. Create a new folder in `case-studies/`
2. Use the templates from `case-study-template/`
3. Follow the established documentation and data management patterns
4. Ensure your analysis scripts are well-documented and reproducible

### Improving Existing Case Studies
1. Fork the repository
2. Make improvements to analysis, documentation, or reproducibility
3. Submit a pull request with clear description of changes

### Developing Shared Tools
1. Consider extracting common functionality into shared utilities
2. Ensure tools are well-documented and tested
3. Submit contributions that benefit multiple case studies

## Contact

For questions about the PathOS project or this toolkit:
- Open an issue in this GitHub repository
- Contact the PathOS project team [add specific contact information]
