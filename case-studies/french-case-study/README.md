# French Case Study : Exploring OpenEdition & HAL Connection Logs

[TOC]

This folder contains the code, documentation, and analysis results for the French case study, which is part of the PathOS project.

## Full Code Repositories

### French Open Science Log Explorer  
- **GitLab Repository:** [path-os/ouestware – code](https://gitlab.huma-num.fr/path-os/ouestware/-/tree/main/code?ref_type=heads)  
- **Website:** [pathos.cis.cnrs.fr](https://pathos.cis.cnrs.fr/) 
- **.tar Zenodo Release:** [DOI: 10.5281/zenodo.17465446](https://doi.org/10.5281/zenodo.17465446)

### French Open Science Log Processing Tools  
- **GitLab – Docker Configuration Files:** [path-os/ouestware – docker](https://gitlab.huma-num.fr/path-os/ouestware/-/tree/main/docker?ref_type=heads)  
- **GitLab – Data Enrichment Scripts:** [path-os/ouestware – scripts](https://gitlab.huma-num.fr/path-os/ouestware/-/tree/main/scripts?ref_type=heads)  
- **.tar Zenodo Release:** [DOI: 10.5281/zenodo.17465446](https://doi.org/10.5281/zenodo.17465446)

### Llama 3.3 for Classification  
- **Llama 3.3 Model:** [ollama.com/library/llama3.3](https://ollama.com/library/llama3.3)  
- **GitLab Repository:** [automated-nace-classifier](https://gitlab.huma-num.fr/path-os/automated-nace-classifier)  
- **.tar Zenodo Release:** [DOI: 10.5281/zenodo.17471199](https://doi.org/10.5281/zenodo.17471199)

## Overview

The [Log Explorer](https://pathos.cis.cnrs.fr/) has been developed by the [CNRS Center for Internet & Society](https://cis.cnrs.fr/) and the [PathOS Project](https://pathos-project.eu/), in close collaboration with [HAL](https://hal.science/) and [OpenEdition](https://www.openedition.org/).  

It enables the investigation of the very first step in the impact pathways of Open Science: **the access to open and closed scientific publications**.

A *connection log* is a large text file that records all connection attempts made to a specific website. The log explorer allows users to explore one year of these logs, collected by the HAL and OpenEdition platforms (*January–December 2023* for OpenEdition journals, and *September 2023–August 2024* for HAL).

More specifically, the log explorer enables exploration of a **final aggregated dataset** of the logs, enriched with information about:
- **Resources** (based on DOI and matched with [OpenAlex](https://docs.openalex.org/api-entities/works)), and  
- **Users** (based on IP addresses, matched with [IPinfo](http://ipinfo.io/contracts/academic-research-agreement) and classified into NACE activity sectors using a LLama3.3-based script)  

It allows filtering the output data by **academic discipline**, **socio-economic sector**, and **country**, and calculating an indicator called the [Open Access Advantage](https://handbook.pathos-project.eu/sections/3_societal_impact/OS_access_advantage.html). This indicator shows, for a given academic discipline, societal sector, country—or any combination thereof—whether open publications were, on average, more or less consulted than closed ones.

Our detailed methodology can be found in [D3.4 Data and tools for the long-term evaluation of open science](https://pathos-project.eu/deliverables-and-publications), section *“French Case Study”*. 

The full data pipeline from raw input data, API and typologies to aggregated dataset and exploration app is shown on the graph below : 

![](https://storage.gra.cloud.ovh.net/v1/AUTH_0f20d409cb2a4c9786c769e2edec0e06/padnumerique/uploads/f8400fbd-c1d0-4913-a9c2-705ef73f68e5.png)

## Ouestware repository Contents

### Analysis Scripts
```bash
.gitignore                         # Git ignore rules
LICENSE                            # Project license
README.md                          # Main project overview
```
#### Web app
```bash
code/                              # Application source
├── package-lock.json              # Locked dependency graph
├── package.json                   # App scripts & deps
├── prettier.config.mjs            # Prettier formatting rules
├── README.md                      # Code folder notes
├── tsconfig.base.json             # Base TS config
├── tsconfig.json                  # Root TS config

```
##### front-end
```bash
├── client/                        # Frontend app (React/Vite)
│   ├── .prettierignore            # Prettier ignore list
│   ├── eslint.config.js           # ESLint config
│   ├── index.html                 # HTML entry point
│   ├── package.json               # Client deps & scripts
│   ├── tsconfig.app.json          # TS config for app
│   ├── tsconfig.json              # Local TS config
│   ├── tsconfig.node.json         # TS config for Node tools
│   ├── vite.config.ts             # Vite build/dev config
│   │
│   ├── src/                       # Frontend source
│   │   ├── config.ts              # Client config
│   │   ├── main.tsx               # React bootstrap
│   │   ├── vite-env.d.ts          # Vite type defs
│   │   │
│   │   ├── components/            # UI components
│   │   │   ├── AccessCountStackedchart.tsx   # Stacked access chart
│   │   │   ├── Facets.tsx                   # Facets UI
│   │   │   ├── ItemsList.tsx                # Items list view
│   │   │   ├── OpenInBibliograph.tsx        # Link to Bibliograph
│   │   │   ├── OpennessMetric.tsx           # Open access advantage calculator
│   │   │   ├── WhoFacets.tsx                # “Who” facet widget
│   │   │   │
│   │   │   └── layout/                      # Layout elements
│   │   │       └── NavBar.tsx               # Top navigation bar
│   │   │
│   │   ├── core/                            # Core app logic
│   │   │   ├── api.ts                       # HTTP client wrappers
│   │   │   ├── consts.tsx                   # Constants
│   │   │   ├── context.ts                   # React contexts
│   │   │   ├── facets.ts                    # Facet helpers
│   │   │   ├── router.tsx                   # Client routes
│   │   │   └── types.ts                     # Shared types
│   │   │
│   │   ├── generated/                       # Codegen outputs
│   │   │   └── openapi.d.ts                 # OpenAPI TS types
│   │   │
│   │   ├── pages/                           # Route pages
│   │   │   ├── Explore.tsx                  # Explore view
│   │   │   └── Root.tsx                     # Root layout/page
│   │   │
│   │   ├── styles/                          # SCSS styles
│   │   │   ├── index.scss                   # Styles entry
│   │   │   ├── _base.scss                   # Base styles
│   │   │   ├── _dataset.scss                # Dataset styles
│   │   │   ├── _search.scss                 # Search styles
│   │   │   ├── _variables-override.scss     # Overrides
│   │   │   └── _variables-set.scss          # Variables
│   │   │
│   │   └── utils/                           # Frontend utils
│   │       └── error.ts                     # Error helpers
│
```
##### back-end
``` bash
└── server/                                 # Backend API (TS/TSOA)
    ├── .env.sample                         # Env vars template
    ├── .gitignore                          # Server ignore rules
    ├── .prettierignore                     # Prettier ignore
    ├── application-2025-04-17.log          # Sample app log
    ├── eslint.config.mjs                   # ESLint config
    ├── nodemon.json                        # Nodemon settings
    ├── package.json                        # Server deps & scripts
    ├── prettier.config.mjs                 # Prettier rules
    ├── tsconfig.json                       # TS config
    ├── tsoa.json                           # TSOA/OpenAPI config
    │
    ├── data/                               # Server data dir
    │   └── .gitkeep                        # Keep empty dir
    │
    └── src/                                # Server source
        ├── error.ts                        # Error handling
        ├── index.ts                        # App entrypoint
        ├── ioc.ts                          # DI container setup
        ├── types.ts                        # Server types
        │
        ├── bin/                            # CLI scripts
        │   ├── dataset-export.ts           # Export datasets
        │   ├── dataset-import.ts           # Import datasets
        │   ├── openalex-collect.ts         # Collect OpenAlex data
        │   └── unique_openalex_ids.bash    # Deduplicate OpenAlex IDs
        │
        ├── config/                         # Config loader
        │   └── index.ts                    # Build config object
        │
        ├── controllers/                    # HTTP controllers
        │   └── datasets.ts                 # Dataset endpoints
        │
        ├── generated/                      # Generated server code
        │   ├── routes.ts                   # TSOA routes
        │   └── swagger.json                # OpenAPI schema
        │
        ├── services/                       # Domain services
        │   ├── dataset.ts                  # Dataset logic
        │   ├── elastic.ts                  # Elasticsearch client
        │   ├── filesystem.ts               # FS utilities
        │   ├── import.ts                   # Import workflow
        │   ├── index.ts                    # Services registry
        │   └── logger.ts                   # Logger setup
        │
        ├── shared/                         # Shared modules
        │   └── facets.ts                   # Facets shared logic
        │
        └── utils/                          # Server utilities
            ├── number.ts                   # Number helpers
            └── sections.ts                 # Section helpers
``` 
### Dockerisation of Elastisearch for indexing the enriched raw logs
```bash
docker/                                    # Containerization files
├── .env                                   # Docker env vars
├── docker-compose.override.yml            # Local overrides
├── docker-compose.yml                     # Compose stack
├── nginx.site.conf                        # Nginx vhost
│
├── elasticsearch/                         # ES node config
│   └── data/
│       └── .gitkeep                       # Keep data dir
│
└── project/
    ├── Dockerfile                         # Project image build
    └── entrypoint.sh                      # Container entrypoint
```
### Docs
```bash
docs/                                      # Documentation
└── pathos_online_explorer.md              # PathOS explorer doc
```
### Raw log enrichment scripts
```bash
scripts/                                   # Data/scripts toolbox
├── .env.sample                            # Env template
├── addIpInfoToCSV.ts                      # Enrich CSV with IP info API
├── addOpenAlexToCSV.ts                    # Enrich CSV with OpenAlex API
├── aiTopics.ts                            # AI topics mapping
├── dumpIpRangeData.sh                     # Dump IP ranges
├── ipInfo.ts                              # IP info client
├── ipRanges.ts                            # IP ranges helpers
├── openAlex.ts                            # OpenAlex utils
├── package-lock.json                      # Locked deps for scripts
├── package.json                           # Scripts package config
├── test_gz_files.bash                     # GZ validation
├── tsconfig.json                          # TS config for scripts
├── types.ts                               # Shared types
├── utils.ts                               # Script utilities
├── whoIs.ts                               # WHOIS lookup
│
├── aggregations/                          # Aggregation scripts
│   ├── addNaceCodes.ts                    # Add NACE codes
│   ├── aggregateAccessByOpenAlexDocument.ts  # Aggregate access per doc
│   ├── listOpenAlexDocuments.ts           # List OpenAlex docs
│   ├── notes.md                           # Notes
│   └── queries.ts                         # Aggregation queries
│
├── data/                                  # Input datasets
│   ├── aiTerms.csv                        # AI terms list
│   ├── OpenAlex_topic_mapping_table_final_topic_field_subfield_table.csv   # Topic→field mapping
│   ├── OpenAlex_topic_mapping_table_final_topic_field_subfield_table_ai.csv # AI-specific mapping
│   └── README.md                          # Data readme
│
├── ezpaarse/                              # ezPAARSE helpers
│   └── treat_hal_logs.sh                  # Process HAL logs
│
└── logstash_elk_openedition/              # Logstash pipelines
    ├── add_openalex_openEdition_2022.sh   # Add OpenEdition 2022 data
    ├── create-indices.sh                  # Create ES indices
    ├── geoindex.sh                        # Build geo index
    ├── hal.logstash.conf                  # Logstash for HAL
    ├── hal_2023_paths.csv                 # HAL input paths 2023
    ├── index_hal.sh                       # Index HAL data
    ├── index_openedition_journals.sh      # Index OE journals
    ├── load-data.sh                       # Load data runner
    ├── logstash_create_conf_and_run.sh    # Generate+run config
    ├── logstash_filter.txt                # Logstash filters
    ├── model.json                         # Index mapping model
       open_edition_journals.logstash.conf  # Logstash journals config
    ├── open_edition_journals_memory_test.logstash.conf # Memory test config
    ├── prepare-data-ipinfo.sh             # Prep IP info data
    ├── prepare-data-openalex-repair-2023.sh # Repair OpenAlex 2023
    ├── prepare-data.sh                    # Generic prep
    ├── prepare-hal-data-ipinfo.sh         # Prep HAL IP info
    └── ...                                # Other scripts (misc)
```

### Automated Nace Classifier

```bash
.env.sample                           # Example environment variables
.gitignore                            # Files ignored by Git
LICENSE                               # Project license
package-lock.json                     # Locked dependency versions (npm)
package.json                          # Node.js project dependencies & scripts
README.md                             # Project documentation
yarn.lock                             # Locked dependencies (Yarn)

input/                                # A folder to store a list of domains names to be classified, in order to test accuracy of prompt and model configuration
├── ReferenceTable.csv                # A random set of input domain names used to test the classifying prompt
└── sections.js                       # the Nace Level 1 Categories

output/                               # Generated output results
├── classified_results_batch.json      # Batch classification results
├── classified_results_withkeywords.json  # Results including extracted keywords
└── classified_results_withoutkeywords.json # Results excluding keywords

scripts/                              # Processing and extraction scripts
├── accuracyTest.js                    # Accuracy evaluation script
├── keywordExtractorWithOllama.js      # Keyword extraction using Ollama model
├── keywordExtractorWithRetext.js      # Keyword extraction using Retext NLP
└── sortWithOllama.js                  # Sorting domain names with Llama3.3
```


### Data
- See `DATA_ACCESS.md` for information on accessing the data used in this case study
- Processed datasets and results are deposited in Zenodo (see DATA_ACCESS.md for DOI)

### Results
Results are generated dynamically using the web app, a subset of results are also contained in a static form in the above mentionned dataset. See D3.4 for more explanations about the difference between static and dynamic results. 

## Usage

### To reproduce the analysis:
1. Obtain authorization to access raw connection logs from the desired open science platform, and store them on an SSH-secured virtual machine.  
2. Dockerize Elasticsearch and Kibana on this machine.  
3. Dockerize Llama 3.3 on a virtual machine with sufficient VRAM, or use an API endpoint connected to a Llama 3.3 instance. Note that, to comply with GDPR, the Llama 3.3 instance must be hosted on a secure European server.  
4. Preprocess the raw logs using the provided Node.js enrichment scripts. Note that this requires an academic access agreement to use IPinfo data.  
5. Index the enriched logs in Elasticsearch and start visually exploring them using Kibana to find the most suitable aggregations and queries.
6. Run a KQL query to extract the desired aggregated output dataset.  
7. Set up a second server with one Elasticsearch instance (to host the aggregated dataset according to the backend setup) and one React/Vite frontend instance.  
8. Purchase or configure an existing domain name to point to your second server’s public IP address.  
9. Configure DevOps pipelines between your GitLab or GitHub repository and both the frontend and backend.


### To extend the analysis

This method could be improved by... 

#### BROADENING THE SCOPE OF COLLECTED LOGS.

1.	Spawn over multiple years, over multiple platforms worldwide, containing final and intermediary open science artifacts (articles, input and output data, communications, teaching materials, reports, software etc). 
2.	Include resources from pirate platforms such as Sci-Hub (for instance the download log for 2017 has been deposited on Zenodo ) and from institutional libraries which grant access to paywalled resources, allowing for insightful and meaningful comparison between (legal, official) open science, closed science, and black open access . 
3.	After collecting the user’s consent, the logs could include more precise usage data such as heatmaps, cross-site tracking, or fingerprinting. It is important to note, however, that this is ethically disputable. A dedicated participatory research add-on for web browsers such as Horus  could be developed.

#### DEVELOPING GRAPH-BASED DATA EXPLORATION TECHNIQUES.

4.	The referers contained in the logs could be analyzed in order to generate a network of all the websites pointing to the studied platform, and a web app for web exploration similar to the one we developed could be designed in order to allow serendipitous discoveries.
5.	Similar to altmetrics which focus on citation of publications on social media, systematic crawling of specific regions of the internet (like Wikipedia, or a set of websites from NGOs with consequent web traffic, or selected governmental websites) could help find out whether the publications available on one or several targeted open science platforms are effectively mentioned on those websites, what graph and what conclusions about their use and “penetration rate” can be drawn from it.

#### REFINING THE CHARACTERIZATION OF THE SOCIO-ECONOMIC STATUS OF USERS.

6.	A survey could be designed and administered to the users of the studied platforms, with the option of linking responses to their IP address. This would make it possible to combine survey data with log data and thereby. 
7.	Should user fingerprints be collected, additional socio-economic profiling data could be acquired through data broker services. Note that this solution is invasive and potentially expensive.
8.	Another way could be to rely on the geolocation derived from their IP address. This information can then be linked to external databases — for example, to the World Inequality Database  at the country level, or to the Filosofi dataset from Insee  at the regional level in France — in order to capture geographical inequalities. 

#### ACHIEVING HIGHER PRECISION IN AUTOMATED NACE CLASSIFICATION OR ANY MORE SUITABLE TAXONOMY.

9.	Expanding the manual annotation dataset size and annotator pool to improve ground truth reliability, 
10.	Deploying sophisticated LLM architectures beyond prompt engineering—including retrieval-augmented generation, model fine-tuning, and embedding-based approaches
11.	Expanding classification depth from 22 sections to 88 divisions to capture sector nuances more effectively

#### PERFORMING MORE REFINED STATISTICAL ANALYSIS.

12.	Adequately control for confounding factors using techniques such as linear regression.
13.	Apply optimization algorithms to manage combinatorial explosions that surpass computational limits, prioritizing relevant parameter subsets and targeted queries.

#### STANDARDIZING LOG ANALYSIS TECHNIQUES AND AIMING AT REAL-TIME IMPLEMENTATION.

14.	As sketched above, the required legal, technical and managerial processes for collecting, storing, analysing the logs on a dedicated, secured server and the multi-stakeholder log exploration methods such as datasprints could be further specified and translated into a dedicated handbook, addressing common challenges and giving mitigation techniques to lift constraints.
15.	The log analysis methods could be implemented in real-time or near-real time in order to design monitoring dashboards that could serve short-termed iterative feedback for platform improvement, open science policy refinement, and raising awareness about the necessity to have free and open access to knowledge.

#### BRIDGING STRICTLY QUANTITATIVE ACCESS METRICS WITH MORE QUALITATIVE, USAGE-FOCUSED ANALYSES. 

16.	Based on sole log data, long-term navigation sessions could be modelized thanks to Markov chains  or other methods based on sessionizing and navigation pattern analysis. 
17.	Study the cognitive interaction of the users with the platforms. For instance, in a study conducted on Gallica platform, researchers have implemented a method called “Subjective Evidence-Based Ethnography” by using glasses equipped with a camera to study the interaction between user and computer while they were using the online library Gallica.

#### COMPLEMENT USAGE-FOCUSED ANALYSES WITH ETHNOGRAPHICAL FIELDWORK. 

18. Using the methods of ethnography to study contextual appropriation and reuse of scientific resources, not only as the direct interaction with digital interfaces, but as longer-term changes, be they “conceptual (e.g. changes to awareness, understanding or perspective), or attitudinal or cultural (e.g. behavioural changes) [or] knowledge, skills gain or the development of relationships between diverse stakeholders” 

## External Dependencies

- LLama3.3  (https://ollama.com/library/llama3.3, https://gitlab.huma-num.fr/path-os/automated-nace-classifier, https://doi.org/10.5281/zenodo.17471199)
- ElasticSearch and Kibana (https://www.elastic.co/fr/elasticsearch, https://www.elastic.co/fr/kibana)
- Node.js (https://nodejs.org/fr)
- IPinfo’s Academic Research Program (https://ipinfo.io/use-cases/ip-data-for-academic-research)
- Eurostat NACE Rev. 2.1 (https://ec.europa.eu/eurostat/web/nace)

➡️ For detailed Node.js dependencies, see directly above in package.json files. 

## Contact

simon.apartis@cnrs.fr
tommaso.venturini@cnrs.fr
melanie.dulong@cnrs.fr
paul@ouestware.com