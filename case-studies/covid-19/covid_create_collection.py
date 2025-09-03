import os
import json
import glob
import gzip
import requests
import pandas as pd
from tqdm import trange, tqdm

"""

We have all the paper files from Semantic Scholar that are also in CORD-19. Now we will find the metadata for the papers to use with our tools.

NOTE: Here we have identified all the semantic scholar IDs from CORD-19 and downloaded their metadata from Semantic Scholar Academic Graph. This requires access to the full data from Semantic Scholar Academic Graph.

"""

# Read the folder with the parquet files
p_files = sorted(glob.glob('PATH_TO_SS_COVID_PAPERS/*.parquet'))

# Load all the parquet files
papers_covid_df = pd.concat([pd.read_parquet(p_files[i]) for i in trange(len(p_files))]).drop_duplicates(['id'])

# Filter papers until the year 2021 to avoid citation bias toward recent publications
papers_covid_df = papers_covid_df[papers_covid_df['year'] <= 2021]

""" Find the open access info from OpenAIRE """

# NOTE: This requires a token for OpenAIRE Graph: https://develop.openaire.eu/user-info?errorCode=1&redirectUrl=%2Fpersonal-token

# Function to get the dois info from the openaire api
def get_dois_info_from_openaire_api(dois_list):
    url = 'https://api.openaire.eu/search/publications?format=json&size=100'
    url += '&doi=' + ','.join(dois_list)
    headers = {
        'Authorization': 'Bearer MY_OPENAIRE_ID_TOKEN'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['response']['results']['result']
    else:
        print('Error dois:', dois_list)
        raise Exception('Error getting the dois info from the openaire api')

find_closed = False

if find_closed:
    # Define the output file path for the OpenAIRE API results
    openaire_api_results_file = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_nonoa_openaire_api_results.jsonl.gz'

    # Define a file to store the list of queried DOIs
    queried_dois_file = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_nonoa_openaire_queried_dois.txt'

    # Get the dois from the papers_covid_df that 'isopenaccess' is False
    api_dois_list = papers_covid_df[papers_covid_df['isopenaccess'] == False]['doi'].dropna().unique().tolist()

    # Convert them to lowercase
    api_dois_list = [x.lower() for x in api_dois_list]

    # Check if the queried DOIs file exists to load previously processed DOIs
    queried_dois = set()
    if os.path.exists(queried_dois_file):
        try:
            with open(queried_dois_file, 'r') as f:
                for line in f:
                    queried_dois.add(line.strip())
            print(f"Loaded {len(queried_dois)} previously queried DOIs")
        except Exception as e:
            print(f"Error loading previously queried DOIs: {e}")

    # Function to load existing results
    def load_existing_results():
        if not os.path.exists(openaire_api_results_file):
            return []
        
        existing_results = []
        try:
            with gzip.open(openaire_api_results_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        existing_results.append(json.loads(line))
            print(f"Loaded {len(existing_results)} existing results")
            return existing_results
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []

    # Load existing results
    existing_results = load_existing_results()

    # Filter out already processed DOIs
    new_dois_list = [doi.strip() for doi in api_dois_list if doi not in queried_dois]
    print(f"Processing {len(new_dois_list)} new DOIs out of {len(api_dois_list)} total")

    # In batches of 50 dois, get the dois info from the openaire api
    api_dois_info = existing_results.copy()
    try:
        for i in trange(0, len(new_dois_list), 50):
            api_dois_batch = new_dois_list[i:i+50]
            try:
                batch_results = get_dois_info_from_openaire_api(api_dois_batch)
                api_dois_info.extend(batch_results)
                
                # Append new results to file
                with gzip.open(openaire_api_results_file, 'at', encoding='utf-8') as f:
                    for result in batch_results:
                        f.write(json.dumps(result) + '\n')
                
                # Append new DOIs to file
                with open(queried_dois_file, 'a') as f:
                    for doi in api_dois_batch:
                        f.write(doi + '\n')
                
                # Update the set of queried DOIs with this batch
                queried_dois.update(api_dois_batch)
            except Exception as e:
                print('Error getting the dois info from the openaire api:', e)
                break
    except Exception as e:
        print(f"Unexpected error: {e}")

    exit()

find_open = False
if find_open:
    # Define the output file path for the OpenAIRE API results
    openaire_api_results_file = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_oa_openaire_api_results.jsonl.gz'

    # Define a file to store the list of queried DOIs
    queried_dois_file = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_oa_openaire_queried_dois.txt'

    # Get the dois from the papers_covid_df that 'isopenaccess' is False
    api_dois_list = papers_covid_df[papers_covid_df['isopenaccess'] == True]['doi'].dropna().unique().tolist()

    # Convert them to lowercase
    api_dois_list = [x.lower() for x in api_dois_list]

    # Check if the queried DOIs file exists to load previously processed DOIs
    queried_dois = set()
    if os.path.exists(queried_dois_file):
        try:
            with open(queried_dois_file, 'r') as f:
                for line in f:
                    queried_dois.add(line.strip())
            print(f"Loaded {len(queried_dois)} previously queried DOIs")
        except Exception as e:
            print(f"Error loading previously queried DOIs: {e}")

    # Function to load existing results
    def load_existing_results():
        if not os.path.exists(openaire_api_results_file):
            return []
        
        existing_results = []
        try:
            with gzip.open(openaire_api_results_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        existing_results.append(json.loads(line))
            print(f"Loaded {len(existing_results)} existing results")
            return existing_results
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []

    # Load existing results
    existing_results = load_existing_results()

    # Filter out already processed DOIs
    new_dois_list = [doi.strip() for doi in api_dois_list if doi not in queried_dois]
    print(f"Processing {len(new_dois_list)} new DOIs out of {len(api_dois_list)} total")

    # In batches of 50 dois, get the dois info from the openaire api
    api_dois_info = existing_results.copy()
    try:
        for i in trange(0, len(new_dois_list), 50):
            api_dois_batch = new_dois_list[i:i+50]
            try:
                batch_results = get_dois_info_from_openaire_api(api_dois_batch)
                api_dois_info.extend(batch_results)
                
                # Append new results to file
                with gzip.open(openaire_api_results_file, 'at', encoding='utf-8') as f:
                    for result in batch_results:
                        f.write(json.dumps(result) + '\n')
                
                # Append new DOIs to file
                with open(queried_dois_file, 'a') as f:
                    for doi in api_dois_batch:
                        f.write(doi + '\n')
                
                # Update the set of queried DOIs with this batch
                queried_dois.update(api_dois_batch)
            except Exception as e:
                print('Error getting the dois info from the openaire api:', e)
                break
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    exit()


# Load the dois info from the openaire api for open access papers
dois_info = []
with gzip.open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_oa_openaire_api_results.jsonl.gz', 'rt', encoding='utf-8') as f:
    # Count lines first to get total for progress bar
    total_lines = sum(1 for _ in f)
    f.seek(0)  # Reset file pointer to beginning
    
    for line in tqdm(f, total=total_lines, desc="Loading OA DOIs"):
        if line.strip():  # Skip empty lines
            dois_info.append(json.loads(line))

# Load the dois info from the openaire api for non-open access papers
dois_info_nonoa = []
with gzip.open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_nonoa_openaire_api_results.jsonl.gz', 'rt', encoding='utf-8') as f:
    # Count lines first to get total for progress bar
    total_lines = sum(1 for _ in f)
    f.seek(0)  # Reset file pointer to beginning
    
    for line in tqdm(f, total=total_lines, desc="Loading non-OA DOIs"):
        if line.strip():  # Skip empty lines
            dois_info_nonoa.append(json.loads(line))

# Combine the dois info from the openaire api
dois_info = dois_info + dois_info_nonoa

# Find the colors of open access for the dois info
doi_colors_info = {}
for doi_info in tqdm(dois_info):
    # Find the doi
    pid = doi_info['metadata']['oaf:entity']['oaf:result']['pid']
    dois = list()
    if isinstance(pid, dict):
        pid = [pid]
    for x in pid:
        if x['@classid'] == 'doi':
            dois.append(x['$'])

    isgreen = doi_info['metadata']['oaf:entity']['oaf:result']['isgreen']['$'] if 'isgreen' in doi_info['metadata']['oaf:entity']['oaf:result'] else None
    openaccesscolor = doi_info['metadata']['oaf:entity']['oaf:result']['openaccesscolor']['$'] if 'openaccesscolor' in doi_info['metadata']['oaf:entity']['oaf:result'] else None
    isindiamondjournal = doi_info['metadata']['oaf:entity']['oaf:result']['isindiamondjournal']['$'] if 'isindiamondjournal' in doi_info['metadata']['oaf:entity']['oaf:result'] else None
    
    # Extract bestaccessright information
    bestaccessright = doi_info['metadata']['oaf:entity']['oaf:result']['bestaccessright']['@classid'] if 'bestaccessright' in doi_info['metadata']['oaf:entity']['oaf:result'] else None

    for doi in dois:
        doi_colors_info[doi] = {
            'isgreen': isgreen, 
            'openaccesscolor': openaccesscolor, 
            'isindiamondjournal': isindiamondjournal, 
            'bestaccessright': bestaccessright,
            'doi_info': doi_info
        }

# Free up memory
del dois_info
del dois_info_nonoa

# Check those that are not green, do not have any color and are not in diamond journal
unknown_color_dois = {x: doi_colors_info[x] for x in doi_colors_info if not (doi_colors_info[x]['isgreen'] or doi_colors_info[x]['isindiamondjournal'] or (doi_colors_info[x]['openaccesscolor'] is not None))}

# Add a new column to the collection_openaccess dataframe with the best color info
# colors : green (isgreen) , bronze (openaccesscolor) , hybrid (openaccesscolor) , gold (openaccesscolor) , diamond (isindiamondjournal)
# NOTE: lower() is used as the openaire api returns the dois as lowercase
papers_covid_df['green'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['isgreen'] if x is not None and x.lower() in doi_colors_info else None)
papers_covid_df['bronze'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='bronze' if x is not None and x.lower() in doi_colors_info else None)
papers_covid_df['hybrid'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='hybrid' if x is not None and x.lower() in doi_colors_info else None)
papers_covid_df['gold'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='gold' if x is not None and x.lower() in doi_colors_info else None)
papers_covid_df['diamond'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['isindiamondjournal'] if x is not None and x.lower() in doi_colors_info else None)

papers_covid_df['only_green'] = papers_covid_df['green'] & (papers_covid_df['bronze'].isin([False, None])) & (papers_covid_df['hybrid'].isin([False, None])) & (papers_covid_df['gold'].isin([False, None])) & (papers_covid_df['diamond'].isin([False, None]))
papers_covid_df['any_color'] = papers_covid_df['green'] | (papers_covid_df['bronze'] == True) | (papers_covid_df['hybrid'] == True) | (papers_covid_df['gold'] == True) | (papers_covid_df['diamond'] == True)
papers_covid_df['green_others'] = papers_covid_df['green'] & ((papers_covid_df['bronze'] == True) | (papers_covid_df['hybrid'] == True) | (papers_covid_df['gold'] == True) | (papers_covid_df['diamond'] == True))
papers_covid_df['only_others'] = (papers_covid_df['green'].isin([False, None])) & ((papers_covid_df['bronze'] == True) | (papers_covid_df['hybrid'] == True) | (papers_covid_df['gold'] == True) | (papers_covid_df['diamond'] == True))
papers_covid_df['no_color'] = (papers_covid_df['green'].isin([False, None])) & (papers_covid_df['bronze'].isin([False, None])) & (papers_covid_df['hybrid'].isin([False, None])) & (papers_covid_df['gold'].isin([False, None])) & (papers_covid_df['diamond'].isin([False, None]))

# If there is any OA color (green, bronze, hybrid, gold, diamond) then change the openaccess info to True (even if it was False)
papers_covid_df['isopenaccess_oaire'] = papers_covid_df['isopenaccess'].where(papers_covid_df['any_color'].isin([False, None]), True)

# Assign the best access right to the openaccess column
papers_covid_df['bestaccessright_oaire'] = papers_covid_df['doi'].map(lambda x: doi_colors_info[x.lower()]['bestaccessright'] if x is not None and x.lower() in doi_colors_info else None)

""" 

Find the pmids of the articles that cite the covid papers through Semantic Scholar 

NOTE: This requires the citations data from Semantic Scholar Academic Graph that have been identified to cite papers from the collection that we created.

"""

find_pmids = False

if find_pmids:
    covid_citations_folder = 'PATH_TO_SS_COVID_PAPER_CITATIONS/papers_covid_citations'

    # Get the list of all parque files in the folder
    parquet_files = glob.glob(f"{covid_citations_folder}/*.parquet")

    # List of all relevant source semantic scholar ids (or papers that cite the covid papers)
    source_ids = set()

    for file in tqdm(parquet_files):
        # Read the parquet file (it has source and dest for citations)
        df = pd.read_parquet(file)

        # Filter the dataframe to only keep the rows where the source is in the papers_covid_df
        df = df[df['dest'].isin(papers_covid_df['id'])]

        # Get the dest ids
        source_ids.update(df['source'].unique())
    
    # Load all the papers from Semantic Scholar Academic Graph
    # NOTE: This requires access to the full data from Semantic Scholar Academic Graph
    all_papers_folder = 'PATH_TO_SS_PAPERS'

    # Get the list of all parquet files in the folder
    all_papers_files = sorted(glob.glob(f"{all_papers_folder}/*.parquet"))

    for file in tqdm(all_papers_files):
        # Read the parquet file
        df = pd.read_parquet(file).drop_duplicates(['id', 'pmid'])

        # Filter the dataframe to only keep the rows where the id is in the source_ids
        df = df[df['id'].isin(source_ids)]

        # Keep only those that have pmid
        df = df[df['pmid'].notnull()]

        # Save the df to a new parquet file in 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citations_w_pmids' with the same name as the original file
        output_file = os.path.join('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citations_w_pmids', os.path.basename(file))
        df.to_parquet(output_file, index=False)


# Find the relevant pmids in the parsed pubmed data and classify them as clinical trials or clinical guidelines
find_relevant_pmids = False

if find_relevant_pmids:

    def classify_publication_type(publication_type: str) -> str:
        """
        Classifies a publication type as a Clinical Trial, Clinical Guideline, or Other.
        
        :param publication_type: The publication type to classify.
        :return: A string indicating the category ("Clinical Trial", "Clinical Guideline", "Other").
        """
        clinical_trials = {
            "Adaptive Clinical Trial",
            "Clinical Trial",
            "Clinical Trial Protocol",
            "Clinical Trial, Phase I",
            "Clinical Trial, Phase II",
            "Clinical Trial, Phase III",
            "Clinical Trial, Phase IV",
            "Clinical Trial, Veterinary",
            "Controlled Clinical Trial",
            "Equivalence Trial",
            "Pragmatic Clinical Trial",
            "Randomized Controlled Trial",
            "Randomized Controlled Trial, Veterinary"
        }
        
        clinical_guidelines = {
            "Guideline",
            "Practice Guideline",
            "Consensus Development Conference",
            "Consensus Development Conference, NIH"
        }
        
        if publication_type in clinical_trials:
            return "Clinical Trial"
        elif publication_type in clinical_guidelines:
            return "Clinical Guideline"
        else:
            return "Other"

    # Read the parquet files with the citations with pmids
    pmid_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citations_w_pmids/*.parquet'))
    pmid_df = pd.concat([pd.read_parquet(pmid_files[i]) for i in trange(len(pmid_files))]).drop_duplicates(['id', 'pmid'])

    # Get the set of pmids in the citations
    pmids_set = set(pmid_df['pmid'].dropna().unique())

    # Read the pubmed parsed files to find the relevant pmids and whether they are clinical trials or clinical guidelines
    # Create an empty dataframe to store the relevant pmids
    # NOTE: This requires access to the pubmed data and to parse them into parquet files
    pubmed_relevant_df = pd.DataFrame(columns=['pmid', 'clinical_trial', 'clinical_guideline'])
    pubmed_files = sorted(glob.glob('PATH_TO_PUBMED_PARSED_FILES/*.parquet'))
    for file in tqdm(pubmed_files):
        # Read the parquet file
        df = pd.read_parquet(file)

        # Filter the dataframe to only keep the rows where the pmid is in the pmid_df
        df = df[df['PMID'].isin(pmids_set)]

        # Classify if they are clinical trials or clinical guidelines
        df['clinical_trial'] = df['PublicationTypes'].str.split(';').apply(lambda x: any(classify_publication_type(y.strip()) == 'Clinical Trial' for y in x))
        df['clinical_guideline'] = df['PublicationTypes'].str.split(';').apply(lambda x: any(classify_publication_type(y.strip()) == 'Clinical Guideline' for y in x))
        
        # Keep only the relevant columns
        df = df[['PMID', 'clinical_trial', 'clinical_guideline']]

        # Rename the columns
        df = df.rename(columns={'PMID': 'pmid', 'clinical_trial': 'clinical_trial', 'clinical_guideline': 'clinical_guideline'})

        # Append the dataframe to the pubmed_relevant_df
        pubmed_relevant_df = pd.concat([pubmed_relevant_df, df], ignore_index=True)

    # Remove duplicates
    pubmed_relevant_df = pubmed_relevant_df.drop_duplicates(['pmid'])

    # Save the dataframe to a parquet file
    pubmed_relevant_df.to_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_pubmed_relevant_pmids.parquet', index=False)


"""

We have the final collection of papers. Now we need to gather the outcomes from the tools (e.g. SciNoBo) that have already been run.

NOTE: This requires access to the full data from Semantic Scholar Academic Graph and to the intermediate results from the tools that have been run (e.g. SciNoBo)

"""

print('Loading the outcomes from the tools:')

tools_path = "PATH_TO_TOOL_RESULTS"


""" 

OUTCOME: Interdisciplinarity 

NOTE: This was computed using SciNoBo.

"""

print('Loading the interdisciplinarity scores...', end='')

# Get the interdisciplinarity scores for the papers
interdisciplinarity_files = sorted(glob.glob(tools_path + '/interdisciplinarity/*.parquet'))
interdisciplinarity_df = pd.concat([pd.read_parquet(interdisciplinarity_files[i]) for i in trange(len(interdisciplinarity_files))]).drop_duplicates(['id'])

# Create a dictionary from id to interdisciplinarity
interdisciplinarity_df = interdisciplinarity_df.set_index('id')

# Create a new column in the collection_df with the interdisciplinarity scores
papers_covid_df['interdisciplinarity_macro'] = papers_covid_df['id'].map(interdisciplinarity_df['macro']).astype(int)
papers_covid_df['interdisciplinarity_meso'] = papers_covid_df['id'].map(interdisciplinarity_df['meso']).astype(int)

""" 

OUTCOME: FWCI

NOTE: This was computed using SciNoBo.

"""

print('Loading the FWCI scores...', end='')

# Get the FWCI scores for the papers
fwci_files = sorted(glob.glob(tools_path + '/fwci/*.parquet'))
fwci_df = pd.concat([pd.read_parquet(fwci_files[i]) for i in trange(len(fwci_files))])

# Create a dictionary from id to fwci
fwci_df = fwci_df.set_index('id')

# Create a new column in the collection_df with the fwci scores
papers_covid_df['fwci'] = papers_covid_df['id'].map(fwci_df['fwci'])

""" 

OUTCOME: Affiliation Analysis

NOTE: Please check "covid_find_paper_openaireids.py" and "covid_find_paper_affiliations.py"  for more details.

NOTE: This affiliation analysis will be integrated into the SciNoBo toolkit.

"""

# Get the openaire ids for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_openaireids.json', 'r') as f:
    doi_openaireids = json.load(f)

# Convert it to a dictionary from doi --> records . Each doi may contain multiple records
doi_openaireids_dict = {}
for x in doi_openaireids:
    if x['pid'] not in doi_openaireids_dict:
        doi_openaireids_dict[x['pid']] = list()
    doi_openaireids_dict[x['pid']].append(x)

# Get the affiliated orgs for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_affiliated_org_oaids.json', 'r') as f:
    affiliated_org_oaids = json.load(f)

# Convert it to a dictionary from id --> records . Each id may contain multiple records
affiliated_org_oaids_dict = {}
for x in affiliated_org_oaids:
    if x['id'] not in affiliated_org_oaids_dict:
        affiliated_org_oaids_dict[x['id']] = list()
    affiliated_org_oaids_dict[x['id']].append(x)

# Get the affiliated orgs metadata for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_affiliated_org_oaids_meta.json', 'r') as f:
    affiliated_org_oaids_meta = json.load(f)

# Convert it to a dictionary from id --> metadata
affiliated_org_oaids_meta_dict = {}
for x in affiliated_org_oaids_meta:
    if x['id'] not in affiliated_org_oaids_meta_dict:
        affiliated_org_oaids_meta_dict[x['id']] = x
    else:
        print('Duplicate id:', x['id'])

# Get the affiliated orgs for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_affiliated_org_oaids_pids.json', 'r') as f:
    affiliated_org_oaids_pids = json.load(f)

# Convert it to a dictionary from id --> records . Each id may contain multiple records
affiliated_org_oaids_pids_dict = {}
for x in affiliated_org_oaids_pids:
    if x['id'] not in affiliated_org_oaids_pids_dict:
        affiliated_org_oaids_pids_dict[x['id']] = list()
    affiliated_org_oaids_pids_dict[x['id']].append(x)

# Synthesize everything from doi to affiliated orgs with metadata and pids
doi_affiliated_orgs = {}
for doi in tqdm(doi_openaireids_dict):
    openaire_ids = [x['result'] for x in doi_openaireids_dict[doi]]
    # For each openaire id find the affiliated orgs
    affiliated_orgs = []
    for openaire_id in openaire_ids:
        if openaire_id in affiliated_org_oaids_dict:
            affiliated_orgs.extend([x['organization'] for x in affiliated_org_oaids_dict[openaire_id]])
    # Create a dictionary for the affiliated orgs to keep the metadata and pids
    affiliated_orgs_dict = {}
    # For each affiliated org find the metadata
    for x in affiliated_orgs:
        affiliated_orgs_dict[x] = dict()
        if x in affiliated_org_oaids_meta_dict:
            affiliated_orgs_dict[x]['metadata'] = affiliated_org_oaids_meta_dict[x]
        if x in affiliated_org_oaids_pids_dict:
            affiliated_orgs_dict[x]['pids'] = affiliated_org_oaids_pids_dict[x]
    # Add the dictionary to the doi_affiliated_orgs
    doi_affiliated_orgs[doi] = affiliated_orgs_dict

# Load the ROR data to find the type of each organization
# NOTE: This requires access to the ROR data
with open('PATH_TO_ROR_DATA/v1.54-2024-10-21-ror-data_schema_v2.json', 'r') as f:
    ror_data = json.load(f)

# Convert to a dictionary from id to the record
ror_data_dict = {x['id']: x for x in ror_data}

# Find the ror id for each affiliated org
for doi in tqdm(doi_affiliated_orgs):
    for org in doi_affiliated_orgs[doi]:
        if 'pids' in doi_affiliated_orgs[doi][org]:
            for pid in doi_affiliated_orgs[doi][org]['pids']:
                if pid['type']=='ROR':
                    # Find the ror id in the ror_data
                    ror_id = pid['pid']
                    ror_instance = ror_data_dict[ror_id] if ror_id in ror_data_dict else None
                    if ror_instance:
                        if 'ror' not in doi_affiliated_orgs[doi][org]:
                            doi_affiliated_orgs[doi][org]['ror'] = [ror_instance]
                        else:
                            doi_affiliated_orgs[doi][org]['ror'].append(ror_instance)

# All types of affiliated orgs: {'nonprofit', 'education', 'other', 'facility', 'government', 'archive', 'healthcare', 'company', 'funder'}
# Education: A university or similar institution involved in providing education and educating/employing researchers
# Healthcare: A medical care facility such as hospital or medical clinic. Excludes medical schools, which should be categorized as “Education”.
# Company: A private for-profit corporate entity involved in conducting or sponsoring research.
# Archive: An organization involved in stewarding research and cultural heritage materials. Includes libraries, museums, and zoos.
# Nonprofit: A non-profit and non-governmental organization involved in conducting or funding research.
# Government: An organization that is part of or operated by a national or regional government and that conducts or supports research.
# Facility: A specialized facility where research takes place, such as a laboratory or telescope or dedicated research area.
# Other: Use this category for any organization that does not fit the categories above.

# # Find the types of affiliated orgs for each doi
# # NOTE: Uncomment this if you want the orgs as well
# doi_affiliated_orgs_types = {}
# for doi in tqdm(doi_affiliated_orgs):
#     doi_affiliated_orgs_types[doi] = set()
#     for org in doi_affiliated_orgs[doi]:
#         if 'ror' in doi_affiliated_orgs[doi][org]:
#             for ror_instance in doi_affiliated_orgs[doi][org]['ror']:
#                 if 'types' in ror_instance:
#                     for ror_type in ror_instance['types']:
#                         if ror_type not in doi_affiliated_orgs_types[doi]:
#                             doi_affiliated_orgs_types[doi][ror_type] = set()
#                         doi_affiliated_orgs_types[doi][ror_type].add(org)

# Find the types of affiliated orgs for each doi
doi_affiliated_orgs_types = {}
for doi in tqdm(doi_affiliated_orgs):
    doi_affiliated_orgs_types[doi] = set()
    for org in doi_affiliated_orgs[doi]:
        if 'ror' in doi_affiliated_orgs[doi][org]:
            for ror_instance in doi_affiliated_orgs[doi][org]['ror']:
                if 'types' in ror_instance:
                    for ror_type in ror_instance['types']:
                        doi_affiliated_orgs_types[doi].add(ror_type)
    doi_affiliated_orgs_types[doi] = sorted(doi_affiliated_orgs_types[doi])

# Create a dictionary from doi to whether there is science-industry collaboration or not, we consider 
# Science: 'Education', 'Healthcare', 'Nonprofit', 'Government', 'Facility', 'Archive'
# Industry: 'Company'
# NOTE: 'Other' is not considered
# Classify as True if there is at least one Science and one Industry
doi_science_industry_collaboration = {}
for doi in doi_affiliated_orgs_types:
    if 'company' in doi_affiliated_orgs_types[doi] and any(x in doi_affiliated_orgs_types[doi] for x in ['education', 'healthcare', 'nonprofit', 'government', 'facility', 'archive']):
        doi_science_industry_collaboration[doi] = True
    else:
        doi_science_industry_collaboration[doi] = False

# Map the dois to the collection_df
papers_covid_df['affiated_org_types'] = papers_covid_df['doi'].map(doi_affiliated_orgs_types)
papers_covid_df['science_industry_collaboration'] = papers_covid_df['doi'].map(doi_science_industry_collaboration)

""" 

OUTCOME: Publication-Patent Citation Analysis 

NOTE: This requires access to the PATSTAT data, parsed and processed by OPIX PC.

NOTE: This affiliation analysis will be integrated into the SciNoBo toolkit.

"""

# Load the doi-patents data
doi_patents = pd.read_parquet('PATH_TO_PATSTAT_DATA/1-doi_patent.parquet')

# Load the patent metadata
# Index(['appln_id', 'appln_auth', 'appln_nr', 'appln_kind', 'appln_filing_date',
#        'appln_filing_year', 'appln_nr_epodoc', 'appln_nr_original', 'ipr_type',
#        'receiving_office', 'internat_appln_id', 'int_phase', 'reg_phase',
#        'nat_phase', 'earliest_filing_date', 'earliest_filing_year',
#        'earliest_filing_id', 'earliest_publn_date', 'earliest_publn_year',
#        'earliest_pat_publn_id', 'granted', 'docdb_family_id',
#        'inpadoc_family_id', 'docdb_family_size', 'nb_citing_docdb_fam',
#        'nb_applicants', 'nb_inventors', 'appln_title_lg', 'appln_title',
#        'appln_abstract_lg', 'appln_abstract', 'ipc', 'cpc', 'nace2',
#        'applicant', 'inventor', 'publication', 'technologies'],
#       dtype='object')
patent_metadata = pd.read_parquet('PATH_TO_PATSTAT_DATA/2-doi_patent_with_metadata.parquet')

# Convert to dictionary from doi to patents
doi_patents_dict = doi_patents.groupby('doi')['appln_id'].apply(list).to_dict()

# Find how many patent citations each paper has
# NOTE: if a doi is not found in the doi_patents_dict then it has NaN patent citations, so that we do not include them in any future aggregation
papers_covid_df['patent_citations'] = papers_covid_df['doi'].map(lambda x: len(doi_patents_dict[x]) if x in doi_patents_dict else (0 if x is not None else None))


""" 

OUTCOME: Citance Analysis 

NOTE: This was computed using SciNoBo and requires access to the Semantic Scholar Academic Graph paper citations with their respective contexts (citances).

"""

calculate_citance_analysis = False

if calculate_citance_analysis:
    # Use pandas tqdm
    tqdm.pandas()

    citance_analysis_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citance_results/*.json.gz'))
    citance_analysis_df = pd.concat([pd.read_json(citance_analysis_files[i], compression='gzip') for i in trange(len(citance_analysis_files))])

    # For each paper in the papers_covid_df, we need to find the following:
    # - Number of Supporting, Neutral, and Refuting Citances (in general)
    # - Number of Supporting, Neutral, and Refuting Citances (for Reuse - Artifact)

    # Keep in mind that we must group the citances as: 
    # - inbound to the paper (cited by others -- the paper "id" in the "dest" column) 
    # - outbound from the paper (citing others -- the paper "id" in the "source" column)

    def extract_polarity_counts(citance_results):
        supporting, neutral, refuting = 0, 0, 0
        supporting_reuse_artifact, neutral_reuse_artifact, refuting_reuse_artifact = 0, 0, 0

        for citance in citance_results:
            citation_marks = [x for x in citance if x != '']
            citation_mark = citation_marks[0] if citation_marks else ''

            polarity = citance[citation_mark]['polarity']
            semantics = citance[citation_mark]['semantics']
            intent = citance[citation_mark]['intent']

            # General citances
            if polarity == 'Supporting':
                supporting += 1
            elif polarity == 'Neutral':
                neutral += 1
            elif polarity == 'Refuting':
                refuting += 1

            # Reuse - Artifact citances
            if semantics == 'Artifact' and intent == 'Reuse':
                if polarity == 'Supporting':
                    supporting_reuse_artifact += 1
                elif polarity == 'Neutral':
                    neutral_reuse_artifact += 1
                elif polarity == 'Refuting':
                    refuting_reuse_artifact += 1

        return pd.Series([supporting, neutral, refuting, supporting_reuse_artifact, neutral_reuse_artifact, refuting_reuse_artifact],
                        index=['supporting', 'neutral', 'refuting', 'supporting_reuse_artifact', 'neutral_reuse_artifact', 'refuting_reuse_artifact'])

    # Apply preprocessing to extract all counts in one go
    citance_analysis_df[['supporting', 'neutral', 'refuting', 'supporting_reuse_artifact', 'neutral_reuse_artifact', 'refuting_reuse_artifact']] = citance_analysis_df['results'].progress_apply(extract_polarity_counts)

    # WARNING: THE BELOW CODE TAKES ~15 MINS WITHOUT PROGRESS BAR

    print('Grouping the results...')

    # Group by inbound (dest) and outbound (source) to count citations
    inbound_counts = citance_analysis_df.groupby('dest')[['supporting', 'neutral', 'refuting', 'supporting_reuse_artifact', 'neutral_reuse_artifact', 'refuting_reuse_artifact']].sum()
    outbound_counts = citance_analysis_df.groupby('source')[['supporting', 'neutral', 'refuting', 'supporting_reuse_artifact', 'neutral_reuse_artifact', 'refuting_reuse_artifact']].sum()

    # Save the inbound and outbound counts as a parquet file (for future use -- only the counts)
    inbound_counts.to_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citance_inbound_counts.parquet')
    outbound_counts.to_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citance_outbound_counts.parquet')

# Load the inbound and outbound counts
inbound_counts = pd.read_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citance_inbound_counts.parquet')
outbound_counts = pd.read_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citance_outbound_counts.parquet')

print('Merging the results...')

# Rename columns before merging to ensure correct naming
inbound_counts = inbound_counts.add_suffix('_inbound')
outbound_counts = outbound_counts.add_suffix('_outbound')

# Merge the results with `papers_covid_df`
papers_covid_df = papers_covid_df.merge(inbound_counts, how='left', left_on='id', right_index=True)
papers_covid_df = papers_covid_df.merge(outbound_counts, how='left', left_on='id', right_index=True)


""" 

OUTCOME: Research Artefact Analysis (RAA)

NOTE: This was computed using SciNoBo and requries access to the Semantic Scholar Academic Graph full-text papers.

"""

import numpy as np
import gzip

calculate_artifact_analysis = False

if calculate_artifact_analysis:
    from raa_reevaluate import reevaluate

    raa_files = sorted(glob.glob('PATH_TO_RAA_OUTPUT/*.json.gz'))

    # Create an empty dict to store the results
    results = {}

    # For each file, load the data and do the analysis
    for raa_file in tqdm(raa_files):
        raa_df = pd.read_json(raa_file, compression='gzip')

        # Iterate over the rows
        for i in range(raa_df.shape[0]):
            row = raa_df.iloc[i]

            # Get the id of the paper
            paper_id = row['pdf_metadata']['id']

            # Get the artifacts
            artifacts = row['research_artifacts']['grouped_clusters']
            
            # Reevaluate the and get the results
            artifact_thresholds = {
                'artifact_answer': 0.8,
                'ownership_answer': 0.8,
                'reuse_answer': 0.8,
            }
            artifacts_simple = reevaluate(artifacts, new_thresholds=artifact_thresholds, verbose=False)

            # Research Artifacts have a field called "RA Cluster". This is the cluster number that the artifact belongs to.
            # Group the artifacts by the cluster id and have lists of same cluster artifacts
            artifact_clusters = {}
            for artifact in artifacts_simple['research_artifacts']:
                if artifact['RA Cluster'] not in artifact_clusters:
                    artifact_clusters[artifact['RA Cluster']] = []
                artifact_clusters[artifact['RA Cluster']].append(artifact)
            # Filter out the unnamed clusters and those that have only one artifact
            artifact_clusters = {k: v for k, v in artifact_clusters.items() if '_unnamed' not in k and len(v) > 1}
            # Group and re-evaluate the artifacts in the clusters and add them to a new list
            clustered_artifacts = []
            for cluster_id in artifact_clusters:
                cluster = artifact_clusters[cluster_id]
                # Ownership
                owned = 'Yes' if np.mean([x['Owned Score'] for x in cluster]) >= artifact_thresholds['ownership_answer'] else 'No' 
                # Reuse
                reused = 'Yes' if np.mean([x['Reused Score'] for x in cluster]) >= artifact_thresholds['reuse_answer'] else 'No'
                # Find the longest name
                longest_name = max([x['Research Artifact'] for x in cluster], key=len)
                # Add to the dictionary
                clustered_artifacts.append({
                    'RA Cluster': cluster_id,
                    'Research Artifact': longest_name,
                    'Type': cluster[0]['Type'],
                    'Research Artifact Score': np.mean([x['Research Artifact Score'] for x in cluster]),
                    'Owned': owned,
                    'Owned Percentage': np.mean([x['Owned Percentage'] for x in cluster]),
                    'Owned Score': np.mean([x['Owned Score'] for x in cluster]),
                    'Reused': reused,
                    'Reused Percentage': np.mean([x['Reused Percentage'] for x in cluster]),
                    'Reused Score': np.mean([x['Reused Score'] for x in cluster]),
                    'Licenses': '\n'.join(sorted(set([x['Licenses'] for x in cluster]))),
                    'Versions': '\n'.join(sorted(set([x['Versions'] for x in cluster]))),
                    'URLs': '\n'.join(sorted(set([x['URLs'] for x in cluster]))),
                    'Citations': '\n'.join(sorted(set([x['Citations'] for x in cluster]))),
                    'Mentions Count': sum([x['Mentions Count'] for x in cluster])
                })

            # Remove the artifacts in the above clusters from the original list
            artifacts_simple['research_artifacts'] = [x for x in artifacts_simple['research_artifacts'] if x['RA Cluster'] not in artifact_clusters]
            # Add the clustered artifacts to the list
            artifacts_simple['research_artifacts'].extend(clustered_artifacts)

            # Find number of reused artifacts
            named_datasets = [x for x in artifacts_simple['research_artifacts'] if 'dataset' in x['Type'].lower() and 'Unnamed_' not in x['Research Artifact']]
            named_software = [x for x in artifacts_simple['research_artifacts'] if 'software' in x['Type'].lower() and 'Unnamed_' not in x['Research Artifact']]
            unnamed_datasets = [x for x in artifacts_simple['research_artifacts'] if 'dataset' in x['Type'].lower() and 'Unnamed_' in x['Research Artifact']]
            unnamed_software = [x for x in artifacts_simple['research_artifacts'] if 'software' in x['Type'].lower() and 'Unnamed_' in x['Research Artifact']]

            named_datasets_reused = sum([1 for x in named_datasets if x['Reused'].lower() == 'yes'])
            named_software_reused = sum([1 for x in named_software if x['Reused'].lower() == 'yes'])
            unnamed_datasets_reused = sum([1 for x in unnamed_datasets if x['Reused'].lower() == 'yes'])
            unnamed_software_reused = sum([1 for x in unnamed_software if x['Reused'].lower() == 'yes'])

            named_datasets_created = sum([1 for x in named_datasets if x['Owned'].lower() == 'yes'])
            named_software_created = sum([1 for x in named_software if x['Owned'].lower() == 'yes'])
            unnamed_datasets_created = sum([1 for x in unnamed_datasets if x['Owned'].lower() == 'yes'])
            unnamed_software_created = sum([1 for x in unnamed_software if x['Owned'].lower() == 'yes'])

            named_datasets_reused_names = [x['Research Artifact'] for x in named_datasets if x['Reused'].lower() == 'yes']
            named_software_reused_names = [x['Research Artifact'] for x in named_software if x['Reused'].lower() == 'yes']
            named_datasets_created_names = [x['Research Artifact'] for x in named_datasets if x['Owned'].lower() == 'yes']
            named_software_created_names = [x['Research Artifact'] for x in named_software if x['Owned'].lower() == 'yes']

            # Add all these to the results
            results[paper_id] = {
                'named_datasets_reused': named_datasets_reused,
                'named_software_reused': named_software_reused,
                'unnamed_datasets_reused': unnamed_datasets_reused,
                'unnamed_software_reused': unnamed_software_reused,
                'named_datasets_created': named_datasets_created,
                'named_software_created': named_software_created,
                'unnamed_datasets_created': unnamed_datasets_created,
                'unnamed_software_created': unnamed_software_created,
                'named_datasets_reused_names': named_datasets_reused_names,
                'named_software_reused_names': named_software_reused_names,
                'named_datasets_created_names': named_datasets_created_names,
                'named_software_created_names': named_software_created_names,
            }

    # Save the results to a json file for future use (to not run the above code each time)
    with gzip.open('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_artifact_analysis_results_final_08.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(results, f)
    
    exit()
    
# Load the results
raa_results_file = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_artifact_analysis_results_final_08.json.gz'

with gzip.open(raa_results_file, 'rt', encoding='utf-8') as f:
    results = json.load(f)

# Convert keys from str to int
results = {int(k): v for k, v in results.items()}

# Map these new columns to the papers_covid_df
papers_covid_df['named_datasets_reused'] = papers_covid_df['id'].map(lambda x: results[x]['named_datasets_reused'] if x in results else None)
papers_covid_df['named_software_reused'] = papers_covid_df['id'].map(lambda x: results[x]['named_software_reused'] if x in results else None)
papers_covid_df['unnamed_datasets_reused'] = papers_covid_df['id'].map(lambda x: results[x]['unnamed_datasets_reused'] if x in results else None)
papers_covid_df['unnamed_software_reused'] = papers_covid_df['id'].map(lambda x: results[x]['unnamed_software_reused'] if x in results else None)
papers_covid_df['named_datasets_created'] = papers_covid_df['id'].map(lambda x: results[x]['named_datasets_created'] if x in results else None)
papers_covid_df['named_software_created'] = papers_covid_df['id'].map(lambda x: results[x]['named_software_created'] if x in results else None)
papers_covid_df['unnamed_datasets_created'] = papers_covid_df['id'].map(lambda x: results[x]['unnamed_datasets_created'] if x in results else None)
papers_covid_df['unnamed_software_created'] = papers_covid_df['id'].map(lambda x: results[x]['unnamed_software_created'] if x in results else None)
papers_covid_df['named_datasets_reused_names'] = papers_covid_df['id'].map(lambda x: results[x]['named_datasets_reused_names'] if x in results else None)
papers_covid_df['named_software_reused_names'] = papers_covid_df['id'].map(lambda x: results[x]['named_software_reused_names'] if x in results else None)
papers_covid_df['named_datasets_created_names'] = papers_covid_df['id'].map(lambda x: results[x]['named_datasets_created_names'] if x in results else None)
papers_covid_df['named_software_created_names'] = papers_covid_df['id'].map(lambda x: results[x]['named_software_created_names'] if x in results else None)

""" 

OUTCOME: Citations from Clinical Trials and Clinical Guidelines 

NOTE: This requires access to PubMed data (see lines 291-412).

NOTE: This will be integrated into the SciNoBo toolkit.

"""

# Method: Take the citation DOIs, PMIDs, etc from the citing papers of each paper in the collection and check which of those are clinical trials or clinical guidelines through Pubmed

# Load the citations data
covid_citations_folder = 'PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citations'
parquet_files = glob.glob(f"{covid_citations_folder}/*.parquet")
covid_citations_df = pd.concat([pd.read_parquet(parquet_files[i])[['source', 'dest', 'isinfluential', 'citationid']] for i in trange(len(parquet_files))])

# Load the papers of the citations to the collection with pmids
pmid_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_citations_w_pmids/*.parquet'))
pmid_df = pd.concat([pd.read_parquet(pmid_files[i]) for i in trange(len(pmid_files))]).drop_duplicates(['id', 'pmid'])

# Load the relevant pmids
pubmed_relevant_df = pd.read_parquet('PATH_TO_INTERMEDIATE_RESULTS/papers_covid_pubmed_relevant_pmids.parquet')

# Check if there is a clinical trial that is also a clinical guideline
# This is proof that we can perform total_clinical_citations = clinical_trial_citations + clinical_guideline_citations
assert not (pubmed_relevant_df['clinical_trial'] & pubmed_relevant_df['clinical_guideline']).any(), "There are pmids that are both clinical trials and clinical guidelines"

# Map the clinical trial and clinical guideline columns to the pmid_df using the pmid column
pmid_df = pmid_df.merge(pubmed_relevant_df, on='pmid', how='left')
pmid_df['clinical_trial'] = pmid_df['clinical_trial'].fillna(False)
pmid_df['clinical_guideline'] = pmid_df['clinical_guideline'].fillna(False)

# Merge citation data with pmid_df to get clinical trial/guideline information
citations_with_metadata = covid_citations_df.merge(
    pmid_df[['id', 'clinical_trial', 'clinical_guideline']],
    left_on='source',  # source = papers doing the citing
    right_on='id',
    how='left'
)

# Create separate columns for influential and non-influential citations
citations_with_metadata['clinical_trial_influential'] = citations_with_metadata['clinical_trial'] & citations_with_metadata['isinfluential']
citations_with_metadata['clinical_guideline_influential'] = citations_with_metadata['clinical_guideline'] & citations_with_metadata['isinfluential']
citations_with_metadata['clinical_trial_non_influential'] = citations_with_metadata['clinical_trial'] & ~citations_with_metadata['isinfluential']
citations_with_metadata['clinical_guideline_non_influential'] = citations_with_metadata['clinical_guideline'] & ~citations_with_metadata['isinfluential']

# Group by destination papers (papers being cited) and count clinical trials/guidelines citations
citation_counts = citations_with_metadata.groupby('dest').agg({
    'clinical_trial': 'sum',  # Boolean sum counts True values
    'clinical_guideline': 'sum',
    'clinical_trial_influential': 'sum',
    'clinical_guideline_influential': 'sum',
    'clinical_trial_non_influential': 'sum',
    'clinical_guideline_non_influential': 'sum'
}).reset_index()

# Convert boolean results to integers
for col in ['clinical_trial', 'clinical_guideline', 'clinical_trial_influential', 
           'clinical_guideline_influential', 'clinical_trial_non_influential', 'clinical_guideline_non_influential']:
    citation_counts[col] = citation_counts[col].astype(int)

# Rename columns for clarity
citation_counts = citation_counts.rename(columns={
    'clinical_trial': 'clinical_trial_citations',
    'clinical_guideline': 'clinical_guideline_citations',
    'clinical_trial_influential': 'clinical_trial_citations_influential',
    'clinical_guideline_influential': 'clinical_guideline_citations_influential',
    'clinical_trial_non_influential': 'clinical_trial_citations_non_influential',
    'clinical_guideline_non_influential': 'clinical_guideline_citations_non_influential'
})

# Merge these counts into papers_covid_df
papers_covid_df = papers_covid_df.merge(
    citation_counts,
    left_on='id',
    right_on='dest',
    how='left'
)

# Fill NaN values with 0 (for papers with no clinical trial/guideline citations)
for col in ['clinical_trial_citations', 'clinical_guideline_citations', 
'clinical_trial_citations_influential', 'clinical_guideline_citations_influential',
           'clinical_trial_citations_non_influential', 'clinical_guideline_citations_non_influential']:
    papers_covid_df[col] = papers_covid_df[col].fillna(0).astype(int)

# Add total clinical citations column
papers_covid_df['total_clinical_citations'] = papers_covid_df['clinical_trial_citations'] + papers_covid_df['clinical_guideline_citations']
papers_covid_df['total_clinical_citations_influential'] = papers_covid_df['clinical_trial_citations_influential'] + papers_covid_df['clinical_guideline_citations_influential']
papers_covid_df['total_clinical_citations_non_influential'] = papers_covid_df['clinical_trial_citations_non_influential'] + papers_covid_df['clinical_guideline_citations_non_influential']


""" CREATE A COMPLETE COLLECTION DF FOR SAVING WITH ALL CALCULATED OUTCOMES """

# Create an aggregation of any artifact (dataset or software) by adding the numbers for the datasets and software columns to create new columns
papers_covid_df['named_artifacts_created'] = papers_covid_df['named_datasets_created'] + papers_covid_df['named_software_created']
papers_covid_df['unnamed_artifacts_created'] = papers_covid_df['unnamed_datasets_created'] + papers_covid_df['unnamed_software_created']

# Add-up the supporting, neutral and refuting citances for the reuse artifacts
papers_covid_df['reuse_artifact_inbound'] = papers_covid_df['supporting_reuse_artifact_inbound'] + papers_covid_df['neutral_reuse_artifact_inbound'] + papers_covid_df['refuting_reuse_artifact_inbound']

# Remove columns, so that we keep only the ones that are useful for our analyses
cols_remove = ['title', 'S2Url', 'doi', 'pmid', 'magId', 'externalids', 'publicationtypes', 'journalName', 'journalPages', 'journalVolume', 'venue', 'publicationvenueid', 'paperAbstract']
for col in cols_remove:
    if col in papers_covid_df.columns:
        papers_covid_df.drop(columns=[col], inplace=True)

# Add author count if not already present
if 'authorcount' not in papers_covid_df.columns and 'authors' in papers_covid_df.columns:
    papers_covid_df['authorcount'] = papers_covid_df['authors'].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )
    papers_covid_df['authorcount'] = papers_covid_df['authorcount'].fillna(0).astype(int)

# Save the complete collection df as an excel file and as a parquet file
papers_covid_df.to_excel(os.path.join('PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes', 'complete_collection_df_fix.xlsx'), index=False)
papers_covid_df.to_parquet(os.path.join('PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes', 'complete_collection_df_fix.parquet'), index=False)
