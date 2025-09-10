"""

In this file we are going to find affiliations of the papers using the openaire ids

NOTE: This requires access to the Semantic Scholar Academic Graph dump, the OpenAIRE Graph dump and SciNoBo results

"""

import json
import glob
import pandas as pd
from tqdm import tqdm

""" LOAD THE PAPERS """

# Load all the parquet files for GROUP C
# NOTE: This requires access to the papers that we identified for the Impact of Open Access Colors on Topic Persistence case study using Semantic Scholar and SciNoBo
p_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C/*.parquet'))
papers_emergingtopics_group_C_df = pd.concat([pd.read_parquet(p_file) for p_file in p_files]).drop_duplicates(['id'])

# Define the collection of papers for the tools
collection_df = papers_emergingtopics_group_C_df

""" FIRST PASS - GET THE ORG OA IDS """

# Get the openaire ids for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_openaireids.json', 'r') as f:
    doi_openaireids = json.load(f)

# Get the list of openaire ids
openaire_ids = set([d['result'] for d in doi_openaireids])

# Find openaire ids for those dois
parquet_files = sorted(glob.glob('PATH_TO_OAIRE_DUMP/result_affiliation/*'))

found_data = []
for p_file in tqdm(parquet_files):
    # Load the data
    data = pd.read_parquet(p_file)

    # Filter the rows that have openaire ids in the collection (filter the id column)
    data = data[data['id'].apply(lambda x: x in openaire_ids)]

    # Convert the data to a list
    data_list = data.to_dict(orient='records')

    # Add the data to the found_data list
    found_data.extend(data_list)

# Save the found data as a json
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids.json', 'w') as f:
    json.dump(found_data, f, indent=1)

""" SECOND PASS - GET THE ORG NAMES AND IDENTIFIERS"""

# Load the affiliations org oaids
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids.json', 'r') as f:
    org_oaids = json.load(f)

# Create a set of all organization openaire ids
org_oaids_set = set([d['organization'] for d in org_oaids])

# Find pids for those org oaids
parquet_files = sorted(glob.glob('PATH_TO_OAIRE_DUMP/organization/*'))

found_data_meta = []
for p_file in tqdm(parquet_files):
    # Load the data
    data = pd.read_parquet(p_file)

    # Filter the rows that have openaire ids in the collection (filter the id column)
    data = data[data['id'].apply(lambda x: x in org_oaids_set)]

    # Convert the data to a list
    data_list = data.to_dict(orient='records')

    # Add the data to the found_data list
    found_data_meta.extend(data_list)

# Save the found data as a json
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids_meta.json', 'w') as f:
    json.dump(found_data_meta, f, indent=1)

parquet_files = sorted(glob.glob('PATH_TO_OAIRE_DUMP/organization_pids/*'))

found_data_pids = []
for p_file in tqdm(parquet_files):
    # Load the data
    data = pd.read_parquet(p_file)

    # Filter the rows that have openaire ids in the collection (filter the id column)
    data = data[data['id'].apply(lambda x: x in org_oaids_set)]

    # Convert the data to a list
    data_list = data.to_dict(orient='records')

    # Add the data to the found_data list
    found_data_pids.extend(data_list)

# Save the found data as a json
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids_pids.json', 'w') as f:
    json.dump(found_data_pids, f, indent=1)
