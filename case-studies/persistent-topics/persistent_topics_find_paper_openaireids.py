"""

In this file we are going to find the openaire ids of those papers that have dois

NOTE: This requires access to the Semantic Scholar Academic Graph dump and the OpenAIRE Graph dump

"""

import json
import glob
import pandas as pd
from tqdm import tqdm

# Load all the parquet files for GROUP C
# NOTE: This requires access to the papers that we identified for the Impact of Open Access Colors on Topic Persistence case study using Semantic Scholar and SciNoBo
p_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C/*.parquet'))
papers_emergingtopics_group_C_df = pd.concat([pd.read_parquet(p_file) for p_file in p_files]).drop_duplicates(['id'])

# Define the collection of papers for the tools
collection_df = papers_emergingtopics_group_C_df

# Get the dois of the papers
collection_dois = set(collection_df['doi'].dropna().apply(str.lower).unique())

# Find openaire ids for those dois
parquet_files = sorted(glob.glob('PATH_TO_OAIRE_DUMP/result_instance_pid/*'))

found_data = []
for p_file in tqdm(parquet_files):
    # Load the data
    data = pd.read_parquet(p_file)

    # Filter the rows that have dois in the collection (filter the pid column)
    data = data[data['pid'].apply(lambda x: x.lower() in collection_dois)]

    # Convert the data to a list
    data_list = data.to_dict(orient='records')

    # Add the data to the found_data list
    found_data.extend(data_list)

# Save the found data as a json
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_openaireids.json', 'w') as f:
    json.dump(found_data, f, indent=1)
