"""

In this file we are going to find the openaire ids of those papers that have dois

NOTE: This requires access to the Semantic Scholar Academic Graph dump and the OpenAIRE Graph dump

"""

import json
import glob
import pandas as pd
from tqdm import tqdm, trange

# Read the folder with the parquet files
# NOTE: This requires access to the papers that we identified for the Impact of Artefact Reuse in COVID-19 Publications case study using CORD-19 and Semantic Scholar
p_files = sorted(glob.glob('PATH_TO_SS_COVID_PAPERS/*.parquet'))

# Load all the parquet files
collection_df = pd.concat([pd.read_parquet(p_files[i]) for i in trange(len(p_files))]).drop_duplicates(['id'])

# Get the dois of the papers
collection_dois = set(collection_df['doi'].dropna().apply(str.lower).unique())

# Find openaire ids for those dois
# NOTE: This requires a dump of the OpenAIRE Graph
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
with open('PATH_TO_INTEMEDIATE_RESULTS/papers_covid_openaireids.json', 'w') as f:
    json.dump(found_data, f, indent=1)
