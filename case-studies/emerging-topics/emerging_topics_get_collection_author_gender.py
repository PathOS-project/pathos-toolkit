"""

In this file we are going to get the gender from the authors of the collections used in the Emerging Topics Case Study.

Date: 19/6/2024
Collection: GROUP C

"""

import glob
import json
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Load all the parquet files for GROUP C
p_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/*.parquet'))
papers_emergingtopics_group_C_df = pd.concat([pd.read_parquet(p_file) for p_file in p_files]).drop_duplicates(['id'])

# Define the collection of papers for the tools
collection_df = papers_emergingtopics_group_C_df

# Get all names of all authors
all_author_names = sorted(set([i['name'] for s in [x.tolist() for x in collection_df['authors']] for i in s]))

# Load gender classification model
pipe = pipeline("text-classification", model="padmajabfrl/Gender-Classification", device_map='cuda')

# Create a dictionary with the author name as key and the gender as value
author_name_gender_dict = dict()
for author_name in tqdm(all_author_names):
    author_gender = pipe(author_name)
    author_name_gender_dict[author_name] = author_gender[0]

# Save the dictionary
with open('PATH_TO_INTERMEDIATE_RESULTS/author_name_gender_dict_groupC.json', 'w') as f:
    json.dump(author_name_gender_dict, f)
