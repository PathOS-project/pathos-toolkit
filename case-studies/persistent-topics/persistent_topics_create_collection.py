import os
import glob
import json
import requests
import pandas as pd
from tqdm import tqdm, trange
import numpy as np

"""

NOTE: Here we have identified all the papers from Semantic Scholar that are AI-Climate related (group C) and downloaded their metadata from Semantic Scholar Academic Graph. This requires access to the full data from Semantic Scholar Academic Graph.

"""
# Load all the parquet files for GROUP C
p_files = sorted(glob.glob('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C/*.parquet'))
papers_emergingtopics_group_C_df = pd.concat([pd.read_parquet(p_file) for p_file in p_files]).drop_duplicates(['id'])

# Filter papers until the year 2021 to avoid citation bias toward recent publications
papers_emergingtopics_group_C_df = papers_emergingtopics_group_C_df[papers_emergingtopics_group_C_df['year'] <= 2021]
print(f"Filtered collection to papers until 2021. Collection size: {len(papers_emergingtopics_group_C_df)}")

# Define the collection of papers for the tools
collection_df = papers_emergingtopics_group_C_df

"""
> collection_df.columns

Index(['authors', 'id', 'title', 'S2Url', 'year', 'doi', 'pmid', 'magId',
       'externalids', 'publicationtypes', 'publicationdate', 'journalName',
       'journalPages', 'journalVolume', 'venue', 'publicationvenueid',
       'isopenaccess', 'referencecount', 'citationcount',
       'influentialcitationcount', 'paperAbstract', 'openaccessinfo'],
      dtype='object')
"""

"""

We have the final collection of papers. Now we need to gather the outcomes from the tools that have already been run.

NOTE: This requires access to the full data from Semantic Scholar Academic Graph and to the intermediate results from the tools that have been run (e.g. SciNoBo)

"""

print('Loading the outcomes from the tools:')

tools_path = "PATH_TO_TOOL_RESULTS"

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
collection_df['fwci'] = collection_df['id'].map(fwci_df['fwci'])

print('Done!')

""" 

OUTCOME: Gender Diversity 

NOTE: See the _emeging_zenodo_author_gender.py file for more details.

"""

with open('PATH_TO_INTERMEDIATE_RESULTS/author_name_gender_dict_groupC.json') as f:
    author_name_gender_dict = json.load(f)

# Add a column to the collection_df if there is at least one women author
# also convert them to int to be able to see stats with describe
collection_df['has_women_authors'] = collection_df['authors'].map(lambda x: any([author_name['name'] in author_name_gender_dict and author_name_gender_dict[author_name['name']]['label']=='Female' for author_name in x]))
collection_df['has_women_authors'] = collection_df['has_women_authors'].astype(int)
collection_df['has_woman_first_author'] = collection_df['authors'].map(lambda x: x[0]['name'] in author_name_gender_dict and author_name_gender_dict[x[0]['name']]['label']=='Female' if len(x)>0 else False)
collection_df['has_woman_first_author'] = collection_df['has_woman_first_author'].astype(int)
# Add a column to the collection_df if there is a woman last author
collection_df['has_woman_last_author'] = collection_df['authors'].map(lambda x: x[-1]['name'] in author_name_gender_dict and author_name_gender_dict[x[-1]['name']]['label']=='Female' if len(x)>0 else False)
collection_df['has_woman_last_author'] = collection_df['has_woman_last_author'].astype(int)
# Add a column to the collection_df if it has only women authors
collection_df['only_women_authors'] = collection_df['authors'].map(lambda x: all([author_name['name'] in author_name_gender_dict and author_name_gender_dict[author_name['name']]['label']=='Female' for author_name in x]))
collection_df['only_women_authors'] = collection_df['only_women_authors'].astype(int)

# First, identify papers with empty or None authors
empty_authors_mask = collection_df['authors'].apply(lambda x: x is None or len(x) == 0)

# Handle the edge case of empty authors for gender metrics
# Set all gender metrics to None where authors is None or empty
collection_df.loc[empty_authors_mask, 'has_women_authors'] = None
collection_df.loc[empty_authors_mask, 'has_woman_first_author'] = None
collection_df.loc[empty_authors_mask, 'has_woman_last_author'] = None
collection_df.loc[empty_authors_mask, 'only_women_authors'] = None

# Print the number of papers with empty authors list
print(f"Found {empty_authors_mask.sum()} papers with empty authors list")

""" 

OUTCOME: SDG Classification 

NOTE: This was computed using SciNoBo.

"""

# Get the SDG output
all_sdg_files = sorted(glob.glob('PATH_TO_TOOL_RESULTS/sdg/*.parquet'))

# Load the SDG classification
sdg_df = pd.concat([pd.read_parquet(sdg_file) for sdg_file in tqdm(all_sdg_files)])

# Filter the sdg_df to only include the papers in the collection_df
sdg_df = sdg_df[sdg_df['id'].isin(collection_df['id'])]

# Get the main sdg category for each id in the collection_df
collection_df['main_sdg_category'] = collection_df['id'].map(sdg_df.groupby('id').apply(lambda x: x.loc[x['score'].idxmax()]['sdg_category']))

# Get all the sdg categories for each id in the collection_df
collection_df['sdg_categories'] = collection_df['id'].map(sdg_df.groupby('id')['sdg_category'].apply(list))

""" 

OUTCOME: FOS Topics (L4-L6) 

NOTE: This was computed using SciNoBo.

"""

# Load the FOS data
fos_emergingtopics_df = pd.read_parquet('PATH_TO_TOOL_RESULTS/fos_emergingtopics.parquet')

# Filter the fos_emergingtopics_df to only include the papers in the collection_df
fos_emergingtopics_df = fos_emergingtopics_df[fos_emergingtopics_df['id'].isin(collection_df['id'])]

# Load the FOS taxonomy
with open('fos_taxonomy_v0.1.2.json', 'r') as f:
    fos_taxonomy = json.load(f)

# Create a mapping from 'level_5_id' to 'level_5_name'
fos_taxonomy_l5_mapping = {}
for x in fos_taxonomy:
    if x['level_5_id'] not in fos_taxonomy_l5_mapping:
        fos_taxonomy_l5_mapping[x['level_5_id']] = x['level_5_name']
    else:
        continue

# Create reverse mappings for L4, L5, L6 to (L1, L2, L3)
fos_taxonomy_l4_mapping_reverse = {}
fos_taxonomy_l5_mapping_reverse = {}

for x in fos_taxonomy:
    if x['level_4'] not in fos_taxonomy_l4_mapping_reverse:
        fos_taxonomy_l4_mapping_reverse[x['level_4']] = list()
    else:
        fos_taxonomy_l4_mapping_reverse[x['level_4']].append((x['level_1'], x['level_2'], x['level_3']))
    
    if x['level_5_name'] not in fos_taxonomy_l5_mapping_reverse:
        fos_taxonomy_l5_mapping_reverse[x['level_5_name']] = list()
    else:
        fos_taxonomy_l5_mapping_reverse[x['level_5_name']].append((x['level_1'], x['level_2'], x['level_3']))

# Deduplicate using set
for x in fos_taxonomy_l4_mapping_reverse:
    fos_taxonomy_l4_mapping_reverse[x] = list(set(fos_taxonomy_l4_mapping_reverse[x]))

for x in fos_taxonomy_l5_mapping_reverse:
    fos_taxonomy_l5_mapping_reverse[x] = list(set(fos_taxonomy_l5_mapping_reverse[x]))

# Map the ids of the collection_df to the names of the FOS L4, L5 and L6
collection_df['FOS L4'] = collection_df['id'].map(fos_emergingtopics_df.groupby('id')['L4'].apply(lambda x: [y for y in x if y != 'N/A']))
collection_df['FOS L5'] = collection_df['id'].map(fos_emergingtopics_df.groupby('id')['L5'].apply(lambda x: [fos_taxonomy_l5_mapping[y] for y in x if y in fos_taxonomy_l5_mapping]))
collection_df['FOS L6'] = collection_df['id'].map(fos_emergingtopics_df.groupby('id')['L6'].apply(lambda x: [y for y in x if str(y) != 'None']))


""" 

OUTCOME: Publication-Patent Citation Analysis 

NOTE: This Publication-Patent Citation Analysis will be integrated into the SciNoBo toolkit.

"""

# Load the doi-patents data
# NOTE: This requires access to the processed PATSTAT data from OPIX PC
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
collection_df['patent_citations'] = collection_df['doi'].map(lambda x: len(doi_patents_dict[x]) if x in doi_patents_dict else (0 if x is not None else None))


""" 

OUTCOME: Affiliation Analysis

NOTE: Please check "persistent_topics_find_paper_openaireids.py" and "persistent_topics_find_paper_affiliations.py"  for more details.

NOTE: This affiliation analysis will be integrated into the SciNoBo toolkit.

"""


# Get the openaire ids for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_openaireids.json', 'r') as f:
    doi_openaireids = json.load(f)

# Convert it to a dictionary from doi --> records . Each doi may contain multiple records
doi_openaireids_dict = {}
for x in doi_openaireids:
    if x['pid'] not in doi_openaireids_dict:
        doi_openaireids_dict[x['pid']] = list()
    doi_openaireids_dict[x['pid']].append(x)

# Get the affiliated orgs for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids.json', 'r') as f:
    affiliated_org_oaids = json.load(f)

# Convert it to a dictionary from id --> records . Each id may contain multiple records
affiliated_org_oaids_dict = {}
for x in affiliated_org_oaids:
    if x['id'] not in affiliated_org_oaids_dict:
        affiliated_org_oaids_dict[x['id']] = list()
    affiliated_org_oaids_dict[x['id']].append(x)

# Get the affiliated orgs metadata for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids_meta.json', 'r') as f:
    affiliated_org_oaids_meta = json.load(f)

# Convert it to a dictionary from id --> metadata
affiliated_org_oaids_meta_dict = {}
for x in affiliated_org_oaids_meta:
    if x['id'] not in affiliated_org_oaids_meta_dict:
        affiliated_org_oaids_meta_dict[x['id']] = x
    else:
        print('Duplicate id:', x['id'])

# Get the affiliated orgs for the papers
with open('PATH_TO_INTERMEDIATE_RESULTS/papers_emergingtopics_group_C_affiliated_org_oaids_pids.json', 'r') as f:
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
collection_df['affiated_org_types'] = collection_df['doi'].map(doi_affiliated_orgs_types)
collection_df['science_industry_collaboration'] = collection_df['doi'].map(doi_science_industry_collaboration)

""" GROUPING: We have added the results from our tools (outcomes) to the collection of papers. Now we need to find the open access papers and their colors. """

print('Finding the open access papers and their colors...', end='')

# Find open access vs non-open access papers
collection_openaccess = collection_df[collection_df['isopenaccess']==True]
collection_nonopenaccess = collection_df[collection_df['isopenaccess']==False]

# For non-open access gather the dois
collection_nonopenaccess_dois = collection_nonopenaccess['doi'].dropna().unique()

# For open access gather the dois
collection_openaccess_dois = collection_openaccess['doi'].dropna().unique()

# Function to get the dois info from the openaire api
def get_dois_info_from_openaire_api(dois_list):
    # Clean the dois_list from incorrect values
    dois_list = [x for x in dois_list if '&' not in x]

    url = 'https://api.openaire.eu/search/publications?format=json&size=100'
    url += '&doi=' + ','.join(dois_list)
    headers = {
        'Authorization': 'Bearer MY_OAIRE_API_KEY'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['response']['results']['result']
    else:
        print('Error dois:', dois_list)
        raise Exception('Error getting the dois info from the openaire api')

# Get the dois info from the openaire api and save it to PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_nonoa_info
if not os.path.exists('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_nonoa_info.json'):
    dois_info = []
    for i in trange(0, len(collection_nonopenaccess_dois), 50):
        dois_info += get_dois_info_from_openaire_api(collection_nonopenaccess_dois[i:i+50])
    # Save to json
    with open('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_nonoa_info.json', 'w') as f:
        json.dump(dois_info, f)

# Get the dois info from the openaire api and save it to PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_oa_info
if not os.path.exists('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_oa_info.json'):
    dois_info = []
    for i in trange(0, len(collection_openaccess_dois), 50):
        dois_info += get_dois_info_from_openaire_api(collection_openaccess_dois[i:i+50])
    # Save to json
    with open('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_oa_info.json', 'w') as f:
        json.dump(dois_info, f)

# Load the dois info from the openaire api
# NOTE: 34,112 of these 'CLOSED ACCESS' papers are found as 'OPEN ACCESS' by the OpenAIRE API
with open('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_nonoa_info.json', 'r') as f:
    nonoa_dois_info = json.load(f)

# Check again the 'CLOSED ACCESS' papers that are found as 'OPEN ACCESS' by the OpenAIRE API
# NOTE: available open access rights are: OPEN, CLOSED, UNKNOWN, RESTRICTED, EMBARGO
nonoa_foundoa_dois_info = [x for x in nonoa_dois_info if x['metadata']['oaf:entity']['oaf:result']['bestaccessright']['@classid'] == 'OPEN']

# Convert these to dois
nonoa_foundoa_dois = [[x['metadata']['oaf:entity']['oaf:result']['pid']['$'] if x['metadata']['oaf:entity']['oaf:result']['pid']['@classid']=='doi' else ''] if isinstance(x['metadata']['oaf:entity']['oaf:result']['pid'], dict) else [y['$'] for y in x['metadata']['oaf:entity']['oaf:result']['pid'] if y and y['@classid']=='doi'] for x in nonoa_foundoa_dois_info]
nonoa_foundoa_dois = sorted(set([i for s in nonoa_foundoa_dois for i in s if i.strip()!='']))

# Update the collection_openaccess and collection_nonopenaccess dataframes based on the found dois
# - first pop the dois that are found in the nonoa_foundoa_dois from the collection_nonopenaccess
# - then add those to the collection_openaccess
# NOTE: ignore case for the dois
# NOTE: the number of these dois is 34,119 (different from 31,112 probably due to the case sensitivity)
collection_openaccess = pd.concat([collection_openaccess, collection_nonopenaccess[collection_nonopenaccess['doi'].str.lower().isin(nonoa_foundoa_dois)]])
collection_nonopenaccess = collection_nonopenaccess[~collection_nonopenaccess['doi'].str.lower().isin(nonoa_foundoa_dois)]

# Load the dois info from the openaire api
with open('PATH_TO_INTERMEDIATE_RESULTS/collection_emergingtopics_oa_info.json', 'r') as f:
    oa_dois_info = json.load(f)

# Combine the dois info
dois_info = nonoa_foundoa_dois_info + oa_dois_info

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

    for doi in dois:
        doi_colors_info[doi] = {'isgreen': isgreen, 'openaccesscolor': openaccesscolor, 'isindiamondjournal': isindiamondjournal, 'doi_info': doi_info}

# Check those that are not green, do not have any color and are not in diamond journal
unknown_color_dois = {x: doi_colors_info[x] for x in doi_colors_info if not (doi_colors_info[x]['isgreen'] or doi_colors_info[x]['isindiamondjournal'] or (doi_colors_info[x]['openaccesscolor'] is not None))}

# Add a new column to the collection_openaccess dataframe with the best color info
# colors : green (isgreen) , bronze (openaccesscolor) , hybrid (openaccesscolor) , gold (openaccesscolor) , diamond (isindiamondjournal)
# NOTE: lower() is used as the openaire api returns the dois as lowercase
collection_openaccess['green'] = collection_openaccess['doi'].map(lambda x: doi_colors_info[x.lower()]['isgreen'] if x.lower() in doi_colors_info else None)
collection_openaccess['bronze'] = collection_openaccess['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='bronze' if x.lower() in doi_colors_info else None)
collection_openaccess['hybrid'] = collection_openaccess['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='hybrid' if x.lower() in doi_colors_info else None)
collection_openaccess['gold'] = collection_openaccess['doi'].map(lambda x: doi_colors_info[x.lower()]['openaccesscolor']=='gold' if x.lower() in doi_colors_info else None)
collection_openaccess['diamond'] = collection_openaccess['doi'].map(lambda x: doi_colors_info[x.lower()]['isindiamondjournal'] if x.lower() in doi_colors_info else None)

print('Done!')

print('Finding the funding info for the H2020 papers...', end='')

# Find the funding info from the dois info (focus on H2020)
doi_funding_info = {}
excluded_ids = set()
excluded_names = set()
for doi_info in tqdm(dois_info):
    # Find the doi
    pid = doi_info['metadata']['oaf:entity']['oaf:result']['pid']
    dois = list()
    if isinstance(pid, dict):
        pid = [pid]
    for x in pid:
        if x['@classid'] == 'doi':
            dois.append(x['$'])

    # Easy search for H2020
    easy_search = 'h2020' in str(doi_info['metadata']['oaf:entity']['oaf:result']).lower()

    # Context search for H2020
    context_search = False
    context = doi_info['metadata']['oaf:entity']['oaf:result']['context'] if 'context' in doi_info['metadata']['oaf:entity']['oaf:result'] else None
    if context:
        if isinstance(context, dict):
            context = [context]
        funding_context = [x for x in context if x and '@type' in x and x['@type']=='funding']
        # Specifically search for H2020
        for x in funding_context:
            if 'category' not in x:
                if x['@id'] == 'EC::H2020':
                    context_search = True
                    break
                else:
                    excluded_ids.add(x['@id'])
            elif isinstance(x['category'], dict):
                if x['category']['@id'] == 'EC::H2020':
                    context_search = True
                    break
                else:
                    excluded_ids.add(x['category']['@id'])
            else:
                for y in x['category']:
                    if y['@id'] == 'EC::H2020':
                        context_search = True
                        break
                    else:
                        excluded_ids.add(y['@id'])
                if context_search:
                    break

    # Rels search for H2020
    rels_search = False
    rels = doi_info['metadata']['oaf:entity']['oaf:result']['rels'] if 'rels' in doi_info['metadata']['oaf:entity']['oaf:result'] else None
    rels = rels['rel'] if rels and 'rel' in rels else None
    if rels:
        if isinstance(rels, dict):
            rels = [rels]
        funding_rels = [x for x in rels if x and 'funding' in x]
        # Specifically search for H2020
        for x in funding_rels:
            x_new = x
            if isinstance(x_new['funding'], dict):
                x_new = [x_new['funding']]
            else:
                for y in x_new['funding']:
                    if 'funding_level_0' in y:
                        if y['funding_level_0']['@name'] == 'H2020':
                            rels_search = True
                            break
                        else:
                            excluded_names.add(y['funding_level_0']['@name'])
                    else:
                        if y['funder']['@name'] == 'H2020':
                            rels_search = True
                            break
                        else:
                            excluded_names.add(y['funder']['@name'])
                if rels_search:
                    break
    
    # Add the funding info to the dictionary
    for doi in dois:
        if doi not in doi_funding_info:
            doi_funding_info[doi] = list()
        doi_funding_info[doi].append({'easy_search': easy_search, 'context_search': context_search, 'rels_search': rels_search, 'doi_info': doi_info})

# Get the easy search dois
assert {x: doi_funding_info[x] for x in doi_funding_info if any((not y['easy_search']) and (y['context_search'] or y['rels_search']) for y in doi_funding_info[x])} == dict()
doi_funding_info_h2020 = {x: doi_funding_info[x] for x in doi_funding_info if any(y['easy_search'] and (y['context_search'] or y['rels_search']) for y in doi_funding_info[x])}
doi_funding_info_easy_excluded = {x: doi_funding_info[x] for x in doi_funding_info if any(y['easy_search'] and not (y['context_search'] or y['rels_search']) for y in doi_funding_info[x])}

# I also checked running this into the search api (the excluded ones):
# http://api.openaire.eu/search/publications?format=json&fundingStream=H2020&doi=10.18653%2Fv1%2Fs18-1027,10.46586%2Ftches.v2019.i2.225-255,10.13154%2Ftches.v2019.i2.225-255,10.21437%2Finterspeech.2020-2870,10.48550%2Farxiv.2007.15916,10.1109%2Fmlsp.2019.8918842,10.48550%2Farxiv.1908.11030,10.3389%2Ffdigh.2018.00015,10.24963%2Fijcai.2018%2F783,10.1007%2F978-3-319-89656-4_8,10.48550%2Farxiv.1802.09059,10.1051%2Fmatecconf%2F201712503004,10.18653%2Fv1%2F2020.findings-emnlp.142,10.48550%2Farxiv.2009.07615,10.18653%2Fv1%2Fe17-2115,10.47412%2Ffzep7016,10.47412%2Fovhj9183
# which returned nothing, meaning validating that those we have excluded are correct.

# Add a new column to the collection_openaccess dataframe with the funding info for H2020 as True or False
# NOTE: lower() is used as the openaire api returns the dois as lowercase
collection_openaccess['funding_h2020'] = collection_openaccess['doi'].map(lambda x: x.lower() in doi_funding_info_h2020)

""" CREATE A COMPLETE COLLECTION DF FOR SAVING WITH ALL CALCULATED OUTCOMES """

complete_collection_df = pd.concat([collection_openaccess, collection_nonopenaccess], axis=0)
# calculate some intermediate columns
complete_collection_df['only_green'] = complete_collection_df['green'] & (complete_collection_df['bronze'].isin([False, None])) & (complete_collection_df['hybrid'].isin([False, None])) & (complete_collection_df['gold'].isin([False, None])) & (complete_collection_df['diamond'].isin([False, None]))
complete_collection_df['any_color'] = complete_collection_df['green'] | (complete_collection_df['bronze'] == True) | (complete_collection_df['hybrid'] == True) | (complete_collection_df['gold'] == True) | (complete_collection_df['diamond'] == True)
complete_collection_df['green_others'] = complete_collection_df['green'] & ((complete_collection_df['bronze'] == True) | (complete_collection_df['hybrid'] == True) | (complete_collection_df['gold'] == True) | (complete_collection_df['diamond'] == True))
complete_collection_df['only_others'] = (complete_collection_df['green'].isin([False, None])) & ((complete_collection_df['bronze'] == True) | (complete_collection_df['hybrid'] == True) | (complete_collection_df['gold'] == True) | (complete_collection_df['diamond'] == True))
complete_collection_df['no_color'] = (complete_collection_df['green'].isin([False, None])) & (complete_collection_df['bronze'].isin([False, None])) & (complete_collection_df['hybrid'].isin([False, None])) & (complete_collection_df['gold'].isin([False, None])) & (complete_collection_df['diamond'].isin([False, None]))

# Create a copy of the df so that we don't delete the original one
complete_collection_df = complete_collection_df.copy()

# Remove columns, so that we keep only the ones that are useful for our analyses
cols_remove = ['title', 'S2Url', 'doi', 'pmid', 'magId', 'externalids', 'publicationtypes', 'journalName', 'journalPages', 'journalVolume', 'venue', 'publicationvenueid', 'paperAbstract']
for col in cols_remove:
    if col in complete_collection_df.columns:
        complete_collection_df.drop(columns=[col], inplace=True)

# Add author count
complete_collection_df['authorcount'] = complete_collection_df['authors'].apply(
    lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
)
complete_collection_df['authorcount'] = complete_collection_df['authorcount'].fillna(0).astype(int)

# If there is any OA color (green, bronze, hybrid, gold, diamond) then change the openaccess info to True (even if it was False)
complete_collection_df['isopenaccess'] = complete_collection_df['isopenaccess'].where(complete_collection_df['any_color'].isin([False, None]), True)

# Remove publications that are not OA and are H2020-funded, as this should not be possible based on the H2020 policy
complete_collection_df = complete_collection_df[~((complete_collection_df['isopenaccess'].isin([False, None])) & (complete_collection_df['funding_h2020'].isin([True, None])))]

# Save the complete collection df as an excel file and as a parquet file
complete_collection_df.to_excel(os.path.join('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes', 'complete_collection_df.xlsx'), index=False)
complete_collection_df.to_parquet(os.path.join('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes', 'complete_collection_df.parquet'), index=False)


""" CREATE TOPIC ATTRIBUTION DATAFRAME -- create a topic attribution df that will have as rows L5 topics (multiple times) and the papers with their calculations - so that we can do pivot tables on them """

print('Creating topic attribution dataframe for pivot table analysis...')

# Initialize an empty list to store rows for our dataframe
topic_attribution_rows = []

# Iterate through each paper in the complete collection
for _, paper in tqdm(complete_collection_df.iterrows(), total=len(complete_collection_df)):
    # Skip papers without L5 topics
    if not isinstance(paper['FOS L5'], list) or not paper['FOS L5']:
        continue
    
    # For each L5 topic in the paper, create a row
    for topic in paper['FOS L5']:
        # Check if topic is AI using L3=='artificial intelligence & image processing', using the fos_taxonomy_l5_mapping_reverse
        ai_topic = False
        if topic in fos_taxonomy_l5_mapping_reverse and any([x[2]=='artificial intelligence & image processing' for x in fos_taxonomy_l5_mapping_reverse[topic]]):
            ai_topic = True

        # Create a row with the topic and all paper attributes
        row = {'topic': topic, 'is_ai_topic': ai_topic}
        
        # Add all columns from the paper (except the ones we don't need to duplicate)
        for col in complete_collection_df.columns:
                row[col] = paper[col]
        
        # Add the row to our list
        topic_attribution_rows.append(row)

# Create the dataframe from our list of rows
topic_attribution_df = pd.DataFrame(topic_attribution_rows)

# Remove the N/A topic
topic_attribution_df = topic_attribution_df[topic_attribution_df['topic'] != 'N/A']

# Remove duplicates based on topic and id
topic_attribution_df = topic_attribution_df.drop_duplicates(subset=['topic', 'id'])

# Reset index
topic_attribution_df.reset_index(drop=True, inplace=True)

""" 

TOPIC PERSISTENCE SCORE 

NOTE: This will be integrated into the SciNoBo toolkit.

"""

print("Calculating topic persistence scores...")

# Function to calculate topic persistence score
def calculate_topic_persistence(topic_attribution_df, max_year=None, recency_weight=0.2):
    """
    Calculate an improved persistence score for each topic in the dataframe.
    
    Parameters:
    - topic_attribution_df: DataFrame with topic attributions
    - max_year: The most recent year in the dataset (for recency calculation)
    - recency_weight: Weight factor for the recency component (0-1)
    
    Returns:
    - Dictionary of topics to their persistence scores
    """
    if max_year is None:
        max_year = topic_attribution_df['year'].max()
        
    # Get unique topics
    topics = topic_attribution_df['topic'].unique()
    
    persistence_scores = {}
    
    for topic in tqdm(topics):
        # Get all papers for this topic
        topic_papers = topic_attribution_df[topic_attribution_df['topic'] == topic]
        
        # Get the years where this topic appears
        topic_years = sorted(topic_papers['year'].unique())
        
        if len(topic_years) <= 1:
            # If topic appears in only one year, calculate base persistence
            year_count = topic_papers['year'].iloc[0]
            papers_in_year = len(topic_papers)
            
            # Add recency component
            recency_factor = 1 + recency_weight * (year_count - max_year + 10) / 10
            
            # Calculate basic score with recency
            persistence_scores[topic] = papers_in_year * recency_factor
            continue
            
        # Find all sequences of consecutive years
        sequences = []
        current_seq = [topic_years[0]]
        
        for i in range(1, len(topic_years)):
            if topic_years[i] == topic_years[i-1] + 1:
                # Year is consecutive to previous
                current_seq.append(topic_years[i])
            else:
                # Break in sequence
                if len(current_seq) > 0:
                    sequences.append(current_seq)
                current_seq = [topic_years[i]]
                
        # Add the last sequence if not empty
        if len(current_seq) > 0:
            sequences.append(current_seq)
        
        # Calculate total persistence score across all sequences
        total_score = 0
        
        for seq in sequences:
            # Papers in this sequence
            papers_in_sequence = topic_papers[topic_papers['year'].isin(seq)]
            papers_count = len(papers_in_sequence)
            
            # Paper growth rate (1 if stable, >1 if growing)
            if len(seq) > 1:
                first_year_papers = len(topic_papers[topic_papers['year'] == seq[0]])
                last_year_papers = len(topic_papers[topic_papers['year'] == seq[-1]])
                growth_factor = (last_year_papers / max(1, first_year_papers)) if first_year_papers > 0 else 1
                # Cap very high growth rates
                growth_factor = min(growth_factor, 3)
            else:
                growth_factor = 1
            
            # Impact component if available (otherwise 1)
            if 'fwci' in topic_papers.columns:
                impact_factor = topic_papers['fwci'].mean() if not pd.isna(topic_papers['fwci'].mean()) else 1
            else:
                impact_factor = 1
            
            # Calculate sequence score with exponential weighting for longer sequences
            seq_length_factor = len(seq) ** 1.5  # Exponential weight for sequence length
            
            # Recency component - more weight to recent sequences
            recency_factor = 1 + recency_weight * (seq[-1] - max_year + 10) / 10
            
            # Combine all factors
            seq_score = seq_length_factor * papers_count * growth_factor * impact_factor * recency_factor
            
            # Add to total score - use a decaying weight for multiple sequences
            total_score += seq_score
        
        persistence_scores[topic] = total_score
    
    return persistence_scores

# Calculate the persistence scores
persistence_scores = calculate_topic_persistence(topic_attribution_df)

# Add persistence score to the topic attribution dataframe
topic_attribution_df['topic_persistence_score'] = topic_attribution_df['topic'].map(persistence_scores)

print(f"Topic persistence scores calculated for {len(persistence_scores)} topics")

""" Save the topic attribution dataframe with the persistence scores """

# Save the topic attribution dataframe
topic_attribution_df.to_excel(os.path.join('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes', 'topic_attribution_df.xlsx'), index=False)
topic_attribution_df.to_parquet(os.path.join('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes', 'topic_attribution_df.parquet'), index=False)
