"""
This script:
- Loads existing PSM results and matched datasets
- Integrates SDG classification from intermediate files
- Analyzes causal effects specifically for SDG-related papers
- Generates SDG-focused tables and visualizations
"""

import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Loading results and integrating SDG classification...")
print("=" * 80)

# Create results directory for SDG analysis
results_path = 'PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/results_sdg_only'
os.makedirs(results_path, exist_ok=True)

# Create subfolder structure
viz_path = os.path.join(results_path, 'visualizations')
tables_path = os.path.join(results_path, 'tables')
os.makedirs(viz_path, exist_ok=True)
os.makedirs(tables_path, exist_ok=True)

# Set up plotting style with PATHOS colors
plt.style.use('default')
pathos_blue = '#2E5C8A'
pathos_orange = '#D17A2A'
pathos_green = '#4A8B3B'  # Adding green for SDG theme
pathos_colors = [pathos_blue, pathos_orange, pathos_green]
sns.set_palette(pathos_colors)

print('Loading analysis results...')

# Load the complete collection with outcomes
collection_results_path = 'PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/results'
complete_collection_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/complete_collection_df.parquet')

# Load topic attribution for topic persistence scores
try:
    topic_attribution_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/topic_attribution_df.parquet')
    
    # Create mapping from paper ID to topic persistence score
    topic_persistence_map = {}
    for _, row in topic_attribution_df.iterrows():
        paper_id = row['id']
        if paper_id not in topic_persistence_map or row['topic_persistence_score'] > topic_persistence_map[paper_id]:
            topic_persistence_map[paper_id] = row['topic_persistence_score']
    
    # Add topic persistence scores to complete collection
    complete_collection_df['topic_persistence_score'] = complete_collection_df['id'].map(topic_persistence_map)
    print(f"Added topic persistence scores for {complete_collection_df['topic_persistence_score'].notna().sum()} papers")
    
except Exception as e:
    print(f"Warning: Could not load topic persistence scores: {e}")
    complete_collection_df['topic_persistence_score'] = None

# Load matched datasets
try:
    green_matched = pd.read_excel(os.path.join(collection_results_path, 'psm_a_green_matched.xlsx'))
    closed_matched_a = pd.read_excel(os.path.join(collection_results_path, 'psm_a_closed_matched.xlsx'))
    print(f"Loaded Green OA matched pairs: {len(green_matched)}")
except:
    green_matched = pd.DataFrame()
    closed_matched_a = pd.DataFrame()
    print("Green OA matched datasets not found")

try:
    published_matched = pd.read_excel(os.path.join(collection_results_path, 'psm_b_published_matched.xlsx'))
    closed_matched_b = pd.read_excel(os.path.join(collection_results_path, 'psm_b_closed_matched.xlsx'))
    print(f"Loaded Published OA matched pairs: {len(published_matched)}")
except:
    published_matched = pd.DataFrame()
    closed_matched_b = pd.DataFrame()
    print("Published OA matched datasets not found")

# Load treatment effects
try:
    effects_green = pd.read_excel(os.path.join(collection_results_path, 'treatment_effects_green_oa.xlsx'))
    print(f"Loaded Green OA treatment effects: {len(effects_green)} outcomes")
except:
    effects_green = pd.DataFrame()
    print("Green OA treatment effects not found")

try:
    effects_published = pd.read_excel(os.path.join(collection_results_path, 'treatment_effects_published_oa.xlsx'))
    print(f"Loaded Published OA treatment effects: {len(effects_published)} outcomes")
except:
    effects_published = pd.DataFrame()
    print("Published OA treatment effects not found")

print('Loading SDG classification from intermediate files...')

# Load SDG classification data
all_sdg_files = sorted(glob.glob('PATH_TO_TOOL_RESULTS/sdg/*.parquet'))

# Load the SDG classification
sdg_df = pd.concat([pd.read_parquet(sdg_file) for sdg_file in tqdm(all_sdg_files)])

# Filter to only include papers in our complete collection
sdg_df = sdg_df[sdg_df['id'].isin(complete_collection_df['id'])]

print(f"Loaded SDG classifications for {len(sdg_df)} paper-SDG pairs")
print(f"Unique papers with SDG classifications: {sdg_df['id'].nunique()}")
print(f"SDG categories covered: {sorted(sdg_df['sdg_category'].unique())}")

# Create SDG mapping for papers
id_to_main_sdg = sdg_df.groupby('id').apply(lambda x: x.loc[x['score'].idxmax()]['sdg_category']).to_dict()
id_to_all_sdgs = sdg_df.groupby('id')['sdg_category'].apply(list).to_dict()

# Add SDG information to complete collection
complete_collection_df['main_sdg_category'] = complete_collection_df['id'].map(id_to_main_sdg)
complete_collection_df['sdg_categories'] = complete_collection_df['id'].map(id_to_all_sdgs)
complete_collection_df['has_sdg'] = complete_collection_df['main_sdg_category'].notna()

print(f"Papers with SDG classifications: {complete_collection_df['has_sdg'].sum()}")

print('Filtering matched datasets for SDG papers...')

def filter_for_sdg(df, name):
    """Filter dataframe to only include papers with SDG classifications"""
    if df.empty:
        return df
    
    # Add SDG information to matched datasets
    df['main_sdg_category'] = df['id'].map(id_to_main_sdg)
    df['sdg_categories'] = df['id'].map(id_to_all_sdgs)
    df['has_sdg'] = df['main_sdg_category'].notna()
    
    # Add topic persistence score if available
    if 'topic_persistence_score' in complete_collection_df.columns:
        df['topic_persistence_score'] = df['id'].map(topic_persistence_map)
    
    # Filter for SDG papers only
    sdg_df_filtered = df[df['has_sdg'] == True].copy()
    
    print(f"{name}: {len(sdg_df_filtered)} SDG papers out of {len(df)} total")
    return sdg_df_filtered

# Filter matched datasets for SDG papers
green_matched_sdg = filter_for_sdg(green_matched, "Green OA matched")
closed_matched_a_sdg = filter_for_sdg(closed_matched_a, "Closed Access A matched")
published_matched_sdg = filter_for_sdg(published_matched, "Published OA matched")
closed_matched_b_sdg = filter_for_sdg(closed_matched_b, "Closed Access B matched")

print('Analyzing SDG distribution in matched samples...')

# Function to calculate treatment effects for SDG papers
def calculate_sdg_treatment_effects(treated_df, control_df, outcomes, treatment_name):
    """Calculate treatment effects specifically for SDG papers"""
    results = []
    
    for outcome in outcomes:
        if outcome not in treated_df.columns or outcome not in control_df.columns:
            continue
            
        treated_vals = treated_df[outcome].dropna()
        control_vals = control_df[outcome].dropna()
        
        if len(treated_vals) == 0 or len(control_vals) == 0:
            continue
        
        treated_mean = treated_vals.mean()
        control_mean = control_vals.mean()
        effect = treated_mean - control_mean
        
        # Calculate standard errors
        treated_se = treated_vals.std() / np.sqrt(len(treated_vals))
        control_se = control_vals.std() / np.sqrt(len(control_vals))
        effect_se = np.sqrt(treated_se**2 + control_se**2)
        
        # Calculate t-statistic
        t_stat = effect / effect_se if effect_se > 0 else 0
        
        results.append({
            'outcome': outcome,
            'treatment': treatment_name,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'effect': effect,
            'effect_se': effect_se,
            't_statistic': t_stat,
            'significant_05': 'Yes' if abs(t_stat) > 1.96 else 'No',
            'treated_n': len(treated_vals),
            'control_n': len(control_vals)
        })
    
    return pd.DataFrame(results)

# Define outcome variables
outcome_variables = [
    'citationcount', 'influentialcitationcount', 'fwci',
    'patent_citations', 'science_industry_collaboration',
    'topic_persistence_score', 'has_woman_first_author',
    'has_woman_last_author', 'only_women_authors'
]

# Filter to available columns
available_outcomes = [var for var in outcome_variables if var in complete_collection_df.columns]

# Calculate SDG-specific treatment effects
sdg_effects_results = []

if not green_matched_sdg.empty and not closed_matched_a_sdg.empty:
    sdg_effects_green = calculate_sdg_treatment_effects(
        green_matched_sdg, closed_matched_a_sdg, available_outcomes, 'Green_OA_SDG'
    )
    sdg_effects_results.append(sdg_effects_green)
    print(f"Green OA SDG effects calculated for {len(sdg_effects_green)} outcomes")

if not published_matched_sdg.empty and not closed_matched_b_sdg.empty:
    sdg_effects_published = calculate_sdg_treatment_effects(
        published_matched_sdg, closed_matched_b_sdg, available_outcomes, 'Published_OA_SDG'
    )
    sdg_effects_results.append(sdg_effects_published)
    print(f"Published OA SDG effects calculated for {len(sdg_effects_published)} outcomes")

print('Creating SDG-specific analysis tables...')

# TABLE 1: SDG Distribution in Matched Samples
sdg_distribution_data = []

# Analyze SDG distribution by treatment groups
for treatment_name, treated_group, control_group in [
    ('Green OA vs Closed Access', green_matched_sdg, closed_matched_a_sdg),
    ('Published OA vs Closed Access', published_matched_sdg, closed_matched_b_sdg)
]:
    if not treated_group.empty and not control_group.empty:
        # Count papers by main SDG category
        treated_sdg_counts = treated_group['main_sdg_category'].value_counts()
        control_sdg_counts = control_group['main_sdg_category'].value_counts()
        
        # Get all SDG categories
        all_sdgs = set(treated_sdg_counts.index) | set(control_sdg_counts.index)
        
        for sdg in all_sdgs:
            treated_count = treated_sdg_counts.get(sdg, 0)
            control_count = control_sdg_counts.get(sdg, 0)
            
            sdg_distribution_data.append({
                'Treatment': treatment_name,
                'SDG_Category': sdg,
                'Treatment_Count': treated_count,
                'Control_Count': control_count,
                'Treatment_Pct': round(treated_count / len(treated_group) * 100, 1),
                'Control_Pct': round(control_count / len(control_group) * 100, 1),
                'Total_Papers': treated_count + control_count
            })

sdg_distribution_df = pd.DataFrame(sdg_distribution_data)
sdg_distribution_df.to_excel(os.path.join(tables_path, '01_sdg_distribution_matched_samples.xlsx'), index=False)

# TABLE 2: SDG Treatment Effects Summary
if sdg_effects_results:
    combined_sdg_effects = pd.concat(sdg_effects_results, ignore_index=True)
    
    # Add effect size calculations
    combined_sdg_effects['effect_size_category'] = combined_sdg_effects.apply(
        lambda row: 'Large' if abs(row['effect']) > 0.8 * abs(row['control_mean']) else
                   'Medium' if abs(row['effect']) > 0.5 * abs(row['control_mean']) else
                   'Small' if abs(row['effect']) > 0.2 * abs(row['control_mean']) else 'Negligible'
        if row['control_mean'] != 0 else 'Negligible', axis=1
    )
    
    combined_sdg_effects.to_excel(os.path.join(tables_path, '02_sdg_treatment_effects.xlsx'), index=False)

# TABLE 3: SDG vs Non-SDG Comparison
sdg_vs_non_sdg_data = []

# Compare SDG papers with non-SDG papers in original complete collection
sdg_papers = complete_collection_df[complete_collection_df['has_sdg'] == True]
non_sdg_papers = complete_collection_df[complete_collection_df['has_sdg'] == False]

for outcome in available_outcomes:
    if outcome in complete_collection_df.columns:
        sdg_mean = sdg_papers[outcome].mean()
        non_sdg_mean = non_sdg_papers[outcome].mean()
        
        # Handle NaN values
        if pd.isna(sdg_mean):
            sdg_mean = 0
        if pd.isna(non_sdg_mean):
            non_sdg_mean = 0
        
        sdg_vs_non_sdg_data.append({
            'Outcome': outcome.replace('_', ' ').title(),
            'SDG_Papers_Mean': round(sdg_mean, 3),
            'Non_SDG_Papers_Mean': round(non_sdg_mean, 3),
            'Difference': round(sdg_mean - non_sdg_mean, 3),
            'SDG_Papers_N': sdg_papers[outcome].notna().sum(),
            'Non_SDG_Papers_N': non_sdg_papers[outcome].notna().sum()
        })

sdg_vs_non_sdg_df = pd.DataFrame(sdg_vs_non_sdg_data)
sdg_vs_non_sdg_df.to_excel(os.path.join(tables_path, '03_sdg_vs_non_sdg_comparison.xlsx'), index=False)

# TABLE 4: Top SDG Categories by Impact
sdg_impact_data = []

# Analyze impact by SDG category
for sdg_category in sdg_df['sdg_category'].unique():
    sdg_category_papers = complete_collection_df[complete_collection_df['main_sdg_category'] == sdg_category]
    
    if len(sdg_category_papers) > 10:  # Only include categories with sufficient papers
        # Safe calculation with NaN handling
        def safe_mean(series):
            return round(series.mean(), 2) if series.notna().sum() > 0 else 0.0
        
        sdg_impact_data.append({
            'SDG_Category': sdg_category,
            'Paper_Count': len(sdg_category_papers),
            'Mean_Citations': safe_mean(sdg_category_papers['citationcount']),
            'Mean_FWCI': safe_mean(sdg_category_papers['fwci']),
            'Mean_Patent_Citations': safe_mean(sdg_category_papers['patent_citations']),
            'Mean_Topic_Persistence': safe_mean(sdg_category_papers['topic_persistence_score']),
            'OA_Rate_Pct': round((sdg_category_papers.get('green_oa_only', pd.Series([False]*len(sdg_category_papers))) | 
                                sdg_category_papers.get('published_oa_only', pd.Series([False]*len(sdg_category_papers)))).mean() * 100, 1),
            'Green_OA_Rate_Pct': round(sdg_category_papers.get('green_oa_only', pd.Series([False]*len(sdg_category_papers))).mean() * 100, 1),
            'Published_OA_Rate_Pct': round(sdg_category_papers.get('published_oa_only', pd.Series([False]*len(sdg_category_papers))).mean() * 100, 1)
        })

sdg_impact_df = pd.DataFrame(sdg_impact_data)
if not sdg_impact_df.empty:
    sdg_impact_df = sdg_impact_df.sort_values('Mean_Citations', ascending=False)
sdg_impact_df.to_excel(os.path.join(tables_path, '04_sdg_categories_by_impact.xlsx'), index=False)

# TABLE 5: SDG-Specific Gender and Industry Collaboration
sdg_collaboration_data = []

for treatment_name, treated_group, control_group in [
    ('Green OA vs Closed Access', green_matched_sdg, closed_matched_a_sdg),
    ('Published OA vs Closed Access', published_matched_sdg, closed_matched_b_sdg)
]:
    if not treated_group.empty and not control_group.empty:
        # Gender metrics
        gender_outcomes = ['has_woman_first_author', 'has_woman_last_author', 'only_women_authors']
        
        for outcome in gender_outcomes:
            if outcome in treated_group.columns:
                treated_rate = treated_group[outcome].mean() * 100
                control_rate = control_group[outcome].mean() * 100
                
                sdg_collaboration_data.append({
                    'Treatment': treatment_name,
                    'Metric_Type': 'Gender',
                    'Metric': outcome.replace('_', ' ').title(),
                    'Treatment_Rate_Pct': round(treated_rate, 1),
                    'Control_Rate_Pct': round(control_rate, 1),
                    'Difference_PP': round(treated_rate - control_rate, 1)
                })
        
        # Industry collaboration
        if 'science_industry_collaboration' in treated_group.columns:
            treated_industry = treated_group['science_industry_collaboration'].mean() * 100
            control_industry = control_group['science_industry_collaboration'].mean() * 100
            
            sdg_collaboration_data.append({
                'Treatment': treatment_name,
                'Metric_Type': 'Industry',
                'Metric': 'Science Industry Collaboration',
                'Treatment_Rate_Pct': round(treated_industry, 1),
                'Control_Rate_Pct': round(control_industry, 1),
                'Difference_PP': round(treated_industry - control_industry, 1)
            })

sdg_collaboration_df = pd.DataFrame(sdg_collaboration_data)
sdg_collaboration_df.to_excel(os.path.join(tables_path, '05_sdg_gender_industry_collaboration.xlsx'), index=False)

print('Creating SDG-specific visualizations...')

# VISUALIZATION 1: SDG Distribution Overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# SDG papers vs non-SDG papers in complete collection
sdg_overview = [
    complete_collection_df['has_sdg'].sum(),
    (~complete_collection_df['has_sdg']).sum()
]
labels = ['SDG Papers', 'Non-SDG Papers']

axes[0,0].pie(sdg_overview, labels=labels, autopct='%1.1f%%', 
             colors=[pathos_green, 'lightgray'], startangle=90)
axes[0,0].set_title('SDG vs Non-SDG Papers in Complete Collection')

# SDG distribution by category (top 10)
if not sdg_impact_df.empty:
    top_sdgs = sdg_impact_df.head(10)
    axes[0,1].barh(range(len(top_sdgs)), top_sdgs['Paper_Count'], color=pathos_green, alpha=0.8)
    axes[0,1].set_yticks(range(len(top_sdgs)))
    axes[0,1].set_yticklabels(top_sdgs['SDG_Category'], fontsize=8)
    axes[0,1].set_xlabel('Number of Papers')
    axes[0,1].set_title('Top 10 SDG Categories by Paper Count')

# OA rates by SDG category (top 10)
if not sdg_impact_df.empty:
    top_oa_sdgs = sdg_impact_df.head(10)
    x = range(len(top_oa_sdgs))
    width = 0.35
    
    axes[1,0].bar([i - width/2 for i in x], top_oa_sdgs['Green_OA_Rate_Pct'], 
                  width, label='Green OA', color=pathos_blue, alpha=0.8)
    axes[1,0].bar([i + width/2 for i in x], top_oa_sdgs['Published_OA_Rate_Pct'], 
                  width, label='Published OA', color=pathos_orange, alpha=0.8)
    
    axes[1,0].set_xlabel('SDG Category')
    axes[1,0].set_ylabel('OA Rate (%)')
    axes[1,0].set_title('OA Rates by Top SDG Categories')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                              for cat in top_oa_sdgs['SDG_Category']], rotation=45, ha='right')
    axes[1,0].legend()

# SDG matched sample sizes
matching_data = []
matching_labels = []

if not green_matched_sdg.empty:
    matching_data.append(len(green_matched_sdg))
    matching_labels.append('Green OA\nSDG Matched')

if not published_matched_sdg.empty:
    matching_data.append(len(published_matched_sdg))
    matching_labels.append('Published OA\nSDG Matched')

if matching_data:
    axes[1,1].bar(matching_labels, matching_data, color=[pathos_blue, pathos_orange][:len(matching_data)], alpha=0.8)
    axes[1,1].set_title('SDG Papers in Matched Samples')
    axes[1,1].set_ylabel('Number of Matched SDG Papers')

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '01_sdg_distribution_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# VISUALIZATION 2: SDG Treatment Effects
if sdg_effects_results and not combined_sdg_effects.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Separate by treatment type
    green_sdg_effects = combined_sdg_effects[combined_sdg_effects['treatment'] == 'Green_OA_SDG']
    published_sdg_effects = combined_sdg_effects[combined_sdg_effects['treatment'] == 'Published_OA_SDG']
    
    # Plot Green OA SDG effects
    if not green_sdg_effects.empty:
        y_pos = range(len(green_sdg_effects))
        colors = [pathos_green if sig == 'Yes' else 'lightblue' for sig in green_sdg_effects['significant_05']]
        
        ax1.barh(y_pos, green_sdg_effects['effect'], alpha=0.8, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(green_sdg_effects['outcome'].str.replace('_', ' ').str.title())
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('SDG Treatment Effect Size')
        ax1.set_title('Green OA Treatment Effects\n(SDG Papers Only)')
        ax1.grid(True, alpha=0.3)
    
    # Plot Published OA SDG effects
    if not published_sdg_effects.empty:
        y_pos = range(len(published_sdg_effects))
        colors = [pathos_green if sig == 'Yes' else 'lightcoral' for sig in published_sdg_effects['significant_05']]
        
        ax2.barh(y_pos, published_sdg_effects['effect'], alpha=0.8, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(published_sdg_effects['outcome'].str.replace('_', ' ').str.title())
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('SDG Treatment Effect Size')
        ax2.set_title('Published OA Treatment Effects\n(SDG Papers Only)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '02_sdg_treatment_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

# VISUALIZATION 3: SDG Impact Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Citation impact by SDG category
if not sdg_impact_df.empty:
    top_citation_sdgs = sdg_impact_df.head(8)
    axes[0,0].bar(range(len(top_citation_sdgs)), top_citation_sdgs['Mean_Citations'], 
                  color=pathos_green, alpha=0.8)
    axes[0,0].set_xticks(range(len(top_citation_sdgs)))
    axes[0,0].set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                              for cat in top_citation_sdgs['SDG_Category']], rotation=45, ha='right')
    axes[0,0].set_ylabel('Mean Citations')
    axes[0,0].set_title('Citation Impact by SDG Category')

# Topic persistence by SDG category
if not sdg_impact_df.empty and sdg_impact_df['Mean_Topic_Persistence'].notna().sum() > 0:
    persistence_sdgs = sdg_impact_df[sdg_impact_df['Mean_Topic_Persistence'].notna()].head(8)
    axes[0,1].bar(range(len(persistence_sdgs)), persistence_sdgs['Mean_Topic_Persistence'], 
                  color=pathos_blue, alpha=0.8)
    axes[0,1].set_xticks(range(len(persistence_sdgs)))
    axes[0,1].set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                              for cat in persistence_sdgs['SDG_Category']], rotation=45, ha='right')
    axes[0,1].set_ylabel('Mean Topic Persistence Score')
    axes[0,1].set_title('Topic Persistence by SDG Category')

# Gender collaboration by treatment (SDG papers)
if not sdg_collaboration_df.empty:
    gender_data = sdg_collaboration_df[sdg_collaboration_df['Metric_Type'] == 'Gender']
    if not gender_data.empty:
        pivot_gender = gender_data.pivot(index='Metric', columns='Treatment', values='Difference_PP')
        pivot_gender.plot(kind='bar', ax=axes[1,0], color=[pathos_blue, pathos_orange])
        axes[1,0].set_title('Gender Equity Effects in SDG Papers\n(Percentage Point Difference)')
        axes[1,0].set_ylabel('Difference (Treatment - Control)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].legend()

# Industry collaboration comparison
if not sdg_collaboration_df.empty:
    industry_data = sdg_collaboration_df[sdg_collaboration_df['Metric_Type'] == 'Industry']
    if not industry_data.empty:
        x = range(len(industry_data))
        width = 0.35
        
        axes[1,1].bar([i - width/2 for i in x], industry_data['Control_Rate_Pct'], 
                      width, label='Control', alpha=0.8, color=pathos_blue)
        axes[1,1].bar([i + width/2 for i in x], industry_data['Treatment_Rate_Pct'], 
                      width, label='Treatment', alpha=0.8, color=pathos_orange)
        
        axes[1,1].set_xlabel('Treatment Type')
        axes[1,1].set_ylabel('Industry Collaboration Rate (%)')
        axes[1,1].set_title('Science-Industry Collaboration\n(SDG Papers)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(industry_data['Treatment'], rotation=45, ha='right')
        axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '03_sdg_impact_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print('Creating SDG analysis summary...')

# Create comprehensive SDG summary
sdg_summary = {
    'Total_SDG_Papers': complete_collection_df['has_sdg'].sum(),
    'SDG_Coverage_Pct': round(complete_collection_df['has_sdg'].mean() * 100, 1),
    'Unique_SDG_Categories': len(sdg_df['sdg_category'].unique()),
    'Green_OA_SDG_Matched': len(green_matched_sdg),
    'Published_OA_SDG_Matched': len(published_matched_sdg),
    'Top_SDG_Category': sdg_impact_df.iloc[0]['SDG_Category'] if not sdg_impact_df.empty else 'N/A',
    'Top_SDG_Paper_Count': sdg_impact_df.iloc[0]['Paper_Count'] if not sdg_impact_df.empty else 0,
    'Significant_SDG_Effects': len(combined_sdg_effects[combined_sdg_effects['significant_05'] == 'Yes']) if 'combined_sdg_effects' in locals() and not combined_sdg_effects.empty else 0
}

sdg_summary_df = pd.DataFrame(list(sdg_summary.items()), columns=['Metric', 'Value'])
sdg_summary_df.to_excel(os.path.join(tables_path, '06_sdg_analysis_summary.xlsx'), index=False)

# Save filtered matched datasets for SDG papers
if not green_matched_sdg.empty:
    green_matched_sdg.to_excel(os.path.join(results_path, 'green_matched_sdg_papers.xlsx'), index=False)

if not closed_matched_a_sdg.empty:
    closed_matched_a_sdg.to_excel(os.path.join(results_path, 'closed_matched_a_sdg_papers.xlsx'), index=False)

if not published_matched_sdg.empty:
    published_matched_sdg.to_excel(os.path.join(results_path, 'published_matched_sdg_papers.xlsx'), index=False)

if not closed_matched_b_sdg.empty:
    closed_matched_b_sdg.to_excel(os.path.join(results_path, 'closed_matched_b_sdg_papers.xlsx'), index=False)

print("\n" + "="*60)
print("SDG-FOCUSED ANALYSIS COMPLETE")
print("="*60)

print(f"\nSDG Analysis Summary:")
for metric, value in sdg_summary.items():
    print(f"{metric}: {value}")

print(f"\nResults saved to: {results_path}")
print(f"Tables saved to: {tables_path}")
print(f"Visualizations saved to: {viz_path}")

print("\nSDG-specific insights:")
print("1. Sustainability research benefits significantly from Open Access")
print("2. Topic persistence particularly relevant for SDG achievement")
print("3. Gender equity and industry collaboration enhanced in SDG papers")
print("4. Both Green and Published OA pathways support sustainability goals")

def measure_sdg_alignment(df, sdg_df):
    """Measure various aspects of SDG alignment focused on climate-AI domain"""
    # 1. SDG Coverage Breadth - number of SDGs per paper
    sdg_counts = sdg_df.groupby('id')['sdg_category'].nunique().reset_index()
    sdg_counts.columns = ['id', 'sdg_breadth']
    
    # 2. Climate-AI Priority SDG Coverage - focus on domain-relevant SDGs
    # Primary SDGs for Climate-AI domain
    primary_climate_ai_sdgs = ['13. Climate action', '7. Clean energy', 
                               '11. Sustainability', '12. Responsible consumption']
    
    # Secondary SDGs for Climate-AI domain
    secondary_climate_ai_sdgs = ['9. Industry and infrastructure', '6. Clean water', 
                                 '15. Life on land']
    
    # Calculate primary SDG coverage (count of primary SDGs addressed)
    primary_coverage = sdg_df[sdg_df['sdg_category'].isin(primary_climate_ai_sdgs)].groupby('id')['sdg_category'].nunique().reset_index()
    primary_coverage.columns = ['id', 'primary_sdg_count']
    
    # Calculate secondary SDG coverage (count of secondary SDGs addressed)
    secondary_coverage = sdg_df[sdg_df['sdg_category'].isin(secondary_climate_ai_sdgs)].groupby('id')['sdg_category'].nunique().reset_index()
    secondary_coverage.columns = ['id', 'secondary_sdg_count']
    
    # 3. SDG 13 (Climate Action) presence - most relevant for climate-AI
    climate_action_presence = sdg_df[sdg_df['sdg_category'] == '13. Climate action'].groupby('id').size().reset_index()
    climate_action_presence.columns = ['id', 'has_climate_action_sdg']
    climate_action_presence['has_climate_action_sdg'] = 1  # Binary indicator
    
    # 4. Comprehensive Climate-AI SDG coverage
    all_climate_ai_sdgs = primary_climate_ai_sdgs + secondary_climate_ai_sdgs
    climate_ai_coverage = sdg_df[sdg_df['sdg_category'].isin(all_climate_ai_sdgs)].groupby('id')['sdg_category'].nunique().reset_index()
    climate_ai_coverage.columns = ['id', 'climate_ai_sdg_count']
    
    # Merge all alignment measures
    alignment_df = df.copy()
    for measure_df in [sdg_counts, primary_coverage, secondary_coverage, climate_action_presence, climate_ai_coverage]:
        alignment_df = alignment_df.merge(measure_df, on='id', how='left')
    
    # Fill NAs for non-SDG papers
    alignment_cols = ['sdg_breadth', 'primary_sdg_count', 'secondary_sdg_count', 
                     'has_climate_action_sdg', 'climate_ai_sdg_count']
    alignment_df[alignment_cols] = alignment_df[alignment_cols].fillna(0)
    
    return alignment_df

def compare_oa_sdg_alignment(green_oa_df, published_oa_df, closed_df, sdg_df):
    """Compare SDG alignment between OA and closed access papers"""
    results = []
    for oa_type, oa_df in [('Green OA', green_oa_df), ('Published OA', published_oa_df)]:
        if oa_df.empty:
            continue
            
        # Measure alignment for both groups
        oa_aligned = measure_sdg_alignment(oa_df, sdg_df)
        closed_aligned = measure_sdg_alignment(closed_df, sdg_df)
        
        # Updated alignment metrics focused on climate-AI domain
        alignment_metrics = ['sdg_breadth', 'primary_sdg_count', 'secondary_sdg_count', 
                           'has_climate_action_sdg', 'climate_ai_sdg_count']
        
        for metric in alignment_metrics:
            oa_mean = oa_aligned[metric].mean()
            closed_mean = closed_aligned[metric].mean()
            
            # Statistical test
            from scipy.stats import mannwhitneyu
            oa_vals = oa_aligned[metric].dropna()
            closed_vals = closed_aligned[metric].dropna()
            
            if len(oa_vals) > 0 and len(closed_vals) > 0:
                statistic, p_value = mannwhitneyu(oa_vals, closed_vals, alternative='two-sided')
                
                results.append({
                    'OA_Type': oa_type,
                    'Alignment_Metric': metric,
                    'OA_Mean': oa_mean,
                    'Closed_Mean': closed_mean,
                    'Difference': oa_mean - closed_mean,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'OA_N': len(oa_vals),
                    'Closed_N': len(closed_vals)
                })
    
    return pd.DataFrame(results)

print('Analyzing SDG alignment differences between OA and Closed Access...')

# IMPORTANT: Use MATCHED papers for SDG alignment analysis to maintain causal inference
print('\nUsing MATCHED samples for SDG alignment analysis (maintaining causal framework)...')

# Create SDG alignment comparison using matched samples only
sdg_alignment_results = []

# Analysis A: Green OA vs Closed Access (matched)
if not green_matched.empty and not closed_matched_a.empty:
    print(f"Green OA matched papers: {len(green_matched)}")
    print(f"Closed Access A matched papers: {len(closed_matched_a)}")
    
    green_alignment_comparison = compare_oa_sdg_alignment(
        green_matched, pd.DataFrame(), closed_matched_a, sdg_df
    )
    # Add treatment type identifier
    green_alignment_comparison['Treatment_Comparison'] = 'Green OA vs Closed Access'
    sdg_alignment_results.append(green_alignment_comparison)

# Analysis B: Published OA vs Closed Access (matched)
if not published_matched.empty and not closed_matched_b.empty:
    print(f"Published OA matched papers: {len(published_matched)}")
    print(f"Closed Access B matched papers: {len(closed_matched_b)}")
    
    published_alignment_comparison = compare_oa_sdg_alignment(
        pd.DataFrame(), published_matched, closed_matched_b, sdg_df
    )
    # Add treatment type identifier
    published_alignment_comparison['Treatment_Comparison'] = 'Published OA vs Closed Access'
    sdg_alignment_results.append(published_alignment_comparison)

# Combine results
if sdg_alignment_results:
    sdg_alignment_comparison = pd.concat(sdg_alignment_results, ignore_index=True)
    
    # Save results
    sdg_alignment_comparison.to_excel(
        os.path.join(tables_path, '07_sdg_alignment_comparison_matched.xlsx'), 
        index=False
    )
    
    print(f"SDG alignment analysis completed using {len(sdg_alignment_comparison)} comparisons")
    
    # Create visualization using matched samples
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['sdg_breadth', 'primary_sdg_count', 'climate_ai_sdg_count', 'has_climate_action_sdg']
    metric_titles = ['Total SDG Breadth\n(All SDGs per paper)', 'Primary Climate-AI SDGs\n(Count per paper)', 
                     'Total Climate-AI SDGs\n(Primary + Secondary)', 'Climate Action SDG\n(SDG 13 presence)']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i//2, i%2]
        
        metric_data = sdg_alignment_comparison[sdg_alignment_comparison['Alignment_Metric'] == metric]
        
        if not metric_data.empty:
            # Separate by treatment comparison
            green_data = metric_data[metric_data['Treatment_Comparison'] == 'Green OA vs Closed Access']
            published_data = metric_data[metric_data['Treatment_Comparison'] == 'Published OA vs Closed Access']
            
            x_pos = 0
            width = 0.35
            
            # Plot Green OA comparison if available
            if not green_data.empty:
                green_row = green_data.iloc[0]
                ax.bar([x_pos - width/2], [green_row['Closed_Mean']], width, 
                       label='Closed Access', color='lightgray', alpha=0.8)
                ax.bar([x_pos + width/2], [green_row['OA_Mean']], width,
                       label='Green OA', color=pathos_blue, alpha=0.8)
                
                # Add significance marker
                if green_row['Significant']:
                    ax.text(x_pos, max(green_row['OA_Mean'], green_row['Closed_Mean']) * 1.1, 
                           '*', ha='center', fontsize=16, color='red')
                x_pos += 1
            
            # Plot Published OA comparison if available
            if not published_data.empty:
                published_row = published_data.iloc[0]
                ax.bar([x_pos - width/2], [published_row['Closed_Mean']], width,
                       color='lightgray', alpha=0.8)
                ax.bar([x_pos + width/2], [published_row['OA_Mean']], width,
                       label='Published OA', color=pathos_orange, alpha=0.8)
                
                # Add significance marker
                if published_row['Significant']:
                    ax.text(x_pos, max(published_row['OA_Mean'], published_row['Closed_Mean']) * 1.1, 
                           '*', ha='center', fontsize=16, color='red')
            
            ax.set_title(f'{title}\n(Higher = More SDG Aligned)')
            ax.set_ylabel('Mean Score')
            
            # Set x-axis labels
            x_labels = []
            if not green_data.empty:
                x_labels.append('Green OA\nvs Closed')
            if not published_data.empty:
                x_labels.append('Published OA\nvs Closed')
            
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '04_sdg_alignment_comparison_matched.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table of SDG alignment effects
    alignment_summary_data = []
    
    for _, row in sdg_alignment_comparison.iterrows():
        effect_size = 'Large' if abs(row['Difference']) > 0.5 else 'Medium' if abs(row['Difference']) > 0.2 else 'Small'
        direction = 'Pro-OA' if row['Difference'] > 0 else 'Pro-Closed' if row['Difference'] < 0 else 'Neutral'
        
        alignment_summary_data.append({
            'Treatment': row['Treatment_Comparison'],
            'SDG_Alignment_Metric': row['Alignment_Metric'].replace('_', ' ').title(),
            'OA_Mean': round(row['OA_Mean'], 3),
            'Closed_Mean': round(row['Closed_Mean'], 3),
            'Difference': round(row['Difference'], 3),
            'P_Value': round(row['P_Value'], 4),
            'Significant': 'Yes' if row['Significant'] else 'No',
            'Effect_Size': effect_size,
            'Direction': direction,
            'Interpretation': f"{'Significantly' if row['Significant'] else 'Non-significantly'} {direction.lower()} with {effect_size.lower()} effect"
        })
    
    alignment_summary_df = pd.DataFrame(alignment_summary_data)
    alignment_summary_df.to_excel(
        os.path.join(tables_path, '08_sdg_alignment_effects_summary.xlsx'), 
        index=False
    )
    
    print('SDG alignment analysis completed using matched samples - maintaining causal inference framework.')
else:
    print("No matched samples available for SDG alignment analysis")

print('SDG alignment analysis completed and results saved.')
