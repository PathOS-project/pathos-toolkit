"""
This script implements the refined causal inference design with:
- Two separate PSM analyses: Green OA vs Closed Access, Published OA vs Closed Access  
- Exclusion of dual-mode OA and Bronze OA for treatment purity
- Focus on topic persistence as a novel outcome measure
- Proper causal identification strategy
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

print("=" * 80)
print("Loading data and preparing for PSM analyses...")
print("=" * 80)

# Create results directory
results_path = 'PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/results'
os.makedirs(results_path, exist_ok=True)

# Create visualizations subfolder
viz_path = os.path.join(results_path, 'visualizations')
os.makedirs(viz_path, exist_ok=True)

# Create tables subfolder
tables_path = os.path.join(results_path, 'tables')
os.makedirs(tables_path, exist_ok=True)

# Set up plotting style with PATHOS colors
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
pathos_blue = '#2E5C8A'    # Medium dark blue
pathos_orange = '#D17A2A'  # Medium dark orange
pathos_colors = [pathos_blue, pathos_orange]
sns.set_palette([pathos_blue, pathos_orange])

print('Loading the complete collection...')

# Load the complete collection with all outcomes calculated
complete_collection_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/complete_collection_df.parquet')

print(f"Loaded collection with {len(complete_collection_df)} papers")

# Load topic attribution dataframe for persistence scores
topic_attribution_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/topic_attribution_df.parquet')

print(f"Loaded topic attribution with {len(topic_attribution_df)} topic-paper pairs")

print('Preparing data according to causal inference specifications...')

# Create a working copy
df = complete_collection_df.copy()

# Define clear OA categories according to the case study specifications
print('Defining OA categories...')

# Green OA: only green, no other colors
df['green_oa_only'] = (df['green'] == True) & (df['bronze'].isin([False, None])) & (df['hybrid'].isin([False, None])) & (df['gold'].isin([False, None])) & (df['diamond'].isin([False, None]))

# Published OA: gold, hybrid, or diamond (but not green)
df['published_oa_only'] = (df['green'].isin([False, None])) & ((df['gold'] == True) | (df['hybrid'] == True) | (df['diamond'] == True))

# Dual-mode OA: both green and any other color (to be excluded)
df['dual_mode_oa'] = (df['green'] == True) & ((df['gold'] == True) | (df['hybrid'] == True) | (df['diamond'] == True))

# Bronze OA: excluded entirely per specifications
df['bronze_oa'] = (df['bronze'] == True)

# Closed Access: no OA color at all
df['closed_access'] = (df['isopenaccess'] == False)

# Print distribution of OA categories
print("\nOA Category Distribution:")
print(f"Green OA only: {df['green_oa_only'].sum():,}")
print(f"Published OA only: {df['published_oa_only'].sum():,}")
print(f"Dual-mode OA (excluded): {df['dual_mode_oa'].sum():,}")
print(f"Bronze OA (excluded): {df['bronze_oa'].sum():,}")
print(f"Closed Access: {df['closed_access'].sum():,}")

print('\nUsing complete dataset for causal inference analysis...')

# Load FOS taxonomy for field dummy variables (still needed for matching)
with open('fos_taxonomy_v0.1.2.json', 'r') as f:
    fos_taxonomy = json.load(f)

# Use complete dataset
df_filtered = df.copy()

print(f"Total papers in analysis: {len(df_filtered):,}")

print('Adding topic persistence scores...')

# Create mapping from paper ID to dominant topic and persistence score
topic_persistence_map = {}
for _, row in topic_attribution_df.iterrows():
    paper_id = row['id']
    if paper_id not in topic_persistence_map or row['topic_persistence_score'] > topic_persistence_map[paper_id]['score']:
        topic_persistence_map[paper_id] = {
            'topic': row['topic'],
            'score': row['topic_persistence_score'],
            'is_ai_topic': row['is_ai_topic']
        }

# Add to main dataframe
df_filtered['dominant_topic'] = df_filtered['id'].map(lambda x: topic_persistence_map.get(x, {}).get('topic'))
df_filtered['topic_persistence_score'] = df_filtered['id'].map(lambda x: topic_persistence_map.get(x, {}).get('score'))
df_filtered['dominant_topic_is_ai'] = df_filtered['id'].map(lambda x: topic_persistence_map.get(x, {}).get('is_ai_topic', False))

print(f"Papers with topic persistence scores: {df_filtered['topic_persistence_score'].notna().sum():,}")

print('Preparing matching variables...')

# Define matching variables per the case study specifications
matching_variables = [
    # Research Quality
    'citationcount', 'influentialcitationcount', 'fwci',
    # Citation Exposure  
    'year',
    # Collaboration Scale
    'authorcount',  # Now we have this available
    # Gender Composition (for balance only, per specifications)
    'has_women_authors',
    # NOT including science_industry_collaboration per new specifications
]

# We can also use referencecount as additional proxy for collaboration
if 'referencecount' in df_filtered.columns:
    matching_variables.append('referencecount')

# Filter matching variables to only include those that exist in the dataframe
matching_variables = [var for var in matching_variables if var in df_filtered.columns]

print(f"Total matching variables: {len(matching_variables)}")
print(f"Available matching variables: {matching_variables}")

def propensity_score_match(df, treatment_col, features, caliper=0.25, verbose=True):
    """
    Perform propensity score matching for causal inference
    """
    # Create working copy
    df_work = df.copy().reset_index(drop=True)
    
    # Prepare data
    X = df_work[features]
    y = df_work[treatment_col]
    
    if verbose:
        print(f"Treatment group size: {y.sum()}")
        print(f"Control group size: {(~y).sum()}")
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    try:
        logit = LogisticRegression(max_iter=1000, solver='liblinear')
        logit.fit(X_scaled, y)
        df_work['propensity_score'] = logit.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Split groups
    treated = df_work[df_work[treatment_col] == True].copy()
    control = df_work[df_work[treatment_col] == False].copy()
    
    if len(treated) == 0 or len(control) == 0:
        print("Empty treatment or control group")
        return pd.DataFrame(), pd.DataFrame()
    
    # Matching
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    valid_pairs = distances.flatten() <= caliper
    
    if not any(valid_pairs):
        print(f"No matches found within caliper {caliper}")
        return pd.DataFrame(), pd.DataFrame()
    
    matched_treated = treated[valid_pairs].copy()
    matched_control_indices = indices.flatten()[valid_pairs]
    matched_control = control.iloc[matched_control_indices].copy()
    
    if verbose:
        print(f"Matched {len(matched_treated)} pairs (caliper={caliper})")
    
    return matched_treated, matched_control

print("\n" + "="*60)
print("PSM ANALYSIS A: GREEN OA vs CLOSED ACCESS")
print("="*60)

# Create dataset for PSM A
psm_a_df = df_filtered[
    (df_filtered['green_oa_only'] == True) | 
    (df_filtered['closed_access'] == True)
].copy()

# Remove papers with missing matching variables
psm_a_df = psm_a_df.dropna(subset=matching_variables)

print(f"PSM A dataset size: {len(psm_a_df):,}")
print(f"Green OA only: {psm_a_df['green_oa_only'].sum():,}")
print(f"Closed Access: {psm_a_df['closed_access'].sum():,}")

# Perform matching
green_matched, closed_matched_a = propensity_score_match(
    psm_a_df, 'green_oa_only', matching_variables, caliper=0.25
)

if not green_matched.empty:
    print(f"\nMatched pairs: {len(green_matched):,}")
    
    # Save matched datasets
    green_matched.to_excel(os.path.join(results_path, 'psm_a_green_matched.xlsx'), index=False)
    closed_matched_a.to_excel(os.path.join(results_path, 'psm_a_closed_matched.xlsx'), index=False)
    
    # Check balance
    balance_a = pd.DataFrame({
        'Green_OA_mean': green_matched[matching_variables].mean(),
        'Closed_mean': closed_matched_a[matching_variables].mean()
    })
    balance_a['std_diff'] = (balance_a['Green_OA_mean'] - balance_a['Closed_mean']) / \
                           np.sqrt((green_matched[matching_variables].var() + 
                                   closed_matched_a[matching_variables].var()) / 2)
    
    balance_a.to_excel(os.path.join(results_path, 'psm_a_balance_check.xlsx'))
    print("Balance statistics saved")

print("\n" + "="*60)
print("PSM ANALYSIS B: PUBLISHED OA vs CLOSED ACCESS")
print("="*60)

# Create dataset for PSM B
psm_b_df = df_filtered[
    (df_filtered['published_oa_only'] == True) | 
    (df_filtered['closed_access'] == True)
].copy()

# Remove papers with missing matching variables
psm_b_df = psm_b_df.dropna(subset=matching_variables)

print(f"PSM B dataset size: {len(psm_b_df):,}")
print(f"Published OA only: {psm_b_df['published_oa_only'].sum():,}")
print(f"Closed Access: {psm_b_df['closed_access'].sum():,}")

# Perform matching
published_matched, closed_matched_b = propensity_score_match(
    psm_b_df, 'published_oa_only', matching_variables, caliper=0.25
)

if not published_matched.empty:
    print(f"\nMatched pairs: {len(published_matched):,}")
    
    # Save matched datasets
    published_matched.to_excel(os.path.join(results_path, 'psm_b_published_matched.xlsx'), index=False)
    closed_matched_b.to_excel(os.path.join(results_path, 'psm_b_closed_matched.xlsx'), index=False)
    
    # Check balance
    balance_b = pd.DataFrame({
        'Published_OA_mean': published_matched[matching_variables].mean(),
        'Closed_mean': closed_matched_b[matching_variables].mean()
    })
    balance_b['std_diff'] = (balance_b['Published_OA_mean'] - balance_b['Closed_mean']) / \
                           np.sqrt((published_matched[matching_variables].var() + 
                                   closed_matched_b[matching_variables].var()) / 2)
    
    balance_b.to_excel(os.path.join(results_path, 'psm_b_balance_check.xlsx'))
    print("Balance statistics saved")

print("\n" + "="*60)
print("OUTCOME ANALYSIS")
print("="*60)

# Define outcome variables per case study specifications - only use available columns
available_outcomes = [
    # Academic Impact
    'citationcount', 'influentialcitationcount', 'fwci',
    # Economic Impact  
    'patent_citations',
    # Science-Industry Links (moved from matching variables)
    'science_industry_collaboration',
    # Topic Persistence
    'topic_persistence_score',
    # Gender Representation (outcomes, not confounders per specifications)
    'has_woman_first_author',
    'has_woman_last_author', 
    'only_women_authors'
]

# Filter to only include columns that exist in the dataframe
outcome_variables = [var for var in available_outcomes if var in df_filtered.columns]

print(f"Available outcome variables: {outcome_variables}")

def calculate_treatment_effects(treated_df, control_df, outcomes, treatment_name):
    """Calculate treatment effects for given outcomes"""
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
        
        # Calculate standard errors (assuming normal distribution)
        treated_se = treated_vals.std() / np.sqrt(len(treated_vals))
        control_se = control_vals.std() / np.sqrt(len(control_vals))
        effect_se = np.sqrt(treated_se**2 + control_se**2)
        
        results.append({
            'outcome': outcome,
            'treatment': treatment_name,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'effect': effect,
            'effect_se': effect_se,
            'treated_n': len(treated_vals),
            'control_n': len(control_vals)
        })
    
    return pd.DataFrame(results)

# Calculate effects for PSM A (Green OA)
if not green_matched.empty:
    effects_a = calculate_treatment_effects(
        green_matched, closed_matched_a, outcome_variables, 'Green_OA'
    )
    effects_a.to_excel(os.path.join(results_path, 'treatment_effects_green_oa.xlsx'), index=False)
    print("Green OA treatment effects calculated")

# Calculate effects for PSM B (Published OA)  
if not published_matched.empty:
    effects_b = calculate_treatment_effects(
        published_matched, closed_matched_b, outcome_variables, 'Published_OA'
    )
    effects_b.to_excel(os.path.join(results_path, 'treatment_effects_published_oa.xlsx'), index=False)
    print("Published OA treatment effects calculated")

print("\n" + "="*60)
print("DESCRIPTIVE ANALYSIS: OVERALL OA vs CLOSED")
print("="*60)

# Create overall OA category (for descriptive purposes only)
df_filtered['any_oa'] = (df_filtered['green_oa_only'] | df_filtered['published_oa_only'])

# Calculate descriptive differences (not causal estimates)
descriptive_df = df_filtered[
    (df_filtered['any_oa'] == True) | (df_filtered['closed_access'] == True)
].copy()

descriptive_effects = calculate_treatment_effects(
    descriptive_df[descriptive_df['any_oa'] == True],
    descriptive_df[descriptive_df['closed_access'] == True],
    outcome_variables,
    'Any_OA_vs_Closed'
)

descriptive_effects.to_excel(os.path.join(results_path, 'descriptive_effects_any_oa.xlsx'), index=False)

print("\n" + "="*60)
print("GENERATING SUMMARY STATISTICS")
print("="*60)

# Create comprehensive summary
summary_stats = {
    'Total_Papers': len(df_filtered),
    'Green_OA_Only': df_filtered['green_oa_only'].sum(),
    'Published_OA_Only': df_filtered['published_oa_only'].sum(),
    'Closed_Access': df_filtered['closed_access'].sum(),
    'Dual_Mode_Excluded': df_filtered['dual_mode_oa'].sum(),
    'Bronze_Excluded': df_filtered['bronze_oa'].sum(),
    'PSM_A_Matched_Pairs': len(green_matched) if not green_matched.empty else 0,
    'PSM_B_Matched_Pairs': len(published_matched) if not published_matched.empty else 0,
    'Papers_with_Topic_Persistence': df_filtered['topic_persistence_score'].notna().sum()
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_excel(os.path.join(results_path, 'summary_statistics.xlsx'), index=False)

print("\nSummary Statistics:")
for metric, value in summary_stats.items():
    print(f"{metric}: {value:,}")

print(f"\n{'='*60}")
print("PATHEOS CASE STUDY ANALYSIS COMPLETE")
print(f"Results saved to: {results_path}")
print(f"{'='*60}")

print("\n" + "="*60)
print("CREATING COMPREHENSIVE PRESENTATION TABLES")
print("="*60)

# TABLE 1: Executive Summary
executive_summary = {
    'Metric': [
        'Total Sample Size',
        'Green OA Only Papers',
        'Published OA Only Papers', 
        'Closed Access Papers',
        'Dual-Mode OA (Excluded)',
        'Bronze OA (Excluded)',
        'Green OA vs Closed - Matched Pairs',
        'Published OA vs Closed - Matched Pairs',
        'Papers with Topic Persistence Scores',
        'Median Topic Persistence Score',
        'Treatment Effect Power (Green OA)',
        'Treatment Effect Power (Published OA)'
    ],
    'Value': [
        len(df_filtered),
        df_filtered['green_oa_only'].sum(),
        df_filtered['published_oa_only'].sum(),
        df_filtered['closed_access'].sum(),
        df_filtered['dual_mode_oa'].sum(),
        df_filtered['bronze_oa'].sum(),
        len(green_matched) if not green_matched.empty else 0,
        len(published_matched) if not published_matched.empty else 0,
        df_filtered['topic_persistence_score'].notna().sum(),
        round(df_filtered['topic_persistence_score'].median(), 2) if df_filtered['topic_persistence_score'].notna().sum() > 0 else 0,
        'Sufficient' if (not green_matched.empty and len(green_matched) >= 100) else 'Limited',
        'Sufficient' if (not published_matched.empty and len(published_matched) >= 100) else 'Limited'
    ]
}

executive_summary_df = pd.DataFrame(executive_summary)
executive_summary_df.to_excel(os.path.join(tables_path, '01_executive_summary.xlsx'), index=False)

# TABLE 2: Treatment Group Characteristics
if not green_matched.empty and not closed_matched_a.empty:
    green_characteristics = {
        'Group': 'Green OA (Treatment)',
        'Sample_Size': len(green_matched),
        'Mean_Citations': round(green_matched['citationcount'].mean(), 1),
        'Mean_FWCI': round(green_matched['fwci'].mean(), 2),
        'Mean_Authors': round(green_matched['authorcount'].mean(), 1),
        'Women_Authors_Pct': round(green_matched['has_women_authors'].mean() * 100, 1),
        'Women_First_Author_Pct': round(green_matched['has_woman_first_author'].mean() * 100, 1),
        'Industry_Collaboration_Pct': round(green_matched['science_industry_collaboration'].mean() * 100, 1),
        'Mean_Patent_Citations': round(green_matched['patent_citations'].mean(), 2),
        'Mean_Topic_Persistence': round(green_matched['topic_persistence_score'].mean(), 2)
    }
    
    closed_a_characteristics = {
        'Group': 'Closed Access (Control A)',
        'Sample_Size': len(closed_matched_a),
        'Mean_Citations': round(closed_matched_a['citationcount'].mean(), 1),
        'Mean_FWCI': round(closed_matched_a['fwci'].mean(), 2),
        'Mean_Authors': round(closed_matched_a['authorcount'].mean(), 1),
        'Women_Authors_Pct': round(closed_matched_a['has_women_authors'].mean() * 100, 1),
        'Women_First_Author_Pct': round(closed_matched_a['has_woman_first_author'].mean() * 100, 1),
        'Industry_Collaboration_Pct': round(closed_matched_a['science_industry_collaboration'].mean() * 100, 1),
        'Mean_Patent_Citations': round(closed_matched_a['patent_citations'].mean(), 2),
        'Mean_Topic_Persistence': round(closed_matched_a['topic_persistence_score'].mean(), 2)
    }

if not published_matched.empty and not closed_matched_b.empty:
    published_characteristics = {
        'Group': 'Published OA (Treatment)',
        'Sample_Size': len(published_matched),
        'Mean_Citations': round(published_matched['citationcount'].mean(), 1),
        'Mean_FWCI': round(published_matched['fwci'].mean(), 2),
        'Mean_Authors': round(published_matched['authorcount'].mean(), 1),
        'Women_Authors_Pct': round(published_matched['has_women_authors'].mean() * 100, 1),
        'Women_First_Author_Pct': round(published_matched['has_woman_first_author'].mean() * 100, 1),
        'Industry_Collaboration_Pct': round(published_matched['science_industry_collaboration'].mean() * 100, 1),
        'Mean_Patent_Citations': round(published_matched['patent_citations'].mean(), 2),
        'Mean_Topic_Persistence': round(published_matched['topic_persistence_score'].mean(), 2)
    }
    
    closed_b_characteristics = {
        'Group': 'Closed Access (Control B)',
        'Sample_Size': len(closed_matched_b),
        'Mean_Citations': round(closed_matched_b['citationcount'].mean(), 1),
        'Mean_FWCI': round(closed_matched_b['fwci'].mean(), 2),
        'Mean_Authors': round(closed_matched_b['authorcount'].mean(), 1),
        'Women_Authors_Pct': round(closed_matched_b['has_women_authors'].mean() * 100, 1),
        'Women_First_Author_Pct': round(closed_matched_b['has_woman_first_author'].mean() * 100, 1),
        'Industry_Collaboration_Pct': round(closed_matched_b['science_industry_collaboration'].mean() * 100, 1),
        'Mean_Patent_Citations': round(closed_matched_b['patent_citations'].mean(), 2),
        'Mean_Topic_Persistence': round(closed_matched_b['topic_persistence_score'].mean(), 2)
    }

# Combine characteristics
characteristics_data = []
if not green_matched.empty:
    characteristics_data.extend([green_characteristics, closed_a_characteristics])
if not published_matched.empty:
    characteristics_data.extend([published_characteristics, closed_b_characteristics])

if characteristics_data:
    characteristics_df = pd.DataFrame(characteristics_data)
    characteristics_df.to_excel(os.path.join(tables_path, '02_treatment_group_characteristics.xlsx'), index=False)

# TABLE 3: Causal Effects Summary
causal_effects_summary = []

# Process Green OA effects
if not green_matched.empty and 'effects_a' in locals():
    for _, effect in effects_a.iterrows():
        # Calculate Cohen's d for effect size
        if effect['control_n'] > 0 and effect['treated_n'] > 0:
            pooled_std = np.sqrt(((effect['treated_n']-1) * (effect['effect_se'] * np.sqrt(effect['treated_n']))**2 + 
                                 (effect['control_n']-1) * (effect['effect_se'] * np.sqrt(effect['control_n']))**2) / 
                                (effect['treated_n'] + effect['control_n'] - 2))
            cohens_d = effect['effect'] / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
            
        # Calculate t-statistic for significance
        t_stat = effect['effect'] / effect['effect_se'] if effect['effect_se'] > 0 else 0
        
        causal_effects_summary.append({
            'Treatment': 'Green OA vs Closed Access',
            'Outcome': effect['outcome'].replace('_', ' ').title(),
            'Treatment_Mean': round(effect['treated_mean'], 3),
            'Control_Mean': round(effect['control_mean'], 3),
            'Causal_Effect': round(effect['effect'], 3),
            'Effect_SE': round(effect['effect_se'], 3),
            'Cohens_D': round(cohens_d, 3),
            'T_Statistic': round(t_stat, 2),
            'Significant_05': 'Yes' if abs(t_stat) > 1.96 else 'No',
            'Effect_Size_Interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small' if abs(cohens_d) > 0.2 else 'Negligible',
            'Sample_Size': f"{effect['treated_n']} vs {effect['control_n']}"
        })

# Process Published OA effects
if not published_matched.empty and 'effects_b' in locals():
    for _, effect in effects_b.iterrows():
        # Calculate Cohen's d for effect size
        if effect['control_n'] > 0 and effect['treated_n'] > 0:
            pooled_std = np.sqrt(((effect['treated_n']-1) * (effect['effect_se'] * np.sqrt(effect['treated_n']))**2 + 
                                 (effect['control_n']-1) * (effect['effect_se'] * np.sqrt(effect['control_n']))**2) / 
                                (effect['treated_n'] + effect['control_n'] - 2))
            cohens_d = effect['effect'] / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
            
        t_stat = effect['effect'] / effect['effect_se'] if effect['effect_se'] > 0 else 0
        
        causal_effects_summary.append({
            'Treatment': 'Published OA vs Closed Access',
            'Outcome': effect['outcome'].replace('_', ' ').title(),
            'Treatment_Mean': round(effect['treated_mean'], 3),
            'Control_Mean': round(effect['control_mean'], 3),
            'Causal_Effect': round(effect['effect'], 3),
            'Effect_SE': round(effect['effect_se'], 3),
            'Cohens_D': round(cohens_d, 3),
            'T_Statistic': round(t_stat, 2),
            'Significant_05': 'Yes' if abs(t_stat) > 1.96 else 'No',
            'Effect_Size_Interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small' if abs(cohens_d) > 0.2 else 'Negligible',
            'Sample_Size': f"{effect['treated_n']} vs {effect['control_n']}"
        })

if causal_effects_summary:
    causal_effects_df = pd.DataFrame(causal_effects_summary)
    causal_effects_df.to_excel(os.path.join(tables_path, '03_causal_effects_summary.xlsx'), index=False)

# TABLE 4: Topic Persistence Analysis
topic_persistence_analysis = []

# Overall topic persistence by OA status
for oa_type, oa_filter in [
    ('Green OA Only', df_filtered['green_oa_only'] == True),
    ('Published OA Only', df_filtered['published_oa_only'] == True),
    ('Closed Access', df_filtered['closed_access'] == True),
    ('Any OA', (df_filtered['green_oa_only'] | df_filtered['published_oa_only']) == True)
]:
    group_data = df_filtered[oa_filter & df_filtered['topic_persistence_score'].notna()]
    
    if len(group_data) > 0:
        topic_persistence_analysis.append({
            'OA_Status': oa_type,
            'Sample_Size': len(group_data),
            'Mean_Topic_Persistence': round(group_data['topic_persistence_score'].mean(), 2),
            'Median_Topic_Persistence': round(group_data['topic_persistence_score'].median(), 2),
            'Std_Topic_Persistence': round(group_data['topic_persistence_score'].std(), 2),
            'High_Persistence_Rate': round((group_data['topic_persistence_score'] > group_data['topic_persistence_score'].quantile(0.75)).mean() * 100, 1),
            'Top_10pct_Persistence_Rate': round((group_data['topic_persistence_score'] > group_data['topic_persistence_score'].quantile(0.9)).mean() * 100, 1)
        })

topic_persistence_df = pd.DataFrame(topic_persistence_analysis)
topic_persistence_df.to_excel(os.path.join(tables_path, '04_topic_persistence_analysis.xlsx'), index=False)

# TABLE 5: Gender Equity Outcomes
gender_equity_analysis = []

# Analyze gender outcomes for matched samples
for treatment_name, treated_group, control_group in [
    ('Green OA vs Closed Access', green_matched, closed_matched_a),
    ('Published OA vs Closed Access', published_matched, closed_matched_b)
]:
    if not treated_group.empty and not control_group.empty:
        gender_outcomes = ['has_woman_first_author', 'has_woman_last_author', 'only_women_authors']
        
        for outcome in gender_outcomes:
            if outcome in treated_group.columns:
                treated_rate = treated_group[outcome].mean() * 100
                control_rate = control_group[outcome].mean() * 100
                
                gender_equity_analysis.append({
                    'Treatment': treatment_name,
                    'Gender_Outcome': outcome.replace('_', ' ').title(),
                    'Treatment_Rate_Pct': round(treated_rate, 1),
                    'Control_Rate_Pct': round(control_rate, 1),
                    'Difference_PP': round(treated_rate - control_rate, 1),
                    'Relative_Improvement_Pct': round(((treated_rate / control_rate - 1) * 100) if control_rate > 0 else 0, 1),
                    'Treatment_N': treated_group[outcome].notna().sum(),
                    'Control_N': control_group[outcome].notna().sum()
                })

if gender_equity_analysis:
    gender_equity_df = pd.DataFrame(gender_equity_analysis)
    gender_equity_df.to_excel(os.path.join(tables_path, '05_gender_equity_outcomes.xlsx'), index=False)

# TABLE 6: Economic Impact Analysis
economic_impact_analysis = []

for treatment_name, treated_group, control_group in [
    ('Green OA vs Closed Access', green_matched, closed_matched_a),
    ('Published OA vs Closed Access', published_matched, closed_matched_b)
]:
    if not treated_group.empty and not control_group.empty:
        # Patent citations analysis
        treated_patents = treated_group['patent_citations'].mean()
        control_patents = control_group['patent_citations'].mean()
        
        # Science-industry collaboration
        treated_industry = treated_group['science_industry_collaboration'].mean() * 100
        control_industry = control_group['science_industry_collaboration'].mean() * 100
        
        economic_impact_analysis.append({
            'Treatment': treatment_name,
            'Mean_Patent_Citations_Treatment': round(treated_patents, 2),
            'Mean_Patent_Citations_Control': round(control_patents, 2),
            'Patent_Citations_Difference': round(treated_patents - control_patents, 2),
            'Patent_Citations_Uplift_Pct': round(((treated_patents / control_patents - 1) * 100) if control_patents > 0 else 0, 1),
            'Industry_Collaboration_Rate_Treatment': round(treated_industry, 1),
            'Industry_Collaboration_Rate_Control': round(control_industry, 1),
            'Industry_Collaboration_Difference_PP': round(treated_industry - control_industry, 1),
            'Papers_With_Patent_Citations_Treatment': (treated_group['patent_citations'] > 0).sum(),
            'Papers_With_Patent_Citations_Control': (control_group['patent_citations'] > 0).sum()
        })

if economic_impact_analysis:
    economic_impact_df = pd.DataFrame(economic_impact_analysis)
    economic_impact_df.to_excel(os.path.join(tables_path, '06_economic_impact_analysis.xlsx'), index=False)

# TABLE 7: Publication Year Analysis
year_analysis = []
years = sorted(df_filtered['year'].unique())

for year in years[-10:]:  # Last 10 years
    year_data = df_filtered[df_filtered['year'] == year]
    
    year_analysis.append({
        'Year': year,
        'Total_Papers': len(year_data),
        'Green_OA_Papers': (year_data['green_oa_only'] == True).sum(),
        'Published_OA_Papers': (year_data['published_oa_only'] == True).sum(),
        'Closed_Access_Papers': (year_data['closed_access'] == True).sum(),
        'Green_OA_Rate_Pct': round((year_data['green_oa_only'] == True).mean() * 100, 1),
        'Published_OA_Rate_Pct': round((year_data['published_oa_only'] == True).mean() * 100, 1),
        'Total_OA_Rate_Pct': round((year_data['green_oa_only'] | year_data['published_oa_only']).mean() * 100, 1),
        'Mean_Topic_Persistence': round(year_data['topic_persistence_score'].mean(), 2) if year_data['topic_persistence_score'].notna().sum() > 0 else None
    })

year_analysis_df = pd.DataFrame(year_analysis)
year_analysis_df.to_excel(os.path.join(tables_path, '07_publication_year_analysis.xlsx'), index=False)

# TABLE 8: Robustness Check - Alternative Specifications
robustness_analysis = []

# Compare with unmatched samples
if 'descriptive_effects' in locals():
    for _, effect in descriptive_effects.iterrows():
        robustness_analysis.append({
            'Analysis_Type': 'Descriptive (Unmatched)',
            'Comparison': 'Any OA vs Closed Access',
            'Outcome': effect['outcome'].replace('_', ' ').title(),
            'Treatment_Mean': round(effect['treated_mean'], 3),
            'Control_Mean': round(effect['control_mean'], 3),
            'Effect': round(effect['effect'], 3),
            'Sample_Size': f"{effect['treated_n']} vs {effect['control_n']}"
        })

# Add matched results for comparison
for treatment_name, effects_data in [
    ('Green OA vs Closed (Matched)', effects_a if 'effects_a' in locals() else pd.DataFrame()),
    ('Published OA vs Closed (Matched)', effects_b if 'effects_b' in locals() else pd.DataFrame())
]:
    if not effects_data.empty:
        for _, effect in effects_data.iterrows():
            robustness_analysis.append({
                'Analysis_Type': 'Causal (Matched)',
                'Comparison': treatment_name,
                'Outcome': effect['outcome'].replace('_', ' ').title(),
                'Treatment_Mean': round(effect['treated_mean'], 3),
                'Control_Mean': round(effect['control_mean'], 3),
                'Effect': round(effect['effect'], 3),
                'Sample_Size': f"{effect['treated_n']} vs {effect['control_n']}"
            })

if robustness_analysis:
    robustness_df = pd.DataFrame(robustness_analysis)
    robustness_df.to_excel(os.path.join(tables_path, '08_robustness_analysis.xlsx'), index=False)

print("\n" + "="*60)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*60)

# VISUALIZATION 1: Sample Overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# OA Category Distribution
oa_counts = [
    df_filtered['green_oa_only'].sum(),
    df_filtered['published_oa_only'].sum(),
    df_filtered['closed_access'].sum(),
    df_filtered['dual_mode_oa'].sum()
]
oa_labels = ['Green OA\nOnly', 'Published OA\nOnly', 'Closed\nAccess', 'Dual-Mode\n(Excluded)']
colors = [pathos_blue, pathos_orange, 'gray', 'lightcoral']

axes[0,0].pie(oa_counts, labels=oa_labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0,0].set_title('Open Access Category Distribution\n(Treatment Definition)')

# Year distribution
year_counts = df_filtered['year'].value_counts().sort_index()
axes[0,1].bar(year_counts.index, year_counts.values, alpha=0.8, color=pathos_blue)
axes[0,1].set_title('Papers by Publication Year')
axes[0,1].set_xlabel('Year')
axes[0,1].set_ylabel('Number of Papers')
axes[0,1].tick_params(axis='x', rotation=45)

# Matching success rates
if not green_matched.empty and not published_matched.empty:
    matching_data = [
        len(green_matched),
        len(published_matched)
    ]
    matching_labels = ['Green OA\nvs Closed', 'Published OA\nvs Closed']
    
    axes[1,0].bar(matching_labels, matching_data, color=[pathos_blue, pathos_orange], alpha=0.8)
    axes[1,0].set_title('Successful Matches by Treatment Type')
    axes[1,0].set_ylabel('Number of Matched Pairs')

# Topic persistence distribution
if df_filtered['topic_persistence_score'].notna().sum() > 0:
    axes[1,1].hist(df_filtered['topic_persistence_score'].dropna(), bins=50, alpha=0.7, 
                   color=pathos_blue, edgecolor='black')
    axes[1,1].set_title('Distribution of Topic Persistence Scores')
    axes[1,1].set_xlabel('Topic Persistence Score')
    axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '01_sample_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# VISUALIZATION 2: Causal Effects
if causal_effects_summary:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Treatment effects plot
    causal_df = pd.DataFrame(causal_effects_summary)
    
    # Separate by treatment type
    green_effects = causal_df[causal_df['Treatment'] == 'Green OA vs Closed Access']
    published_effects = causal_df[causal_df['Treatment'] == 'Published OA vs Closed Access']
    
    # Plot effect sizes
    if not green_effects.empty:
        y_pos = range(len(green_effects))
        ax1.barh(y_pos, green_effects['Causal_Effect'], alpha=0.8, color=pathos_blue)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(green_effects['Outcome'])
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Causal Effect Size')
        ax1.set_title('Green OA Treatment Effects')
        ax1.grid(True, alpha=0.3)
    
    if not published_effects.empty:
        y_pos = range(len(published_effects))
        ax2.barh(y_pos, published_effects['Causal_Effect'], alpha=0.8, color=pathos_orange)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(published_effects['Outcome'])
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Causal Effect Size')
        ax2.set_title('Published OA Treatment Effects')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '02_causal_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

# VISUALIZATION 3: Topic Persistence Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Topic persistence by OA status
if not topic_persistence_df.empty:
    axes[0,0].bar(topic_persistence_df['OA_Status'], topic_persistence_df['Mean_Topic_Persistence'],
                  color=[pathos_blue, pathos_orange, 'gray', 'lightgreen'], alpha=0.8)
    axes[0,0].set_title('Mean Topic Persistence by OA Status')
    axes[0,0].set_ylabel('Mean Topic Persistence Score')
    axes[0,0].tick_params(axis='x', rotation=45)

# Treatment vs Control comparison for topic persistence
if not green_matched.empty and not closed_matched_a.empty:
    data_to_plot = [
        closed_matched_a['topic_persistence_score'].dropna().values,
        green_matched['topic_persistence_score'].dropna().values
    ]
    labels = ['Closed Access', 'Green OA']
    
    box_plot = axes[0,1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = [pathos_colors[0], pathos_colors[1]]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0,1].set_title('Topic Persistence: Green OA vs Closed Access')
    axes[0,1].set_ylabel('Topic Persistence Score')

# Gender equity outcomes
if not gender_equity_df.empty:
    gender_pivot = gender_equity_df.pivot(index='Gender_Outcome', columns='Treatment', values='Difference_PP')
    gender_pivot.plot(kind='bar', ax=axes[1,0], color=[pathos_blue, pathos_orange])
    axes[1,0].set_title('Gender Equity Effects (Percentage Point Difference)')
    axes[1,0].set_ylabel('Difference (Treatment - Control)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].legend()

# Economic impact
if not economic_impact_df.empty:
    x = range(len(economic_impact_df))
    width = 0.35
    
    axes[1,1].bar([i - width/2 for i in x], economic_impact_df['Mean_Patent_Citations_Control'], 
                  width, label='Control', alpha=0.8, color=pathos_blue)
    axes[1,1].bar([i + width/2 for i in x], economic_impact_df['Mean_Patent_Citations_Treatment'], 
                  width, label='Treatment', alpha=0.8, color=pathos_orange)
    
    axes[1,1].set_xlabel('Treatment Type')
    axes[1,1].set_ylabel('Mean Patent Citations')
    axes[1,1].set_title('Economic Impact: Patent Citations')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(economic_impact_df['Treatment'], rotation=45, ha='right')
    axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '03_outcome_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# VISUALIZATION 4: Temporal Trends
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# OA adoption over time
if not year_analysis_df.empty:
    axes[0,0].plot(year_analysis_df['Year'], year_analysis_df['Green_OA_Rate_Pct'], 
                   marker='o', label='Green OA', color=pathos_blue)
    axes[0,0].plot(year_analysis_df['Year'], year_analysis_df['Published_OA_Rate_Pct'], 
                   marker='s', label='Published OA', color=pathos_orange)
    axes[0,0].plot(year_analysis_df['Year'], year_analysis_df['Total_OA_Rate_Pct'], 
                   marker='^', label='Total OA', color='green')
    
    axes[0,0].set_title('Open Access Adoption Rates Over Time')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('OA Rate (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

# Topic persistence over time
if not year_analysis_df.empty and year_analysis_df['Mean_Topic_Persistence'].notna().sum() > 0:
    valid_years = year_analysis_df[year_analysis_df['Mean_Topic_Persistence'].notna()]
    axes[0,1].bar(valid_years['Year'], valid_years['Mean_Topic_Persistence'], 
                  alpha=0.8, color=pathos_blue)
    axes[0,1].set_title('Topic Persistence by Publication Year')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Mean Topic Persistence')

# Balance check visualization
if not green_matched.empty and not closed_matched_a.empty:
    balance_vars = ['citationcount', 'fwci', 'authorcount', 'has_women_authors']
    available_balance_vars = [var for var in balance_vars if var in green_matched.columns]
    
    green_means = [green_matched[var].mean() for var in available_balance_vars]
    closed_means = [closed_matched_a[var].mean() for var in available_balance_vars]
    
    x = range(len(available_balance_vars))
    width = 0.35
    
    axes[1,0].bar([i - width/2 for i in x], closed_means, width, 
                  label='Closed Access', alpha=0.8, color=pathos_blue)
    axes[1,0].bar([i + width/2 for i in x], green_means, width,
                  label='Green OA', alpha=0.8, color=pathos_orange)
    
    axes[1,0].set_xlabel('Matching Variables')
    axes[1,0].set_ylabel('Mean Value')
    axes[1,0].set_title('Balance Check: Green OA vs Closed Access')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([var.replace('_', ' ').title() for var in available_balance_vars], rotation=45)
    axes[1,0].legend()

# Effect size comparison
if causal_effects_summary:
    causal_df = pd.DataFrame(causal_effects_summary)
    significant_effects = causal_df[causal_df['Significant_05'] == 'Yes']
    
    if not significant_effects.empty:
        axes[1,1].scatter(significant_effects['Causal_Effect'], significant_effects['Cohens_D'], 
                         c=[pathos_blue if 'Green' in x else pathos_orange for x in significant_effects['Treatment']], 
                         s=100, alpha=0.7)
        
        axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Causal Effect Size')
        axes[1,1].set_ylabel('Cohen\'s D (Effect Size)')
        axes[1,1].set_title('Significant Effects: Magnitude vs Statistical Significance')
        axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '04_temporal_and_balance.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("GENERATING ANALYSIS CONCLUSIONS")
print("="*60)

# Calculate some summary statistics for the conclusions
total_papers = len(df_filtered)
green_papers = df_filtered['green_oa_only'].sum()
published_papers = df_filtered['published_oa_only'].sum()
closed_papers = df_filtered['closed_access'].sum()
dual_mode_papers = df_filtered['dual_mode_oa'].sum()
bronze_papers = df_filtered['bronze_oa'].sum()
green_matched_count = len(green_matched) if not green_matched.empty else 0
published_matched_count = len(published_matched) if not published_matched.empty else 0
papers_with_persistence = df_filtered['topic_persistence_score'].notna().sum()
median_persistence = df_filtered['topic_persistence_score'].median() if df_filtered['topic_persistence_score'].notna().sum() > 0 else 0
significant_effects_count = len([x for x in causal_effects_summary if x['Significant_05'] == 'Yes']) if causal_effects_summary else 0

print(f"\nAll tables saved to: {tables_path}")
print(f"All visualizations saved to: {viz_path}")
print(f"Main results saved to: {results_path}")
