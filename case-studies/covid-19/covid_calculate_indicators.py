"""

For the COVID case study due to the openness issues with the CORD-19 dataset, we use a regression-based model to analyze downstream impact conditional on artifact creation and citations.

"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


print('Loading COVID dataset and preparing regression-based analysis...')

# Load the complete dataset parquet file
papers_covid_df = pd.read_parquet(os.path.join('PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes', 'complete_collection_df_fix.parquet'))

# Create v4b subfolder for all outputs
results_path = 'PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes/results'
os.makedirs(results_path, exist_ok=True)

# Create visualizations subfolder
viz_path = os.path.join(results_path, 'visualizations')
os.makedirs(viz_path, exist_ok=True)

# Create tables subfolder
tables_path = os.path.join(results_path, 'tables')
os.makedirs(tables_path, exist_ok=True)

# Set up plotting style with PATHOS colors
plt.style.use('default')
pathos_blue = '#2E5C8A'    # Medium dark blue
pathos_orange = '#D17A2A'  # Medium dark orange
pathos_colors = [pathos_blue, pathos_orange]
sns.set_palette([pathos_blue, pathos_orange])

print(f"Initial dataset size: {len(papers_covid_df)}")
print(f"Available columns: {sorted(papers_covid_df.columns.tolist())}")

# STEP 1: Filter sample according to design criteria
print("Filtering sample based on design criteria...")

# Filter 1: Papers that created a dataset or software (shared an artifact)
artifact_filter = (
    (papers_covid_df['named_datasets_created'] > 0) |
    (papers_covid_df['unnamed_datasets_created'] > 0) |
    (papers_covid_df['named_software_created'] > 0) |
    (papers_covid_df['unnamed_software_created'] > 0)
)

# Filter 2: Papers that were cited at least once
citation_filter = papers_covid_df['citationcount'] > 0

# Apply both filters
filtered_df = papers_covid_df[artifact_filter & citation_filter].copy()

print(f"Sample size after filtering (artifact creators + cited): {len(filtered_df)}")

# STEP 2: Create main explanatory variable
# Binary indicator for having at least one Reuse-Artifact citance
filtered_df['has_reuse_artifact_citance'] = (filtered_df['reuse_artifact_inbound'] > 0).astype(int)

print(f"Papers with reuse-artifact citances: {filtered_df['has_reuse_artifact_citance'].sum()}")
print(f"Papers without reuse-artifact citances: {(filtered_df['has_reuse_artifact_citance'] == 0).sum()}")

# STEP 3: Prepare outcome variables
outcome_vars = [
    # Main clinical impact measures
    'clinical_trial_citations',
    'clinical_guideline_citations',
    'total_clinical_citations',
    
    # Influential vs non-influential breakdown
    'clinical_trial_citations_influential',
    'clinical_trial_citations_non_influential', 
    'clinical_guideline_citations_influential',
    'clinical_guideline_citations_non_influential',
    'total_clinical_citations_influential',
    'total_clinical_citations_non_influential',
    
    # Other outcomes
    'patent_citations',
    'science_industry_collaboration'
]

# Verify outcome variables exist
missing_outcomes = [var for var in outcome_vars if var not in filtered_df.columns]
if missing_outcomes:
    print(f"Missing outcome variables: {missing_outcomes}")
    outcome_vars = [var for var in outcome_vars if var in filtered_df.columns]
    print(f"Using available outcome variables: {outcome_vars}")

# STEP 4: Prepare control variables
# Total artifact count
filtered_df['total_artifacts'] = (
    filtered_df['named_datasets_created'].fillna(0) + 
    filtered_df['unnamed_datasets_created'].fillna(0) +
    filtered_df['named_software_created'].fillna(0) + 
    filtered_df['unnamed_software_created'].fillna(0)
)

control_vars = [
    'citationcount',  # Total citations (absorbs general visibility/quality)
    'fwci',  # Field-weighted citation impact
    'total_artifacts',  # Artifact count
    'authorcount',  # Number of authors
    'year'  # Publication year
]

# Verify control variables exist
missing_controls = [var for var in control_vars if var not in filtered_df.columns]
if missing_controls:
    print(f"Missing control variables: {missing_controls}")
    control_vars = [var for var in control_vars if var in filtered_df.columns]
    print(f"Using available control variables: {control_vars}")

# STEP 5: Clean data for regression
regression_vars = ['has_reuse_artifact_citance'] + outcome_vars + control_vars

# First, ensure all variables are numeric
print("Converting variables to numeric types...")
for var in regression_vars:
    if var in filtered_df.columns:
        # Convert to numeric, coercing errors to NaN
        filtered_df[var] = pd.to_numeric(filtered_df[var], errors='coerce')

regression_df = filtered_df.dropna(subset=regression_vars).copy()

# Additional cleaning: remove infinite values
print("Removing infinite values...")
for var in regression_vars:
    if var in regression_df.columns:
        regression_df = regression_df[np.isfinite(regression_df[var])]

print(f"Final regression sample size: {len(regression_df)}")

if len(regression_df) == 0:
    print("ERROR: No valid observations for regression analysis!")
    print("Checking data quality...")
    for var in regression_vars:
        print(f"{var}: {filtered_df[var].isnull().sum()} missing, {np.isinf(filtered_df[var]).sum()} infinite")
    exit(1)

# STEP 6: Run regression models
print("Running regression models...")

regression_results = {}

for outcome in outcome_vars:
    print(f"\nRegressing {outcome}...")
    
    if outcome not in regression_df.columns:
        print(f"  Skipping {outcome} - not found in data")
        continue
    
    # Prepare data
    y = regression_df[outcome].astype(float)
    X = regression_df[['has_reuse_artifact_citance'] + control_vars].astype(float)
    X = sm.add_constant(X)  # Add intercept
    
    # Handle any remaining missing values
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    if len(X_clean) > 0:
        try:
            # Ensure all data is float64
            X_clean = X_clean.astype(np.float64)
            y_clean = y_clean.astype(np.float64)
            
            # Fit OLS regression
            model = sm.OLS(y_clean, X_clean)
            results = model.fit()
            regression_results[outcome] = results
            
            print(f"  N = {len(X_clean)}")
            print(f"  R-squared = {results.rsquared:.4f}")
            print(f"  Reuse coefficient = {results.params['has_reuse_artifact_citance']:.4f}")
            print(f"  p-value = {results.pvalues['has_reuse_artifact_citance']:.4f}")
            
        except Exception as e:
            print(f"  Error fitting model: {e}")
            regression_results[outcome] = None
    else:
        print(f"  No valid observations for {outcome}")
        regression_results[outcome] = None

# STEP 7: Save results and data for Excel analysis
print("\nSaving results...")

# Save the regression sample
regression_df.to_excel(os.path.join(results_path, 'regression_sample_covid.xlsx'), index=False)
regression_df.to_parquet(os.path.join(results_path, 'regression_sample_covid.parquet'), index=False)

# Create regression results summary
results_summary = []
for outcome, results in regression_results.items():
    if results is not None:
        coef = results.params['has_reuse_artifact_citance']
        pvalue = results.pvalues['has_reuse_artifact_citance']
        ci_lower = results.conf_int().loc['has_reuse_artifact_citance', 0]
        ci_upper = results.conf_int().loc['has_reuse_artifact_citance', 1]
        
        results_summary.append({
            'outcome': outcome,
            'coefficient': coef,
            'p_value': pvalue,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': results.nobs,
            'r_squared': results.rsquared
        })

results_summary_df = pd.DataFrame(results_summary)
results_summary_df.to_excel(os.path.join(results_path, 'regression_results_summary_covid.xlsx'), index=False)

# Create descriptive statistics by treatment group
desc_stats = []
for outcome in outcome_vars:
    if outcome not in regression_df.columns:
        continue
        
    # Treatment group (has reuse citances)
    treated = regression_df[regression_df['has_reuse_artifact_citance'] == 1][outcome]
    # Control group (no reuse citances)  
    control = regression_df[regression_df['has_reuse_artifact_citance'] == 0][outcome]
    
    desc_stats.append({
        'outcome': outcome,
        'treated_mean': treated.mean(),
        'treated_std': treated.std(),
        'treated_n': len(treated),
        'control_mean': control.mean(),
        'control_std': control.std(), 
        'control_n': len(control),
        'difference': treated.mean() - control.mean()
    })

desc_stats_df = pd.DataFrame(desc_stats)
desc_stats_df.to_excel(os.path.join(results_path, 'descriptive_stats_by_treatment_covid.xlsx'), index=False)

# Save detailed regression output to text files
for outcome, results in regression_results.items():
    if results is not None:
        with open(os.path.join(results_path, f'regression_output_{outcome}_covid.txt'), 'w') as f:
            f.write(str(results.summary()))

# Additional Excel files for manual analysis
print("Creating additional Excel files for manual analysis...")

# 1. Treatment group breakdown with key variables
treatment_breakdown = regression_df.groupby('has_reuse_artifact_citance').agg({
    'clinical_trial_citations': ['count', 'mean', 'std', 'min', 'max'],
    'clinical_guideline_citations': ['count', 'mean', 'std', 'min', 'max'],
    'total_clinical_citations': ['count', 'mean', 'std', 'min', 'max'],
    'clinical_trial_citations_influential': ['count', 'mean', 'std', 'min', 'max'],
    'clinical_trial_citations_non_influential': ['count', 'mean', 'std', 'min', 'max'],
    'clinical_guideline_citations_influential': ['count', 'mean', 'std', 'min', 'max'],
    'clinical_guideline_citations_non_influential': ['count', 'mean', 'std', 'min', 'max'],
    'total_clinical_citations_influential': ['count', 'mean', 'std', 'min', 'max'],
    'total_clinical_citations_non_influential': ['count', 'mean', 'std', 'min', 'max'],
    'patent_citations': ['count', 'mean', 'std', 'min', 'max'],
    'science_industry_collaboration': ['count', 'mean', 'std', 'min', 'max'],
    'citationcount': ['mean', 'std'],
    'fwci': ['mean', 'std'],
    'total_artifacts': ['mean', 'std'],
    'authorcount': ['mean', 'std']
}).round(4)

# Flatten column names
treatment_breakdown.columns = ['_'.join(col).strip() for col in treatment_breakdown.columns]
treatment_breakdown.to_excel(os.path.join(results_path, 'treatment_breakdown_covid.xlsx'))

# 2. Correlation matrix
correlation_vars = ['has_reuse_artifact_citance'] + outcome_vars + control_vars
correlation_matrix = regression_df[correlation_vars].corr().round(4)
correlation_matrix.to_excel(os.path.join(results_path, 'correlation_matrix_covid.xlsx'))

# STEP 8: Create comprehensive presentation-ready tables
print("\nCreating presentation-ready tables...")

# Define treated and control groups for table calculations
treated_group = regression_df[regression_df['has_reuse_artifact_citance'] == 1]
control_group = regression_df[regression_df['has_reuse_artifact_citance'] == 0]

# Calculate effect sizes (Cohen's d) for key outcomes
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

key_outcomes = ['clinical_trial_citations', 'clinical_guideline_citations', 'patent_citations', 'total_clinical_citations']
effect_sizes = []

for outcome in key_outcomes:
    if outcome in regression_df.columns:
        effect_size = cohens_d(treated_group[outcome], control_group[outcome])
        effect_sizes.append((outcome, effect_size))

# STEP 8.5: INTERACTION EFFECTS ANALYSIS
print("\nConducting interaction effects analysis...")

# Create interaction terms
print("Creating interaction terms...")
regression_df['reuse_x_total_artifacts'] = regression_df['has_reuse_artifact_citance'] * regression_df['total_artifacts']
regression_df['reuse_x_citationcount'] = regression_df['has_reuse_artifact_citance'] * regression_df['citationcount']
regression_df['reuse_x_year'] = regression_df['has_reuse_artifact_citance'] * regression_df['year']
regression_df['reuse_x_fwci'] = regression_df['has_reuse_artifact_citance'] * regression_df['fwci']

# Center continuous variables for interaction interpretation
regression_df['citationcount_centered'] = regression_df['citationcount'] - regression_df['citationcount'].mean()
regression_df['total_artifacts_centered'] = regression_df['total_artifacts'] - regression_df['total_artifacts'].mean()
regression_df['year_centered'] = regression_df['year'] - regression_df['year'].mean()
regression_df['fwci_centered'] = regression_df['fwci'] - regression_df['fwci'].mean()

# Create centered interaction terms
regression_df['reuse_x_artifacts_centered'] = regression_df['has_reuse_artifact_citance'] * regression_df['total_artifacts_centered']
regression_df['reuse_x_citations_centered'] = regression_df['has_reuse_artifact_citance'] * regression_df['citationcount_centered']
regression_df['reuse_x_year_centered'] = regression_df['has_reuse_artifact_citance'] * regression_df['year_centered']
regression_df['reuse_x_fwci_centered'] = regression_df['has_reuse_artifact_citance'] * regression_df['fwci_centered']

# Define interaction models
interaction_models = {
    'artifacts': {
        'name': 'Reuse × Total Artifacts',
        'vars': ['has_reuse_artifact_citance', 'total_artifacts_centered', 'reuse_x_artifacts_centered'] + 
                [v for v in control_vars if v != 'total_artifacts'],
        'interpretation': 'Does reuse effect vary by number of artifacts created?'
    },
    'citations': {
        'name': 'Reuse × Citation Count',
        'vars': ['has_reuse_artifact_citance', 'citationcount_centered', 'reuse_x_citations_centered'] + 
                [v for v in control_vars if v != 'citationcount'],
        'interpretation': 'Does reuse effect vary by paper visibility?'
    },
    'year': {
        'name': 'Reuse × Publication Year',
        'vars': ['has_reuse_artifact_citance', 'year_centered', 'reuse_x_year_centered'] + 
                [v for v in control_vars if v != 'year'],
        'interpretation': 'Did reuse effect change over time during pandemic?'
    },
    'fwci': {
        'name': 'Reuse × FWCI',
        'vars': ['has_reuse_artifact_citance', 'fwci_centered', 'reuse_x_fwci_centered'] + 
                [v for v in control_vars if v != 'fwci'],
        'interpretation': 'Does reuse effect vary by paper quality?'
    }
}

# Run interaction models for key outcomes
key_interaction_outcomes = [
    # Main clinical impact measures
    'clinical_trial_citations',
    'clinical_guideline_citations',
    'total_clinical_citations',
    
    # Influential vs non-influential breakdown
    'clinical_trial_citations_influential',
    'clinical_trial_citations_non_influential', 
    'clinical_guideline_citations_influential',
    'clinical_guideline_citations_non_influential',
    'total_clinical_citations_influential',
    'total_clinical_citations_non_influential',
    
    # Other outcomes
    'patent_citations',
    'science_industry_collaboration'
]

interaction_results = {}

for outcome in key_interaction_outcomes:
    if outcome not in regression_df.columns:
        continue
    
    print(f"\nRunning interaction models for {outcome}...")
    interaction_results[outcome] = {}
    
    for model_name, model_spec in interaction_models.items():
        print(f"  - {model_spec['name']}")
        
        # Prepare data
        y = regression_df[outcome].astype(float)
        X_vars = [v for v in model_spec['vars'] if v in regression_df.columns]
        X = regression_df[X_vars].astype(float)
        X = sm.add_constant(X)
        
        # Handle missing values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) > 10:  # Minimum sample size
            try:
                model = sm.OLS(y_clean, X_clean)
                results = model.fit()
                interaction_results[outcome][model_name] = {
                    'results': results,
                    'interpretation': model_spec['interpretation'],
                    'interaction_term': [v for v in X_vars if 'reuse_x_' in v][0]
                }
                
                # Get interaction coefficient
                interaction_var = [v for v in X_vars if 'reuse_x_' in v][0]
                if interaction_var in results.params.index:
                    coef = results.params[interaction_var]
                    pval = results.pvalues[interaction_var]
                    print(f"    Interaction coefficient: {coef:.4f} (p={pval:.4f})")
                
            except Exception as e:
                print(f"    Error: {e}")
                interaction_results[outcome][model_name] = None

# Create interaction effects summary table
print("\nCreating interaction effects summary table...")
interaction_summary = []

for outcome, models in interaction_results.items():
    for model_name, result in models.items():
        if result is not None and result['results'] is not None:
            res = result['results']
            interaction_term = result['interaction_term']
            
            if interaction_term in res.params.index:
                interaction_summary.append({
                    'outcome': outcome,
                    'interaction_type': model_name,
                    'interaction_name': interaction_models[model_name]['name'],
                    'interpretation': result['interpretation'],
                    'main_effect_coef': res.params.get('has_reuse_artifact_citance', np.nan),
                    'main_effect_pval': res.pvalues.get('has_reuse_artifact_citance', np.nan),
                    'interaction_coef': res.params[interaction_term],
                    'interaction_pval': res.pvalues[interaction_term],
                    'interaction_significant': res.pvalues[interaction_term] < 0.05,
                    'r_squared': res.rsquared,
                    'n_obs': res.nobs
                })

interaction_summary_df = pd.DataFrame(interaction_summary)
if not interaction_summary_df.empty:
    interaction_summary_df.to_excel(os.path.join(tables_path, '16_interaction_effects_summary.xlsx'), index=False)

# MARGINAL EFFECTS ANALYSIS
print("\nCalculating marginal effects for significant interactions...")

marginal_effects_data = []

for outcome, models in interaction_results.items():
    for model_name, result in models.items():
        if (result is not None and result['results'] is not None and 
            result['interaction_term'] in result['results'].params.index):
            
            res = result['results']
            interaction_term = result['interaction_term']
            
            # Only proceed if interaction is significant
            if res.pvalues[interaction_term] < 0.10:  # Use 10% threshold for exploration
                
                # Calculate marginal effects at different values of the moderator
                if 'artifacts' in model_name:
                    moderator_values = [0, 1, 2, 3, 5]  # Different artifact counts
                    moderator_name = 'total_artifacts'
                elif 'citations' in model_name:
                    moderator_values = regression_df['citationcount'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
                    moderator_name = 'citationcount'
                elif 'year' in model_name:
                    moderator_values = sorted(regression_df['year'].unique())
                    moderator_name = 'year'
                elif 'fwci' in model_name:
                    moderator_values = regression_df['fwci'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
                    moderator_name = 'fwci'
                
                main_effect = res.params.get('has_reuse_artifact_citance', 0)
                interaction_effect = res.params[interaction_term]
                
                for mod_val in moderator_values:
                    # For centered variables, use the deviation from mean
                    if 'centered' in interaction_term:
                        mean_val = regression_df[moderator_name].mean()
                        centered_val = mod_val - mean_val
                        marginal_effect = main_effect + interaction_effect * centered_val
                    else:
                        marginal_effect = main_effect + interaction_effect * mod_val
                    
                    marginal_effects_data.append({
                        'outcome': outcome,
                        'interaction_type': model_name,
                        'moderator': moderator_name,
                        'moderator_value': mod_val,
                        'marginal_effect': marginal_effect,
                        'main_effect': main_effect,
                        'interaction_coef': interaction_effect
                    })

marginal_effects_df = pd.DataFrame(marginal_effects_data)
if not marginal_effects_df.empty:
    marginal_effects_df.to_excel(os.path.join(tables_path, '17_marginal_effects_analysis.xlsx'), index=False)

# STRATIFIED ANALYSIS FOR INTERACTION INTERPRETATION
print("\nConducting stratified analysis for interaction interpretation...")

stratified_analysis = []

# Artifacts stratification
artifact_strata = [
    ('Low artifacts (1)', regression_df['total_artifacts'] == 1),
    ('Medium artifacts (2-3)', (regression_df['total_artifacts'] >= 2) & (regression_df['total_artifacts'] <= 3)),
    ('High artifacts (4+)', regression_df['total_artifacts'] >= 4)
]

for stratum_name, stratum_mask in artifact_strata:
    stratum_data = regression_df[stratum_mask]
    if len(stratum_data) > 20:  # Minimum sample size
        treated = stratum_data[stratum_data['has_reuse_artifact_citance'] == 1]
        control = stratum_data[stratum_data['has_reuse_artifact_citance'] == 0]
        
        for outcome in key_interaction_outcomes:
            if outcome in stratum_data.columns:
                stratified_analysis.append({
                    'stratum_type': 'artifacts',
                    'stratum_name': stratum_name,
                    'outcome': outcome,
                    'treated_mean': treated[outcome].mean() if len(treated) > 0 else np.nan,
                    'control_mean': control[outcome].mean() if len(control) > 0 else np.nan,
                    'difference': (treated[outcome].mean() - control[outcome].mean()) if len(treated) > 0 and len(control) > 0 else np.nan,
                    'treated_n': len(treated),
                    'control_n': len(control),
                    'total_n': len(stratum_data)
                })

# Citation impact stratification
citation_strata = [
    ('Low citations (≤10)', regression_df['citationcount'] <= 10),
    ('Medium citations (11-50)', (regression_df['citationcount'] > 10) & (regression_df['citationcount'] <= 50)),
    ('High citations (>50)', regression_df['citationcount'] > 50)
]

for stratum_name, stratum_mask in citation_strata:
    stratum_data = regression_df[stratum_mask]
    if len(stratum_data) > 20:
        treated = stratum_data[stratum_data['has_reuse_artifact_citance'] == 1]
        control = stratum_data[stratum_data['has_reuse_artifact_citance'] == 0]
        
        for outcome in key_interaction_outcomes:
            if outcome in stratum_data.columns:
                stratified_analysis.append({
                    'stratum_type': 'citations',
                    'stratum_name': stratum_name,
                    'outcome': outcome,
                    'treated_mean': treated[outcome].mean() if len(treated) > 0 else np.nan,
                    'control_mean': control[outcome].mean() if len(control) > 0 else np.nan,
                    'difference': (treated[outcome].mean() - control[outcome].mean()) if len(treated) > 0 and len(control) > 0 else np.nan,
                    'treated_n': len(treated),
                    'control_n': len(control),
                    'total_n': len(stratum_data)
                })

# Year stratification
year_strata = [
    ('Early pandemic (2020)', regression_df['year'] == 2020),
    ('Mid pandemic (2021)', regression_df['year'] == 2021),
    ('Later pandemic (2022+)', regression_df['year'] >= 2022)
]

for stratum_name, stratum_mask in year_strata:
    stratum_data = regression_df[stratum_mask]
    if len(stratum_data) > 20:
        treated = stratum_data[stratum_data['has_reuse_artifact_citance'] == 1]
        control = stratum_data[stratum_data['has_reuse_artifact_citance'] == 0]
        
        for outcome in key_interaction_outcomes:
            if outcome in stratum_data.columns:
                stratified_analysis.append({
                    'stratum_type': 'year',
                    'stratum_name': stratum_name,
                    'outcome': outcome,
                    'treated_mean': treated[outcome].mean() if len(treated) > 0 else np.nan,
                    'control_mean': control[outcome].mean() if len(control) > 0 else np.nan,
                    'difference': (treated[outcome].mean() - control[outcome].mean()) if len(treated) > 0 and len(control) > 0 else np.nan,
                    'treated_n': len(treated),
                    'control_n': len(control),
                    'total_n': len(stratum_data)
                })

stratified_df = pd.DataFrame(stratified_analysis)
if not stratified_df.empty:
    stratified_df.to_excel(os.path.join(tables_path, '18_stratified_interaction_analysis.xlsx'), index=False)

# Save detailed interaction model outputs
for outcome, models in interaction_results.items():
    for model_name, result in models.items():
        if result is not None and result['results'] is not None:
            filename = f'interaction_model_{outcome}_{model_name}_covid.txt'
            with open(os.path.join(results_path, filename), 'w') as f:
                f.write(f"INTERACTION MODEL: {result['interpretation']}\n")
                f.write("="*60 + "\n\n")
                f.write(str(result['results'].summary()))

# Create interaction effects visualizations
print("\nCreating interaction effects visualizations...")

# Create interactions subfolder
interactions_viz_path = os.path.join(viz_path, 'interactions')
os.makedirs(interactions_viz_path, exist_ok=True)

# VISUALIZATION 1: Interaction effects coefficients
if not interaction_summary_df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, outcome in enumerate(key_interaction_outcomes[:4]):
        if i >= 4:
            break
        row, col = i // 2, i % 2
        
        outcome_data = interaction_summary_df[interaction_summary_df['outcome'] == outcome]
        
        if not outcome_data.empty:
            # Plot interaction coefficients
            x_pos = range(len(outcome_data))
            colors = [pathos_orange if sig else 'gray' for sig in outcome_data['interaction_significant']]
            
            bars = axes[row, col].bar(x_pos, outcome_data['interaction_coef'], 
                                     color=colors, alpha=0.8)
            
            # Add significance indicators
            for j, (idx, row_data) in enumerate(outcome_data.iterrows()):
                if row_data['interaction_significant']:
                    axes[row, col].text(j, row_data['interaction_coef'], '*', 
                                       ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            axes[row, col].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[row, col].set_title(f'{outcome.replace("_", " ").title()}\nInteraction Effects')
            axes[row, col].set_xticks(x_pos)
            axes[row, col].set_xticklabels([name.replace('_', '\n') for name in outcome_data['interaction_type']], 
                                          rotation=0, ha='center')
            axes[row, col].set_ylabel('Interaction Coefficient')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(interactions_viz_path, '01_interaction_coefficients.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# VISUALIZATION 2: Marginal effects plots
if not marginal_effects_df.empty:
    unique_interactions = marginal_effects_df[['outcome', 'interaction_type']].drop_duplicates()
    
    n_plots = len(unique_interactions)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (_, interaction) in enumerate(unique_interactions.iterrows()):
        if i >= len(axes):
            break
            
        data = marginal_effects_df[
            (marginal_effects_df['outcome'] == interaction['outcome']) & 
            (marginal_effects_df['interaction_type'] == interaction['interaction_type'])
        ]
        
        if not data.empty:
            axes[i].plot(data['moderator_value'], data['marginal_effect'], 
                        marker='o', linewidth=3, markersize=8, color=pathos_orange)
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[i].set_xlabel(data['moderator'].iloc[0].replace('_', ' ').title())
            axes[i].set_ylabel('Marginal Effect of Reuse Citations')
            axes[i].set_title(f'{interaction["outcome"].replace("_", " ").title()}\nvs {interaction["interaction_type"].replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(unique_interactions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(interactions_viz_path, '02_marginal_effects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# VISUALIZATION 3: Stratified analysis heatmap
if not stratified_df.empty:
    # Create pivot table for heatmap
    for stratum_type in ['artifacts', 'citations', 'year']:
        stratum_data = stratified_df[stratified_df['stratum_type'] == stratum_type]
        
        if not stratum_data.empty:
            pivot_data = stratum_data.pivot_table(
                values='difference', 
                index='stratum_name', 
                columns='outcome', 
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.2f', cbar_kws={'label': 'Treatment Effect'}, ax=ax)
            ax.set_title(f'Treatment Effects by {stratum_type.title()} Strata\n(Difference: Treated - Control)')
            ax.set_xlabel('Outcome Variables')
            ax.set_ylabel(f'{stratum_type.title()} Strata')
            
            plt.tight_layout()
            plt.savefig(os.path.join(interactions_viz_path, f'03_stratified_heatmap_{stratum_type}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

# VISUALIZATION 4: Interaction significance summary
if not interaction_summary_df.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Count significant interactions by type
    sig_counts = interaction_summary_df.groupby('interaction_type')['interaction_significant'].sum()
    total_counts = interaction_summary_df.groupby('interaction_type').size()
    
    sig_rates = (sig_counts / total_counts * 100).fillna(0)
    
    ax1.bar(range(len(sig_rates)), sig_rates.values, color=pathos_orange, alpha=0.8)
    ax1.set_title('Significant Interaction Effects by Type\n(% of outcomes with p < 0.05)')
    ax1.set_xticks(range(len(sig_rates)))
    ax1.set_xticklabels([name.replace('_', '\n') for name in sig_rates.index])
    ax1.set_ylabel('Percentage Significant (%)')
    ax1.grid(True, alpha=0.3)
    
    # Distribution of interaction effect sizes
    ax2.hist(interaction_summary_df['interaction_coef'], bins=20, alpha=0.7, 
             color=pathos_blue, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
    ax2.set_title('Distribution of Interaction Effect Sizes')
    ax2.set_xlabel('Interaction Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(interactions_viz_path, '04_interaction_significance_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# Create comprehensive interaction interpretation table
interaction_interpretation = []

for outcome in key_interaction_outcomes:
    if outcome in interaction_results:
        for model_name, result in interaction_results[outcome].items():
            if result is not None and result['results'] is not None:
                res = result['results']
                interaction_term = result['interaction_term']
                
                if interaction_term in res.params.index:
                    # Determine interpretation
                    main_effect = res.params.get('has_reuse_artifact_citance', 0)
                    interaction_coef = res.params[interaction_term]
                    interaction_pval = res.pvalues[interaction_term]
                    
                    if interaction_pval < 0.05:
                        if interaction_coef > 0:
                            interpretation = "Positive interaction: reuse effect increases with moderator"
                            practical_meaning = f"Reuse citations are MORE beneficial for papers with higher {model_name}"
                        else:
                            interpretation = "Negative interaction: reuse effect decreases with moderator"
                            practical_meaning = f"Reuse citations are LESS beneficial for papers with higher {model_name}"
                    else:
                        interpretation = "No significant interaction"
                        practical_meaning = f"Reuse effect is consistent across levels of {model_name}"
                    
                    interaction_interpretation.append({
                        'Outcome': outcome.replace('_', ' ').title(),
                        'Moderator': interaction_models[model_name]['name'],
                        'Main_Effect': round(main_effect, 4),
                        'Interaction_Coefficient': round(interaction_coef, 4),
                        'Interaction_P_Value': round(interaction_pval, 4),
                        'Statistical_Interpretation': interpretation,
                        'Practical_Meaning': practical_meaning,
                        'Significant': interaction_pval < 0.05
                    })

interaction_interpretation_df = pd.DataFrame(interaction_interpretation)
if not interaction_interpretation_df.empty:
    interaction_interpretation_df.to_excel(os.path.join(tables_path, '19_interaction_interpretation.xlsx'), index=False)

print(f"\nInteraction effects analysis completed!")
print(f"Files saved to:")
print(f"  - Tables: {tables_path}")
print(f"  - Visualizations: {interactions_viz_path}")
print(f"  - Model outputs: {results_path}")

# TABLE 1: Executive Summary Table
executive_summary = {
    'Metric': [
        'Total Sample Size',
        'Papers with Reuse Citations',
        'Papers without Reuse Citations',
        'Reuse Citation Rate (%)',
        'Mean Clinical Impact (Treatment)',
        'Mean Clinical Impact (Control)',
        'Clinical Impact Increase (%)',
        'Mean Patent Citations (Treatment)',
        'Mean Patent Citations (Control)',
        'Patent Impact Increase (%)',
        'Science-Industry Collaboration Rate (Treatment)',
        'Science-Industry Collaboration Rate (Control)',
        'Collaboration Rate Difference (pp)'
    ],
    'Value': [
        len(regression_df),
        regression_df['has_reuse_artifact_citance'].sum(),
        (regression_df['has_reuse_artifact_citance'] == 0).sum(),
        round(regression_df['has_reuse_artifact_citance'].mean() * 100, 1),
        round(treated_group['total_clinical_citations'].mean(), 2),
        round(control_group['total_clinical_citations'].mean(), 2),
        round(((treated_group['total_clinical_citations'].mean() / control_group['total_clinical_citations'].mean() - 1) * 100) if control_group['total_clinical_citations'].mean() > 0 else 0, 1),
        round(treated_group['patent_citations'].mean(), 2),
        round(control_group['patent_citations'].mean(), 2),
        round(((treated_group['patent_citations'].mean() / control_group['patent_citations'].mean() - 1) * 100) if control_group['patent_citations'].mean() > 0 else 0, 1),
        round(treated_group['science_industry_collaboration'].mean() * 100, 1),
        round(control_group['science_industry_collaboration'].mean() * 100, 1),
        round((treated_group['science_industry_collaboration'].mean() - control_group['science_industry_collaboration'].mean()) * 100, 1)
    ]
}

executive_summary_df = pd.DataFrame(executive_summary)
executive_summary_df.to_excel(os.path.join(tables_path, '01_executive_summary.xlsx'), index=False)

# TABLE 2: Impact by Artifact Type
artifact_impact_data = []
artifact_types = ['named_datasets_created', 'unnamed_datasets_created', 'named_software_created', 'unnamed_software_created']

for artifact_type in artifact_types:
    # Papers that created this type of artifact
    creators = regression_df[regression_df[artifact_type] > 0]
    
    if len(creators) > 0:
        # Split by reuse citation status
        creators_with_reuse = creators[creators['has_reuse_artifact_citance'] == 1]
        creators_without_reuse = creators[creators['has_reuse_artifact_citance'] == 0]
        
        artifact_impact_data.append({
            'Artifact_Type': artifact_type.replace('_', ' ').title(),
            'Total_Creators': len(creators),
            'Creators_With_Reuse': len(creators_with_reuse),
            'Creators_Without_Reuse': len(creators_without_reuse),
            'Reuse_Rate_Percent': round(len(creators_with_reuse) / len(creators) * 100, 1),
            'Clinical_Impact_With_Reuse': round(creators_with_reuse['total_clinical_citations'].mean() if len(creators_with_reuse) > 0 else 0, 2),
            'Clinical_Impact_Without_Reuse': round(creators_without_reuse['total_clinical_citations'].mean() if len(creators_without_reuse) > 0 else 0, 2),
            'Patent_Impact_With_Reuse': round(creators_with_reuse['patent_citations'].mean() if len(creators_with_reuse) > 0 else 0, 2),
            'Patent_Impact_Without_Reuse': round(creators_without_reuse['patent_citations'].mean() if len(creators_without_reuse) > 0 else 0, 2)
        })

artifact_impact_df = pd.DataFrame(artifact_impact_data)
artifact_impact_df.to_excel(os.path.join(tables_path, '02_impact_by_artifact_type.xlsx'), index=False)

# TABLE 3: Statistical Significance Summary
if results_summary:
    significance_summary = []
    for result in results_summary:
        significance_summary.append({
            'Outcome': result['outcome'].replace('_', ' ').title(),
            'Coefficient': round(result['coefficient'], 4),
            'P_Value': round(result['p_value'], 4),
            'Significant_005': 'Yes' if result['p_value'] < 0.05 else 'No',
            'Significant_001': 'Yes' if result['p_value'] < 0.01 else 'No',
            'CI_Lower': round(result['ci_lower'], 4),
            'CI_Upper': round(result['ci_upper'], 4),
            'R_Squared': round(result['r_squared'], 4),
            'Sample_Size': int(result['n_obs']),
            'Effect_Direction': 'Positive' if result['coefficient'] > 0 else 'Negative',
            'Effect_Magnitude': 'Large' if abs(result['coefficient']) > 1 else 'Medium' if abs(result['coefficient']) > 0.5 else 'Small'
        })
    
    significance_df = pd.DataFrame(significance_summary)
    significance_df.to_excel(os.path.join(tables_path, '03_statistical_significance_summary.xlsx'), index=False)

# TABLE 4: Clinical Impact Breakdown
clinical_breakdown = []
clinical_outcomes = ['clinical_trial_citations', 'clinical_guideline_citations', 
                    'clinical_trial_citations_influential', 'clinical_trial_citations_non_influential',
                    'clinical_guideline_citations_influential', 'clinical_guideline_citations_non_influential']

for outcome in clinical_outcomes:
    if outcome in regression_df.columns:
        treated_mean = treated_group[outcome].mean()
        control_mean = control_group[outcome].mean()
        
        # Calculate success rates (having any citations of this type)
        treated_success = (treated_group[outcome] > 0).mean() * 100
        control_success = (control_group[outcome] > 0).mean() * 100
        
        clinical_breakdown.append({
            'Clinical_Outcome': outcome.replace('_', ' ').title(),
            'Treatment_Mean': round(treated_mean, 2),
            'Control_Mean': round(control_mean, 2),
            'Difference': round(treated_mean - control_mean, 2),
            'Percent_Increase': round(((treated_mean / control_mean - 1) * 100) if control_mean > 0 else 0, 1),
            'Treatment_Success_Rate': round(treated_success, 1),
            'Control_Success_Rate': round(control_success, 1),
            'Success_Rate_Difference': round(treated_success - control_success, 1)
        })

clinical_breakdown_df = pd.DataFrame(clinical_breakdown)
clinical_breakdown_df.to_excel(os.path.join(tables_path, '04_clinical_impact_breakdown.xlsx'), index=False)

# TABLE 5: Dose-Response Analysis
dose_response_data = []
reuse_counts = regression_df['reuse_artifact_inbound'].values
max_reuse = min(max(reuse_counts), 10)  # Cap at 10 for meaningful bins

for dose in range(int(max_reuse) + 1):
    dose_group = regression_df[regression_df['reuse_artifact_inbound'] == dose]
    
    if len(dose_group) > 0:
        dose_response_data.append({
            'Reuse_Citations': dose,
            'Sample_Size': len(dose_group),
            'Percent_of_Sample': round(len(dose_group) / len(regression_df) * 100, 1),
            'Mean_Clinical_Citations': round(dose_group['total_clinical_citations'].mean(), 2),
            'Mean_Patent_Citations': round(dose_group['patent_citations'].mean(), 2),
            'Mean_Total_Citations': round(dose_group['citationcount'].mean(), 1),
            'Mean_FWCI': round(dose_group['fwci'].mean(), 2),
            'Industry_Collaboration_Rate': round(dose_group['science_industry_collaboration'].mean() * 100, 1)
        })

dose_response_df = pd.DataFrame(dose_response_data)
dose_response_df.to_excel(os.path.join(tables_path, '05_dose_response_analysis.xlsx'), index=False)

# TABLE 6: Top Performers Analysis
top_performers = []

# Top clinical impact papers
top_clinical = regression_df.nlargest(10, 'total_clinical_citations')[
    ['id', 'year', 'total_clinical_citations', 'patent_citations', 'has_reuse_artifact_citance', 
     'reuse_artifact_inbound', 'total_artifacts', 'citationcount', 'fwci']
].copy()
top_clinical['Category'] = 'Top Clinical Impact'

# Top patent impact papers  
top_patent = regression_df.nlargest(10, 'patent_citations')[
    ['id', 'year', 'total_clinical_citations', 'patent_citations', 'has_reuse_artifact_citance',
     'reuse_artifact_inbound', 'total_artifacts', 'citationcount', 'fwci']
].copy()
top_patent['Category'] = 'Top Patent Impact'

# Top reuse citation papers
top_reuse = regression_df.nlargest(10, 'reuse_artifact_inbound')[
    ['id', 'year', 'total_clinical_citations', 'patent_citations', 'has_reuse_artifact_citance',
     'reuse_artifact_inbound', 'total_artifacts', 'citationcount', 'fwci']
].copy()
top_reuse['Category'] = 'Top Reuse Citations'

top_performers_df = pd.concat([top_clinical, top_patent, top_reuse], ignore_index=True)
top_performers_df.to_excel(os.path.join(tables_path, '06_top_performers_analysis.xlsx'), index=False)

# TABLE 7: Cross-tabulation Analysis
# Year vs Treatment Status
year_treatment_crosstab = pd.crosstab(
    regression_df['year'], 
    regression_df['has_reuse_artifact_citance'],
    margins=True,
    normalize='index'
) * 100

# Check actual column count and rename accordingly
if year_treatment_crosstab.shape[1] == 3:
    year_treatment_crosstab.columns = ['No_Reuse_Percent', 'Has_Reuse_Percent', 'Total']
elif year_treatment_crosstab.shape[1] == 2:
    year_treatment_crosstab.columns = ['No_Reuse_Percent', 'Has_Reuse_Percent']

year_treatment_crosstab.to_excel(os.path.join(tables_path, '07_year_treatment_crosstab.xlsx'))

# Industry Collaboration vs Treatment Status
if 'science_industry_collaboration' in regression_df.columns:
    collab_treatment_crosstab = pd.crosstab(
        regression_df['science_industry_collaboration'],
        regression_df['has_reuse_artifact_citance'],
        margins=True,
        normalize='columns'
    ) * 100
    
    # Check actual dimensions and rename accordingly
    if collab_treatment_crosstab.shape[0] == 3:
        collab_treatment_crosstab.index = ['No_Industry_Collaboration', 'Has_Industry_Collaboration', 'Total']
    elif collab_treatment_crosstab.shape[0] == 2:
        collab_treatment_crosstab.index = ['No_Industry_Collaboration', 'Has_Industry_Collaboration']
    
    if collab_treatment_crosstab.shape[1] == 3:
        collab_treatment_crosstab.columns = ['No_Reuse_Citations', 'Has_Reuse_Citations', 'Total']
    elif collab_treatment_crosstab.shape[1] == 2:
        collab_treatment_crosstab.columns = ['No_Reuse_Citations', 'Has_Reuse_Citations']
    
    collab_treatment_crosstab.to_excel(os.path.join(tables_path, '08_collaboration_treatment_crosstab.xlsx'))

# TABLE 8: Quartile Analysis
quartile_analysis = []
for outcome in ['total_clinical_citations', 'patent_citations', 'citationcount', 'fwci']:
    if outcome in regression_df.columns:
        # Calculate quartiles for the entire sample
        quartiles = regression_df[outcome].quantile([0.25, 0.5, 0.75]).values
        
        for i, (q_name, q_val) in enumerate(zip(['Q1', 'Q2', 'Q3', 'Q4'], [0] + list(quartiles) + [float('inf')])):
            if i < 3:
                mask = (regression_df[outcome] >= q_val) & (regression_df[outcome] < quartiles[i])
            else:
                mask = regression_df[outcome] >= quartiles[2]
            
            quartile_group = regression_df[mask]
            
            if len(quartile_group) > 0:
                quartile_analysis.append({
                    'Outcome': outcome.replace('_', ' ').title(),
                    'Quartile': q_name,
                    'Sample_Size': len(quartile_group),
                    'Reuse_Citation_Rate': round(quartile_group['has_reuse_artifact_citance'].mean() * 100, 1),
                    'Mean_Reuse_Citations': round(quartile_group['reuse_artifact_inbound'].mean(), 2),
                    'Mean_Artifacts_Created': round(quartile_group['total_artifacts'].mean(), 2),
                    'Industry_Collaboration_Rate': round(quartile_group['science_industry_collaboration'].mean() * 100, 1)
                })

quartile_df = pd.DataFrame(quartile_analysis)
quartile_df.to_excel(os.path.join(tables_path, '09_quartile_analysis.xlsx'), index=False)

# TABLE 9: Effect Size Summary
if effect_sizes:
    effect_size_summary = []
    for outcome, effect_size in effect_sizes:
        # Add interpretation
        if abs(effect_size) < 0.2:
            interpretation = 'Negligible'
        elif abs(effect_size) < 0.5:
            interpretation = 'Small'
        elif abs(effect_size) < 0.8:
            interpretation = 'Medium'
        else:
            interpretation = 'Large'
        
        effect_size_summary.append({
            'Outcome': outcome.replace('_', ' ').title(),
            'Cohens_D': round(effect_size, 3),
            'Effect_Size_Interpretation': interpretation,
            'Treatment_Mean': round(treated_group[outcome].mean(), 2),
            'Control_Mean': round(control_group[outcome].mean(), 2),
            'Absolute_Difference': round(treated_group[outcome].mean() - control_group[outcome].mean(), 2),
            'Relative_Difference_Percent': round(((treated_group[outcome].mean() / control_group[outcome].mean() - 1) * 100) if control_group[outcome].mean() > 0 else 0, 1)
        })
    
    effect_size_df = pd.DataFrame(effect_size_summary)
    effect_size_df.to_excel(os.path.join(tables_path, '10_effect_size_summary.xlsx'), index=False)

# TABLE 10: Success Rate Comparison
success_rate_comparison = []
binary_outcomes = [
    ('Any_Clinical_Citation', regression_df['total_clinical_citations'] > 0),
    ('High_Clinical_Impact', regression_df['total_clinical_citations'] > 2),
    ('Any_Patent_Citation', regression_df['patent_citations'] > 0),
    ('Multiple_Patent_Citations', regression_df['patent_citations'] > 1),
    ('Industry_Collaboration', regression_df['science_industry_collaboration'] == 1),
    ('High_Citation_Impact', regression_df['citationcount'] > regression_df['citationcount'].median()),
    ('High_FWCI', regression_df['fwci'] > 1)
]

for outcome_name, outcome_mask in binary_outcomes:
    treated_success = outcome_mask[regression_df['has_reuse_artifact_citance'] == 1].mean() * 100
    control_success = outcome_mask[regression_df['has_reuse_artifact_citance'] == 0].mean() * 100
    
    # Calculate odds ratio
    treated_yes = outcome_mask[regression_df['has_reuse_artifact_citance'] == 1].sum()
    treated_no = (~outcome_mask[regression_df['has_reuse_artifact_citance'] == 1]).sum()
    control_yes = outcome_mask[regression_df['has_reuse_artifact_citance'] == 0].sum()
    control_no = (~outcome_mask[regression_df['has_reuse_artifact_citance'] == 0]).sum()
    
    if treated_no > 0 and control_no > 0:
        odds_ratio = (treated_yes * control_no) / (treated_no * control_yes)
    else:
        odds_ratio = float('inf')
    
    success_rate_comparison.append({
        'Outcome': outcome_name.replace('_', ' '),
        'Treatment_Success_Rate': round(treated_success, 1),
        'Control_Success_Rate': round(control_success, 1),
        'Success_Rate_Difference': round(treated_success - control_success, 1),
        'Odds_Ratio': round(odds_ratio, 2) if odds_ratio != float('inf') else 'Inf',
        'Treatment_N_Success': treated_yes,
        'Control_N_Success': control_yes
    })

success_rate_df = pd.DataFrame(success_rate_comparison)
success_rate_df.to_excel(os.path.join(tables_path, '11_success_rate_comparison.xlsx'), index=False)

# TABLE 11: Artifact Creation Patterns
artifact_patterns = []
for has_reuse in [0, 1]:
    group = regression_df[regression_df['has_reuse_artifact_citance'] == has_reuse]
    group_name = 'Has Reuse Citations' if has_reuse else 'No Reuse Citations'
    
    artifact_patterns.append({
        'Group': group_name,
        'Sample_Size': len(group),
        'Mean_Named_Datasets': round(group['named_datasets_created'].mean(), 2),
        'Mean_Unnamed_Datasets': round(group['unnamed_datasets_created'].mean(), 2),
        'Mean_Named_Software': round(group['named_software_created'].mean(), 2),
        'Mean_Unnamed_Software': round(group['unnamed_software_created'].mean(), 2),
        'Mean_Total_Artifacts': round(group['total_artifacts'].mean(), 2),
        'Percent_Multiple_Artifacts': round((group['total_artifacts'] > 1).mean() * 100, 1),
        'Percent_Dataset_Creators': round((group['named_datasets_created'] + group['unnamed_datasets_created'] > 0).mean() * 100, 1),
        'Percent_Software_Creators': round((group['named_software_created'] + group['unnamed_software_created'] > 0).mean() * 100, 1)
    })

artifact_patterns_df = pd.DataFrame(artifact_patterns)
artifact_patterns_df.to_excel(os.path.join(tables_path, '12_artifact_creation_patterns.xlsx'), index=False)

# TABLE 12: Publication Year Impact Analysis
year_impact_analysis = []
for year in sorted(regression_df['year'].unique()):
    year_group = regression_df[regression_df['year'] == year]
    if len(year_group) > 0:
        year_treated = year_group[year_group['has_reuse_artifact_citance'] == 1]
        year_control = year_group[year_group['has_reuse_artifact_citance'] == 0]
        
        year_impact_analysis.append({
            'Year': year,
            'Total_Papers': len(year_group),
            'Papers_With_Reuse': len(year_treated),
            'Papers_Without_Reuse': len(year_control),
            'Reuse_Rate_Percent': round(len(year_treated) / len(year_group) * 100, 1),
            'Clinical_Impact_With_Reuse': round(year_treated['total_clinical_citations'].mean() if len(year_treated) > 0 else 0, 2),
            'Clinical_Impact_Without_Reuse': round(year_control['total_clinical_citations'].mean() if len(year_control) > 0 else 0, 2),
            'Patent_Impact_With_Reuse': round(year_treated['patent_citations'].mean() if len(year_treated) > 0 else 0, 2),
            'Patent_Impact_Without_Reuse': round(year_control['patent_citations'].mean() if len(year_control) > 0 else 0, 2)
        })

year_impact_df = pd.DataFrame(year_impact_analysis)
year_impact_df.to_excel(os.path.join(tables_path, '13_publication_year_impact_analysis.xlsx'), index=False)

# TABLE 13: Citation Quality Analysis
citation_quality_data = []
citation_thresholds = [1, 5, 10, 25, 50, 100]

for threshold in citation_thresholds:
    high_cited = regression_df[regression_df['citationcount'] >= threshold]
    
    if len(high_cited) > 0:
        high_cited_treated = high_cited[high_cited['has_reuse_artifact_citance'] == 1]
        high_cited_control = high_cited[high_cited['has_reuse_artifact_citance'] == 0]
        
        citation_quality_data.append({
            'Citation_Threshold': f'≥{threshold}',
            'Total_Papers': len(high_cited),
            'Papers_With_Reuse': len(high_cited_treated),
            'Papers_Without_Reuse': len(high_cited_control),
            'Reuse_Rate_Percent': round(len(high_cited_treated) / len(high_cited) * 100, 1),
            'Mean_Clinical_Citations_Treated': round(high_cited_treated['total_clinical_citations'].mean() if len(high_cited_treated) > 0 else 0, 2),
            'Mean_Clinical_Citations_Control': round(high_cited_control['total_clinical_citations'].mean() if len(high_cited_control) > 0 else 0, 2),
            'Mean_FWCI_Treated': round(high_cited_treated['fwci'].mean() if len(high_cited_treated) > 0 else 0, 2),
            'Mean_FWCI_Control': round(high_cited_control['fwci'].mean() if len(high_cited_control) > 0 else 0, 2)
        })

citation_quality_df = pd.DataFrame(citation_quality_data)
citation_quality_df.to_excel(os.path.join(tables_path, '14_citation_quality_analysis.xlsx'), index=False)

# TABLE 14: Robustness Check - Multiple Artifact Types
artifact_combination_analysis = []
artifact_combinations = [
    ('Only_Datasets', lambda x: (x['named_datasets_created'] + x['unnamed_datasets_created'] > 0) & 
                                (x['named_software_created'] + x['unnamed_software_created'] == 0)),
    ('Only_Software', lambda x: (x['named_software_created'] + x['unnamed_software_created'] > 0) & 
                                (x['named_datasets_created'] + x['unnamed_datasets_created'] == 0)),
    ('Both_Types', lambda x: (x['named_datasets_created'] + x['unnamed_datasets_created'] > 0) & 
                             (x['named_software_created'] + x['unnamed_software_created'] > 0)),
    ('Named_Only', lambda x: (x['named_datasets_created'] + x['named_software_created'] > 0) & 
                             (x['unnamed_datasets_created'] + x['unnamed_software_created'] == 0)),
    ('Unnamed_Only', lambda x: (x['unnamed_datasets_created'] + x['unnamed_software_created'] > 0) & 
                               (x['named_datasets_created'] + x['named_software_created'] == 0))
]

for combo_name, combo_filter in artifact_combinations:
    combo_group = regression_df[combo_filter(regression_df)]
    
    if len(combo_group) > 0:
        combo_treated = combo_group[combo_group['has_reuse_artifact_citance'] == 1]
        combo_control = combo_group[combo_group['has_reuse_artifact_citance'] == 0]
        
        artifact_combination_analysis.append({
            'Artifact_Combination': combo_name.replace('_', ' '),
            'Total_Papers': len(combo_group),
            'Papers_With_Reuse': len(combo_treated),
            'Reuse_Rate_Percent': round(len(combo_treated) / len(combo_group) * 100, 1),
            'Clinical_Impact_Treated': round(combo_treated['total_clinical_citations'].mean() if len(combo_treated) > 0 else 0, 2),
            'Clinical_Impact_Control': round(combo_control['total_clinical_citations'].mean() if len(combo_control) > 0 else 0, 2),
            'Patent_Impact_Treated': round(combo_treated['patent_citations'].mean() if len(combo_treated) > 0 else 0, 2),
            'Patent_Impact_Control': round(combo_control['patent_citations'].mean() if len(combo_control) > 0 else 0, 2)
        })

artifact_combination_df = pd.DataFrame(artifact_combination_analysis)
artifact_combination_df.to_excel(os.path.join(tables_path, '15_artifact_combination_analysis.xlsx'), index=False)

print(f"All presentation tables saved to: {tables_path}")

# STEP 9: Create comprehensive visualizations with analysis conclusions
print("\nCreating visualizations and analysis conclusions...")

# Update colors in existing visualizations
# 1. Sample size and treatment distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Treatment distribution
treatment_counts = regression_df['has_reuse_artifact_citance'].value_counts()
axes[0,0].pie(treatment_counts.values, labels=['No Reuse Citations', 'Has Reuse Citations'], 
              autopct='%1.1f%%', startangle=90, colors=[pathos_blue, pathos_orange])
axes[0,0].set_title('Distribution of Treatment Groups\n(Reuse-Artifact Citations)')

# Sample size by year
year_counts = regression_df['year'].value_counts().sort_index()
axes[0,1].bar(year_counts.index, year_counts.values, alpha=0.8, color=pathos_blue)
axes[0,1].set_title('Sample Distribution by Publication Year')
axes[0,1].set_xlabel('Year')
axes[0,1].set_ylabel('Number of Papers')
axes[0,1].tick_params(axis='x', rotation=45)

# Artifact creation distribution
artifact_data = [
    regression_df['named_datasets_created'].sum(),
    regression_df['unnamed_datasets_created'].sum(),
    regression_df['named_software_created'].sum(),
    regression_df['unnamed_software_created'].sum()
]
artifact_labels = ['Named\nDatasets', 'Unnamed\nDatasets', 'Named\nSoftware', 'Unnamed\nSoftware']
axes[1,0].bar(artifact_labels, artifact_data, alpha=0.7, color=pathos_blue)
axes[1,0].set_title('Total Artifacts Created by Type')
axes[1,0].set_ylabel('Number of Artifacts')
axes[1,0].tick_params(axis='x', rotation=45)

# Citation distribution
axes[1,1].hist(regression_df['citationcount'], bins=50, alpha=0.7, edgecolor='black', color=pathos_blue)
axes[1,1].set_title('Distribution of Citation Counts')
axes[1,1].set_xlabel('Citation Count')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '01_sample_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Outcome distributions by treatment group
n_outcomes = len(outcome_vars)
n_cols = 3
n_rows = (n_outcomes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for i, outcome in enumerate(outcome_vars):
    if outcome in regression_df.columns:
        treated = regression_df[regression_df['has_reuse_artifact_citance'] == 1][outcome]
        control = regression_df[regression_df['has_reuse_artifact_citance'] == 0][outcome]
        
        # Box plot comparison
        data_to_plot = [control.values, treated.values]
        box_plot = axes[i].boxplot(data_to_plot, labels=['No Reuse', 'Has Reuse'], patch_artist=True)
        
        # Color the boxes
        colors = pathos_colors
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[i].set_title(f'{outcome.replace("_", " ").title()}')
        axes[i].set_ylabel('Count')
        
        # Add mean values as text
        axes[i].text(1, control.mean(), f'Mean: {control.mean():.2f}', 
                    ha='center', va='bottom', fontweight='bold')
        axes[i].text(2, treated.mean(), f'Mean: {treated.mean():.2f}', 
                    ha='center', va='bottom', fontweight='bold')

# Remove empty subplots
for i in range(n_outcomes, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '02_outcome_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Regression coefficients visualization
if results_summary:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Coefficient plot with confidence intervals
    outcomes = [r['outcome'] for r in results_summary]
    coefficients = [r['coefficient'] for r in results_summary]
    ci_lower = [r['ci_lower'] for r in results_summary]
    ci_upper = [r['ci_upper'] for r in results_summary]
    p_values = [r['p_value'] for r in results_summary]
    
    # Color by significance
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    y_pos = range(len(outcomes))
    ax1.errorbar(coefficients, y_pos, 
                xerr=[(c - l) for c, l in zip(coefficients, ci_lower)], 
                fmt='o', color='black', capsize=5)
    
    for i, (coef, color) in enumerate(zip(coefficients, colors)):
        ax1.scatter(coef, i, color=color, s=100, zorder=3)
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([o.replace('_', ' ').title() for o in outcomes])
    ax1.set_xlabel('Coefficient Estimate')
    ax1.set_title('Regression Coefficients\n(Red = p < 0.05)')
    ax1.grid(True, alpha=0.3)
    
    # P-value plot
    ax2.barh(y_pos, [-np.log10(p) for p in p_values], color=colors)
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([o.replace('_', ' ').title() for o in outcomes])
    ax2.set_xlabel('-log10(p-value)')
    ax2.set_title('Statistical Significance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '03_regression_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix_data = regression_df[correlation_vars].corr()
mask = np.triu(np.ones_like(correlation_matrix_data, dtype=bool))
sns.heatmap(correlation_matrix_data, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Correlation Matrix of Variables')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '04_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Treatment vs Control means comparison
if desc_stats:
    desc_df = pd.DataFrame(desc_stats)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(desc_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], desc_df['control_mean'], width, 
           label='No Reuse Citations', alpha=0.8, color=pathos_blue)
    ax.bar([i + width/2 for i in x], desc_df['treated_mean'], width,
           label='Has Reuse Citations', alpha=0.8, color=pathos_orange)
    
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Mean Value')
    ax.set_title('Treatment vs Control Group Means')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', ' ').title() for o in desc_df['outcome']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '05_treatment_control_means.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 6. Distribution of key control variables
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

control_vars_plot = ['citationcount', 'fwci', 'total_artifacts', 'authorcount', 'year']
for i, var in enumerate(control_vars_plot):
    if var in regression_df.columns and i < len(axes):
        treated = regression_df[regression_df['has_reuse_artifact_citance'] == 1][var]
        control = regression_df[regression_df['has_reuse_artifact_citance'] == 0][var]
        
        axes[i].hist(control.values, alpha=0.7, label='No Reuse', bins=30, density=True, color=pathos_blue)
        axes[i].hist(treated.values, alpha=0.7, label='Has Reuse', bins=30, density=True, color=pathos_orange)
        axes[i].set_title(f'{var.replace("_", " ").title()}')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# Remove empty subplot
if len(control_vars_plot) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '06_control_variables_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Clinical impact breakdown
clinical_outcomes = [var for var in outcome_vars if 'clinical' in var]
if clinical_outcomes:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Clinical trials vs guidelines
    if 'clinical_trial_citations' in regression_df.columns and 'clinical_guideline_citations' in regression_df.columns:
        treated_trials = regression_df[regression_df['has_reuse_artifact_citance'] == 1]['clinical_trial_citations']
        treated_guidelines = regression_df[regression_df['has_reuse_artifact_citance'] == 1]['clinical_guideline_citations']
        control_trials = regression_df[regression_df['has_reuse_artifact_citance'] == 0]['clinical_trial_citations']
        control_guidelines = regression_df[regression_df['has_reuse_artifact_citance'] == 0]['clinical_guideline_citations']
        
        categories = ['Clinical Trials', 'Clinical Guidelines']
        treated_means = [treated_trials.mean(), treated_guidelines.mean()]
        control_means = [control_trials.mean(), control_guidelines.mean()]
        
        x = range(len(categories))
        width = 0.35
        
        axes[0,0].bar([i - width/2 for i in x], control_means, width, label='No Reuse', alpha=0.8, color=pathos_blue)
        axes[0,0].bar([i + width/2 for i in x], treated_means, width, label='Has Reuse', alpha=0.8, color=pathos_orange)
        axes[0,0].set_title('Clinical Impact by Type')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(categories)
        axes[0,0].legend()
        axes[0,0].set_ylabel('Mean Citations')
    
    # Influential vs Non-influential breakdown
    if 'total_clinical_citations_influential' in regression_df.columns:
        treated_inf = regression_df[regression_df['has_reuse_artifact_citance'] == 1]['total_clinical_citations_influential']
        treated_non_inf = regression_df[regression_df['has_reuse_artifact_citance'] == 1]['total_clinical_citations_non_influential']
        control_inf = regression_df[regression_df['has_reuse_artifact_citance'] == 0]['total_clinical_citations_influential']
        control_non_inf = regression_df[regression_df['has_reuse_artifact_citance'] == 0]['total_clinical_citations_non_influential']
        
        categories = ['Influential', 'Non-Influential']
        treated_means = [treated_inf.mean(), treated_non_inf.mean()]
        control_means = [control_inf.mean(), control_non_inf.mean()]
        
        x = range(len(categories))
        axes[0,1].bar([i - width/2 for i in x], control_means, width, label='No Reuse', alpha=0.8, color=pathos_blue)
        axes[0,1].bar([i + width/2 for i in x], treated_means, width, label='Has Reuse', alpha=0.8, color=pathos_orange)
        axes[0,1].set_title('Clinical Citations: Influential vs Non-Influential')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(categories)
        axes[0,1].legend()
        axes[0,1].set_ylabel('Mean Citations')
    
    # Scatter plot: Total citations vs Clinical citations
    if 'total_clinical_citations' in regression_df.columns:
        treated_data = regression_df[regression_df['has_reuse_artifact_citance'] == 1]
        control_data = regression_df[regression_df['has_reuse_artifact_citance'] == 0]
        
        axes[1,0].scatter(control_data['citationcount'], control_data['total_clinical_citations'], 
                         alpha=0.6, label='No Reuse', s=20, color=pathos_blue)
        axes[1,0].scatter(treated_data['citationcount'], treated_data['total_clinical_citations'], 
                         alpha=0.6, label='Has Reuse', s=20, color=pathos_orange)
        axes[1,0].set_xlabel('Total Citations')
        axes[1,0].set_ylabel('Clinical Citations')
        axes[1,0].set_title('Clinical vs Total Citations')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Patent citations comparison
    if 'patent_citations' in regression_df.columns:
        treated_patents = regression_df[regression_df['has_reuse_artifact_citance'] == 1]['patent_citations']
        control_patents = regression_df[regression_df['has_reuse_artifact_citance'] == 0]['patent_citations']
        
        data_to_plot = [control_patents.values, treated_patents.values]
        box_plot = axes[1,1].boxplot(data_to_plot, labels=['No Reuse', 'Has Reuse'], patch_artist=True)
        
        colors = pathos_colors
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1,1].set_title('Patent Citations Distribution')
        axes[1,1].set_ylabel('Patent Citations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '07_clinical_impact_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 8. Artifact creation patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Artifact types by treatment group
artifact_cols = ['named_datasets_created', 'unnamed_datasets_created', 
                'named_software_created', 'unnamed_software_created']
treated_artifacts = regression_df[regression_df['has_reuse_artifact_citance'] == 1][artifact_cols].mean()
control_artifacts = regression_df[regression_df['has_reuse_artifact_citance'] == 0][artifact_cols].mean()

x = range(len(artifact_cols))
width = 0.35
axes[0,0].bar([i - width/2 for i in x], control_artifacts, width, label='No Reuse', alpha=0.8, color=pathos_blue)
axes[0,0].bar([i + width/2 for i in x], treated_artifacts, width, label='Has Reuse', alpha=0.8, color=pathos_orange)
axes[0,0].set_title('Mean Artifacts Created by Type')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels([col.replace('_', '\n').title() for col in artifact_cols])
axes[0,0].legend()
axes[0,0].set_ylabel('Mean Count')

# Total artifacts distribution
axes[0,1].hist(regression_df[regression_df['has_reuse_artifact_citance'] == 0]['total_artifacts'], 
              alpha=0.7, label='No Reuse', bins=20, density=True, color=pathos_blue)
axes[0,1].hist(regression_df[regression_df['has_reuse_artifact_citance'] == 1]['total_artifacts'], 
              alpha=0.7, label='Has Reuse', bins=20, density=True, color=pathos_orange)
axes[0,1].set_title('Total Artifacts Distribution')
axes[0,1].set_xlabel('Total Artifacts')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()

# Scatter: Artifacts vs Outcomes
if 'total_clinical_citations' in regression_df.columns:
    treated_data = regression_df[regression_df['has_reuse_artifact_citance'] == 1]
    control_data = regression_df[regression_df['has_reuse_artifact_citance'] == 0]
    
    axes[1,0].scatter(control_data['total_artifacts'], control_data['total_clinical_citations'], 
                     alpha=0.6, label='No Reuse', s=20, color=pathos_blue)
    axes[1,0].scatter(treated_data['total_artifacts'], treated_data['total_clinical_citations'], 
                     alpha=0.6, label='Has Reuse', s=20, color=pathos_orange)
    axes[1,0].set_xlabel('Total Artifacts Created')
    axes[1,0].set_ylabel('Clinical Citations')
    axes[1,0].set_title('Artifacts vs Clinical Impact')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

# Science-industry collaboration
if 'science_industry_collaboration' in regression_df.columns:
    collab_data = regression_df.groupby(['has_reuse_artifact_citance', 'science_industry_collaboration']).size().unstack(fill_value=0)
    collab_pct = collab_data.div(collab_data.sum(axis=1), axis=0) * 100
    
    collab_pct.plot(kind='bar', ax=axes[1,1], color=pathos_colors)
    axes[1,1].set_title('Science-Industry Collaboration Rates')
    axes[1,1].set_xlabel('Has Reuse Citations')
    axes[1,1].set_ylabel('Percentage')
    axes[1,1].set_xticklabels(['No', 'Yes'], rotation=0)
    axes[1,1].legend(['No Collaboration', 'Has Collaboration'])

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '08_artifact_patterns.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9. Summary statistics visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Create summary table visualization
summary_data = {
    'Metric': ['Sample Size', 'Treatment Group', 'Control Group', 'Mean Citations (Treatment)', 
               'Mean Citations (Control)', 'Mean FWCI (Treatment)', 'Mean FWCI (Control)'],
    'Value': [
        len(regression_df),
        regression_df['has_reuse_artifact_citance'].sum(),
        (regression_df['has_reuse_artifact_citance'] == 0).sum(),
        regression_df[regression_df['has_reuse_artifact_citance'] == 1]['citationcount'].mean(),
        regression_df[regression_df['has_reuse_artifact_citance'] == 0]['citationcount'].mean(),
        regression_df[regression_df['has_reuse_artifact_citance'] == 1]['fwci'].mean(),
        regression_df[regression_df['has_reuse_artifact_citance'] == 0]['fwci'].mean()
    ]
}

summary_df_viz = pd.DataFrame(summary_data)
summary_df_viz['Value'] = summary_df_viz['Value'].round(2)

# Create table
table = ax.table(cellText=summary_df_viz.values, colLabels=summary_df_viz.columns,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Style the table
for i in range(len(summary_df_viz.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.axis('off')
ax.set_title('Regression Analysis Summary Statistics', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(viz_path, '09_summary_statistics.png'), dpi=300, bbox_inches='tight')
plt.close()

# NEW: Analysis conclusions with supporting figures
print("\nGenerating analysis conclusions...")

# CONCLUSION 1: Effect size analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate effect sizes (Cohen's d) for key outcomes
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

key_outcomes = ['clinical_trial_citations', 'clinical_guideline_citations', 'patent_citations', 'total_clinical_citations']
effect_sizes = []
treated_group = regression_df[regression_df['has_reuse_artifact_citance'] == 1]
control_group = regression_df[regression_df['has_reuse_artifact_citance'] == 0]

for outcome in key_outcomes:
    if outcome in regression_df.columns:
        effect_size = cohens_d(treated_group[outcome], control_group[outcome])
        effect_sizes.append((outcome, effect_size))

# Plot effect sizes
outcomes_for_plot = [x[0] for x in effect_sizes]
effect_values = [x[1] for x in effect_sizes]

axes[0,0].barh(range(len(outcomes_for_plot)), effect_values, color=pathos_orange, alpha=0.8)
axes[0,0].set_yticks(range(len(outcomes_for_plot)))
axes[0,0].set_yticklabels([o.replace('_', ' ').title() for o in outcomes_for_plot])
axes[0,0].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Small Effect')
axes[0,0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Effect')
axes[0,0].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large Effect')
axes[0,0].set_xlabel("Cohen's d (Effect Size)")
axes[0,0].set_title('Effect Sizes: Reuse Citations Impact')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# CONCLUSION 2: Success rate analysis
success_metrics = {
    'Any Clinical Citation': (regression_df['total_clinical_citations'] > 0).astype(int),
    'High Clinical Impact (>2)': (regression_df['total_clinical_citations'] > 2).astype(int),
    'Patent Citation': (regression_df['patent_citations'] > 0).astype(int),
    'Industry Collaboration': regression_df['science_industry_collaboration'].astype(int)
}

success_rates_treated = []
success_rates_control = []
metric_names = []

for metric_name, metric_data in success_metrics.items():
    treated_rate = metric_data[regression_df['has_reuse_artifact_citance'] == 1].mean()
    control_rate = metric_data[regression_df['has_reuse_artifact_citance'] == 0].mean()
    
    success_rates_treated.append(treated_rate * 100)
    success_rates_control.append(control_rate * 100)
    metric_names.append(metric_name)

x = np.arange(len(metric_names))
width = 0.35

axes[0,1].bar(x - width/2, success_rates_control, width, label='No Reuse Citations', 
              color=pathos_blue, alpha=0.8)
axes[0,1].bar(x + width/2, success_rates_treated, width, label='Has Reuse Citations', 
              color=pathos_orange, alpha=0.8)

axes[0,1].set_ylabel('Success Rate (%)')
axes[0,1].set_title('Success Rates by Treatment Group')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(metric_names, rotation=45, ha='right')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# CONCLUSION 3: Dose-response relationship
reuse_counts = regression_df['reuse_artifact_inbound'].values
clinical_impact = regression_df['total_clinical_citations'].values

# Create bins for reuse counts
reuse_bins = [0, 1, 2, 5, 10, max(reuse_counts) + 1]
reuse_binned = pd.cut(reuse_counts, bins=reuse_bins, right=False, 
                     labels=['0', '1', '2-4', '5-9', '10+'])

bin_means = []
bin_labels = []
bin_counts = []

for bin_label in ['0', '1', '2-4', '5-9', '10+']:
    mask = reuse_binned == bin_label
    if mask.sum() > 0:
        bin_means.append(clinical_impact[mask].mean())
        bin_labels.append(bin_label)
        bin_counts.append(mask.sum())

axes[1,0].bar(range(len(bin_labels)), bin_means, color=pathos_orange, alpha=0.8)
axes[1,0].set_xticks(range(len(bin_labels)))
axes[1,0].set_xticklabels([f'{label}\n(n={count})' for label, count in zip(bin_labels, bin_counts)])
axes[1,0].set_xlabel('Number of Reuse-Artifact Citations')
axes[1,0].set_ylabel('Mean Clinical Citations')
axes[1,0].set_title('Dose-Response: Reuse Citations vs Clinical Impact')
axes[1,0].grid(True, alpha=0.3)

# CONCLUSION 4: Quality vs quantity analysis
if 'total_clinical_citations_influential' in regression_df.columns:
    treated_group = regression_df[regression_df['has_reuse_artifact_citance'] == 1]
    control_group = regression_df[regression_df['has_reuse_artifact_citance'] == 0]
    
    quality_metrics = ['total_clinical_citations_influential', 'total_clinical_citations_non_influential']
    treated_quality = [treated_group[metric].mean() for metric in quality_metrics]
    control_quality = [control_group[metric].mean() for metric in quality_metrics]
    
    x = np.arange(len(quality_metrics))
    axes[1,1].bar(x - width/2, control_quality, width, label='No Reuse Citations', 
                  color=pathos_blue, alpha=0.8)
    axes[1,1].bar(x + width/2, treated_quality, width, label='Has Reuse Citations', 
                  color=pathos_orange, alpha=0.8)
    
    axes[1,1].set_ylabel('Mean Citations')
    axes[1,1].set_title('Quality of Clinical Impact:\nInfluential vs Non-Influential Citations')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(['Influential', 'Non-Influential'])
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '10_analysis_conclusions.png'), dpi=300, bbox_inches='tight')
plt.close()

# CONCLUSION 5: Statistical significance summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Significance levels and effect directions
if results_summary:
    outcomes = [r['outcome'] for r in results_summary]
    p_values = [r['p_value'] for r in results_summary]
    coefficients = [r['coefficient'] for r in results_summary]
    
    # Categorize results
    significant_positive = sum(1 for p, c in zip(p_values, coefficients) if p < 0.05 and c > 0)
    significant_negative = sum(1 for p, c in zip(p_values, coefficients) if p < 0.05 and c < 0)
    non_significant = sum(1 for p in p_values if p >= 0.05)
    
    # Pie chart of significance
    significance_data = [significant_positive, significant_negative, non_significant]
    significance_labels = [f'Significant Positive\n(n={significant_positive})', 
                          f'Significant Negative\n(n={significant_negative})', 
                          f'Non-Significant\n(n={non_significant})']
    colors = [pathos_orange, 'red', 'gray']
    
    ax1.pie(significance_data, labels=significance_labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Distribution of Statistical Results')
    
    # Effect magnitude vs significance
    scatter_colors = [pathos_orange if p < 0.05 else 'gray' for p in p_values]
    ax2.scatter(coefficients, [-np.log10(p) for p in p_values], 
               c=scatter_colors, s=100, alpha=0.7)
    
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Coefficient Estimate')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Effect Size vs Statistical Significance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '11_statistical_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("REGRESSION ANALYSIS COMPLETED")
print("="*60)
print(f"Final sample size: {len(regression_df)}")
print(f"Papers with reuse citances: {regression_df['has_reuse_artifact_citance'].sum()}")
print(f"Papers without reuse citances: {(regression_df['has_reuse_artifact_citance'] == 0).sum()}")

if not interaction_summary_df.empty:
    print(f"Interaction effects: {interaction_summary_df['interaction_significant'].sum()} significant out of {len(interaction_summary_df)} tested")

print("\nFiles saved:")
print("- regression_sample_covid.xlsx (main dataset for Excel analysis)")
print("- regression_results_summary_covid.xlsx (coefficient estimates)")  
print("- descriptive_stats_by_treatment_covid.xlsx (means by group)")
print("- treatment_breakdown_covid.xlsx (detailed treatment group statistics)")
print("- correlation_matrix_covid.xlsx (variable correlations)")
print("- regression_output_[outcome]_covid.txt (detailed results)")
print("- 16_interaction_effects_summary.xlsx (interaction analysis)")
print("- 17_marginal_effects_analysis.xlsx (marginal effects)")
print("- 18_stratified_interaction_analysis.xlsx (stratified analysis)")
print("- 19_interaction_interpretation.xlsx (interaction interpretation)")
