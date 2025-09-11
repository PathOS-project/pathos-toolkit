import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style with PATHOS colors
plt.style.use('default')
pathos_blue = '#2E5C8A'    # Medium dark blue
pathos_orange = '#D17A2A'  # Medium dark orange
pathos_green = '#4CAF50'   # Green for neutral/supporting
pathos_red = '#F44336'     # Red for refuting
pathos_colors = [pathos_blue, pathos_orange]
sns.set_palette([pathos_blue, pathos_orange])

# Load the regression sample data
print("Loading COVID-19 regression sample data...")
data_path = 'PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes/results'
regression_df = pd.read_parquet(os.path.join(data_path, 'regression_sample_covid_v4b.parquet'))

# Create final visualization directory - NEW PATH
final_viz_path = 'PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes/results/final_visualization_data_figures'
os.makedirs(final_viz_path, exist_ok=True)

# Create subdirectories for organization
viz_path = os.path.join(final_viz_path, 'figures')
data_export_path = os.path.join(final_viz_path, 'data')
os.makedirs(viz_path, exist_ok=True)
os.makedirs(data_export_path, exist_ok=True)

print(f"Sample size: {len(regression_df)}")
print(f"Years covered: {regression_df['year'].min()}-{regression_df['year'].max()}")

# Load the complete COVID dataset for the first visualization (created artifact share)
print("Loading complete COVID dataset for artifact creation share...")
complete_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/covid_collection_w_outcomes/complete_collection_df_fix.parquet')

# 1. Created artifact share
print("Creating Visualization 1: Created artifact share")

# Calculate artifact creation by year
artifact_creators = (
    (complete_df['named_datasets_created'] > 0) |
    (complete_df['unnamed_datasets_created'] > 0) |
    (complete_df['named_software_created'] > 0) |
    (complete_df['unnamed_software_created'] > 0)
)

artifact_share_by_year = complete_df.groupby('year').agg({
    'id': 'count',  # Total papers
}).rename(columns={'id': 'total_papers'})

artifact_creators_by_year = complete_df[artifact_creators].groupby('year').agg({
    'id': 'count'  # Papers that created artifacts
}).rename(columns={'id': 'artifact_creators'})

artifact_share = artifact_creators_by_year.join(artifact_share_by_year, how='right').fillna(0)
artifact_share['artifact_share_pct'] = (artifact_share['artifact_creators'] / artifact_share['total_papers']) * 100

# Save data for this visualization
artifact_share.to_excel(os.path.join(data_export_path, '01_created_artifact_share_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(artifact_share.index, artifact_share['artifact_share_pct'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue)
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of COVID Papers that Created ≥1 Dataset/Software', fontsize=12)
ax.set_title('Created Artifact Share Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(artifact_share['artifact_share_pct']) * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '01_created_artifact_share.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Observable reuse rate
print("Creating Visualization 2: Observable reuse rate")

reuse_rate_by_year = regression_df.groupby('year').agg({
    'has_reuse_artifact_citance': ['count', 'sum']
}).round(2)

reuse_rate_by_year.columns = ['total_papers', 'papers_with_reuse']
reuse_rate_by_year['reuse_rate_pct'] = (reuse_rate_by_year['papers_with_reuse'] / reuse_rate_by_year['total_papers']) * 100

# Save data for this visualization
reuse_rate_by_year.to_excel(os.path.join(data_export_path, '02_observable_reuse_rate_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(reuse_rate_by_year.index, reuse_rate_by_year['reuse_rate_pct'], 
        marker='o', linewidth=3, markersize=8, color=pathos_orange)
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of Artifact-Creating Papers with ≥1 Reuse-Artifact Citance', fontsize=12)
ax.set_title('Observable Reuse Rate Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(reuse_rate_by_year['reuse_rate_pct']) * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '02_observable_reuse_rate.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Clinical-trial citation intensity
print("Creating Visualization 3: Clinical-trial citation intensity")

clinical_trial_by_year = regression_df.groupby(['year', 'has_reuse_artifact_citance'])['clinical_trial_citations'].mean().unstack()
clinical_trial_by_year.columns = ['No Reuse', 'Has Reuse']

# Save data for this visualization
clinical_trial_by_year.to_excel(os.path.join(data_export_path, '03_clinical_trial_intensity_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(clinical_trial_by_year.index, clinical_trial_by_year['No Reuse'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='No Reuse')
ax.plot(clinical_trial_by_year.index, clinical_trial_by_year['Has Reuse'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Has Reuse')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Mean Clinical-Trial Citations per Paper', fontsize=12)
ax.set_title('Clinical-Trial Citation Intensity Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '03_clinical_trial_intensity.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Clinical-guideline citation intensity
print("Creating Visualization 4: Clinical-guideline citation intensity")

clinical_guideline_by_year = regression_df.groupby(['year', 'has_reuse_artifact_citance'])['clinical_guideline_citations'].mean().unstack()
clinical_guideline_by_year.columns = ['No Reuse', 'Has Reuse']

# Save data for this visualization
clinical_guideline_by_year.to_excel(os.path.join(data_export_path, '04_clinical_guideline_intensity_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(clinical_guideline_by_year.index, clinical_guideline_by_year['No Reuse'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='No Reuse')
ax.plot(clinical_guideline_by_year.index, clinical_guideline_by_year['Has Reuse'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Has Reuse')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Mean Clinical-Guideline Citations per Paper', fontsize=12)
ax.set_title('Clinical-Guideline Citation Intensity Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '04_clinical_guideline_intensity.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Patent citations
print("Creating Visualization 5: Patent citations")

patent_by_year = regression_df.groupby(['year', 'has_reuse_artifact_citance'])['patent_citations'].mean().unstack()
patent_by_year.columns = ['No Reuse', 'Has Reuse']

# Save data for this visualization
patent_by_year.to_excel(os.path.join(data_export_path, '05_patent_citations_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(patent_by_year.index, patent_by_year['No Reuse'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='No Reuse')
ax.plot(patent_by_year.index, patent_by_year['Has Reuse'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Has Reuse')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Mean Patent Citations per Paper', fontsize=12)
ax.set_title('Patent Citations Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '05_patent_citations.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Science-industry collaboration rate
print("Creating Visualization 6: Science-industry collaboration rate")

collab_by_year = regression_df.groupby(['year', 'has_reuse_artifact_citance'])['science_industry_collaboration'].mean().unstack() * 100
collab_by_year.columns = ['No Reuse', 'Has Reuse']

# Save data for this visualization
collab_by_year.to_excel(os.path.join(data_export_path, '06_science_industry_collaboration_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(collab_by_year.index, collab_by_year['No Reuse'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='No Reuse')
ax.plot(collab_by_year.index, collab_by_year['Has Reuse'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Has Reuse')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of Papers with Industry Collaboration', fontsize=12)
ax.set_title('Science-Industry Collaboration Rate Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '06_science_industry_collaboration.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Community stance on focal papers
print("Creating Visualization 7: Community stance on focal papers")

# Calculate total inbound citations by stance
regression_df['total_inbound'] = (regression_df['supporting_inbound'] + 
                                 regression_df['neutral_inbound'] + 
                                 regression_df['refuting_inbound'])

stance_by_year = regression_df.groupby('year').agg({
    'supporting_inbound': 'sum',
    'neutral_inbound': 'sum', 
    'refuting_inbound': 'sum',
    'total_inbound': 'sum'
})

# Calculate percentages
stance_pct_by_year = stance_by_year.div(stance_by_year['total_inbound'], axis=0) * 100
stance_pct_by_year = stance_pct_by_year.drop('total_inbound', axis=1)

# Save data for this visualization
stance_pct_by_year.to_excel(os.path.join(data_export_path, '07_community_stance_focal_papers_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stance_pct_by_year.index, stance_pct_by_year['supporting_inbound'], 
        marker='o', linewidth=3, markersize=8, color=pathos_green, label='Supporting')
ax.plot(stance_pct_by_year.index, stance_pct_by_year['neutral_inbound'], 
        marker='s', linewidth=3, markersize=8, color=pathos_blue, label='Neutral')
ax.plot(stance_pct_by_year.index, stance_pct_by_year['refuting_inbound'], 
        marker='^', linewidth=3, markersize=8, color=pathos_red, label='Refuting')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of Inbound Citations in Each Stance Category', fontsize=12)
ax.set_title('Community Stance on Focal Papers Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '07_community_stance_focal_papers.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. Stance on artifact-reuse citances
print("Creating Visualization 8: Stance on artifact-reuse citances")

# Filter to papers with reuse citances
reuse_papers = regression_df[regression_df['has_reuse_artifact_citance'] == 1]

# Calculate total reuse citances by stance
reuse_papers_copy = reuse_papers.copy()
reuse_papers_copy['total_reuse_inbound'] = (reuse_papers_copy['supporting_reuse_artifact_inbound'] + 
                                           reuse_papers_copy['neutral_reuse_artifact_inbound'] + 
                                           reuse_papers_copy['refuting_reuse_artifact_inbound'])

reuse_stance_by_year = reuse_papers_copy.groupby('year').agg({
    'supporting_reuse_artifact_inbound': 'sum',
    'neutral_reuse_artifact_inbound': 'sum',
    'refuting_reuse_artifact_inbound': 'sum',
    'total_reuse_inbound': 'sum'
})

# Calculate percentages
reuse_stance_pct_by_year = reuse_stance_by_year.div(reuse_stance_by_year['total_reuse_inbound'], axis=0) * 100
reuse_stance_pct_by_year = reuse_stance_pct_by_year.drop('total_reuse_inbound', axis=1)

# Save data for this visualization
reuse_stance_pct_by_year.to_excel(os.path.join(data_export_path, '08_stance_artifact_reuse_citances_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(reuse_stance_pct_by_year.index, reuse_stance_pct_by_year['supporting_reuse_artifact_inbound'], 
        marker='o', linewidth=3, markersize=8, color=pathos_green, label='Supporting')
ax.plot(reuse_stance_pct_by_year.index, reuse_stance_pct_by_year['neutral_reuse_artifact_inbound'], 
        marker='s', linewidth=3, markersize=8, color=pathos_blue, label='Neutral')
ax.plot(reuse_stance_pct_by_year.index, reuse_stance_pct_by_year['refuting_reuse_artifact_inbound'], 
        marker='^', linewidth=3, markersize=8, color=pathos_red, label='Refuting')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of Reuse-Artifact Citances in Each Stance Category', fontsize=12)
ax.set_title('Stance on Artifact-Reuse Citances Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '08_stance_artifact_reuse_citances.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9. Dose-response curve
print("Creating Visualization 9: Dose-response curve")

# Create dose bins
max_reuse = min(regression_df['reuse_artifact_inbound'].max(), 15)  # Cap for meaningful visualization
dose_bins = [0, 1, 2, 5, 10, max_reuse + 1]
dose_labels = ['0', '1', '2-4', '5-9', '10+']

regression_df['reuse_dose_bin'] = pd.cut(regression_df['reuse_artifact_inbound'], 
                                        bins=dose_bins, right=False, labels=dose_labels)

dose_response = regression_df.groupby('reuse_dose_bin').agg({
    'clinical_trial_citations': ['mean', 'count', 'std']
}).round(3)

dose_response.columns = ['mean_clinical_trial', 'count', 'std_clinical_trial']
dose_response = dose_response.reset_index()

# Calculate standard errors
dose_response['se_clinical_trial'] = dose_response['std_clinical_trial'] / np.sqrt(dose_response['count'])

# Save data for this visualization
dose_response.to_excel(os.path.join(data_export_path, '09_dose_response_curve_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(dose_response))
bars = ax.bar(x_pos, dose_response['mean_clinical_trial'], 
              yerr=dose_response['se_clinical_trial'], capsize=5,
              color=pathos_orange, alpha=0.8, edgecolor='black')

# Add sample size labels on bars
for i, (bar, count) in enumerate(zip(bars, dose_response['count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + dose_response['se_clinical_trial'].iloc[i],
            f'n={int(count)}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Number of Reuse-Artifact Citances', fontsize=12)
ax.set_ylabel('Mean Clinical-Trial Citations', fontsize=12)
ax.set_title('Dose-Response: Reuse Citations vs Clinical Impact', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(dose_response['reuse_dose_bin'])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '09_dose_response_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 10. Quality-modulated reuse premium (FWCI strata)
print("Creating Visualization 10: Quality-modulated reuse premium")

# Create FWCI strata
fwci_terciles = regression_df['fwci'].quantile([0.33, 0.67]).values
regression_df['fwci_stratum'] = pd.cut(regression_df['fwci'], 
                                      bins=[-np.inf, fwci_terciles[0], fwci_terciles[1], np.inf],
                                      labels=['Low FWCI', 'Medium FWCI', 'High FWCI'])

# Calculate marginal effects for each stratum
marginal_effects = []
stratum_names = []
stratum_data_summary = []

for stratum in ['Low FWCI', 'Medium FWCI', 'High FWCI']:
    stratum_data = regression_df[regression_df['fwci_stratum'] == stratum]
    
    if len(stratum_data) > 20:  # Minimum sample size
        treated = stratum_data[stratum_data['has_reuse_artifact_citance'] == 1]['clinical_trial_citations']
        control = stratum_data[stratum_data['has_reuse_artifact_citance'] == 0]['clinical_trial_citations']
        
        if len(treated) > 0 and len(control) > 0:
            marginal_effect = treated.mean() - control.mean()
            marginal_effects.append(marginal_effect)
            stratum_names.append(stratum)
            
            # Store detailed data for export
            stratum_data_summary.append({
                'stratum': stratum,
                'marginal_effect': marginal_effect,
                'treated_mean': treated.mean(),
                'control_mean': control.mean(),
                'treated_n': len(treated),
                'control_n': len(control)
            })

# Save data for this visualization
pd.DataFrame(stratum_data_summary).to_excel(os.path.join(data_export_path, '10_quality_modulated_reuse_premium_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(stratum_names)), marginal_effects, 
              color=pathos_orange, alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar, effect in zip(bars, marginal_effects):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{effect:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('FWCI Stratum', fontsize=12)
ax.set_ylabel('Marginal Effect of Reuse on Clinical-Trial Citations (β̂)', fontsize=12)
ax.set_title('Quality-Modulated Reuse Premium', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(stratum_names)))
ax.set_xticklabels(stratum_names)
ax.grid(True, alpha=0.3, axis='y')

# Add sample size information
for i, stratum in enumerate(stratum_names):
    stratum_data = regression_df[regression_df['fwci_stratum'] == stratum]
    treated_n = (stratum_data['has_reuse_artifact_citance'] == 1).sum()
    control_n = (stratum_data['has_reuse_artifact_citance'] == 0).sum()
    ax.text(i, ax.get_ylim()[0] + 0.05, f'T:{treated_n}\nC:{control_n}', 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(viz_path, '10_quality_modulated_reuse_premium.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a summary figure with key statistics
print("Creating summary statistics table...")

summary_stats = {
    'Metric': [
        'Total Sample Size',
        'Papers with Reuse Citations (%)',
        'Mean Clinical Trial Citations (Reuse)',
        'Mean Clinical Trial Citations (No Reuse)',
        'Reuse Premium (Clinical Trials)',
        'Mean Clinical Guideline Citations (Reuse)',
        'Mean Clinical Guideline Citations (No Reuse)',
        'Reuse Premium (Clinical Guidelines)',
        'Mean Patent Citations (Reuse)',
        'Mean Patent Citations (No Reuse)',
        'Reuse Premium (Patents)',
        'Industry Collaboration Rate (Reuse)',
        'Industry Collaboration Rate (No Reuse)'
    ],
    'Value': [
        f"{len(regression_df):,}",
        f"{regression_df['has_reuse_artifact_citance'].mean()*100:.1f}%",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['clinical_trial_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==0]['clinical_trial_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['clinical_trial_citations'].mean() - regression_df[regression_df['has_reuse_artifact_citance']==0]['clinical_trial_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['clinical_guideline_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==0]['clinical_guideline_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['clinical_guideline_citations'].mean() - regression_df[regression_df['has_reuse_artifact_citance']==0]['clinical_guideline_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['patent_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==0]['patent_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['patent_citations'].mean() - regression_df[regression_df['has_reuse_artifact_citance']==0]['patent_citations'].mean():.3f}",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==1]['science_industry_collaboration'].mean()*100:.1f}%",
        f"{regression_df[regression_df['has_reuse_artifact_citance']==0]['science_industry_collaboration'].mean()*100:.1f}%"
    ]
}

summary_df = pd.DataFrame(summary_stats)

# Save summary data to Excel
summary_df.to_excel(os.path.join(data_export_path, '11_summary_statistics_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(12, 8))
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style the table
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor(pathos_blue)
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.axis('off')
ax.set_title('Impact of Artefact Reuse in COVID-19 Publications Case Study: Key Statistics Summary', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(viz_path, '11_summary_statistics_table.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a comprehensive data export file with all visualization data
print("Creating comprehensive data export...")

# Create a workbook with multiple sheets
with pd.ExcelWriter(os.path.join(data_export_path, 'all_visualization_data.xlsx'), engine='openpyxl') as writer:
    artifact_share.to_excel(writer, sheet_name='01_artifact_share')
    reuse_rate_by_year.to_excel(writer, sheet_name='02_reuse_rate')
    clinical_trial_by_year.to_excel(writer, sheet_name='03_clinical_trial')
    clinical_guideline_by_year.to_excel(writer, sheet_name='04_clinical_guideline')
    patent_by_year.to_excel(writer, sheet_name='05_patent')
    collab_by_year.to_excel(writer, sheet_name='06_collaboration')
    stance_pct_by_year.to_excel(writer, sheet_name='07_stance_focal')
    reuse_stance_pct_by_year.to_excel(writer, sheet_name='08_stance_reuse')
    dose_response.to_excel(writer, sheet_name='09_dose_response', index=False)
    pd.DataFrame(stratum_data_summary).to_excel(writer, sheet_name='10_fwci_strata', index=False)
    summary_df.to_excel(writer, sheet_name='11_summary_stats', index=False)

print(f"\nAll visualizations completed and saved to: {viz_path}")
print(f"All data files saved to: {data_export_path}")
print("\nGenerated visualizations:")
print("01_created_artifact_share.png - % of COVID papers creating artifacts by year")
print("02_observable_reuse_rate.png - % of artifact creators with reuse citations by year") 
print("03_clinical_trial_intensity.png - Mean clinical trial citations by year and reuse status")
print("04_clinical_guideline_intensity.png - Mean clinical guideline citations by year and reuse status")
print("05_patent_citations.png - Mean patent citations by year and reuse status")
print("06_science_industry_collaboration.png - Industry collaboration rates by year and reuse status")
print("07_community_stance_focal_papers.png - Citation stance distribution over time")
print("08_stance_artifact_reuse_citances.png - Reuse citation stance distribution over time")
print("09_dose_response_curve.png - Clinical impact by number of reuse citations")
print("10_quality_modulated_reuse_premium.png - Reuse effect by paper quality (FWCI)")
print("11_summary_statistics_table.png - Key statistics summary table")

print(f"\nData files created:")
print("- Individual Excel files for each visualization (01-11)")
print("- all_visualization_data.xlsx - Comprehensive file with all data in separate sheets")
print(f"\nAll files saved in: {final_viz_path}")
