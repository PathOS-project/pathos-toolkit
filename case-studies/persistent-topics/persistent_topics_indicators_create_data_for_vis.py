import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

print("Loading Impact of Open Access Routes on Topic Persistence Case Study data...")

# Load the main results and datasets
data_path = 'PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/results'
complete_collection_df = pd.read_parquet('PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/complete_collection_df.parquet')

# Create final visualization directory
final_viz_path = 'PATH_TO_INDICATOR_RESULTS/persistent_topics_collection_w_outcomes/results/final_visualization_data_figures'
os.makedirs(final_viz_path, exist_ok=True)

# Create subdirectories for organization
viz_path = os.path.join(final_viz_path, 'figures')
data_export_path = os.path.join(final_viz_path, 'data')
os.makedirs(viz_path, exist_ok=True)
os.makedirs(data_export_path, exist_ok=True)

# Load matched datasets
try:
    green_matched = pd.read_excel(os.path.join(data_path, 'psm_a_green_matched.xlsx'))
    closed_matched_a = pd.read_excel(os.path.join(data_path, 'psm_a_closed_matched.xlsx'))
    published_matched = pd.read_excel(os.path.join(data_path, 'psm_b_published_matched.xlsx'))
    closed_matched_b = pd.read_excel(os.path.join(data_path, 'psm_b_closed_matched.xlsx'))
    print("Matched datasets loaded successfully")
except FileNotFoundError as e:
    print(f"Warning: Could not load matched datasets: {e}")
    green_matched = pd.DataFrame()
    closed_matched_a = pd.DataFrame()
    published_matched = pd.DataFrame()
    closed_matched_b = pd.DataFrame()

print(f"Complete collection size: {len(complete_collection_df)}")
print(f"Years covered: {complete_collection_df['year'].min()}-{complete_collection_df['year'].max()}")

# Define OA categories consistently with the main analysis
complete_collection_df['green_oa_only'] = (
    (complete_collection_df['green'] == True) & 
    (complete_collection_df['bronze'].isin([False, None])) & 
    (complete_collection_df['hybrid'].isin([False, None])) & 
    (complete_collection_df['gold'].isin([False, None])) & 
    (complete_collection_df['diamond'].isin([False, None]))
)

complete_collection_df['published_oa_only'] = (
    (complete_collection_df['green'].isin([False, None])) & 
    ((complete_collection_df['gold'] == True) | 
     (complete_collection_df['hybrid'] == True) | 
     (complete_collection_df['diamond'] == True))
)

complete_collection_df['closed_access'] = (complete_collection_df['isopenaccess'] == False)

print(f"Green OA only: {complete_collection_df['green_oa_only'].sum():,}")
print(f"Published OA only: {complete_collection_df['published_oa_only'].sum():,}")
print(f"Closed Access: {complete_collection_df['closed_access'].sum():,}")

# 1. Share of Green OA & Share of Published OA over time
print("Creating Visualization 1: OA shares over time")

oa_shares_by_year = complete_collection_df.groupby('year').agg({
    'id': 'count',  # Total papers
    'green_oa_only': 'sum',
    'published_oa_only': 'sum'
}).rename(columns={'id': 'total_papers'})

oa_shares_by_year['green_oa_share_pct'] = (oa_shares_by_year['green_oa_only'] / oa_shares_by_year['total_papers']) * 100
oa_shares_by_year['published_oa_share_pct'] = (oa_shares_by_year['published_oa_only'] / oa_shares_by_year['total_papers']) * 100

# Save data for this visualization
oa_shares_by_year.to_excel(os.path.join(data_export_path, '01_oa_shares_over_time_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(oa_shares_by_year.index, oa_shares_by_year['green_oa_share_pct'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='Green OA')
ax.plot(oa_shares_by_year.index, oa_shares_by_year['published_oa_share_pct'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Published OA')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('% of AI-Climate Papers', fontsize=12)
ax.set_title('Open Access Adoption Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '01_oa_shares_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Mean Citation Count by access route over time
print("Creating Visualization 2: Mean citation count over time")

citation_by_year = complete_collection_df.groupby(['year', 'green_oa_only', 'published_oa_only', 'closed_access']).agg({
    'citationcount': 'mean'
}).reset_index()

# Create separate series for each access route
green_citations = citation_by_year[citation_by_year['green_oa_only'] == True].groupby('year')['citationcount'].mean()
published_citations = citation_by_year[citation_by_year['published_oa_only'] == True].groupby('year')['citationcount'].mean()
closed_citations = citation_by_year[citation_by_year['closed_access'] == True].groupby('year')['citationcount'].mean()

citation_data = pd.DataFrame({
    'Green_OA': green_citations,
    'Published_OA': published_citations,
    'Closed_Access': closed_citations
}).fillna(0)

# Save data for this visualization
citation_data.to_excel(os.path.join(data_export_path, '02_citation_count_over_time_data.xlsx'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(citation_data.index, citation_data['Green_OA'], 
        marker='o', linewidth=3, markersize=8, color=pathos_blue, label='Green OA')
ax.plot(citation_data.index, citation_data['Published_OA'], 
        marker='s', linewidth=3, markersize=8, color=pathos_orange, label='Published OA')
ax.plot(citation_data.index, citation_data['Closed_Access'], 
        marker='^', linewidth=3, markersize=8, color='gray', label='Closed Access')
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Mean Citations per Paper', fontsize=12)
ax.set_title('Citation Impact by Access Route Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '02_citation_count_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Field-Weighted Citation Impact (FWCI) comparison
print("Creating Visualization 3: FWCI comparison")

# Calculate FWCI means for matched samples
fwci_comparison = []

if not green_matched.empty and not closed_matched_a.empty:
    fwci_comparison.extend([
        {'Access_Route': 'Green OA', 'Mean_FWCI': green_matched['fwci'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Mean_FWCI': closed_matched_a['fwci'].mean(), 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty:
    fwci_comparison.extend([
        {'Access_Route': 'Published OA', 'Mean_FWCI': published_matched['fwci'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Mean_FWCI': closed_matched_b['fwci'].mean(), 'Group': 'Control'}
    ])

fwci_df = pd.DataFrame(fwci_comparison)

# Save data for this visualization
fwci_df.to_excel(os.path.join(data_export_path, '03_fwci_comparison_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
treatment_data = fwci_df[fwci_df['Group'] == 'Treatment']
control_data = fwci_df[fwci_df['Group'] == 'Control']

x_pos = range(len(treatment_data))
width = 0.35

bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Mean_FWCI'], 
               width, label='Control (Closed)', color='gray', alpha=0.8)
bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Mean_FWCI'], 
               width, label='Treatment (OA)', color=pathos_orange, alpha=0.8)

ax.set_xlabel('Access Route Comparison', fontsize=12)
ax.set_ylabel('Mean Field-Weighted Citation Impact (FWCI)', fontsize=12)
ax.set_title('Quality-Adjusted Citation Impact by Access Route', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '03_fwci_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Topic-Persistence Score comparison
print("Creating Visualization 4: Topic persistence score")

persistence_comparison = []

if not green_matched.empty and not closed_matched_a.empty:
    persistence_comparison.extend([
        {'Access_Route': 'Green OA', 'Mean_Persistence': green_matched['topic_persistence_score'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Mean_Persistence': closed_matched_a['topic_persistence_score'].mean(), 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty:
    persistence_comparison.extend([
        {'Access_Route': 'Published OA', 'Mean_Persistence': published_matched['topic_persistence_score'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Mean_Persistence': closed_matched_b['topic_persistence_score'].mean(), 'Group': 'Control'}
    ])

persistence_df = pd.DataFrame(persistence_comparison)

# Save data for this visualization
persistence_df.to_excel(os.path.join(data_export_path, '04_topic_persistence_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
treatment_data = persistence_df[persistence_df['Group'] == 'Treatment']
control_data = persistence_df[persistence_df['Group'] == 'Control']

x_pos = range(len(treatment_data))
width = 0.35

bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Mean_Persistence'], 
               width, label='Control (Closed)', color='gray', alpha=0.8)
bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Mean_Persistence'], 
               width, label='Treatment (OA)', color=pathos_blue, alpha=0.8)

ax.set_xlabel('Access Route Comparison', fontsize=12)
ax.set_ylabel('Mean Topic Persistence Score', fontsize=12)
ax.set_title('Long-Term Topic Staying Power by Access Route', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '04_topic_persistence.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. High-Persistence Topic Rate (Top 25%)
print("Creating Visualization 5: High-persistence topic rate")

# Calculate top 25% threshold
if 'topic_persistence_score' in complete_collection_df.columns:
    persistence_threshold = complete_collection_df['topic_persistence_score'].quantile(0.75)
    
    high_persistence_rates = []
    
    if not green_matched.empty and not closed_matched_a.empty:
        green_high_rate = (green_matched['topic_persistence_score'] > persistence_threshold).mean() * 100
        closed_a_high_rate = (closed_matched_a['topic_persistence_score'] > persistence_threshold).mean() * 100
        
        high_persistence_rates.extend([
            {'Access_Route': 'Green OA', 'High_Persistence_Rate': green_high_rate, 'Group': 'Treatment'},
            {'Access_Route': 'Closed (vs Green)', 'High_Persistence_Rate': closed_a_high_rate, 'Group': 'Control'}
        ])
    
    if not published_matched.empty and not closed_matched_b.empty:
        published_high_rate = (published_matched['topic_persistence_score'] > persistence_threshold).mean() * 100
        closed_b_high_rate = (closed_matched_b['topic_persistence_score'] > persistence_threshold).mean() * 100
        
        high_persistence_rates.extend([
            {'Access_Route': 'Published OA', 'High_Persistence_Rate': published_high_rate, 'Group': 'Treatment'},
            {'Access_Route': 'Closed (vs Published)', 'High_Persistence_Rate': closed_b_high_rate, 'Group': 'Control'}
        ])
    
    high_persistence_df = pd.DataFrame(high_persistence_rates)
    
    # Save data for this visualization
    high_persistence_df.to_excel(os.path.join(data_export_path, '05_high_persistence_rate_data.xlsx'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    treatment_data = high_persistence_df[high_persistence_df['Group'] == 'Treatment']
    control_data = high_persistence_df[high_persistence_df['Group'] == 'Control']
    
    x_pos = range(len(treatment_data))
    width = 0.35
    
    bars1 = ax.bar([x - width/2 for x in x_pos], control_data['High_Persistence_Rate'], 
                   width, label='Control (Closed)', color='gray', alpha=0.8)
    bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['High_Persistence_Rate'], 
                   width, label='Treatment (OA)', color=pathos_green, alpha=0.8)
    
    ax.set_xlabel('Access Route Comparison', fontsize=12)
    ax.set_ylabel('% of Papers in Top 25% Persistence', fontsize=12)
    ax.set_title('High-Persistence Topic Rate by Access Route', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '05_high_persistence_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 6. Science-Industry Collaboration Rate
print("Creating Visualization 6: Science-industry collaboration rate")

collab_comparison = []

if not green_matched.empty and not closed_matched_a.empty:
    collab_comparison.extend([
        {'Access_Route': 'Green OA', 'Collaboration_Rate': green_matched['science_industry_collaboration'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Collaboration_Rate': closed_matched_a['science_industry_collaboration'].mean() * 100, 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty:
    collab_comparison.extend([
        {'Access_Route': 'Published OA', 'Collaboration_Rate': published_matched['science_industry_collaboration'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Collaboration_Rate': closed_matched_b['science_industry_collaboration'].mean() * 100, 'Group': 'Control'}
    ])

collab_df = pd.DataFrame(collab_comparison)

# Save data for this visualization
collab_df.to_excel(os.path.join(data_export_path, '06_science_industry_collaboration_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
treatment_data = collab_df[collab_df['Group'] == 'Treatment']
control_data = collab_df[collab_df['Group'] == 'Control']

x_pos = range(len(treatment_data))
width = 0.35

bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Collaboration_Rate'], 
               width, label='Control (Closed)', color='gray', alpha=0.8)
bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Collaboration_Rate'], 
               width, label='Treatment (OA)', color=pathos_orange, alpha=0.8)

ax.set_xlabel('Access Route Comparison', fontsize=12)
ax.set_ylabel('% of Papers with Industry Collaboration', fontsize=12)
ax.set_title('Science-Industry Collaboration Rate by Access Route', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '06_science_industry_collaboration.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Mean Patent Citations per Paper
print("Creating Visualization 7: Patent citations")

patent_comparison = []

if not green_matched.empty and not closed_matched_a.empty:
    patent_comparison.extend([
        {'Access_Route': 'Green OA', 'Mean_Patent_Citations': green_matched['patent_citations'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Mean_Patent_Citations': closed_matched_a['patent_citations'].mean(), 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty:
    patent_comparison.extend([
        {'Access_Route': 'Published OA', 'Mean_Patent_Citations': published_matched['patent_citations'].mean(), 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Mean_Patent_Citations': closed_matched_b['patent_citations'].mean(), 'Group': 'Control'}
    ])

patent_df = pd.DataFrame(patent_comparison)

# Save data for this visualization
patent_df.to_excel(os.path.join(data_export_path, '07_patent_citations_data.xlsx'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
treatment_data = patent_df[patent_df['Group'] == 'Treatment']
control_data = patent_df[patent_df['Group'] == 'Control']

x_pos = range(len(treatment_data))
width = 0.35

bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Mean_Patent_Citations'], 
               width, label='Control (Closed)', color='gray', alpha=0.8)
bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Mean_Patent_Citations'], 
               width, label='Treatment (OA)', color=pathos_red, alpha=0.8)

ax.set_xlabel('Access Route Comparison', fontsize=12)
ax.set_ylabel('Mean Patent Citations per Paper', fontsize=12)
ax.set_title('Technological Translation by Access Route', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(viz_path, '07_patent_citations.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. Women as Last Author
print("Creating Visualization 8: Women as last author")

women_last_comparison = []

if not green_matched.empty and not closed_matched_a.empty and 'has_woman_last_author' in green_matched.columns:
    women_last_comparison.extend([
        {'Access_Route': 'Green OA', 'Women_Last_Author_Rate': green_matched['has_woman_last_author'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Women_Last_Author_Rate': closed_matched_a['has_woman_last_author'].mean() * 100, 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty and 'has_woman_last_author' in published_matched.columns:
    women_last_comparison.extend([
        {'Access_Route': 'Published OA', 'Women_Last_Author_Rate': published_matched['has_woman_last_author'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Women_Last_Author_Rate': closed_matched_b['has_woman_last_author'].mean() * 100, 'Group': 'Control'}
    ])

if women_last_comparison:
    women_last_df = pd.DataFrame(women_last_comparison)
    
    # Save data for this visualization
    women_last_df.to_excel(os.path.join(data_export_path, '08_women_last_author_data.xlsx'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    treatment_data = women_last_df[women_last_df['Group'] == 'Treatment']
    control_data = women_last_df[women_last_df['Group'] == 'Control']
    
    x_pos = range(len(treatment_data))
    width = 0.35
    
    bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Women_Last_Author_Rate'], 
                   width, label='Control (Closed)', color='gray', alpha=0.8)
    bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Women_Last_Author_Rate'], 
                   width, label='Treatment (OA)', color='purple', alpha=0.8)
    
    ax.set_xlabel('Access Route Comparison', fontsize=12)
    ax.set_ylabel('% of Papers with Women as Last Author', fontsize=12)
    ax.set_title('Gender Equity: Senior Authorship by Access Route', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '08_women_last_author.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 9. Only-Women Author Teams
print("Creating Visualization 9: Only-women author teams")

women_only_comparison = []

if not green_matched.empty and not closed_matched_a.empty and 'only_women_authors' in green_matched.columns:
    women_only_comparison.extend([
        {'Access_Route': 'Green OA', 'Only_Women_Teams_Rate': green_matched['only_women_authors'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Green)', 'Only_Women_Teams_Rate': closed_matched_a['only_women_authors'].mean() * 100, 'Group': 'Control'}
    ])

if not published_matched.empty and not closed_matched_b.empty and 'only_women_authors' in published_matched.columns:
    women_only_comparison.extend([
        {'Access_Route': 'Published OA', 'Only_Women_Teams_Rate': published_matched['only_women_authors'].mean() * 100, 'Group': 'Treatment'},
        {'Access_Route': 'Closed (vs Published)', 'Only_Women_Teams_Rate': closed_matched_b['only_women_authors'].mean() * 100, 'Group': 'Control'}
    ])

if women_only_comparison:
    women_only_df = pd.DataFrame(women_only_comparison)
    
    # Save data for this visualization
    women_only_df.to_excel(os.path.join(data_export_path, '09_only_women_teams_data.xlsx'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    treatment_data = women_only_df[women_only_df['Group'] == 'Treatment']
    control_data = women_only_df[women_only_df['Group'] == 'Control']
    
    x_pos = range(len(treatment_data))
    width = 0.35
    
    bars1 = ax.bar([x - width/2 for x in x_pos], control_data['Only_Women_Teams_Rate'], 
                   width, label='Control (Closed)', color='gray', alpha=0.8)
    bars2 = ax.bar([x + width/2 for x in x_pos], treatment_data['Only_Women_Teams_Rate'], 
                   width, label='Treatment (OA)', color='magenta', alpha=0.8)
    
    ax.set_xlabel('Access Route Comparison', fontsize=12)
    ax.set_ylabel('% of Papers with Only-Women Author Teams', fontsize=12)
    ax.set_title('Team Composition Diversity by Access Route', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Green OA vs Closed', 'Published OA vs Closed'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, '09_only_women_teams.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create summary statistics table
print("Creating summary statistics table...")

summary_stats = {
    'Metric': [
        'Total Sample Size',
        'Green OA Only Papers',
        'Published OA Only Papers',
        'Closed Access Papers',
        'Green OA vs Closed - Matched Pairs',
        'Published OA vs Closed - Matched Pairs',
        'Mean Topic Persistence (Green OA)',
        'Mean Topic Persistence (Closed vs Green)',
        'Mean Topic Persistence (Published OA)',
        'Mean Topic Persistence (Closed vs Published)',
        'Mean FWCI (Green OA)',
        'Mean FWCI (Published OA)',
        'Industry Collaboration Rate (Green OA)',
        'Industry Collaboration Rate (Published OA)'
    ],
    'Value': [
        f"{len(complete_collection_df):,}",
        f"{complete_collection_df['green_oa_only'].sum():,}",
        f"{complete_collection_df['published_oa_only'].sum():,}",
        f"{complete_collection_df['closed_access'].sum():,}",
        f"{len(green_matched):,}" if not green_matched.empty else "0",
        f"{len(published_matched):,}" if not published_matched.empty else "0",
        f"{green_matched['topic_persistence_score'].mean():.3f}" if not green_matched.empty else "N/A",
        f"{closed_matched_a['topic_persistence_score'].mean():.3f}" if not closed_matched_a.empty else "N/A",
        f"{published_matched['topic_persistence_score'].mean():.3f}" if not published_matched.empty else "N/A",
        f"{closed_matched_b['topic_persistence_score'].mean():.3f}" if not closed_matched_b.empty else "N/A",
        f"{green_matched['fwci'].mean():.3f}" if not green_matched.empty else "N/A",
        f"{published_matched['fwci'].mean():.3f}" if not published_matched.empty else "N/A",
        f"{green_matched['science_industry_collaboration'].mean()*100:.1f}%" if not green_matched.empty else "N/A",
        f"{published_matched['science_industry_collaboration'].mean()*100:.1f}%" if not published_matched.empty else "N/A"
    ]
}

summary_df = pd.DataFrame(summary_stats)

# Save summary data to Excel
summary_df.to_excel(os.path.join(data_export_path, '10_summary_statistics_data.xlsx'), index=False)

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
ax.set_title('Impact of Open Access Routes on Topic Persistence Case Study: Key Statistics Summary', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(viz_path, '10_summary_statistics_table.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a comprehensive data export file with all visualization data
print("Creating comprehensive data export...")

# Create a workbook with multiple sheets
with pd.ExcelWriter(os.path.join(data_export_path, 'all_visualization_data.xlsx'), engine='openpyxl') as writer:
    oa_shares_by_year.to_excel(writer, sheet_name='01_oa_shares')
    citation_data.to_excel(writer, sheet_name='02_citations')
    fwci_df.to_excel(writer, sheet_name='03_fwci', index=False)
    persistence_df.to_excel(writer, sheet_name='04_persistence', index=False)
    if 'high_persistence_df' in locals():
        high_persistence_df.to_excel(writer, sheet_name='05_high_persistence', index=False)
    collab_df.to_excel(writer, sheet_name='06_collaboration', index=False)
    patent_df.to_excel(writer, sheet_name='07_patents', index=False)
    if 'women_last_df' in locals():
        women_last_df.to_excel(writer, sheet_name='08_women_last', index=False)
    if 'women_only_df' in locals():
        women_only_df.to_excel(writer, sheet_name='09_women_only', index=False)
    summary_df.to_excel(writer, sheet_name='10_summary', index=False)

print(f"\nAll visualizations completed and saved to: {viz_path}")
print(f"All data files saved to: {data_export_path}")
print("\nGenerated visualizations:")
print("01_oa_shares_over_time.png - OA adoption rates by year")
print("02_citation_count_over_time.png - Mean citations by access route over time")
print("03_fwci_comparison.png - Quality-adjusted citation impact comparison")
print("04_topic_persistence.png - Topic persistence score comparison")
print("05_high_persistence_rate.png - High-persistence topic rate comparison")
print("06_science_industry_collaboration.png - Industry collaboration rate comparison")
print("07_patent_citations.png - Patent citations comparison")
print("08_women_last_author.png - Women as last author rate comparison")
print("09_only_women_teams.png - Only-women author teams rate comparison")
print("10_summary_statistics_table.png - Key statistics summary table")

print(f"\nData files created:")
print("- Individual Excel files for each visualization (01-10)")
print("- all_visualization_data.xlsx - Comprehensive file with all data in separate sheets")
print(f"\nAll files saved in: {final_viz_path}")
