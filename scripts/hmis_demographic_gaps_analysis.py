import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from matplotlib.ticker import PercentFormatter
import ast  # For parsing list fields like RaceEthnicity and Gender

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the data
data_dir = "hmis_synthetic_data"

clients = pd.read_csv(f"{data_dir}/clients.csv")
enrollments = pd.read_csv(f"{data_dir}/enrollments.csv")
projects = pd.read_csv(f"{data_dir}/projects.csv")
living_situation = pd.read_csv(f"{data_dir}/living_situation.csv")

# Convert date columns to datetime
date_columns = {
    'clients': ['DOB', 'CreatedDate'],
    'enrollments': ['EntryDate', 'ExitDate'],
}

for df_name, columns in date_columns.items():
    df = locals()[df_name]
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

# Convert string representations of lists to actual lists for multi-select fields
def parse_list_field(field):
    if pd.isna(field):
        return []
    try:
        return ast.literal_eval(field)
    except (ValueError, SyntaxError):
        return []

clients['RaceEthnicity'] = clients['RaceEthnicity'].apply(parse_list_field)
clients['Gender'] = clients['Gender'].apply(parse_list_field)

# Merge project information into enrollments
enrollments = enrollments.merge(
    projects[['ProjectID', 'ProjectName', 'ProjectType', 'BedCount']], 
    on='ProjectID', how='left'
)

# Add living situation data to enrollments
enrollments = enrollments.merge(
    living_situation[['EnrollmentID', 'LivingSituation']], 
    on='EnrollmentID', how='left'
)

# Categorize situations
def categorize_situation(code):
    if pd.isna(code):
        return "Unknown/Still in program"
    
    # Permanent Housing
    if code in [410, 435, 421, 411, 422, 423, 426]:
        return "Permanent Housing"
    
    # Temporary Housing
    elif code in [302, 329, 314, 332, 312, 313, 336, 335]:
        return "Temporary Housing"
    
    # Institutional
    elif code in [215, 206, 207, 225, 204, 205]:
        return "Institutional"
    
    # Homeless
    elif code in [116, 101, 118]:
        return "Homeless"
    
    # Other/Unknown
    else:
        return "Other/Unknown"

# Add categorized columns
enrollments['DestinationCategory'] = enrollments['Destination'].apply(categorize_situation)
enrollments['PriorSituationCategory'] = enrollments['LivingSituation'].apply(categorize_situation)

# Calculate ages at entry
def calculate_age(dob, reference_date):
    if pd.isna(dob):
        return None
    return (reference_date - dob).days / 365.25

# Merge clients with enrollments
client_enrollments = enrollments.merge(clients[['PersonalID', 'DOB', 'RaceEthnicity', 'Gender', 'VeteranStatus']], 
                                     on='PersonalID', how='left')

# Calculate age at enrollment
client_enrollments['Age'] = client_enrollments.apply(
    lambda row: calculate_age(row['DOB'], row['EntryDate']), axis=1
)

# Age categories
def age_category(age):
    if pd.isna(age):
        return "Unknown"
    elif age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 55:
        return "35-54"
    else:
        return "55+"

client_enrollments['AgeCategory'] = client_enrollments['Age'].apply(age_category)

# Add entry year
client_enrollments['EntryYear'] = client_enrollments['EntryDate'].dt.year

# Intilialize empty global df for race and gender

# ==========================================
# DEMOGRAPHIC ANALYSIS
# ==========================================
def analyze_demographics():
    """
    Analyze the demographic composition of clients in the HMIS data
    and identify any disparities in outcomes or service access
    """
    print("\n===== DEMOGRAPHIC ANALYSIS =====\n")
    
    # Race/Ethnicity analysis
    race_codes = {
        1: "American Indian/Alaska Native/Indigenous",
        2: "Asian/Asian American",
        3: "Black/African American/African",
        4: "Native Hawaiian/Pacific Islander",
        5: "White",
        6: "Hispanic/Latina/e/o",
        7: "Middle Eastern/North African",
        8: "Client doesn't know",
        9: "Client prefers not to answer",
        99: "Data not collected"
    }
    
    # Extract race/ethnicity info
    def extract_race_counts(row):
        race_list = row['RaceEthnicity']
        race_counts = {}
        
        if not race_list:
            return {}
            
        # Handle "don't know/refused" options first
        if any(code in [8, 9, 99] for code in race_list):
            # Just pick the first unknown code
            for code in race_list:
                if code in [8, 9, 99]:
                    race_counts[race_codes[code]] = 1
                    break
        else:
            # Otherwise count each race/ethnicity
            for code in race_list:
                if code in race_codes:
                    race_counts[race_codes[code]] = 1
        
        return race_counts
    
    race_data = []
    
    for _, row in client_enrollments.iterrows():
        race_counts = extract_race_counts(row)
        
        for race, count in race_counts.items():
            race_data.append({
                'EnrollmentID': row['EnrollmentID'],
                'PersonalID': row['PersonalID'],
                'Race': race,
                'EntryYear': row['EntryYear'],
                'ProjectType': row['ProjectType'],
                'AgeCategory': row['AgeCategory'],
                'Destination': row['DestinationCategory'] if pd.notna(row['ExitDate']) else "Still in program"
            })
    
    race_df = pd.DataFrame(race_data)
    
    # Overall race/ethnicity distribution
    race_dist = race_df['Race'].value_counts()
    race_dist_pct = race_df['Race'].value_counts(normalize=True) * 100
    
    print("RACE/ETHNICITY DISTRIBUTION:")
    for race, percent in race_dist_pct.items():
        if race not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]:
            print(f"  {race}: {percent:.1f}% ({race_dist[race]} enrollments)")
    
    # Race/ethnicity by year
    race_by_year = pd.crosstab(race_df['EntryYear'], race_df['Race'], normalize='index') * 100
    
    print("\nRACE/ETHNICITY TRENDS BY YEAR (PERCENTAGES):")
    print(race_by_year.drop(columns=["Client doesn't know", "Client prefers not to answer", "Data not collected"], errors='ignore'))
    
    # Create visualization of race/ethnicity trends
    race_cols = [col for col in race_by_year.columns if col not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]]
    
    plt.figure(figsize=(14, 8))
    race_by_year[race_cols].plot(kind='bar', stacked=True)
    plt.title('Race/Ethnicity Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend(title='Race/Ethnicity')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/race_ethnicity_by_year.png')
    
    # Gender analysis
    gender_codes = {
        0: "Woman/Girl",
        1: "Man/Boy",
        2: "Culturally Specific Identity",
        3: "Different Identity",
        4: "Non-Binary",
        5: "Transgender",
        6: "Questioning",
        8: "Client doesn't know",
        9: "Client prefers not to answer",
        99: "Data not collected"
    }
    
    # Extract gender info
    def extract_gender_counts(row):
        gender_list = row['Gender']
        gender_counts = {}
        
        if not gender_list:
            return {}
            
        # Handle "don't know/refused" options first
        if any(code in [8, 9, 99] for code in gender_list):
            # Just pick the first unknown code
            for code in gender_list:
                if code in [8, 9, 99]:
                    gender_counts[gender_codes[code]] = 1
                    break
        else:
            # Check for transgender
            is_trans = 5 in gender_list
            
            # If transgender and also man/woman, create combined category
            if is_trans:
                if 0 in gender_list:
                    gender_counts["Transgender Woman"] = 1
                elif 1 in gender_list:
                    gender_counts["Transgender Man"] = 1
                else:
                    gender_counts[gender_codes[5]] = 1
            else:
                # Otherwise count each gender
                for code in gender_list:
                    if code in gender_codes:
                        gender_counts[gender_codes[code]] = 1
        
        return gender_counts
    
    gender_data = []
    
    for _, row in client_enrollments.iterrows():
        gender_counts = extract_gender_counts(row)
        
        for gender, count in gender_counts.items():
            gender_data.append({
                'EnrollmentID': row['EnrollmentID'],
                'PersonalID': row['PersonalID'],
                'Gender': gender,
                'EntryYear': row['EntryYear'],
                'ProjectType': row['ProjectType'],
                'AgeCategory': row['AgeCategory'],
                'Destination': row['DestinationCategory'] if pd.notna(row['ExitDate']) else "Still in program"
            })
    
    gender_df = pd.DataFrame(gender_data)
    
    # Overall gender distribution
    gender_dist = gender_df['Gender'].value_counts()
    gender_dist_pct = gender_df['Gender'].value_counts(normalize=True) * 100
    
    print("\nGENDER DISTRIBUTION:")
    for gender, percent in gender_dist_pct.items():
        if gender not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]:
            print(f"  {gender}: {percent:.1f}% ({gender_dist[gender]} enrollments)")
    
    # Gender by year
    gender_by_year = pd.crosstab(gender_df['EntryYear'], gender_df['Gender'], normalize='index') * 100
    
    print("\nGENDER TRENDS BY YEAR (PERCENTAGES):")
    gender_cols = [col for col in gender_by_year.columns if col not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]]
    print(gender_by_year[gender_cols])
    
    # Create visualization of gender trends
    plt.figure(figsize=(14, 8))
    gender_by_year[gender_cols].plot(kind='bar', stacked=True)
    plt.title('Gender Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend(title='Gender')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/gender_by_year.png')
    
    # Age distribution
    age_dist = client_enrollments['AgeCategory'].value_counts()
    age_dist_pct = client_enrollments['AgeCategory'].value_counts(normalize=True) * 100
    
    print("\nAGE DISTRIBUTION:")
    for age, percent in age_dist_pct.items():
        if age != "Unknown":
            print(f"  {age}: {percent:.1f}% ({age_dist[age]} enrollments)")
    
    # Age by year
    age_by_year = pd.crosstab(client_enrollments['EntryYear'], client_enrollments['AgeCategory'], normalize='index') * 100
    
    print("\nAGE TRENDS BY YEAR (PERCENTAGES):")
    age_cols = [col for col in age_by_year.columns if col != "Unknown"]
    print(age_by_year[age_cols])
    
    # Veteran status
    vet_dist = client_enrollments['VeteranStatus'].value_counts()
    vet_dist_pct = client_enrollments['VeteranStatus'].value_counts(normalize=True) * 100
    
    print("\nVETERAN STATUS DISTRIBUTION:")
    vet_labels = {0: "Non-Veteran", 1: "Veteran"}
    for status, percent in vet_dist_pct.items():
        if status in vet_labels:
            print(f"  {vet_labels[status]}: {percent:.1f}% ({vet_dist[status]} enrollments)")
    
    # Disabling condition
    disability_dist = client_enrollments['DisablingCondition'].value_counts()
    disability_dist_pct = client_enrollments['DisablingCondition'].value_counts(normalize=True) * 100
    
    print("\nDISABLING CONDITION DISTRIBUTION:")
    disability_labels = {0: "No Disability", 1: "Has Disability"}
    for status, percent in disability_dist_pct.items():
        if status in disability_labels:
            print(f"  {disability_labels[status]}: {percent:.1f}% ({disability_dist[status]} enrollments)")
    
    # Create demographic summary plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Age
    plt.subplot(2, 2, 1)
    age_dist_pct = client_enrollments['AgeCategory'].value_counts(normalize=True) * 100
    age_dist_pct = age_dist_pct[age_dist_pct.index != "Unknown"]
    age_dist_pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Age Distribution')
    plt.ylabel('')
    
    # Subplot 2: Gender (simplified)
    plt.subplot(2, 2, 2)
    gender_simple = gender_df['Gender'].copy()
    gender_simple = gender_simple.replace([
        "Client doesn't know", "Client prefers not to answer", "Data not collected"
    ], "Unknown")
    gender_dist_pct = gender_simple.value_counts(normalize=True) * 100
    gender_dist_pct = gender_dist_pct[gender_dist_pct.index != "Unknown"]
    gender_dist_pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.ylabel('')
    
    # Subplot 3: Race/Ethnicity (simplified)
    plt.subplot(2, 2, 3)
    race_simple = race_df['Race'].copy()
    race_simple = race_simple.replace([
        "Client doesn't know", "Client prefers not to answer", "Data not collected"
    ], "Unknown")
    race_dist_pct = race_simple.value_counts(normalize=True) * 100
    race_dist_pct = race_dist_pct[race_dist_pct.index != "Unknown"]
    race_dist_pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Race/Ethnicity Distribution')
    plt.ylabel('')
    
    # Subplot 4: Veteran Status
    plt.subplot(2, 2, 4)
    vet_dist_pct = client_enrollments['VeteranStatus'].value_counts(normalize=True) * 100
    vet_dist_pct = vet_dist_pct[vet_dist_pct.index.isin([0, 1])]
    vet_dist_pct.index = ["Non-Veteran", "Veteran"]
    vet_dist_pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Veteran Status')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('plots/demographic_summary.png')
    
    return {
        'race_df': race_df,
        'gender_df': gender_df,
        'age_dist': age_dist,
        'vet_dist': vet_dist,
        'disability_dist': disability_dist
    }

# ==========================================
# DEMOGRAPHIC OUTCOME DISPARITIES
# ==========================================
def analyze_outcome_disparities(race_df, gender_df):
    """
    Analyze disparities in outcomes based on demographic factors
    """
    print("\n===== DEMOGRAPHIC OUTCOME DISPARITIES =====\n")
    
    # Filter to exits only
    exits = client_enrollments[client_enrollments['ExitDate'].notna()].copy()
    
    # Define success as exit to permanent housing
    exits['Success'] = (exits['DestinationCategory'] == 'Permanent Housing').astype(int)
    
    # Calculate success rates by demographic factors
    
    # 1. By Age Category
    success_by_age = exits.groupby('AgeCategory')['Success'].agg(['mean', 'count'])
    success_by_age['mean'] = success_by_age['mean'] * 100  # Convert to percentage
    
    print("SUCCESS RATES BY AGE CATEGORY:")
    for age, row in success_by_age.iterrows():
        if age != "Unknown":
            print(f"  {age}: {row['mean']:.1f}% ({row['count']} exits)")
    
    # 2. By Race/Ethnicity (need to aggregate from race_df)
    race_exits = race_df[race_df['Destination'] != "Still in program"].copy()
    race_exits['Success'] = (race_exits['Destination'] == 'Permanent Housing').astype(int)
    
    success_by_race = race_exits.groupby('Race')['Success'].agg(['mean', 'count'])
    success_by_race['mean'] = success_by_race['mean'] * 100  # Convert to percentage
    
    print("\nSUCCESS RATES BY RACE/ETHNICITY:")
    for race, row in success_by_race.iterrows():
        if race not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]:
            print(f"  {race}: {row['mean']:.1f}% ({row['count']} exits)")
    
    # 3. By Gender
    gender_exits = gender_df[gender_df['Destination'] != "Still in program"].copy()
    gender_exits['Success'] = (gender_exits['Destination'] == 'Permanent Housing').astype(int)
    
    success_by_gender = gender_exits.groupby('Gender')['Success'].agg(['mean', 'count'])
    success_by_gender['mean'] = success_by_gender['mean'] * 100  # Convert to percentage
    
    print("\nSUCCESS RATES BY GENDER:")
    for gender, row in success_by_gender.iterrows():
        if gender not in ["Client doesn't know", "Client prefers not to answer", "Data not collected"]:
            print(f"  {gender}: {row['mean']:.1f}% ({row['count']} exits)")
    
    # 4. By Veteran Status
    success_by_veteran = exits.groupby('VeteranStatus')['Success'].agg(['mean', 'count'])
    success_by_veteran['mean'] = success_by_veteran['mean'] * 100  # Convert to percentage
    
    print("\nSUCCESS RATES BY VETERAN STATUS:")
    vet_labels = {0: "Non-Veteran", 1: "Veteran"}
    for status, row in success_by_veteran.iterrows():
        if status in vet_labels:
            print(f"  {vet_labels[status]}: {row['mean']:.1f}% ({row['count']} exits)")
    
    # 5. By Disability Status
    success_by_disability = exits.groupby('DisablingCondition')['Success'].agg(['mean', 'count'])
    success_by_disability['mean'] = success_by_disability['mean'] * 100  # Convert to percentage
    
    print("\nSUCCESS RATES BY DISABILITY STATUS:")
    disability_labels = {0: "No Disability", 1: "Has Disability"}
    for status, row in success_by_disability.iterrows():
        if status in disability_labels:
            print(f"  {disability_labels[status]}: {row['mean']:.1f}% ({row['count']} exits)")
    
    # Create visualizations of outcome disparities
    
    # Plot 1: Success by Age
    plt.figure(figsize=(12, 6))
    age_data = success_by_age.reset_index()
    age_data = age_data[age_data['AgeCategory'] != "Unknown"]
    sns.barplot(x='AgeCategory', y='mean', data=age_data, palette='viridis')
    plt.title('Permanent Housing Success Rates by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Success Rate (%)')
    plt.axhline(exits['Success'].mean() * 100, color='r', linestyle='--', 
               label=f'Overall Average: {exits["Success"].mean() * 100:.1f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/success_by_age.png')
    
    # Plot 2: Success by Race/Ethnicity
    plt.figure(figsize=(12, 6))
    race_data = success_by_race.reset_index()
    race_data = race_data[~race_data['Race'].isin(["Client doesn't know", "Client prefers not to answer", "Data not collected"])]
    sns.barplot(x='Race', y='mean', data=race_data, palette='viridis')
    plt.title('Permanent Housing Success Rates by Race/Ethnicity')
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('Success Rate (%)')
    plt.axhline(race_exits['Success'].mean() * 100, color='r', linestyle='--', 
               label=f'Overall Average: {race_exits["Success"].mean() * 100:.1f}%')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/success_by_race.png')
    
    # Plot 3: Success by Gender
    plt.figure(figsize=(12, 6))
    gender_data = success_by_gender.reset_index()
    gender_data = gender_data[~gender_data['Gender'].isin(["Client doesn't know", "Client prefers not to answer", "Data not collected"])]
    sns.barplot(x='Gender', y='mean', data=gender_data, palette='viridis')
    plt.title('Permanent Housing Success Rates by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Success Rate (%)')
    plt.axhline(gender_exits['Success'].mean() * 100, color='r', linestyle='--', 
               label=f'Overall Average: {gender_exits["Success"].mean() * 100:.1f}%')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/success_by_gender.png')
    
    return {
        'success_by_age': success_by_age,
        'success_by_race': success_by_race,
        'success_by_gender': success_by_gender,
        'success_by_veteran': success_by_veteran,
        'success_by_disability': success_by_disability
    }

# ==========================================
# SERVICE GAPS ANALYSIS
# ==========================================
def analyze_service_gaps(race_df, gender_df):
    """
    Analyze potential gaps in services based on demographic needs, 
    geographic distribution, and service availability
    """
    print("\n===== SERVICE GAPS ANALYSIS =====\n")
    
    # 1. Analyze project types by demographic groups
    print("PROJECT TYPE UTILIZATION BY DEMOGRAPHIC GROUPS:")
    
    # Project type distribution by age
    proj_by_age = pd.crosstab(
        client_enrollments['AgeCategory'], 
        client_enrollments['ProjectType'],
        normalize='index'
    ) * 100
    
    # Rename project types for readability
    proj_type_names = {
        0: 'ES - Entry Exit',
        1: 'ES - Night-by-Night',
        2: 'Transitional Housing',
        3: 'Permanent Supportive Housing',
        13: 'Rapid Re-Housing'
    }
    
    proj_by_age.columns = [proj_type_names.get(col, f'Other ({col})') for col in proj_by_age.columns]
    
    print("\nPROJECT TYPE UTILIZATION BY AGE (PERCENTAGES):")
    print(proj_by_age[proj_by_age.index != "Unknown"])
    
    # Project type distribution by race/ethnicity
    race_by_proj = pd.crosstab(
        race_df['Race'], 
        race_df['ProjectType'],
        normalize='index'
    ) * 100
    
    race_by_proj.columns = [proj_type_names.get(col, f'Other ({col})') for col in race_by_proj.columns]
    
    print("\nPROJECT TYPE UTILIZATION BY RACE/ETHNICITY (PERCENTAGES):")
    print(race_by_proj[~race_by_proj.index.isin(["Client doesn't know", "Client prefers not to answer", "Data not collected"])])
    
    # Project type by veteran status
    vet_by_proj = pd.crosstab(
        client_enrollments['VeteranStatus'], 
        client_enrollments['ProjectType'],
        normalize='index'
    ) * 100

    vet_labels = {0: "Non-Veteran", 1: "Veteran"}
    
    vet_by_proj.columns = [proj_type_names.get(col, f'Other ({col})') for col in vet_by_proj.columns]
    vet_by_proj.index = [vet_labels.get(idx, str(idx)) for idx in vet_by_proj.index]
    
    print("\nPROJECT TYPE UTILIZATION BY VETERAN STATUS (PERCENTAGES):")
    print(vet_by_proj)
    
    # Project type by disability status
    disability_by_proj = pd.crosstab(
        client_enrollments['DisablingCondition'], 
        client_enrollments['ProjectType'],
        normalize='index'
    ) * 100

    disability_labels = {0: "No Disability", 1: "Has Disability"}
    
    disability_by_proj.columns = [proj_type_names.get(col, f'Other ({col})') for col in disability_by_proj.columns]
    disability_by_proj.index = [disability_labels.get(idx, str(idx)) for idx in disability_by_proj.index]
    
    print("\nPROJECT TYPE UTILIZATION BY DISABILITY STATUS (PERCENTAGES):")
    print(disability_by_proj)
    
    # 2. Analyze prior living situations vs. available services
    
    # Count of prior living situations
    prior_situations = client_enrollments['PriorSituationCategory'].value_counts()
    prior_situations_pct = client_enrollments['PriorSituationCategory'].value_counts(normalize=True) * 100
    
    print("\nPRIOR LIVING SITUATION DISTRIBUTION:")
    for situation, percent in prior_situations_pct.items():
        print(f"  {situation}: {percent:.1f}% ({prior_situations[situation]} enrollments)")
    
    # Project capacity relative to need
    # For this synthetic data, we'll use project bed counts as a measure of capacity
    
    project_counts = projects.groupby('ProjectType')['BedCount'].sum()
    project_counts.index = [proj_type_names.get(idx, f'Other ({idx})') for idx in project_counts.index]
    
    print("\nPROJECT CAPACITY BY TYPE:")
    for proj_type, beds in project_counts.items():
        print(f"  {proj_type}: {beds} beds")
    
    # Create visualization comparing needs to resources
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Prior situation distribution
    plt.subplot(2, 1, 1)
    prior_situations.plot(kind='bar')
    plt.title('Distribution of Prior Living Situations')
    plt.xlabel('Prior Situation Category')
    plt.ylabel('Number of Enrollments')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 2: Project capacity distribution
    plt.subplot(2, 1, 2)
    project_counts.plot(kind='bar')
    plt.title('Bed Capacity by Project Type')
    plt.xlabel('Project Type')
    plt.ylabel('Number of Beds')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('plots/needs_vs_resources.png')
    
    # 3. Analyze length of stay distribution by project type
    
    # Filter to exits only
    exits = client_enrollments[client_enrollments['ExitDate'].notna()].copy()
    
    # Calculate length of stay
    exits['LengthOfStay'] = (exits['ExitDate'] - exits['EntryDate']).dt.days
    
    # Calculate bed utilization days by project type
    los_by_project = exits.groupby('ProjectType')['LengthOfStay'].agg(['mean', 'median', 'sum', 'count'])
    los_by_project.index = [proj_type_names.get(idx, f'Other ({idx})') for idx in los_by_project.index]
    
    print("\nTOTAL BED UTILIZATION AND LENGTH OF STAY BY PROJECT TYPE:")
    for proj_type, stats in los_by_project.iterrows():
        print(f"  {proj_type}:")
        print(f"    - Average length of stay: {stats['mean']:.1f} days")
        print(f"    - Median length of stay: {stats['median']:.1f} days")
        print(f"    - Total bed-days utilized: {stats['sum']:.0f}")
        print(f"    - Number of exits: {stats['count']}")
    
    # 4. Identify potential service gaps by demographic group
    
    # Combine prior homeless situations and veteran status
    homeless_counts = client_enrollments[
        client_enrollments['PriorSituationCategory'] == 'Homeless'
    ].groupby('VeteranStatus').size()
    
    veteran_proj_counts = client_enrollments[
        (client_enrollments['VeteranStatus'] == 1) &
        (client_enrollments['ProjectType'].isin([3, 13]))  # PSH and RRH
    ].groupby('ProjectType').size()
    
    vet_beds = projects[
        projects['ProjectName'].str.contains('Veteran', case=False, na=False)
    ]['BedCount'].sum()
    
    print("\nPOTENTIAL SERVICE GAPS:")
    if 1 in homeless_counts:
        print(f"  Homeless veterans identified: {homeless_counts[1]}")
        print(f"  Veteran-specific beds available: {vet_beds}")
    
    # Age-specific needs
    youth_counts = client_enrollments[
        client_enrollments['AgeCategory'].isin(['Under 18', '18-24'])
    ].groupby('PriorSituationCategory').size()
    
    youth_beds = projects[
        projects['ProjectName'].str.contains('Youth|Young Adult', case=False, na=False)
    ]['BedCount'].sum()
    
    print("\n  Youth and young adults by prior situation:")
    for situation, count in youth_counts.items():
        print(f"    - {situation}: {count}")
    print(f"  Youth-specific beds available: {youth_beds}")
    
    # Disability status and PSH
    disabled_counts = client_enrollments[
        client_enrollments['DisablingCondition'] == 1
    ].groupby('PriorSituationCategory').size()
    
    psh_beds = projects[
        projects['ProjectType'] == 3  # PSH
    ]['BedCount'].sum()
    
    print("\n  Clients with disabilities by prior situation:")
    for situation, count in disabled_counts.items():
        print(f"    - {situation}: {count}")
    print(f"  Permanent Supportive Housing beds available: {psh_beds}")
    
    # Create visualization of potential service gaps
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Veterans - needs vs. resources
    plt.subplot(3, 1, 1)
    data = pd.Series({
        'Homeless Veterans': homeless_counts.get(1, 0),
        'Veteran-Specific Beds': vet_beds
    })
    data.plot(kind='bar')
    plt.title('Veterans - Needs vs. Resources')
    plt.ylabel('Count')
    
    # Plot 2: Youth - needs vs. resources
    plt.subplot(3, 1, 2)
    data = pd.Series({
        'Homeless Youth/Young Adults': youth_counts.get('Homeless', 0),
        'Youth-Specific Beds': youth_beds
    })
    data.plot(kind='bar')
    plt.title('Youth - Needs vs. Resources')
    plt.ylabel('Count')
    
    # Plot 3: Disability - needs vs. resources
    plt.subplot(3, 1, 3)
    data = pd.Series({
        'Homeless with Disabilities': disabled_counts.get('Homeless', 0),
        'PSH Beds': psh_beds
    })
    data.plot(kind='bar')
    plt.title('Clients with Disabilities - Needs vs. Resources')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('plots/service_gaps.png')
    
    return {
        'proj_by_age': proj_by_age,
        'race_by_proj': race_by_proj,
        'vet_by_proj': vet_by_proj,
        'disability_by_proj': disability_by_proj,
        'los_by_project': los_by_project
    }

# ==========================================
# MAIN ANALYSIS EXECUTION
# ==========================================
def run_demographic_analyses():
    print("HMIS DEMOGRAPHIC AND SERVICE GAPS ANALYSIS")
    print("==========================================\n")
    
    # Run all analyses
    demographic_data = analyze_demographics()
    disparity_data = analyze_outcome_disparities(demographic_data['race_df'],
                                                 demographic_data['gender_df'])
    gap_data = analyze_service_gaps(demographic_data['race_df'],
                                    demographic_data['gender_df'])
    
    print("\n===== ANALYSIS COMPLETE =====")
    print("Additional plots have been saved to the 'plots' directory")
    
    return {
        'demographic_data': demographic_data,
        'disparity_data': disparity_data,
        'gap_data': gap_data
    }

# Run the analysis
if __name__ == "__main__":
    results = run_demographic_analyses()