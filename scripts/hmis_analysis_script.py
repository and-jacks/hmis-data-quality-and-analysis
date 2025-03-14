import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from IPython.display import display, HTML

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)
os.makedirs('plots', exist_ok=True)  # Create directory for saving plots

# Load the data
data_dir = "hmis_synthetic_data"

organizations = pd.read_csv(f"{data_dir}/organizations.csv")
projects = pd.read_csv(f"{data_dir}/projects.csv")
clients = pd.read_csv(f"{data_dir}/clients.csv")
enrollments = pd.read_csv(f"{data_dir}/enrollments.csv")
living_situation = pd.read_csv(f"{data_dir}/living_situation.csv")

# Convert date columns to datetime
date_columns = {
    'projects': ['Operating_StartDate', 'Operating_EndDate'],
    'clients': ['DOB', 'CreatedDate'],
    'enrollments': ['EntryDate', 'ExitDate'],
    'living_situation': ['DateHomelessStarted']
}

for df_name, columns in date_columns.items():
    df = locals()[df_name]
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

# Function to get friendly names for codes
def get_destination_name(code):
    destinations = {
        # Homeless Situations
        116: 'Place not meant for habitation',
        101: 'Emergency shelter',
        118: 'Safe Haven',
        
        # Institutional Situations
        215: 'Foster care home',
        206: 'Hospital/non-psychiatric',
        207: 'Jail/prison',
        225: 'Long-term care facility',
        204: 'Psychiatric facility',
        205: 'Substance abuse facility',
        
        # Temporary Housing
        302: 'Transitional housing',
        329: 'Residential project',
        314: 'Hotel/motel (no voucher)',
        332: 'Host Home (non-crisis)',
        312: 'Family, temporary',
        313: 'Friends, temporary',
        327: 'HOPWA to HOPWA TH',
        336: 'Staying with friend',
        335: 'Staying with family',
        
        # Permanent Housing
        422: 'Family, permanent',
        423: 'Friends, permanent',
        426: 'HOPWA to HOPWA PH',
        410: 'Rental, no subsidy',
        435: 'Rental, with subsidy',
        421: 'Owned, with subsidy',
        411: 'Owned, no subsidy',
        
        # Other
        30: 'No exit interview',
        17: 'Other',
        24: 'Deceased',
        37: 'Worker unable to determine',
        8: "Client doesn't know",
        9: "Client prefers not to answer",
        99: "Data not collected"
    }
    return destinations.get(code, f"Unknown ({code})")

def get_project_type_name(code):
    project_types = {
        0: 'Emergency Shelter - Entry Exit',
        1: 'Emergency Shelter - Night-by-Night',
        2: 'Transitional Housing',
        3: 'PH - Permanent Supportive Housing',
        13: 'PH - Rapid Re-Housing'
    }
    return project_types.get(code, f"Other ({code})")

# Add descriptive names
enrollments['DestinationName'] = enrollments['Destination'].apply(
    lambda x: get_destination_name(x) if pd.notna(x) else "Still in program"
)

living_situation['LivingSituationName'] = living_situation['LivingSituation'].apply(
    lambda x: get_destination_name(x) if pd.notna(x) else "Unknown"
)

# Merge project information into enrollments
enrollments = enrollments.merge(
    projects[['ProjectID', 'ProjectName', 'ProjectType', 'BedCount']], 
    on='ProjectID', how='left'
)

enrollments['ProjectTypeName'] = enrollments['ProjectType'].apply(get_project_type_name)

# Add year and quarter information for trend analysis
enrollments['EntryYear'] = enrollments['EntryDate'].dt.year
enrollments['EntryQuarter'] = enrollments['EntryDate'].dt.to_period('Q')
enrollments['EntryMonth'] = enrollments['EntryDate'].dt.to_period('M')

# For exits that have occurred
mask = enrollments['ExitDate'].notna()
enrollments.loc[mask, 'ExitYear'] = enrollments.loc[mask, 'ExitDate'].dt.year
enrollments.loc[mask, 'ExitQuarter'] = enrollments.loc[mask, 'ExitDate'].dt.to_period('Q')
enrollments.loc[mask, 'ExitMonth'] = enrollments.loc[mask, 'ExitDate'].dt.to_period('M')

# Calculate length of stay for exited clients
enrollments.loc[mask, 'LengthOfStay'] = (
    enrollments.loc[mask, 'ExitDate'] - enrollments.loc[mask, 'EntryDate']
).dt.days

# Add living situation data to enrollments
enrollments = enrollments.merge(
    living_situation[['EnrollmentID', 'LivingSituation', 'LivingSituationName']], 
    on='EnrollmentID', how='left'
)

# Function to categorize a code into a broader category
def categorize_destination(code):
    if pd.isna(code):
        return "Still in program"
    
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
enrollments['DestinationCategory'] = enrollments['Destination'].apply(categorize_destination)

# Similarly categorize living situations
enrollments['PriorSituationCategory'] = enrollments['LivingSituation'].apply(categorize_destination)

# ==========================================
# DATA QUALITY ASSESSMENT
# ==========================================
def assess_data_quality():
    print("\n===== DATA QUALITY ASSESSMENT =====\n")
    
    # 1. Check for missing values in key fields
    print("MISSING VALUES IN KEY FIELDS:")
    missing_data = {
        'Client Demographics': clients[['PersonalID', 'FirstName', 'LastName', 'SSN', 'DOB', 'Gender', 'RaceEthnicity']].isna().sum(),
        'Enrollment Data': enrollments[['PersonalID', 'ProjectID', 'EntryDate', 'HouseholdID', 'RelationshipToHoH']].isna().sum(),
        'Living Situation': living_situation[['EnrollmentID', 'LivingSituation', 'LengthOfStay']].isna().sum()
    }
    
    for category, missing in missing_data.items():
        print(f"\n{category}:")
        for field, count in missing.items():
            percent = (count / len(missing.index)) * 100
            print(f"  - {field}: {count} missing ({percent:.1f}%)")
    
    # 2. Data integrity - check for logical inconsistencies
    print("\nLOGICAL INCONSISTENCIES:")
    
    # Check if exit dates are before entry dates
    invalid_exits = enrollments[(enrollments['ExitDate'].notna()) & 
                              (enrollments['ExitDate'] < enrollments['EntryDate'])]
    print(f"  - Exit dates before entry dates: {len(invalid_exits)} records")
    
    # Check if veterans are under 18
    minor_veterans = clients[(clients['VeteranStatus'] == 1) & 
                           ((datetime.now() - clients['DOB']).dt.days / 365.25 < 18)]
    print(f"  - Veterans under 18 years old: {len(minor_veterans)} records")
    
    # Check for duplicate enrollments (same person, same project, overlapping dates)
    # This requires more complex analysis - simplified version here
    enrollment_counts = enrollments.groupby(['PersonalID', 'ProjectID']).size()
    potential_duplicates = enrollment_counts[enrollment_counts > 1].reset_index()
    print(f"  - Potential duplicate enrollments (same person, same project): {len(potential_duplicates)} instances")
    
    # 3. Data completion rates for key fields
    print("\nDATA COMPLETION RATES FOR CRITICAL FIELDS:")
    
    # SSN data quality
    ssn_quality = clients['SSNDataQuality'].value_counts(normalize=True) * 100
    print("  - SSN Data Quality:")
    for quality, percent in ssn_quality.items():
        quality_desc = {1: "Full SSN", 2: "Partial SSN", 8: "Client doesn't know", 
                        9: "Client prefers not to answer", 99: "Data not collected"}.get(quality, "Unknown")
        print(f"    {quality_desc}: {percent:.1f}%")
    
    # DOB data quality
    dob_quality = clients['DOBDataQuality'].value_counts(normalize=True) * 100
    print("  - DOB Data Quality:")
    for quality, percent in dob_quality.items():
        quality_desc = {1: "Full DOB", 2: "Partial DOB", 8: "Client doesn't know", 
                        9: "Client prefers not to answer", 99: "Data not collected"}.get(quality, "Unknown")
        print(f"    {quality_desc}: {percent:.1f}%")
    
    # Destination data quality for exits
    dest_quality = enrollments[enrollments['ExitDate'].notna()]['Destination'].apply(
        lambda x: "Valid Destination" if pd.notna(x) and x not in [8, 9, 99, 30] else "Unknown/Missing"
    ).value_counts(normalize=True) * 100
    print("  - Exit Destination Data Quality:")
    for quality, percent in dest_quality.items():
        print(f"    {quality}: {percent:.1f}%")
        
    # 4. Check capacity utilization
    print("\nCAPACITY UTILIZATION BY PROJECT TYPE:")
    
    # Get count of active clients by project and date
    def calculate_utilization(row):
        # Filter enrollments for this project that were active on the 15th of each month
        project_id = row['ProjectID']
        bed_count = row['BedCount']
        
        utilization = []
        
        # Calculate for a sample of dates
        sample_dates = pd.date_range(start='2022-01-15', end='2024-06-15', freq='MS')
        
        for date in sample_dates:
            active_count = len(enrollments[
                (enrollments['ProjectID'] == project_id) & 
                (enrollments['EntryDate'] <= date) & 
                ((enrollments['ExitDate'] > date) | (enrollments['ExitDate'].isna()))
            ])
            
            util_rate = (active_count / bed_count) * 100 if bed_count > 0 else 0
            utilization.append((date, util_rate))
        
        return utilization
    
    # Calculate for a few sample projects
    project_sample = projects.head(3)
    for _, project in project_sample.iterrows():
        util = calculate_utilization(project)
        avg_util = sum([u[1] for u in util]) / len(util)
        print(f"  - {project['ProjectName']} (Type: {get_project_type_name(project['ProjectType'])}): {avg_util:.1f}% average utilization")
    
    # Return dataframes with potential data quality issues for further investigation
    return {
        'invalid_exits': invalid_exits,
        'minor_veterans': minor_veterans,
        'potential_duplicates': potential_duplicates
    }

# ==========================================
# TRENDS ANALYSIS (2022-2024)
# ==========================================
def analyze_trends():
    print("\n===== TRENDS ANALYSIS (2022-2024) =====\n")
    
    # 1. Overall homelessness trends by quarter
    entry_counts = enrollments.groupby(['EntryYear', 'EntryQuarter']).size().reset_index(name='EntryCount')
    entry_counts['YearQuarter'] = entry_counts['EntryQuarter'].astype(str)
    
    print("QUARTERLY ENROLLMENT TRENDS:")
    print(entry_counts)
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=entry_counts, x='YearQuarter', y='EntryCount')
    plt.title('Homeless Program Enrollments by Quarter (2022-2024)')
    plt.xlabel('Year-Quarter')
    plt.ylabel('Number of Enrollments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/quarterly_enrollment_trends.png')
    
    # 2. Trends by project type
    pt_trends = enrollments.groupby(['EntryYear', 'ProjectTypeName']).size().reset_index(name='Count')
    pt_pivot = pt_trends.pivot(index='EntryYear', columns='ProjectTypeName', values='Count').fillna(0)
    
    print("\nENROLLMENTS BY PROJECT TYPE AND YEAR:")
    print(pt_pivot)
    
    plt.figure(figsize=(14, 7))
    pt_pivot.plot(kind='bar', stacked=True)
    plt.title('Enrollments by Project Type (2022-2024)')
    plt.xlabel('Year')
    plt.ylabel('Number of Enrollments')
    plt.legend(title='Project Type')
    plt.tight_layout()
    plt.savefig('plots/project_type_trends.png')
    
    # 3. Length of stay trends
    los_yearly = enrollments[enrollments['LengthOfStay'].notna()].groupby('ExitYear')['LengthOfStay'].agg(
        ['mean', 'median', 'min', 'max']
    ).reset_index()
    
    print("\nLENGTH OF STAY TRENDS BY YEAR:")
    print(los_yearly)
    
    # 4. Demographic shifts over time
    # Age distribution by year
    def calculate_age(dob, reference_date):
        if pd.isna(dob):
            return None
        return (reference_date - dob).days / 365.25
    
    # Merge clients with enrollments
    client_enrollments = enrollments.merge(clients[['PersonalID', 'DOB']], on='PersonalID', how='left')
    
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
    
    age_trends = client_enrollments.groupby(['EntryYear', 'AgeCategory']).size().reset_index(name='Count')
    age_pivot = age_trends.pivot(index='EntryYear', columns='AgeCategory', values='Count').fillna(0)
    
    print("\nAGE DISTRIBUTION TRENDS:")
    print(age_pivot)
    
    plt.figure(figsize=(14, 7))
    age_pivot.plot(kind='bar', stacked=True)
    plt.title('Age Distribution of Clients by Entry Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Clients')
    plt.legend(title='Age Category')
    plt.tight_layout()
    plt.savefig('plots/age_distribution_trends.png')
    
    return {
        'entry_counts': entry_counts,
        'pt_trends': pt_trends,
        'los_yearly': los_yearly,
        'age_trends': age_trends
    }

# ==========================================
# DISCHARGE DESTINATION ANALYSIS
# ==========================================
def analyze_discharge_destinations():
    print("\n===== DISCHARGE DESTINATION ANALYSIS =====\n")
    
    # Filter to only include clients who have exited
    exited_enrollments = enrollments[enrollments['ExitDate'].notna()].copy()
    
    # 1. Overall distribution of discharge destinations
    dest_counts = exited_enrollments['DestinationCategory'].value_counts(normalize=True) * 100
    print("OVERALL DISCHARGE DESTINATION CATEGORIES:")
    for dest, percent in dest_counts.items():
        print(f"  - {dest}: {percent:.1f}%")
    
    plt.figure(figsize=(12, 8))
    dest_counts.plot(kind='bar', color=sns.color_palette("viridis", len(dest_counts)))
    plt.title('Overall Discharge Destination Categories')
    plt.xlabel('Destination Category')
    plt.ylabel('Percentage of Exits')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/overall_discharge_destinations.png')
    
    # 2. Destination by project type
    dest_by_project = pd.crosstab(
        exited_enrollments['ProjectTypeName'], 
        exited_enrollments['DestinationCategory'],
        normalize='index'
    ) * 100
    
    print("\nDISCHARGE DESTINATIONS BY PROJECT TYPE (PERCENTAGES):")
    print(dest_by_project)
    
    plt.figure(figsize=(14, 10))
    dest_by_project.plot(kind='bar', stacked=True)
    plt.title('Discharge Destinations by Project Type')
    plt.xlabel('Project Type')
    plt.ylabel('Percentage')
    plt.legend(title='Destination Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/destinations_by_project_type.png')
    
    # 3. Top specific destinations
    specific_dests = exited_enrollments['DestinationName'].value_counts(normalize=True).head(10) * 100
    
    print("\nTOP 10 SPECIFIC DISCHARGE DESTINATIONS:")
    for dest, percent in specific_dests.items():
        print(f"  - {dest}: {percent:.1f}%")
    
    # 4. Destinations by year (trends)
    dest_by_year = pd.crosstab(
        exited_enrollments['ExitYear'], 
        exited_enrollments['DestinationCategory'],
        normalize='index'
    ) * 100
    
    print("\nDISCHARGE DESTINATION TRENDS BY YEAR (PERCENTAGES):")
    print(dest_by_year)
    
    plt.figure(figsize=(14, 8))
    dest_by_year.plot(kind='bar', stacked=True)
    plt.title('Discharge Destination Trends by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend(title='Destination Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/destinations_by_year.png')
    
    # 5. Success rates (Permanent Housing) by project
    success_rates = exited_enrollments.groupby('ProjectName').apply(
        lambda x: (x['DestinationCategory'] == 'Permanent Housing').mean() * 100
    ).sort_values(ascending=False)
    
    print("\nPERMANENT HOUSING SUCCESS RATES BY PROJECT:")
    for project, rate in success_rates.items():
        print(f"  - {project}: {rate:.1f}%")
    
    # Create a success rate by project type chart
    success_by_type = exited_enrollments.groupby('ProjectTypeName').apply(
        lambda x: (x['DestinationCategory'] == 'Permanent Housing').mean() * 100
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    success_by_type.plot(kind='bar', color='darkgreen')
    plt.title('Permanent Housing Success Rates by Project Type')
    plt.xlabel('Project Type')
    plt.ylabel('Percentage to Permanent Housing')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/success_rates_by_project_type.png')
    
    return {
        'dest_counts': dest_counts,
        'dest_by_project': dest_by_project,
        'specific_dests': specific_dests,
        'dest_by_year': dest_by_year,
        'success_rates': success_rates
    }

# ==========================================
# PRIOR LIVING SITUATION ANALYSIS
# ==========================================
def analyze_prior_living_situations():
    print("\n===== PRIOR LIVING SITUATION ANALYSIS =====\n")
    
    # 1. Overall distribution of prior living situations
    prior_counts = enrollments['PriorSituationCategory'].value_counts(normalize=True) * 100
    print("OVERALL PRIOR LIVING SITUATION CATEGORIES:")
    for situation, percent in prior_counts.items():
        print(f"  - {situation}: {percent:.1f}%")
    
    plt.figure(figsize=(12, 8))
    prior_counts.plot(kind='bar', color=sns.color_palette("mako", len(prior_counts)))
    plt.title('Overall Prior Living Situation Categories')
    plt.xlabel('Living Situation Category')
    plt.ylabel('Percentage of Enrollments')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/overall_prior_situations.png')
    
    # 2. Prior situation by project type
    prior_by_project = pd.crosstab(
        enrollments['ProjectTypeName'], 
        enrollments['PriorSituationCategory'],
        normalize='index'
    ) * 100
    
    print("\nPRIOR LIVING SITUATIONS BY PROJECT TYPE (PERCENTAGES):")
    print(prior_by_project)
    
    plt.figure(figsize=(14, 10))
    prior_by_project.plot(kind='bar', stacked=True)
    plt.title('Prior Living Situations by Project Type')
    plt.xlabel('Project Type')
    plt.ylabel('Percentage')
    plt.legend(title='Prior Situation Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/prior_situations_by_project_type.png')
    
    # 3. Top specific prior situations
    specific_prior = enrollments['LivingSituationName'].value_counts(normalize=True).head(10) * 100
    
    print("\nTOP 10 SPECIFIC PRIOR LIVING SITUATIONS:")
    for situation, percent in specific_prior.items():
        print(f"  - {situation}: {percent:.1f}%")
    
    # 4. Prior situations by year (trends)
    prior_by_year = pd.crosstab(
        enrollments['EntryYear'], 
        enrollments['PriorSituationCategory'],
        normalize='index'
    ) * 100
    
    print("\nPRIOR LIVING SITUATION TRENDS BY YEAR (PERCENTAGES):")
    print(prior_by_year)
    
    plt.figure(figsize=(14, 8))
    prior_by_year.plot(kind='bar', stacked=True)
    plt.title('Prior Living Situation Trends by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend(title='Prior Situation Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/prior_situations_by_year.png')
    
    # 5. Relationship between prior situation and destination
    # Only for exited clients
    exited = enrollments[enrollments['ExitDate'].notna()]
    prior_to_dest = pd.crosstab(
        exited['PriorSituationCategory'], 
        exited['DestinationCategory'],
        normalize='index'
    ) * 100
    
    print("\nRELATIONSHIP BETWEEN PRIOR SITUATION AND DESTINATION (PERCENTAGES):")
    print(prior_to_dest)
    
    plt.figure(figsize=(14, 10))
    prior_to_dest.plot(kind='bar', stacked=True)
    plt.title('Relationship Between Prior Living Situations and Destinations')
    plt.xlabel('Prior Living Situation')
    plt.ylabel('Percentage')
    plt.legend(title='Destination Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()
    plt.savefig('plots/prior_to_destination.png')
    
    return {
        'prior_counts': prior_counts,
        'prior_by_project': prior_by_project,
        'specific_prior': specific_prior,
        'prior_by_year': prior_by_year,
        'prior_to_dest': prior_to_dest
    }

# ==========================================
# SHELTER CAPACITY ANALYSIS
# ==========================================
def analyze_shelter_capacity():
    print("\n===== SHELTER CAPACITY ANALYSIS =====\n")
    
    # Filter for emergency shelters and transitional housing
    shelter_projects = projects[projects['ProjectType'].isin([0, 1, 2])].copy()
    
    print(f"Total shelter and transitional housing projects: {len(shelter_projects)}")
    print(f"Total bed capacity: {shelter_projects['BedCount'].sum()}")
    
    # Group by project type
    capacity_by_type = shelter_projects.groupby('ProjectType')['BedCount'].agg(['sum', 'mean', 'count'])
    capacity_by_type.index = capacity_by_type.index.map(get_project_type_name)
    
    print("\nCAPACITY BY PROJECT TYPE:")
    print(capacity_by_type)
    
    # Calculate utilization over time (example for a few dates)
    sample_dates = [
        datetime(2022, 1, 15),
        datetime(2022, 7, 15),
        datetime(2023, 1, 15),
        datetime(2023, 7, 15),
        datetime(2024, 1, 15),
        datetime(2024, 6, 15)
    ]
    
    utilization_data = []
    
    for date in sample_dates:
        for _, project in shelter_projects.iterrows():
            # Count active clients on this date
            active_count = len(enrollments[
                (enrollments['ProjectID'] == project['ProjectID']) & 
                (enrollments['EntryDate'] <= date) & 
                ((enrollments['ExitDate'] > date) | (enrollments['ExitDate'].isna()))
            ])
            
            util_rate = (active_count / project['BedCount']) * 100 if project['BedCount'] > 0 else 0
            
            utilization_data.append({
                'Date': date,
                'ProjectID': project['ProjectID'],
                'ProjectName': project['ProjectName'],
                'ProjectType': get_project_type_name(project['ProjectType']),
                'BedCount': project['BedCount'],
                'ActiveClients': active_count,
                'UtilizationRate': util_rate
            })
    
    utilization_df = pd.DataFrame(utilization_data)
    
    # Average utilization by project type over time
    util_by_type_time = utilization_df.groupby(['Date', 'ProjectType'])['UtilizationRate'].mean().reset_index()
    util_pivot = util_by_type_time.pivot(index='Date', columns='ProjectType', values='UtilizationRate')
    
    print("\nAVERAGE UTILIZATION RATE BY PROJECT TYPE OVER TIME:")
    print(util_pivot)
    
    plt.figure(figsize=(14, 8))
    for column in util_pivot.columns:
        plt.plot(util_pivot.index, util_pivot[column], marker='o', linewidth=2, label=column)
    
    plt.title('Shelter Utilization Rates Over Time by Project Type')
    plt.xlabel('Date')
    plt.ylabel('Average Utilization Rate (%)')
    plt.legend(title='Project Type')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/shelter_utilization_trends.png')
    
    # Projects with consistently high/low utilization
    avg_util_by_project = utilization_df.groupby('ProjectName')['UtilizationRate'].mean().sort_values(ascending=False)
    
    print("\nPROJECTS WITH HIGHEST AVERAGE UTILIZATION:")
    for project, rate in avg_util_by_project.head(3).items():
        print(f"  - {project}: {rate:.1f}%")
    
    print("\nPROJECTS WITH LOWEST AVERAGE UTILIZATION:")
    for project, rate in avg_util_by_project.tail(3).items():
        print(f"  - {project}: {rate:.1f}%")
    
    return {
        'capacity_by_type': capacity_by_type,
        'utilization_df': utilization_df,
        'util_by_type_time': util_by_type_time,
        'avg_util_by_project': avg_util_by_project
    }

# ==========================================
# MAIN ANALYSIS EXECUTION
# ==========================================
def run_all_analyses():
    print("HMIS DATA ANALYSIS REPORT")
    print("========================\n")
    
    print(f"Analysis period: 2022-01-01 to 2024-06-30")
    print(f"Total clients: {len(clients)}")
    print(f"Total project enrollments: {len(enrollments)}")
    print(f"Total projects: {len(projects)}")
    
    # Run all analyses
    data_quality_issues = assess_data_quality()
    trends_data = analyze_trends()
    destination_data = analyze_discharge_destinations()
    prior_situation_data = analyze_prior_living_situations()
    capacity_data = analyze_shelter_capacity()
    
    print("\n===== ANALYSIS COMPLETE =====")
    print("Plots have been saved to the 'plots' directory")
    
    return {
        'data_quality_issues': data_quality_issues,
        'trends_data': trends_data,
        'destination_data': destination_data,
        'prior_situation_data': prior_situation_data,
        'capacity_data': capacity_data
    }

# Run the analysis
if __name__ == "__main__":
    results = run_all_analyses()