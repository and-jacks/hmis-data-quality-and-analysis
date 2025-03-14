import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 6, 30)  # Current data up to mid-2024

# Helper functions
def random_date(start, end):
    """Generate a random date between start and end dates"""
    days_between = (end - start).days
    random_days = random.randint(0, days_between)
    return start + timedelta(days=random_days)

def generate_entry_exit_dates(start_date, end_date, min_stay=1, max_stay=180):
    """Generate entry and exit dates within the given range"""
    entry_date = random_date(start_date, end_date - timedelta(days=min_stay))
    
    # Some clients might still be in the program
    if random.random() < 0.15:  # 15% of clients still active
        return entry_date, None
    
    # Otherwise, generate an exit date
    max_exit_days = min(max_stay, (end_date - entry_date).days)
    stay_length = random.randint(min_stay, max_exit_days)
    exit_date = entry_date + timedelta(days=stay_length)
    
    return entry_date, exit_date

# Organizations
def generate_organizations(num_orgs=5):
    organization_ids = [str(uuid.uuid4())[:8] for _ in range(num_orgs)]
    organization_names = [
        "Metro Housing Solutions",
        "Community Action Agency",
        "Lighthouse Hope Center",
        "Veterans Support Alliance",
        "Family Crisis Network"
    ]
    
    organizations = pd.DataFrame({
        'OrganizationID': organization_ids,
        'OrganizationName': organization_names[:num_orgs],
        'VictimServiceProvider': [0, 0, 0, 0, 1]  # Only the last one is a VSP
    })
    
    return organizations

# Projects
def generate_projects(organizations_df, num_projects=10):
    project_ids = [str(uuid.uuid4())[:8] for _ in range(num_projects)]
    
    # Ensure we have various project types represented
    project_types = [
        0,  # Emergency Shelter - Entry Exit
        1,  # Emergency Shelter - Night-by-Night
        2,  # Transitional Housing
        3,  # PH - Permanent Supportive Housing
        13,  # PH - Rapid Re-Housing
        2,  # Transitional Housing (another one)
        0,  # Emergency Shelter - Entry Exit (another one)
        13,  # PH - Rapid Re-Housing (another one)
        2,  # Transitional Housing (another one)
        3,  # PH - Permanent Supportive Housing (another one)
    ]
    
    project_names = [
        "Main Street Emergency Shelter",
        "Overflow Night Shelter",
        "New Beginnings Transitional Housing",
        "Permanent Housing First",
        "Rapid Rehousing Program",
        "Second Chance Transitional Housing",
        "Family Emergency Shelter",
        "Young Adult Rapid Rehousing",
        "Veterans Transitional Housing",
        "Supportive Housing for Seniors"
    ]
    
    # Project type to bed capacity mapping (approximately)
    bed_capacities = {
        0: (20, 50),  # Emergency Shelter - Entry Exit
        1: (15, 30),  # Emergency Shelter - Night-by-Night
        2: (10, 25),  # Transitional Housing
        3: (15, 35),  # PH - Permanent Supportive Housing
        13: (10, 20)  # PH - Rapid Re-Housing
    }
    
    # Assign organizations to projects (many-to-one)
    org_ids = organizations_df['OrganizationID'].tolist()
    assigned_orgs = [random.choice(org_ids) for _ in range(num_projects)]
    
    # Generate bed capacities based on project type
    bed_counts = [random.randint(*bed_capacities[pt]) for pt in project_types]
    
    # Generate CoC codes (just using one CoC for simplicity)
    coc_codes = ['NY-123'] * num_projects
    
    # Create project dataframe
    projects = pd.DataFrame({
        'ProjectID': project_ids,
        'ProjectName': project_names[:num_projects],
        'OrganizationID': assigned_orgs,
        'ProjectType': project_types[:num_projects],
        'BedCount': bed_counts,
        'ContinuumCode': coc_codes,
        'Operating_StartDate': [start_date - timedelta(days=random.randint(365, 1095)) for _ in range(num_projects)],
        'Operating_EndDate': [None] * num_projects  # All projects still operating
    })
    
    return projects

# Clients
def generate_clients(num_clients=500):
    client_ids = [str(uuid.uuid4())[:8] for _ in range(num_clients)]
    
    # Generate demographic information
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", 
                  "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Nancy",
                  "Daniel", "Lisa", "Matthew", "Margaret", "Anthony", "Sandra", "Mark", "Ashley", "Donald", "Emily"]
    
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
                 "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
                 "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King"]
    
    ssn_data_quality = np.random.choice([1, 2, 8, 9, 99], size=num_clients, p=[0.75, 0.1, 0.05, 0.05, 0.05])
    
    # Function to generate fake SSN
    def generate_ssn():
        if random.random() < 0.85:  # 85% have full SSN
            return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        else:  # 15% have partial SSN
            return f"{random.randint(100, 999)}-XX-XXXX"
    
    # Generate birthdates (ensuring a variety of age ranges)
    # Most homeless clients are adults, with fewer youth and seniors
    
    def generate_birthdate():
        age_category = np.random.choice(["youth", "young_adult", "adult", "older_adult", "senior"], 
                                      p=[0.08, 0.25, 0.45, 0.15, 0.07])
        
        today = datetime.now()
        if age_category == "youth":
            # 13-17
            years_ago = random.randint(13, 17)
        elif age_category == "young_adult":
            # 18-24
            years_ago = random.randint(18, 24)
        elif age_category == "adult":
            # 25-45
            years_ago = random.randint(25, 45)
        elif age_category == "older_adult":
            # 46-62
            years_ago = random.randint(46, 62)
        else:  # senior
            # 63+
            years_ago = random.randint(63, 85)
            
        birthdate = today - timedelta(days=years_ago*365 + random.randint(-182, 182))
        return birthdate
    
    birthdates = [generate_birthdate() for _ in range(num_clients)]
    
    # Race and ethnicity (using the combined field in FY2024)
    # We'll create a distribution that roughly models national demographics
    # but with higher representation of certain groups based on homelessness statistics
    
    def generate_race_ethnicity():
        races = []
        
        # Define probabilities
        race_probabilities = {
            1: 0.1,  # American Indian, Alaska Native, or Indigenous
            2: 0.04,  # Asian or Asian American
            3: 0.4,  # Black, African American, or African
            6: 0.2,  # Hispanic/Latina/e/o
            7: 0.03,  # Middle Eastern or North African
            4: 0.02,  # Native Hawaiian or Pacific Islander
            5: 0.5,  # White
        }
        
        # Decide for each race category
        for race_code, probability in race_probabilities.items():
            if random.random() < probability:
                races.append(race_code)
        
        # If no race selected, pick one randomly
        if not races:
            races.append(random.choice(list(race_probabilities.keys())))
            
        # Small chance of unknown/refused
        if random.random() < 0.05:
            return [np.random.choice([8, 9, 99])]
        
        return races
    
    race_ethnicity = [generate_race_ethnicity() for _ in range(num_clients)]
    
    # Gender
    def generate_gender():
        gender_options = [
            [0],  # Woman (Girl, if child)
            [1],  # Man (Boy, if child)
            [2],  # Culturally Specific Identity
            [4],  # Non-Binary
            [5],  # Transgender
            [6],  # Questioning
            [0, 5],  # Transgender woman
            [1, 5],  # Transgender man
            [3]  # Different Identity
        ]
        
        probabilities = [0.35, 0.55, 0.01, 0.03, 0.01, 0.01, 0.02, 0.01, 0.01]
        
        # Small chance of unknown/refused
        if random.random() < 0.03:
            return [np.random.choice([8, 9, 99])]
        
        # Use random.choices() instead of np.random.choice() for selecting from a list of lists
        return random.choices(gender_options, weights=probabilities, k=1)[0]

    gender = [generate_gender() for _ in range(num_clients)]
    
    # Veteran status (only for adults)
    veteran_status = []
    for bdate in birthdates:
        age = (end_date - bdate).days / 365.25
        if age < 18:
            veteran_status.append(0)  # Not a veteran (minor)
        else:
            # About 9% of adult homeless population are veterans
            veteran_status.append(np.random.choice([0, 1], p=[0.91, 0.09]))
    
    # Disabling condition
    disabling_condition = np.random.choice([0, 1, 8, 9, 99], size=num_clients, 
                                          p=[0.3, 0.55, 0.05, 0.05, 0.05])
    
    # Create client dataframe
    clients = pd.DataFrame({
        'PersonalID': client_ids,
        'FirstName': [random.choice(first_names) for _ in range(num_clients)],
        'LastName': [random.choice(last_names) for _ in range(num_clients)],
        'SSN': [generate_ssn() for _ in range(num_clients)],
        'SSNDataQuality': ssn_data_quality,
        'DOB': birthdates,
        'DOBDataQuality': np.random.choice([1, 2, 8, 9, 99], size=num_clients, 
                                          p=[0.85, 0.05, 0.05, 0.03, 0.02]),
        'RaceEthnicity': race_ethnicity,
        'Gender': gender,
        'VeteranStatus': veteran_status,
        'DisablingCondition': disabling_condition,
        'CreatedDate': [random_date(start_date - timedelta(days=365), end_date) for _ in range(num_clients)]
    })
    
    return clients

# Enrollments and Exits
def generate_enrollments(clients_df, projects_df, num_enrollments=1000):
    # Some clients will have multiple enrollments
    client_ids = np.random.choice(clients_df['PersonalID'].tolist(), size=num_enrollments, replace=True)
    
    # Project assignments
    project_ids = np.random.choice(projects_df['ProjectID'].tolist(), size=num_enrollments)
    
    # Generate enrollment IDs
    enrollment_ids = [str(uuid.uuid4())[:8] for _ in range(num_enrollments)]
    
    # Generate household IDs (some clients will share HH IDs)
    # Let's assume 65% single adults, 35% in multi-person households
    household_ids = [str(uuid.uuid4())[:8] for _ in range(num_enrollments)]
    
    # Now let's create some shared household IDs
    shared_hh_count = int(num_enrollments * 0.35)
    shared_hh_indices = np.random.choice(range(num_enrollments), size=shared_hh_count, replace=False)
    
    # Group into actual households (2-5 people per household)
    household_groups = []
    remaining_indices = list(shared_hh_indices)
    
    while remaining_indices:
        group_size = min(random.randint(2, 5), len(remaining_indices))
        household_groups.append(remaining_indices[:group_size])
        remaining_indices = remaining_indices[group_size:]
    
    # Assign shared household IDs
    for group in household_groups:
        shared_id = str(uuid.uuid4())[:8]
        for idx in group:
            household_ids[idx] = shared_id
    
    # Generate entry and exit dates
    entry_exit_dates = [generate_entry_exit_dates(start_date, end_date) for _ in range(num_enrollments)]
    entry_dates = [dates[0] for dates in entry_exit_dates]
    exit_dates = [dates[1] for dates in entry_exit_dates]
    
    # Determine destination for exited clients
    def generate_destination(project_type):
        # Distribution will vary based on project type
        if project_type in [3, 13]:  # PSH or RRH (higher success rates)
            options = [
                435,  # Rental with subsidy
                410,  # Rental no subsidy
                431,  # RRH subsidy
                312,  # Family temporary
                313,  # Friends temporary
                422,  # Family permanent
                423,  # Friends permanent
                116,  # Place not meant for habitation
                101,  # Emergency shelter
                206,  # Hospital
                207,  # Jail/prison
                17,   # Other
                8, 9, 99, 30  # Unknown options
            ]
            probabilities = [0.25, 0.15, 0.1, 0.06, 0.05, 0.07, 0.05, 0.04, 0.05, 0.03, 0.02, 0.03, 0.02, 0.02, 0.03, 0.03]
            
            # Make sure options and probabilities have the same length
            if len(options) != len(probabilities):
                # For debugging - print the lengths
                print(f"Options length: {len(options)}, Probabilities length: {len(probabilities)}")
                # Adjust probabilities to match options
                probabilities = probabilities[:len(options)] if len(probabilities) > len(options) else probabilities + [0] * (len(options) - len(probabilities))
                # Normalize probabilities to sum to 1
                probabilities = [p/sum(probabilities) for p in probabilities]
                
            return np.random.choice(options, p=probabilities)
        else:  # ES or TH
            options = [
                435,  # Rental with subsidy
                410,  # Rental no subsidy
                431,  # RRH subsidy
                312,  # Family temporary
                313,  # Friends temporary
                422,  # Family permanent
                423,  # Friends permanent
                116,  # Place not meant for habitation
                101,  # Emergency shelter
                206,  # Hospital
                207,  # Jail/prison
                17,   # Other
                8, 9, 99, 30  # Unknown options
            ]
            probabilities = [0.15, 0.1, 0.08, 0.12, 0.1, 0.05, 0.05, 0.1, 0.08, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02]
            
            # Make sure options and probabilities have the same length
            if len(options) != len(probabilities):
                # For debugging - print the lengths
                print(f"Options length: {len(options)}, Probabilities length: {len(probabilities)}")
                # Adjust probabilities to match options
                probabilities = probabilities[:len(options)] if len(probabilities) > len(options) else probabilities + [0] * (len(options) - len(probabilities))
                # Normalize probabilities to sum to 1
                probabilities = [p/sum(probabilities) for p in probabilities]
                
            return np.random.choice(options, p=probabilities)

    # For each enrollment, get the project type from projects_df
    project_types = [projects_df.loc[projects_df['ProjectID'] == pid, 'ProjectType'].iloc[0] for pid in project_ids]
    
    # Generate destinations
    destinations = [generate_destination(pt) if exit_dates[i] is not None else None 
                    for i, pt in enumerate(project_types)]
    
    # Generate relationship to head of household
    relationship_to_hoh = []
    
    # Track which household members have been processed
    processed_households = {}
    
    for i in range(num_enrollments):
        hh_id = household_ids[i]
        
        if hh_id not in processed_households:
            # This is the first member of the household we're seeing
            processed_households[hh_id] = [i]
            relationship_to_hoh.append(1)  # Head of Household
        else:
            # This household already has a head
            processed_households[hh_id].append(i)
            # Relationship options: 2=child, 3=spouse/partner, 4=other relation, 5=non-relation
            relationship_to_hoh.append(random.choice([2, 3, 4, 5]))
    
    # Generate enrollment data
    enrollments = pd.DataFrame({
        'EnrollmentID': enrollment_ids,
        'PersonalID': client_ids,
        'ProjectID': project_ids,
        'EntryDate': entry_dates,
        'ExitDate': exit_dates,
        'HouseholdID': household_ids,
        'RelationshipToHoH': relationship_to_hoh,
        'EnrollmentCoC': ['NY-123'] * num_enrollments,  # Just using one CoC for simplicity
        'Destination': destinations,
        'DisablingCondition': np.random.choice([0, 1, 8, 9, 99], size=num_enrollments, 
                                             p=[0.3, 0.55, 0.05, 0.05, 0.05])
    })
    
    return enrollments

# Prior Living Situation
def generate_prior_living_situation(enrollments_df, clients_df):
    # Create a dataframe with the same indexes as enrollments
    living_situation = pd.DataFrame(index=enrollments_df.index)
    
    # Copy enrollment IDs and personal IDs
    living_situation['EnrollmentID'] = enrollments_df['EnrollmentID']
    living_situation['PersonalID'] = enrollments_df['PersonalID']
    
    # Generate living situation types
    def generate_living_situation():
        options = [
            116,  # Place not meant for habitation
            101,  # Emergency shelter
            118,  # Safe Haven
            215,  # Foster care
            206,  # Hospital
            207,  # Jail/prison
            204,  # Psychiatric facility
            302,  # Transitional housing
            314,  # Hotel/motel
            312,  # Family temporary
            313,  # Friends temporary
            422,  # Family permanent
            423,  # Friends permanent
            435,  # Rental with subsidy
            410,  # Rental no subsidy
            8, 9, 99  # Unknown options
        ]
        
        probabilities = [0.2, 0.15, 0.02, 0.02, 0.03, 0.05, 0.03, 0.08, 0.05, 0.1, 0.08, 0.04, 0.03, 0.05, 0.03, 0.01, 0.01, 0.02]
        
        # Make sure options and probabilities have the same length
        if len(options) != len(probabilities):
            # For debugging
            print(f"Living Situation - Options length: {len(options)}, Probabilities length: {len(probabilities)}")
            
            # Adjust arrays to match in length
            if len(probabilities) > len(options):
                probabilities = probabilities[:len(options)]
            else:
                probabilities = probabilities + [0] * (len(options) - len(probabilities))
            
            # Normalize probabilities to sum to 1
            probabilities = [p/sum(probabilities) for p in probabilities]
        
        return np.random.choice(options, p=probabilities)
    
    living_situation['LivingSituation'] = [generate_living_situation() for _ in range(len(enrollments_df))]
    
    # Length of stay
    living_situation['LengthOfStay'] = np.random.choice([10, 11, 2, 3, 4, 5, 8, 9, 99], 
                                                      size=len(enrollments_df),
                                                      p=[0.15, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
    
    # Approximate date homelessness started
    # This will be random for now, but in a real scenario would be more correlated with entry date
    living_situation['DateHomelessStarted'] = [
        enrollments_df['EntryDate'].iloc[i] - timedelta(days=random.randint(30, 730))
        if living_situation['LivingSituation'].iloc[i] in [116, 101, 118]  # Homeless situations
        else None
        for i in range(len(enrollments_df))
    ]
    
    # Times homeless in past 3 years
    living_situation['TimesHomelessPastThreeYears'] = np.random.choice([1, 2, 3, 4, 8, 9, 99], 
                                                                     size=len(enrollments_df),
                                                                     p=[0.2, 0.25, 0.2, 0.2, 0.05, 0.05, 0.05])
    
    # Months homeless in past 3 years
    living_situation['MonthsHomelessPastThreeYears'] = [
        np.random.choice(list(range(101, 113)) + [113, 8, 9, 99], 
                        p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05])
        if living_situation['LivingSituation'].iloc[i] in [116, 101, 118]  # Homeless situations
        else None
        for i in range(len(enrollments_df))
    ]
    
    return living_situation

# Generate data
def generate_hmis_data(output_dir="hmis_synthetic_data"):
    # Create organizations and projects
    organizations = generate_organizations(5)
    projects = generate_projects(organizations, 10)
    
    # Create clients
    clients = generate_clients(500)
    
    # Create enrollments
    enrollments = generate_enrollments(clients, projects, 1000)
    
    # Create prior living situation data
    living_situation = generate_prior_living_situation(enrollments, clients)
    
    # Export to CSV
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    organizations.to_csv(f"{output_dir}/organizations.csv", index=False)
    projects.to_csv(f"{output_dir}/projects.csv", index=False)
    clients.to_csv(f"{output_dir}/clients.csv", index=False)
    enrollments.to_csv(f"{output_dir}/enrollments.csv", index=False)
    living_situation.to_csv(f"{output_dir}/living_situation.csv", index=False)
    
    return {
        "organizations": organizations,
        "projects": projects,
        "clients": clients,
        "enrollments": enrollments,
        "living_situation": living_situation
    }

# Generate the data
data = generate_hmis_data()

# Preview the data
for name, df in data.items():
    print(f"\n{name.upper()} - {len(df)} records")
    print(df.head(2))