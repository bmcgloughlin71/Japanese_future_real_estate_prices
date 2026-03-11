import numpy as np
import pandas as pd
import os
import warnings
import re
from glob import glob
from geopy.distance import geodesic
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

os.makedirs("Cleaned_Data_Sets", exist_ok=True)

#Functions#
capital_dict = {
    "Hokkaido": "Sapporo",
    "Aomori Prefecture": "Aomori",
    "Iwate Prefecture": "Morioka",
    "Miyagi Prefecture": "Sendai",
    "Akita Prefecture": "Akita",
    "Yamagata Prefecture": "Yamagata",
    "Fukushima Prefecture": "Fukushima",
    "Ibaraki Prefecture": "Mito",
    "Tochigi Prefecture": "Utsunomiya",
    "Gunma Prefecture": "Maebashi",
    "Saitama Prefecture": "Saitama",
    "Chiba Prefecture": "Chiba",
    "Tokyo": "Tokyo",
    "Kanagawa Prefecture": "Yokohama",
    "Niigata Prefecture": "Niigata",
    "Toyama Prefecture": "Toyama",
    "Ishikawa Prefecture": "Kanazawa",
    "Fukui Prefecture": "Fukui",
    "Yamanashi Prefecture": "Kofu",
    "Nagano Prefecture": "Nagano",
    "Gifu Prefecture": "Gifu",
    "Shizuoka Prefecture": "Shizuoka",
    "Aichi Prefecture": "Nagoya",
    "Mie Prefecture": "Tsu",
    "Shiga Prefecture": "Otsu",
    "Kyoto Prefecture": "Kyoto",
    "Osaka Prefecture": "Osaka",
    "Hyogo Prefecture": "Kobe",
    "Nara Prefecture": "Nara",
    "Wakayama Prefecture": "Wakayama",
    "Tottori Prefecture": "Tottori",
    "Shimane Prefecture": "Matsue",
    "Okayama Prefecture": "Okayama",
    "Hiroshima Prefecture": "Hiroshima",
    "Yamaguchi Prefecture": "Yamaguchi",
    "Tokushima Prefecture": "Tokushima",
    "Kagawa Prefecture": "Takamatsu",
    "Ehime Prefecture": "Matsuyama",
    "Kochi Prefecture": "Kochi",
    "Fukuoka Prefecture": "Fukuoka",
    "Saga Prefecture": "Saga",
    "Nagasaki Prefecture": "Nagasaki",
    "Kumamoto Prefecture": "Kumamoto",
    "Oita Prefecture": "Oita",
    "Miyazaki Prefecture": "Miyazaki",
    "Kagoshima Prefecture": "Kagoshima",
    "Okinawa Prefecture": "Naha"
}


def categorize_municipality(municipality, prefecture):
    if capital_dict[prefecture] in municipality:
        return 4  # Capital
    elif 'City' in municipality:
        return 3  # City
    elif 'Town' in municipality:
        return 2  # Town
    elif 'Village' in municipality:
        return 1  # Village
    elif "Ward" in municipality and prefecture=="Tokyo":
        return 4 # Tokyo is an edge case
    else:
        raise ValueError(f"Could not categorize municipality: {municipality} for prefecture: {prefecture}")

region_map = {
    'Hokkaido': 'Hokkaido',
    'Aomori Prefecture': 'Tohoku', 'Iwate Prefecture': 'Tohoku', 'Miyagi Prefecture': 'Tohoku', 'Akita Prefecture': 'Tohoku', 'Yamagata Prefecture': 'Tohoku', 'Fukushima Prefecture': 'Tohoku',
    'Ibaraki Prefecture': 'Kanto', 'Tochigi Prefecture': 'Kanto', 'Gunma Prefecture': 'Kanto', 'Saitama Prefecture': 'Kanto', 'Chiba Prefecture': 'Kanto', 'Tokyo': 'Kanto', 'Kanagawa Prefecture': 'Kanto',
    'Niigata Prefecture': 'Chubu', 'Toyama Prefecture': 'Chubu', 'Ishikawa Prefecture': 'Chubu', 'Fukui Prefecture': 'Chubu', 'Yamanashi Prefecture': 'Chubu', 'Nagano Prefecture': 'Chubu', 'Gifu Prefecture': 'Chubu', 'Shizuoka Prefecture': 'Chubu', 'Aichi Prefecture': 'Chubu',
    'Mie Prefecture': 'Kansai', 'Shiga Prefecture': 'Kansai', 'Kyoto Prefecture': 'Kansai', 'Osaka Prefecture': 'Kansai', 'Hyogo Prefecture': 'Kansai', 'Nara Prefecture': 'Kansai', 'Wakayama Prefecture': 'Kansai',
    'Tottori Prefecture': 'Chugoku', 'Shimane Prefecture': 'Chugoku', 'Okayama Prefecture': 'Chugoku', 'Hiroshima Prefecture': 'Chugoku', 'Yamaguchi Prefecture': 'Chugoku',
    'Tokushima Prefecture': 'Shikoku', 'Kagawa Prefecture': 'Shikoku', 'Ehime Prefecture': 'Shikoku', 'Kochi Prefecture': 'Shikoku',
    'Fukuoka Prefecture': 'Kyushu', 'Saga Prefecture': 'Kyushu', 'Nagasaki Prefecture': 'Kyushu', 'Kumamoto Prefecture': 'Kyushu', 'Oita Prefecture': 'Kyushu', 'Miyazaki Prefecture': 'Kyushu', 'Kagoshima Prefecture': 'Kyushu', 'Okinawa Prefecture': 'Kyushu'
}

def encode_region(df):

    df['Region'] = df['Prefecture'].map(region_map)

    #Check for NaN values in the 'Region' column 
    if df['Region'].isna().any():
        raise ValueError("Some prefectures did not match any region.")

    region_encoded = pd.get_dummies(df['Region'], prefix='Region')
    df = pd.concat([df, region_encoded], axis=1)

    # Drop the original 'Region' column since it's now encoded
    df.drop(columns=['Region'], inplace=True)

    return df

column_mapping = {
    'City,Town,Ward,Village': 'Location',
    'Total transaction value': 'TotalTransactionValue',
    'Area(㎡)': 'Area',
    'Building : Total floor area': 'TotalFloorArea',
    'Building : Construction year': 'ConstructionYear',
    'Building coverage ratio': 'BuildingCoverageRatio',
    'Floor area ratio': 'FloorAreaRatio',
    'Area_greater_2000_flag': 'FloorAreaGreaterFlag',
    'before_the_war_flag': 'BeforeWarFlag',
    'Average Distance to Station': 'AverageTimeToStation',
    'area_greater_flag': 'AreaGreaterFlag',
    'floor_area_greater_than_2000': 'FloorAreaGreaterFLag',
    'Region_Commercial Area': 'RegionCommercialArea',
    'Region_Industrial Area': 'RegionIndustrialArea',
    'Region_Potential Residential Area': 'RegionPotentialResidentialArea',
    'Region_Residential Area': 'RegionResidentialArea'
}

def convert_to_minutes(distance_str):
    # If it contains a range (e.g., '30-60minutes' or '1H-1H30')
    if '-' in distance_str:
        if 'H' in distance_str:
            # Split the range by "-"
            parts = distance_str.split('-')

            # Split each part by "H" and handle hours and minutes
            start = parts[0].split('H')
            end = parts[1].split('H')
            # Convert start time
            start_hours = int(start[0]) if start[0] else 0
            start_minutes = int(start[1]) if len(start) > 1 and start[1]  else 0
            start_total_minutes = start_hours * 60 + start_minutes

            # Convert end time
            end_hours = int(end[0]) if end[0] else 0
            end_minutes = int(end[1]) if len(end) > 1 and end[1] else 0
            end_total_minutes = end_hours * 60 + end_minutes

            if end_hours == 0 and end_minutes == 0: #deal with edge cases such as "2H-"
                return start_total_minutes
            # Return the average of the two
            return (start_total_minutes + end_total_minutes) // 2

        elif 'minutes' in distance_str:
            parts = distance_str.split('-')

            start = int(parts[0])
            end = int(parts[1].split("minutes")[0])

            return (start + end) // 2

    else:  # If there's only a single value (e.g., '30minutes')
        # Extract the number and convert to minutes (if needed)
        if 'H' in distance_str:  # In case it's in hours
            hours = int(re.search(r'(\d+)', distance_str).group(0))
            return hours * 60  # Convert to minutes
        else:  # Just minutes
            return int(re.search(r'(\d+)', distance_str).group(0))

def find_closest_city_and_distance(lat, lon):
    min_distance = float('inf')
    closest_city = None
    for _, city in designated_cities.iterrows():
        city_coords = (city.Latitude, city.Longitude)
        distance = geodesic((lat, lon), city_coords).kilometers
        if distance < min_distance:
            min_distance = distance
            closest_city = city.CityName
    return pd.Series([min_distance, closest_city])

def assign_nearest_population(row):
    year = row['Year']  # Transaction year
    pop_years = [2005, 2010, 2015, 2020]
    pop_cols = ['Population_2005', 'Population_2010', 'Population_2015', 'Population_2020']

    # Compute absolute differences
    diffs = [abs(year - y) for y in pop_years]

    sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i])

    for idx in sorted_indices:
        pop_value = row[pop_cols[idx]]
        if pd.notna(pop_value) and pop_value != 0:
            return pop_value

    raise ValueError(
        f"No valid population found for Area_Code {row['City,Town,Ward,Village code']} "
        f"at transaction year {year}. Check population dataset."
    )
#Get prefecture codes
prefecture_codes = pd.read_csv("../Data/2005_2024/prefecture_code.csv")

data_dir = "../Data/2005_2024/trade_prices"

#Clean real estate price data
for prefecture_idx, prefecture in prefecture_codes.iterrows():

    prefecture_code = prefecture['Code']
    prefecture_name = prefecture['EnName'].replace(" ", "")
    print(f'Now processing {prefecture_name} prefecture with prefecture code: {prefecture_code}')

    #Load releveant data set
    print("Loading data . . .")
    data = pd.read_csv(f'{data_dir}/{prefecture_code:02d}.csv', encoding="cp932", low_memory=False)

    print(f'Successfully loaded the {prefecture_name} data set.')

    # Remove unwanted columns #
    data.drop(columns=['Price information classification', 'District', 'Nearest station : Name',
                     'City planning', 'Land : Shape', 'Frontage road : Direction', 'Frontage road : Type', 'Frontage road : Width',
                      'Renovation', 'Transaction factors', 'Layout', 'Building : Structure', 'Land : Price per ㎡'], inplace=True)


    Intended_House_condition = (data['Purpose of use'] == "House") | (data['Use'] == "House")
    House_df = data[Intended_House_condition]
    House_df.drop(columns=['Purpose of use', 'Use'], inplace=True) # No longer needed

    print(f'{len(House_df)} entries, ({(len(House_df) / len(data)) * 100:.2f} % of total) in this data set satisfy the housing condition.')



    # Any type that is Land Only will not have a floor size, so we can set the TotalFloorArea to -1. Same logic for other Building stats and frontage
    # Also Agriculutual land
    land_only_condition = (House_df['Type'] == 'Residential Land(Land Only)') | (House_df['Type'] == 'Agricultural Land') | (House_df['Type'] == 'Forest Land')
    House_df.loc[House_df['Building : Total floor area'].isna() & (land_only_condition), 'Building : Total floor area'] = -1
    House_df.loc[House_df['Building : Construction year'].isna() & (land_only_condition), 'Building : Construction year'] = -1
    House_df.loc[House_df['Building coverage ratio'].isna() & (land_only_condition), 'Building coverage ratio'] = -1
    House_df.loc[House_df['Floor area ratio'].isna() & (land_only_condition), 'Floor area ratio'] = -1
    House_df.loc[House_df['Frontage'].isna() & (land_only_condition), 'Frontage'] = -1

    House_df.drop(columns=['Area'], inplace=True) # Decide to drop this since essentially all buildings, intended for housing, are in residential areas

    # For condomoniums etc, we will assume that area = total_floor_area and that frontage = 0.
    is_condomonium = (House_df['Type'] == 'Pre-owned Condominiums, etc.')
    House_df.loc[House_df['Building : Total floor area'].isna() & (is_condomonium), 'Building : Total floor area'] = House_df.loc[House_df['Building : Total floor area'].isna() & (is_condomonium), 'Area(㎡)']
    House_df.loc[House_df['Frontage'].isna() & (is_condomonium), 'Frontage'] = 0.
    House_df['is_condomonium_like'] = is_condomonium
    # Count rows with NaN values in the entire DataFrame
    nan_rows_count = House_df.isna().sum(axis=1)
    rows_with_nan = nan_rows_count[nan_rows_count > 0]
    building_temp_df = House_df[~House_df['Type'].str.contains('Land Only')]

    print(f'Number of rows with NaN values after initial processing: {len(rows_with_nan)} ({100 * (len(rows_with_nan))/(len(House_df)):.2f} %). Removing . . .')
    # Drop unwanted NaN values
    House_df.dropna(inplace=True)

    print("Cleaning timing information . . .")
    House_df['Quarter'] = House_df['Transaction timing'].str.extract(r'(\d)')[0].astype(int)
    House_df['Year'] = House_df['Transaction timing'].str.extract(r'(\d{4})')[0].astype(int)
    House_df.drop(columns=['Transaction timing'], inplace=True)

    #print("One-hot encoding Regions . . .")
    #region_encoded = pd.get_dummies(House_df['Area'], prefix='Region')
    #House_df = pd.concat([House_df, region_encoded], axis=1)
    #House_df.drop(columns=['Area'], inplace=True)

    print("Generating Muncipality Categories . . .")
    House_df['MunicipalityCategory'] = House_df.apply(lambda row: categorize_municipality(row['City,Town,Ward,Village'], row['Prefecture']), axis=1)

    print("Sorting prefecture to region . . .")

    House_df = encode_region(House_df)

    print("Formatting times . . .")
    House_df['Average Distance to Station'] = House_df['Nearest station : Distance'].apply(convert_to_minutes)
    House_df.drop(columns=['Nearest station : Distance'], inplace=True)

    print("Adding greater-than flags")

    # Floor Area
    House_df['Building : Total floor area'] = House_df['Building : Total floor area'].apply(
    lambda x: 10 if 'less than 10' in str(x) else (2000 if 'or greater' in str(x) else x))
    House_df['Building : Total floor area'] = pd.to_numeric(House_df['Building : Total floor area'], errors='coerce')
    House_df['floor_area_greater_than_2000'] = House_df['Building : Total floor area'] >= 2000

    # Construction year
    House_df['Building : Construction year'] = House_df['Building : Construction year'].apply(
    lambda x: 1945 if 'before the war' in str(x) else x)
    House_df['Building : Construction year'] = pd.to_numeric(House_df['Building : Construction year'], errors='coerce')
    House_df['before_the_war_flag'] = House_df['Building : Construction year'] <= 1945

    # Frontage
    House_df['Frontage'] = House_df['Frontage'].apply(
    lambda x: 50 if '50.0m or longer' in str(x) else x)
    House_df['Frontage'] = pd.to_numeric(House_df['Frontage'], errors='coerce')
    House_df['frontage_greater_than_50'] = House_df['Frontage'] >= 50

    # Area
    House_df['Area(㎡)'] = House_df['Area(㎡)'].apply(
    lambda x: 2000 if 'or greater' in str(x) else x)
    House_df['Area(㎡)'] = pd.to_numeric(House_df['Area(㎡)'], errors='coerce')
    House_df['area_greater_flag'] = House_df['Area(㎡)'] >= 2000

    print("Renaming Columns . . .")
    House_df = House_df.rename(columns=column_mapping)

    print("Splitting data set between land only and land with building purchases")

    # Filter the dataset for properties with buildings and without buildings
    land_only_condition = (House_df['Type'] == 'Residential Land(Land Only)') | (House_df['Type'] == 'Agricultural Land') | (House_df['Type'] == 'Forest Land')
    land_only_df = House_df[land_only_condition]
    building_df = House_df[~land_only_condition]

    House_df.drop(columns=['Type'], inplace=True)

    # Save the datasets
    land_only_df.to_csv(f'./Cleaned_Data_Sets/{prefecture_name}_cleaned_test_landOnly.csv', index=False)
    building_df.to_csv(f'./Cleaned_Data_Sets/{prefecture_name}_cleaned_test_buildings.csv', index=False)

    print(f'Finished processing the {prefecture_name} data set! \n')

print(f'All precture real-estate data sets successfully cleaned!\n')

#Merge real estate price data into a single csv file
print("Merging prefectures . . .")

files_to_merge = glob('./Cleaned_Data_Sets/*build*.csv')

dataframes = [pd.read_csv(file) for file in files_to_merge]

all_columns = sorted(set(col for df in dataframes for col in df.columns))

# Ensure all dataframes have the same columns, filling missing ones with False
for i, df in enumerate(dataframes):
    missing_cols = [col for col in all_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = False  # Fill missing columns with False

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv('./Cleaned_Data_Sets/All_prefectures_buildings.csv', index=False)

print(f"Combined {len(files_to_merge)} files into 'All_prefectures_buildings.csv'\n")

#Merge Migration Data
print("Merging Migration and real estate price data . . .")

df_migration = pd.read_csv("../Data/Internal_Migration_Data_2008_2024/Number_of_Migants_to_Muncipalties_per_year.csv")

df_housing_prices = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings.csv")

# First, rename columns to align for merging
df_migration_renamed = df_migration.rename(columns={
    'area_code': 'City,Town,Ward,Village code',
    'value': 'Migration',
    'Year': 'MigrationYear'
})

# Create a 'TargetYear' in housing_prices for joining
df_housing_prices['TargetYear'] = df_housing_prices['Year'] - 1 #assumption is that migration has a 1 year lag on housing prices

# Attempt the direct merge on area_code and target year
merged = pd.merge(
    df_housing_prices,
    df_migration_renamed,
    how='left',
    left_on=['City,Town,Ward,Village code', 'TargetYear'],
    right_on=['City,Town,Ward,Village code', 'MigrationYear']
)

# Now fill missing Migration values using the earliest year per area_code
# First, get the earliest year migration value per area_code
earliest_migration = (
    df_migration_renamed.sort_values('MigrationYear')
    .drop_duplicates('City,Town,Ward,Village code')
    [['City,Town,Ward,Village code', 'Migration']]
)

# Fill in missing Migration values in merged dataframe
merged = pd.merge(
    merged,
    earliest_migration,
    how='left',
    on='City,Town,Ward,Village code',
    suffixes=('', '_earliest')
)

# Use the found migration value or fall back to earliest
merged['Migration'] = merged['Migration'].combine_first(merged['Migration_earliest'])

# Drop helper columns
merged = merged.drop(columns=['Migration_earliest', 'TargetYear', 'MigrationYear'])

merged.to_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration.csv", index=False)

print("Merged migration and real estate price data sets!\n")

#Add Coordinate and Population Data
print("Merging population and coordinate data . . .\n")
Housing_data = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration.csv")
Population_and_coordinate_data = pd.read_csv("../Data/Population_and_coordinate_data/Japanese_Populations_coordinates.csv")

Housing_data['City,Town,Ward,Village code'] = Housing_data['City,Town,Ward,Village code'].astype('Int64')
Population_and_coordinate_data['Area_Codes'] = Population_and_coordinate_data['Area_Codes'].astype('Int64')

merged_df = pd.merge(
    Housing_data,
    Population_and_coordinate_data,
    left_on='City,Town,Ward,Village code',
    right_on='Area_Codes',
    how='inner'
)
print("Merged coordinate data!\n")

print("Adding population data . . .\n")

merged_df['Population'] = merged_df.apply(assign_nearest_population, axis=1)

cols_to_exclude = [
    "Population_2020", "Population_2015", "Population_2010", "Population_2005", "Area_Codes", "Destination"]

merged_df.drop(columns=cols_to_exclude, inplace=True)

merged_df.to_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration_coords_pop.csv", index=False)

df_House_with_coords = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration_coords_pop.csv")

designated_cities =  pd.read_csv("../Data/Population_and_coordinate_data/designated_cities_and_tokyo.txt", sep='\s+', comment='#',
                        names=["CityName", "Latitude", "Longitude"])


#Compute distances from properties to designated cities
print("Calculating distance to nearest designated city . . .\n")
tqdm.pandas()
df_House_with_coords[['Distance_to_designated_city', 'Nearest_designated_city']] = df_House_with_coords.progress_apply(
    lambda row: find_closest_city_and_distance(row['latitude'], row['longitude']), axis=1)

print("Handling special cases...\n")
greater_tokyo_set = {'Yokohama', 'Kawasaki', 'Saitama', 'Chiba', 'Sagamihara'}

df_House_with_coords['Close_to_Tokyo'] = (df_House_with_coords['Nearest_designated_city'] == 'Tokyo').astype(int)
df_House_with_coords['Close_to_greater_Tokyo_area'] = df_House_with_coords['Nearest_designated_city'].isin(greater_tokyo_set).astype(int)
df_House_with_coords['Close_to_designated_city_flag'] = (df_House_with_coords['Distance_to_designated_city'] < 5).astype(int)


df_House_with_coords.to_csv("./Cleaned_Data_Sets/Final_Cleaned_Data_Set.csv", index=False)

print("All Done!")
