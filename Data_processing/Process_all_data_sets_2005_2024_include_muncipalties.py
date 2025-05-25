import numpy as np
import pandas as pd
import os
import warnings
import re
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
        return 4 # Tokyo is an edhe case
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

#Get prefecture codes
prefecture_codes = pd.read_csv("../Data/2005_2024/prefecture_code.csv")

data_dir = "../Data/2005_2024/trade_prices"

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

    print(f'{len(House_df)} entries, ({(len(House_df) / len(data)) * 100} % of total) in this data set satisfy the housing condition.')



    # Any type that is Land Only will not have a floor size, so we can set the TotalFloorArea to -1. Same logic for other Building stats and frontage
    House_df.loc[House_df['Building : Total floor area'].isna() & (House_df['Type'] == 'Residential Land(Land Only)'), 'Building : Total floor area'] = -1
    House_df.loc[House_df['Building : Construction year'].isna() & (House_df['Type'] == 'Residential Land(Land Only)'), 'Building : Construction year'] = -1
    House_df.loc[House_df['Building coverage ratio'].isna() & (House_df['Type'] == 'Residential Land(Land Only)'), 'Building coverage ratio'] = -1
    House_df.loc[House_df['Floor area ratio'].isna() & (House_df['Type'] == 'Residential Land(Land Only)'), 'Floor area ratio'] = -1
    House_df.loc[House_df['Frontage'].isna() & (House_df['Type'] == 'Residential Land(Land Only)'), 'Frontage'] = -1

    # Count rows with NaN values in the entire DataFrame
    nan_rows_count = House_df.isna().sum(axis=1)
    rows_with_nan = nan_rows_count[nan_rows_count > 0]

    print(f'Number of rows with NaN values after initial processing: {len(rows_with_nan)}. Removing . . .')
    # Drop unwanted NaN values
    House_df.dropna(inplace=True)

    print("Cleaning timing information . . .")
    House_df['Quarter'] = House_df['Transaction timing'].str.extract(r'(\d)')[0].astype(int)
    House_df['Year'] = House_df['Transaction timing'].str.extract(r'(\d{4})')[0].astype(int)
    House_df.drop(columns=['Transaction timing'], inplace=True)

    print("One-hot encoding Regions . . .")
    region_encoded = pd.get_dummies(House_df['Area'], prefix='Region')
    House_df = pd.concat([House_df, region_encoded], axis=1)
    House_df.drop(columns=['Area'], inplace=True)

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
    land_only_df = House_df[House_df['Type'].str.contains('Land Only')]  
    building_df = House_df[~House_df['Type'].str.contains('Land Only')]  


    House_df.drop(columns=['Type'], inplace=True)

    # Save the datasets
    land_only_df.to_csv(f'./Cleaned_Data_Sets/2005_2024_with_municipalities/{prefecture_name}_cleaned_test_landOnly.csv', index=False)
    building_df.to_csv(f'./Cleaned_Data_Sets/2005_2024_with_municipalities/{prefecture_name}_cleaned_test_buildings.csv', index=False)

    print(f'Finished processing the {prefecture_name} data set! \n')






