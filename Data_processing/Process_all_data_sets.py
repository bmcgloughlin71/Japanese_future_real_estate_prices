import numpy as np
import pandas as pd
import os
import warnings
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


#Get prefecture codes
prefecture_codes = pd.read_csv("../Data/prefecture_code.csv")

data_dir = "../Data/trade_prices"

for prefecture_idx, prefecture in prefecture_codes.iterrows():

    prefecture_code = prefecture['Code']
    prefecture_name = prefecture['EnName']
    print(f'Now processing {prefecture_name} prefecture with prefecture code: {prefecture_code}')

    #Load releveant data set
    print("Loading data . . .")
    data = pd.read_csv(f'{data_dir}/{prefecture_code:02d}.csv', low_memory=False)
    
    print(f'Successfully loaded the {prefecture_name} data set.')

    # Remove unwanted columns #
    data.drop(columns=['No', 'FloorPlan', 'Remarks', 'Renovation', 
                     'LandShape', 'Structure', 'Direction', 'Classification', 'Breadth', 
                     'CityPlanning', 'CoverageRatio', 'FloorAreaRatio', 'NearestStation', 
                     'TimeToNearestStation', 'Use', 'PricePerTsubo', 'UnitPrice', 'MunicipalityCode', 
                      'Period', 'DistrictName'], inplace=True)
    
    # Combine MinTimeToNearestStation and MaxTimeToNearestStation into AvgTimeToNearestStation
    data['AvgTimeToNearestStation'] = data[['MinTimeToNearestStation', 'MaxTimeToNearestStation']].mean(axis=1)

    data.drop(columns=['MinTimeToNearestStation', 'MaxTimeToNearestStation'], inplace=True)

    # Convert relevant columns to bool type
    bool_columns = ['AreaIsGreaterFlag', 'FrontageIsGreaterFlag', 'TotalFloorAreaIsGreaterFlag', 'PrewarBuilding']

    for col in bool_columns:
        data[col] = data[col].astype(bool)

    #Extract data sets 
    Intended_House_condition = (data["Purpose"] == "House")
    House_data = data[Intended_House_condition]
    House_data.drop(columns=['Purpose'], inplace=True) # No longer needed
    print(f'{len(House_data)} entries, ({(len(House_data) / len(data)) * 100} % of total) of all purchases intended for housing.')

    # Any type that is Land Only will not have a floor size, so we can set the TotalFloorArea to -1. Same logic for Building year and frontage
    House_data.loc[House_data['TotalFloorArea'].isna() & (House_data['Type'] == 'Residential Land(Land Only)'), 'TotalFloorArea'] = -1
    House_data.loc[House_data['BuildingYear'].isna() & (House_data['Type'] == 'Residential Land(Land Only)'), 'BuildingYear'] = -1
    House_data.loc[House_data['Frontage'].isna() & (House_data['Type'] == 'Residential Land(Land Only)'), 'Frontage'] = -1
    

    # Count rows with NaN values in the entire DataFrame
    nan_rows_count = House_data.isna().sum(axis=1)
    rows_with_nan = nan_rows_count[nan_rows_count > 0]

    print(f'Number of rows with NaN values after initial processing: {len(rows_with_nan)}. Removing . . .')
    #Drop unwanted NaN values
    House_data.dropna(inplace=True)

    print("One-hot encoding Regions . . .")

    region_encoded = pd.get_dummies(House_data['Region'], prefix='Region')
    House_data = pd.concat([House_data, region_encoded], axis=1)
    House_data.drop(columns=['Region'], inplace=True)

    print("Generating Muncipality Categories . . .")
    House_data['MunicipalityCategory'] = House_data.apply(lambda row: categorize_municipality(row['Municipality'], row['Prefecture']), axis=1)

    print("Sorting prefecture to region . . .")

    House_data = encode_region(House_data)

    print("Splitting data set between land only and land with building purchases")

    # Filter the dataset for properties with buildings and without buildings
    land_only_df = House_data[House_data['Type_Residential Land(Land Only)'] == True]
    building_df = House_data[House_data['Type_Residential Land(Land Only)'] == False]
    House_data.drop(columns=['Type'], inplace=True)

    # Save the datasets
    land_only_df.to_csv(f'./Cleaned_Data_Sets/{prefecture_name}_cleaned_test_landOnly.csv', index=False)
    building_df.to_csv(f'./Cleaned_Data_Sets/{prefecture_code}_cleaned_test_buildings.csv', index=False)

    print(f'Finished processing the {prefecture_name} data set! \n')

    




