from pathlib import Path
import re
import numpy as np
import pandas as pd
from TCIAClient import Client

##################################### parameter #############################################
base_url = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query'
collection = 'Head-Neck-PET-CT'
output_dir = './' + collection
##################################### parameter #############################################

client = Client(base_url)
output_dir = Path(output_dir)

series_df = pd.DataFrame(data=client.get_json('getSeries', {'Collection': collection}))
series_df = series_df[['StudyInstanceUID', 'SeriesInstanceUID', 'Modality', 'SeriesDescription']]

patient_df = pd.DataFrame(data=client.get_json('getPatientStudy', {'Collection': collection}))
patient_df = patient_df[['PatientID', 'StudyInstanceUID', 'SeriesCount']]

n_patients = len(patient_df['PatientID'].unique())
print('there are {} patients in the data set'.format(n_patients))
n_reg = np.count_nonzero(series_df['Modality'] == 'REG')
print('for {} of them contours were drawn on the RT CT'.format(n_reg))

df = patient_df.merge(series_df, how='outer', on='StudyInstanceUID')

# get the studies that contain a REG file
reg_study_uids = []
for uid in df['StudyInstanceUID'].unique():
    uid_df = df[df['StudyInstanceUID'] == uid]
    if np.count_nonzero(uid_df['Modality'].isin(['REG'])):
        reg_study_uids.append(uid)
reg_study_uids = np.asarray(reg_study_uids)

# change the SeriesDescription in order to distinguish the CT and PET RTSTRUCT files
df['SeriesDescription'].fillna('None', inplace=True)
for ii, row in df[df['Modality']=='RTSTRUCT'].iterrows():
    if re.match('.*->CT.*', row['SeriesDescription']):
        description = 'CT_STRUCT'
    elif re.match('.*->PET.*', row['SeriesDescription']):
        description = 'PET_STRUCT'
    else:
        description = 'None'
    df.loc[ii, 'SeriesDescription'] = description

ct_struct_ids = df[df['SeriesDescription'] == 'CT_STRUCT']['PatientID'].values
pet_struct_ids = df[df['SeriesDescription'] == 'PET_STRUCT']['PatientID'].values

struct_df = df[~df['StudyInstanceUID'].isin(reg_study_uids) & (df['Modality'] == 'RTSTRUCT')].copy()
struct_df['SeriesDescription'].fillna('_', inplace=True)

# as the SeriesDescription for the RTSTRUCT files in mutually exclusive fill the missing ones
iter_df = df[(df['SeriesDescription'] == 'None') & (df['Modality'] == 'RTSTRUCT')]
for ii, row in iter_df.iterrows():
    if row['PatientID'] in ct_struct_ids:
        description = 'PET_STRUCT'
    elif row['PatientID'] in pet_struct_ids:
        description = 'CT_STRUCT'
    else:
        description = 'None'
    df.loc[ii, 'SeriesDescription'] = description

mask = np.isin(df['StudyInstanceUID'].unique(), reg_study_uids)
planning_uids = df['StudyInstanceUID'].unique()[~mask]

# get unique descriptions
sorted_df = df.sort_values(by=['StudyInstanceUID', 'SeriesDescription']).copy()
count = 0
for ii in range(0, len(sorted_df[:-1])):
    kk = sorted_df.index[ii]
    row = sorted_df.iloc[ii]
    next_row = sorted_df.iloc[ii+1]
    desc = row['SeriesDescription']
    next_desc = next_row['SeriesDescription']
    uid = row['StudyInstanceUID']
    next_uid = next_row['StudyInstanceUID']
    if (desc == next_desc) and (desc != 'None') and (uid == next_uid):
        df.loc[kk, 'SeriesDescription'] = desc + '_' + str(count)
        count += 1
    elif count > 0:
        df.loc[kk, 'SeriesDescription'] = desc + '_' + str(count)
        count = 0
    else:
        count = 0

ct_df = df[
    (
        df['StudyInstanceUID'].isin(planning_uids)
    ) & (
        (
            df['Modality'] == 'CT'
        ) | (
            (df['Modality'] == 'RTSTRUCT') & (df['SeriesDescription'].str.contains('CT_'))
        )
    )
]

def get_folder_name(row):
    if row['Modality'] != 'RTSTRUCT':
        return str(row['Modality'])
    else:
        return str(row['SeriesDescription'])

for ii, row in ct_df.iterrows():
    uid = row['SeriesInstanceUID']
    
    target_dir = (output_dir / row['PatientID']) / get_folder_name(row)
    print(f'downloading: {target_dir}')

    target_path = target_dir / (row['PatientID'] + '.zip')
        
    client.get_image(uid, target_path, unzip=True, remove_zip=True)
