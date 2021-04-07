from pathlib import Path
import pandas as pd
from TCIAClient import Client

##################################### parameter #############################################
base_url = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query'
collection = 'OPC-Radiomics' 
output_dir = './' + collection
##################################### parameter #############################################

client = Client(base_url)
output_dir = Path(output_dir)

series_df = pd.DataFrame(data=client.get_json('getSeries', {'Collection': collection}))
series_df = series_df[['StudyInstanceUID', 'SeriesInstanceUID', 'Modality']]

patient_df = pd.DataFrame(data=client.get_json('getPatientStudy', {'Collection': collection}))
patient_df = patient_df[['PatientID', 'StudyInstanceUID', 'SeriesCount']]

# get the StudyUIDs of all CT's with a corresponding RTSTRUCT file
rts_stud_uids = series_df[series_df['Modality'] == 'RTSTRUCT']['StudyInstanceUID'].values
# use only CT cases with a corresponding RTSTRUCT file
series_df = series_df[
    (
        series_df['Modality'].isin(['CT', 'RTSTRUCT'])
    ) & (
        series_df['StudyInstanceUID'].isin(rts_stud_uids)
    )
]

download_df = patient_df.merge(series_df, how='right', on='StudyInstanceUID')

for ii, row in download_df.iterrows():
    uid = row['SeriesInstanceUID']
    
    target_dir = (output_dir / row['PatientID']) / row['Modality']
    
    # check if target_dir already exists, if so append a count number
    if target_dir.exists():
        split = target_dir.stem.split('_')
        if len(split) == 2:
            count = int(split[-1]) + 1
        else:
            count = 1
        target_dir = target_dir.parent / f'{target_dir.stem}_{count}'
    print(f'downloading: {target_dir}')

    target_path = target_dir / (row['PatientID'] + '.zip')
        
    client.get_image(uid, target_path, unzip=True, remove_zip=True)
