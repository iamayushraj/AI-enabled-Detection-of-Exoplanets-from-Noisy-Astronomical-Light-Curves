import pandas as pd
from dashboard.app import run_full_analysis
import os

print('Loading models...')
data_dir = 'data/synthetic'
for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    if os.path.isdir(cls_path):
        files = os.listdir(cls_path)
        if files:
            file_path = os.path.join(cls_path, files[0])
            print(f'\n--- Testing {cls} ---')
            df = pd.read_csv(file_path)
            time = df['time'].values
            flux = df['flux'].values
            res = run_full_analysis(time, flux, filename=files[0])
            c_name = res['classification']['class']
            params = res.get('parameters')
            is_cand = params.is_candidate if params else False
            print(f'SUCCESS! Predicted: {c_name}, Candidate: {is_cand}')
