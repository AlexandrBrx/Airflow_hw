import glob
import pandas as pd
import dill
import json
import os
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')

def predict():
    with open(f'/Users/Alexandr/airflow/airflow_homework/data/models/cars_pipe_202408291023.pkl', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(X)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'/Users/Alexandr/airflow/airflow_homework/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')

print(os.getcwd(), os.path.abspath('..'))

if __name__ == '__main__':
    predict()




