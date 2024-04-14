import pathlib
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

app = FastAPI()

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent
path = f"{home_dir.as_posix()}/model.joblib"
model = joblib.load(path)
scaler = joblib.load(f"{home_dir.as_posix()}/scaler_model.pkl")
encoder = joblib.load(f"{home_dir.as_posix()}/encoder_model.pkl" )
df = pd.read_csv(f"{home_dir.as_posix()}/df.csv")

class Cloth(BaseModel):
    shop_id: float
    title: str
    description: str
    price: float
    type: str
    wear_degree: str
    sex: str
    status: str
    created_at: str
    size: str
    brand: str
    color: str
    material: str
    season: str



# add dependencies = [Depends(api_key_auth)] after 'predict/'
@app.post('/predict')
def predict_wineq(user_input: Cloth) -> dict:

    processed_data = (user_input.model_dump())
    #processed_data = (user_input)

    processed_data = pd.DataFrame(processed_data, columns= ['shop_id', 'title', 'description', 'price', 'type', 'wear_degree', 'sex', 'status', 'created_at', 'size', 'brand', 'color', 'material', 'season'],index=[0])

    #processed_data = processed_data.transpose()
    #processed_data = pd.DataFrame(processed_data, columns= ['shop_id', 'title', 'description', 'price', 'type', 'wear_degree', 'sex', 'status', 'created_at', 'size', 'brand', 'color', 'material', 'season'])

    processed_data = processed_data.drop(["title", "description", "status", "created_at"], axis=1)

    X_test_encoded = pd.DataFrame(encoder.transform(processed_data[["type", "material", "color", 'brand', 'size', 'season', 'sex', 'wear_degree']]).toarray())

    processed_data = processed_data.join(X_test_encoded)
    processed_data.drop(["type", "material", "color", 'brand', 'size', 'season', 'sex', 'wear_degree'], axis= 1 , inplace= True )
    #print(processed_data)
    processed_data.columns = processed_data.columns.astype(str)
    input_data = scaler.transform(processed_data)
    #input_data = np.concatenate((X_test_scaled, X_test_encoded), axis=1)
    arr = model.kneighbors(input_data, n_neighbors= 2)[1]
    prediction = df.iloc[arr[0]]
    prediction = prediction.to_dict()

    return prediction
    

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)

# app_name: api (file name)
# port: 8000 (default)
# cmd: uvicorn service.api:app --reload
