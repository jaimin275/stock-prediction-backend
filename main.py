from datetime import datetime
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.prediction import predictModel
# from models.schemas import BaseResponse, PredictRequest
import numpy as np
# from controller.auth import AuthHandler

origins = ["*"]
app = FastAPI()
# auth_handler = AuthHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
@app.get("/")
def hello():
    return 'Welcome to the Stock Prediction API'

from models.schemas import BaseResponse, PredictRequest
@app.post("/predict")
def predict(req: PredictRequest):
    res = BaseResponse()
    
    # print("request received", req.riskLevel)
    print("request received", req.stock)
    # print("request received", req.stockAmount)
    # arr = []
    # arr.append(req.stock)
    modelFileNameMapping = {
        'AAPL': 'AAPL_model.h5',
        'CNQ': 'CNQ_model.h5',
        'FB': 'FB_model.h5',
        'GOOG': 'GOOG_model.h5',
        'JNJ': 'JNJ_model.h5',
        'MCD': 'MCD_model.h5',
        '^GSPC': 'SP500_model.h5',
    }
    input_data = {
        # "investmentMoney": int(req.stockAmount),
        # "riskLevel": int(req.riskLevel),
        "userSelectedStock": req.stock,
        "daysOfPrediction": 100,
        "modelFileName": modelFileNameMapping[req.stock]
    }
    
    # print("input created", input_data)
    temp = predictModel(input_data)
    # temp = predictFromFB(input_data)
    
    # print("temp received", temp)
    # print("temp received", temp)
    # print("temp received", temp["previous_days"].tolist())
    # print("temp received", type(temp["previous_days"]))
    # print("temp received", type(temp["previous_days"].tolist()))
    # temp: {
    #     'AAPL': 50,
    #     'CNQ': 4,
    #     'FB': 4,
    #     'GOOG': 0
    # }
    response = {
        "original_data": temp["original_data"][-1000:],
        "previous_days_data": temp["previous_days_data"],
        "predicted_days_data": temp["predicted_days_data"][50:]
    }
    res.Success = True
    res.Data = response
    # res.Data = None
    return res
    # return {"data": str(predictModel(np.array(req.data))), "success": "True" }
  