from pydantic import BaseModel, Field
    
class BaseResponse(BaseModel):
    Success: bool = Field(False)
    ResponseCode: int = Field(None)
    Data: dict = Field({})
    ErrorMessage: str = Field(None)
    
class PredictRequest(BaseModel):
    stock: str = Field(None)