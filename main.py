#import backend ----------------------------------------------------------------------------------------------------------
from fastapi import FastAPI

app = FastAPI()




#html render---------------------------------------------------------------------------------------------------------------
from fastapi.templating import Jinja2Templates 
from fastapi import Request 

templates = Jinja2Templates(directory="templates")




#data validation --------------------------------------------------------------------------------------------------------
from fastapi import Form  
from pydantic import BaseModel

class Features(BaseModel):
  fLength : float
  fWidth : float
  fSize : float
  fConc : float
  fConc1 : float
  fAsym : float
  fM3Long : float
  fM3Trans : float
  fAlpha : float
  fDist : float

  @classmethod
  def fetch_form(
    cls,
    feature01 = Form(...),
    feature02 = Form(...),
    feature03 = Form(...),
    feature04 = Form(...),
    feature05 = Form(...),
    feature06 = Form(...),
    feature07 = Form(...),
    feature08 = Form(...),
    feature09 = Form(...),
    feature10 = Form(...)
  ):   #fetch form data from previous route 

    return cls(fLength=feature01, fWidth=feature02,  fSize=feature03,
              fConc=feature04,  fConc1=feature05,  fAsym=feature06,
              fM3Long=feature07, fM3Trans=feature08, fAlpha=feature09,
              fDist=feature10)   #return class.attribute value(alr validation) ...




#import ml model--------------------------------------------------------------------------------------------------------------
import pickle

model = pickle.load(open('model-1.0.pkl', 'rb'))




#backend development ----------------------------------------------------------------------------------------------------
@app.get("/")
async def root(request:Request):

  return templates.TemplateResponse("home.html", {"request": request})


import pandas as pd
import numpy as np
from fastapi import Depends
#2..  x = class.def//return class.attribute value//alr validation

@app.post("/result")
async def root(request:Request, item:Features = Depends(Features.fetch_form)): #1..  x:class = Depends(class.def)
  dict_01 = item.dict()
  values = pd.Series(dict_01.values()).astype(float)
  array = np.reshape(values, (1, -1))
  array = model.predict(array)   
  result = int(array[0,0].item())
  
  return templates.TemplateResponse("result.html", {"request": request, "result":result})