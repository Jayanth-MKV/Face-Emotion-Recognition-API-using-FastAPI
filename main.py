from enum import Enum
from pydantic import BaseModel, Field
from typing import Annotated, Union
import numpy as np
from fastapi import FastAPI, Request, Path,Security, Header, Depends, UploadFile, File, HTTPException
from PIL import Image
import time
from fastapi.responses import JSONResponse, HTMLResponse
from ml import predict_image, mapper, predict_images
from fastapi.security.api_key import APIKeyHeader


class Gender(str, Enum):
    female = "FEMALE"
    male = "MALE"
    all = "ALL"


app = FastAPI()
api_key_header = APIKeyHeader(name='X-Api-Key')


@app.get("/")
async def main(api_key: str = Security(api_key_header)):
    if api_key != "API_KEY":
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Key")

    content = """
<body>
<form action="/process_images/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    # Change here to Logger
    return JSONResponse(
        status_code=500,
        content={
            "message": (
                f"Failed method {request.method} at URL {request.url}."
                f" Exception message is {exc!r}."
            )
        },
    )


# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     return response


@app.get("/{gender}")
async def classify_based_on_gender(gender:  Annotated[Gender, Path(title="The gender to get the model")], api_key: str = Security(api_key_header)):
    if api_key != "API_KEY":
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Key")
    if gender is Gender.female:
        return {"type": gender}

    if gender is Gender.male:
        return {"type": gender}

    return {"type": gender}


@app.post("/predict")
async def predict(file: UploadFile, api_key: str = Security(api_key_header)):
    if api_key != "API_KEY":
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Key")
        
    content_type = file.content_type
    if not content_type or (content_type and not content_type.startswith("image/")):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not an image.")

    predictions_list = await predict_image(file)
    ind = np.argmax(predictions_list[0])

    return {"emotion": mapper[ind]}


@app.post("/process_images")
async def process_images(files: Annotated[
    list[UploadFile], File(description="Multiple files as UploadFile")
], api_key: str = Security(api_key_header)):
    
    if api_key != "API_KEY":
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Key")
    
    predictions_list = await predict_images(files)

    return {"predictions": predictions_list}
