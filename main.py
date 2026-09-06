from contextlib import asynccontextmanager
from fastapi import FastAPI,UploadFile,File
from LoadModels import LoadModels
from fastapi.responses import FileResponse,Response

## function to load models  once when server starts using lifespan
@asynccontextmanager
async def loader(app:FastAPI):
    ## Create an object of containing all the models
    app.state.models=LoadModels()
    print("Models loaded.")
    yield
    print("Power off")

## Load the models when server starts
app=FastAPI(lifespan=loader)


@app.get("/")
async def home():
    return FileResponse("index.html")
## Post API
@app.post("/ITS")
async def ImgToSpeech(image:UploadFile=File(...)):
    ## Get the image sent via request
    input=await image.read()
    ## use the reference to the LoadModels class
    Models=app.state.models
    Text=Models.OCR(input)
    TranslatedText=Models.Translator([Text])
    FinalText=" ".join(TranslatedText)
    audioBuffer=Models.SpeechSynthesize(FinalText)
    audio_data=audioBuffer.getvalue()
    ## Return the audio as wav file
    return Response(content=audioBuffer.getvalue(),media_type="audio/wav")


