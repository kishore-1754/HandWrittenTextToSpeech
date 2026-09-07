from contextlib import asynccontextmanager
from fastapi import FastAPI,UploadFile,File,Form
from LoadModels import LoadModels
from fastapi.responses import FileResponse
from LanguageMap import Languages
import base64
from DetectLines import ExtractLines

## function to load models  once when server starts using lifespan
@asynccontextmanager
async def loader(app:FastAPI):
    ## Create an object of containing all the models
    app.state.models=LoadModels()
    app.state.LanguageMap=Languages
    app.state.Extractor=ExtractLines
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
async def ImgToSpeech(image:UploadFile=File(...),TargetLanguage:str=Form(...)):
    ## Get the image sent via request as Bytes Object
    input=await image.read()
    TargetCode=app.state.LanguageMap[TargetLanguage]["LangCode"]
    Speaker=app.state.LanguageMap[TargetLanguage]["SpeakerID"]
    ExtractedImage=app.state.Extractor(input)
    print("Received TargetLanguage:", TargetLanguage)
    ## Select appropriate Speaker ID
    if "female" in Speaker:
        SpeakerID=Speaker["female"]
    else:
        SpeakerID=Speaker["male"]
    ## use the reference to the LoadModels class
    Models=app.state.models
    TranslatedText=list()
    for Img in ExtractedImage:
        Text=Models.OCR(Img)
        if Text.strip():
            TranslatedBatch=Models.Translator([Text],targetLang=TargetCode)
            TranslatedText.append(TranslatedBatch[0])
    FinalText=" ".join(TranslatedText)
    audioBuffer=Models.SpeechSynthesize(FinalText,speakerID=SpeakerID)
    ## Convert the audio file into a base64 ASCII byte object and then convert it to a utf-8 string
    audioBase64=base64.b64encode(audioBuffer.getvalue()).decode("utf-8")
    ## Return the final response
    return {"TranslatedText":FinalText,"audioText":audioBase64}


