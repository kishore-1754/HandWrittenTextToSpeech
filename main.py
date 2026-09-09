from contextlib import asynccontextmanager
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from LoadModels import LoadModels
from fastapi.responses import FileResponse
from LanguageMap import Languages
import base64
from typing import List
# from DetectLines import ExtractLines
from fastapi.middleware.cors import CORSMiddleware

## function to load models  once when server starts using lifespan
@asynccontextmanager
async def loader(app:FastAPI):
    ## Create an object of containing all the models
    app.state.models=LoadModels()
    app.state.LanguageMap=Languages
    # app.state.Extractor=ExtractLines
    print("Models loaded.")
    yield
    print("Power off")

## Load the models when server starts
app=FastAPI(lifespan=loader)

## Add middleware to enable connectivity between frontend and backend (CROSS ORIGIN RESOURCE SHARING)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], ## Allow requests from any origin
    allow_methods=["POST","GET"], ## Allow requests of POST and GET type,
    allow_credentials=False, ## Don't cookies and other credentials
    allow_headers=["*"] ## Allow all types of headers
)

@app.get("/")
async def home():
    return FileResponse("index.html")
## Post API
@app.post("/ITS")
# async def ImgToSpeech(Input:List[UploadFile]=File(...),TargetLanguage:str=Form(...)):
async def ImgToSpeech(Input:UploadFile=File(...),TargetLanguage:str=Form(...)): ## Single file format
    TargetCode=app.state.LanguageMap[TargetLanguage]["LangCode"]
    Speaker=app.state.LanguageMap[TargetLanguage]["SpeakerID"]
    # ExtractedImage=app.state.Extractor(input)
    print("Received TargetLanguage:", TargetLanguage)
    ## Select appropriate Speaker ID
    if "female" in Speaker:
        SpeakerID=Speaker["female"]
    else:
        SpeakerID=Speaker["male"]
    ## use the reference to the LoadModels object
    Models=app.state.models
    TranslatedText=list()
    ExtractedText=list()
    ## Get and process images sent
    # for image in Input:
        ## Get each image sent via request as Bytes Object
    Img=await Input.read()
    Text=Models.OCR(Img)
    if Text.strip():
        ExtractedText.append(Text) ## Append the text obtained from the image
    ## Join the list of strings into one string so that model retains context
    FinalText=" ".join(ExtractedText)
    print(FinalText)
    ## Translate the text into required language
    TranslatedText=Models.Translator([FinalText],targetLang=TargetCode,maxTokens=500)
    print(TranslatedText)
    ## Pass the translated text into audio
    audioBuffer=Models.SpeechSynthesize(TranslatedText[0],speakerID=SpeakerID)
    ## Convert the audio file into a base64 ASCII byte object and then convert it to a utf-8 string
    audioBase64=base64.b64encode(audioBuffer.getvalue()).decode("utf-8")
    ## Return the final response
    return {"TranslatedText":TranslatedText,"audioText":audioBase64}