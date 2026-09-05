## Writing a class that contains methods to load and return all 3 models.

## Module imports for IndicTrans2
import os
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from IndicTransToolkit import IndicProcessor

class LoadModels:
    ## Defining base path for portability
    BaseDirectory=os.path.dirname(os.path.abspath(__file__))

   ## INDIC model path. (Distilled and quantized) 
    IndicPath=os.path.join(BaseDirectory,"models","Translation","indictrans2_int8_onnx")   
   ## Load Indic Model MetaData to handle out of index tokens
    metaData=json.load(open(os.path.join(IndicPath, "tokenizer_meta.json"),"r"))

   ## Constructor
    def __init__(self):
        self.Indic=self.IndicLoader()
        self.TrOCR=None
        self.Rasa=None

    ## Define target and source language ids
    sourceLang="eng_Latn" ## For English
    targetLang="kan_Knda" ## For Kannada

    ## INDIC Model Loader
    def IndicLoader(self):
        ## Preprocessor for text preprocessing
        preprocssor=IndicProcessor(inference=True)
        '''Consider something like 'hello   world' and 'hello world' both aren't the same
           Hence such sentences have to be preprocessed'''
        ## Import tokenizer -> The tokenizer breaks the sentence into tokens and assigns them IDs based on the vocabulary
        ## Current vocabulary size is 32322 for Source (English) and 122672 for Target (Indic languages)
                
        ## Tokenizer for English (Encoding)
        srcTokenizer=Tokenizer.from_file(os.path.join(self.IndicPath,"tokenizer_src.json"))
        ## Tokenizer for Indic language (Decoding)
        targetTokenizer=Tokenizer.from_file(os.path.join(self.IndicPath,"tokenizer_tgt.json"))

        ## Encoder for English using ORT (Onnx runtime)
        encoder=ort.InferenceSession(os.path.join(self.IndicPath,"encoder_model.onnx"),providers=["CPUExecutionProvider"])

        ## Decoder for Indic (Kannada) 
        decoder=ort.InferenceSession(os.path.join(self.IndicPath,"decoder_model.onnx"),providers=["CPUExecutionProvider"])
        ## Uncomment the below lines if you want to see the vocabulary size
        # print(metaData)
        ## Return the model encoder and decoders
        return {"encoder":encoder,"decoder":decoder}






