## Writing a class that contains methods to load and return all 3 models.

## Time module for optimization
import time
## Module imports for IndicTrans2
import os
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from IndicTransToolkit import IndicProcessor

## Module imports for VITS-RASA TTS model
import io
import sherpa_onnx as so
import soundfile as sf
import sounddevice as sd

## Module imports for TrOCR

class LoadModels:
    ## Defining base path for portability
    BaseDirectory=os.path.dirname(os.path.abspath(__file__))

   ## INDIC TRANSLATOR MODEL PIPELINE

   ## INDIC model path. (Distilled and quantized) 
    IndicPath=os.path.join(BaseDirectory,"models","Translation","indictrans2_int8_onnx")   
   ## Load Indic Model MetaData to handle out of index tokens
    metaData=json.load(open(os.path.join(IndicPath, "tokenizer_meta.json"),"r"))

   ## Constructor
    def __init__(self):
        self.Indic=self.IndicLoader()
        self.TrOCR=None
        self.Rasa=self.RasaLoader()

    ## INDIC Model Loader
    def IndicLoader(self):
        ## processor for text preprocessing
        procssor=IndicProcessor(inference=True)
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
        ## Uncomment the below line if you want to see the vocabulary size
        # print(metaData)

        ## Return the model processor, tokenizers, encoder and decoder.
        return {"encoder":encoder,"decoder":decoder,"processor":procssor,"srcTokenizer":srcTokenizer,"tgtTokenizer":targetTokenizer}

    def Translator(self,Input:list[str],maxTokens=160,sourceLang="eng_Latn",targetLang="kan_Knda"):
        ## preprocess the input string in batches
        batch=self.Indic["processor"].preprocess_batch(Input,src_lang=sourceLang,tgt_lang=targetLang)
        translated=list()

        ## Translation in batches
        for text in batch:
            BatchText=f"{sourceLang} {targetLang} {text}"
            ## Encode the text to an integer ID
            encodedID=self.Indic["srcTokenizer"].encode(BatchText)
            ## Clamp ids to prevent out of bound error
            inputIds=np.array([[i if i<self.metaData["src_dict_size"] else self.metaData["unk_id"] for i in encodedID.ids]],dtype=np.int64)

            ## Attention = dynamically calculated weights. Higher the attention, higher the relationship between words
            ## Attention mask is used to ignore the padded tokens.
            ## The placeholder tokens are appended to the sentences to make the sentences of same length to enable parallel computing for mathematical operations 
            attentionMask=np.array([encodedID.attention_mask],dtype=np.int64)

            ## Encode the Ids into a high dimensional vector
            encoderOutput=self.Indic["encoder"].run(["last_hidden_state"],{"input_ids":inputIds,"attention_mask":attentionMask})[0]
            decoderStartID=2
            eosID=2 ## ID of end of sequence character / tagger
            decoderIDs=np.array([[decoderStartID]],dtype=np.int64)
            generatedTokens=[]
            
            for itertation in range(maxTokens):
                decoderOutput=self.Indic["decoder"].run(None,{"input_ids":decoderIDs,"encoder_hidden_states":encoderOutput,"encoder_attention_mask":attentionMask})
                ## Obtain a list of scores for each word ID that model knows
                logits=decoderOutput[0]

                ## Select the ID that has the highest score
                BestScoreID=int(np.argmax(logits[0,-1,:]))

                if BestScoreID == eosID:
                    break
                ## Concatenate the BestScoreID into the output ID list for ID -> NL convertion
                generatedTokens.append(BestScoreID)
                ## Append the BestScoreID for enabling model to retain the context
                decoderIDs=np.concatenate([decoderIDs,np.array([[BestScoreID]],dtype=np.int64)],axis=1)
            ## Convert IDs to Indic text
            translatedText=self.Indic["tgtTokenizer"].decode(generatedTokens)
            ## Fix the grammar
            finalText=self.Indic["processor"].postprocess_batch([translatedText],lang=targetLang)[0]

            ## Append translated texts of all batches
            translated.append(finalText)
        return translated
#_________________________________________________________________________________________________________________________________________________________
    ## RASA TTS MODEL PIPELINE

    ## Define the RASA model path 
    RasaPath=os.path.join(BaseDirectory,"models","TTS","vits-rasa-13-onnx")

    ## Member method to load Rasa model
    def RasaLoader(self):
        ModelPath=os.path.join(self.RasaPath,"model.onnx")
        ## Tokens file contains the character to Integer ID mapping to enable the model to analyze and generate speech
        TokensPath=os.path.join(self.RasaPath,"tokens.txt")

        ## Configuration for the pipeline using Onnx config
        PipeLineConfig=so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig( ## Model configuration
            vits=so.OfflineTtsVitsModelConfig(## VITS model configuration
                model=ModelPath,
                tokens=TokensPath
            ),
            provider="cpu",  ## RUN the onnx model in CPU only
            num_threads=2,  ## Parallelize the work using 2 threads
            )
        )
        ## Load the model
        model=so.OfflineTts(PipeLineConfig)
        return model

    ## Method to use Rasa Model for TTS
    def SpeechSynthesize(self,Input:str,speakerID=8,speed=1.0): ## 8 = Kannda female voice
        ## Call the neural network to generate the audio
            ## Here Audio contains a list of amplitudes
            Audio=self.Rasa.generate(Input,sid=speakerID,speed=speed)
            ## Check if audio was generated
            if len(Audio.samples)>0:
                ## Trim the samples from the list to remove the noise from the end
                TrimSize=int(Audio.sample_rate*0.45)
                ## Generate audio in RAM
                AudioBuffer=io.BytesIO()
                ## Write audio into a Wav format for sending
                sf.write(AudioBuffer,Audio.samples[:-TrimSize],Audio.sample_rate,format="WAV") 
                ## Move the audio pointer back to start
                AudioBuffer.seek(0)
                return AudioBuffer
            else: raise RuntimeError("Failed to generate audio")



if __name__ == "__main__":
    start=time.perf_counter()
    Object=LoadModels()
    LoadTime=time.perf_counter()
    test=Object.Translator(["Hello, brother, I hope you are doing well."])
    translationTime=time.perf_counter()
    print(test[0])
    ## Get data from the Buffer
    Content,sampleRate=sf.read(Object.SpeechSynthesize(Input=test[0]))
    ## Play the audio
    TTSTime=time.perf_counter()
    sd.play(Content,samplerate=sampleRate)
    sd.wait()
    print(f"Time to Load models:{LoadTime-start}\nTranslation Time: {translationTime-LoadTime}\nTTS Time:{TTSTime-translationTime}")
    