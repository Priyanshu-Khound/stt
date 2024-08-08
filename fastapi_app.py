from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.responses import JSONResponse, RedirectResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tempfile import NamedTemporaryFile

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

app = FastAPI()

@app.post('/whisper')
async def handler(files: List[UploadFile] = File( ... )):
    if not files:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    results = []

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )


    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())

            result = pipe(temp.name,generate_kwargs={"language": "english"})

            results.append(
                {
                    'filename':file.filename,
                    'transcript':result['text']
                }
            )


    return JSONResponse(content={'results':results})
