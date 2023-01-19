from transformers import pipeline

from typing import List

from fastapi import FastAPI, Response

nlp = pipeline(
    "ner",
    model="KB/bert-base-swedish-cased-ner",
    tokenizer="KB/bert-base-swedish-cased-ner",
)

app = FastAPI()


@app.get("/health")
def health(response: Response):
    response.headers["cache-control"] = "no-cache"
    return {"status": "ok"}


@app.post("/ner")
def ner(input: List[str]):
    results = nlp(input)

    for result in results:
        for token in result:
            token["score"] = token["score"].item()  # convert to python type

    return results
