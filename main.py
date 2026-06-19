from fastapi import FastAPI
from pydantic import BaseModel

from fetch1 import run_fetch
from three1 import run_delta

from people import fetch_people
from cleanraw import clean_people


app = FastAPI()


class FullRequest(BaseModel):
    deal_id: int


class DeltaRequest(BaseModel):
    deal_id: int
    latest_call_date: str | None = None
    latest_email_date: str | None = None
    latest_note_date: str | None = None
    latest_activity_date: str | None = None


@app.post("/deal-history")
def deal_history(req: FullRequest):

    return run_fetch(
        req.deal_id
    )


@app.post("/deal-history-delta")
def deal_history_delta(req: DeltaRequest):

    return run_delta(
        req.deal_id,
        req.latest_call_date,
        req.latest_email_date,
        req.latest_note_date,
        req.latest_activity_date
    )


class CleanPeopleRequest(BaseModel):
    people: list

@app.post("/person-fetch")
def person_fetch():

    return fetch_people()


@app.post("/person-clean")
def person_clean(
    req: CleanPeopleRequest
):

    return clean_people(
        req.people
    )


@app.get("/")
def health():

    return {
        "status": "ok",
        "endpoints": [
            "/deal-history",
            "/deal-history-delta",
            "/person-fetch",
            "/person-clean"
        ]
    }