from fastapi import FastAPI
from pydantic import BaseModel

from fetch1 import fetch_history as full_deal_history
from three1 import fetch_history as delta_deal_history

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
    return full_deal_history(req)


@app.post("/deal-history-delta")
def deal_history_delta(req: DeltaRequest):
    return delta_deal_history(req)