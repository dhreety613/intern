from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.responses import FileResponse
import os

#frontend
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


from fetch1 import run_fetch
from three1 import run_delta

from people import fetch_people
from cleanraw import clean_people

from block3 import fetch_existing_cc_emails
from block4 import upload_contacts

from cloudtalk_get_last_seven_days_call import fetch_calls
from filter_calls import filter_the_calls
from download_calls import download_the_calls
from upload_to_fireflies import upload_to_the_fireflies


app = FastAPI(docs_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")










app = FastAPI(docs_url=None)

class UploadContactsRequest(BaseModel):
    contacts: List[Dict[str, Any]]

class FullRequest(BaseModel):
    deal_id: int


class DeltaRequest(BaseModel):
    deal_id: int
    latest_call_date: str | None = None
    latest_email_date: str | None = None
    latest_note_date: str | None = None
    latest_activity_date: str | None = None

class FilterRequest(BaseModel):
    calls: List[Dict[str, Any]]

class DownloadRequest(BaseModel):
    filtered_calls: List[Dict[str, Any]]


class FirefliesRequest(BaseModel):
    files: list


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )





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
@app.post("/constant-contact/existing-emails")
def constant_contact_existing_emails():

    return fetch_existing_cc_emails()

@app.post("/constant-contact/upload")
def constant_contact_upload(request: UploadContactsRequest):

    return upload_contacts(request.contacts)

@app.post("/cloudtalk_lastweek_calls_all")
def fetch_cloudtalk_weekcalls():
    return fetch_calls()


@app.post("/filter_cloudtalk_calls")
def filter_calls_endpoint(request: FilterRequest):

    try:
        return filter_the_calls(request.calls)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.post("/download_cloudtalk_calls")
def download_calls_endpoint(request: DownloadRequest):

    try:
        return download_the_calls(request.filtered_calls)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
  #so as the file stored are accessible to download via public endpoint that fireflies will call and download  
@app.get("/recordings/{filename}")
def get_recording(filename: str):

    filepath = os.path.join(
        "storage",
        "recordings",
        filename
    )

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="Recording not found."
        )

    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename
    )


@app.post("/upload_fireflies")
def upload_fireflies_endpoint(request: FirefliesRequest):

    try:
        return upload_to_the_fireflies(request.files)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/")
def health():

    return {
        "status": "ok",
        "endpoints": [
            "/deal-history",
            "/deal-history-delta",
            "/person-fetch",
            "/person-clean",
            "/constant-contact/existing-emails",
            "/constant-contact/upload",
            "/cloudtalk_lastweek_calls_all",
            "/filter_cloudtalk_calls",
            "/download_cloudtalk_calls",
            "/upload_fireflies"
        ]
    }