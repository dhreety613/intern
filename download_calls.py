import os
import requests
from requests.auth import HTTPBasicAuth

API_ID = os.getenv("CLOUDTALK_KEY_ID")
API_KEY = os.getenv("CLOUDTALK_KEY_SECRET")

BASE_URL = "https://my.cloudtalk.io/api"

OUTPUT_FOLDER = "storage/recordings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

auth = HTTPBasicAuth(API_ID, API_KEY)


def download_the_calls(filtered_calls):

    downloaded = []
    failed = []

    for call in filtered_calls:

        call_id = call["call_id"]

        filename = (
            f'{call["agent"].replace(" ", "_")}_'
            f'{call["date"]}_'
            f'{call["time"]}.wav'
        )

        filepath = os.path.join(OUTPUT_FOLDER, filename)

        url = f"{BASE_URL}/calls/recording/{call_id}.json"

        try:

            response = requests.get(
                url,
                auth=auth,
                stream=True,
                timeout=180
            )

            if response.status_code != 200:
                failed.append({
                    "call_id": call_id,
                    "status": response.status_code
                })
                continue

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            downloaded.append({
                "call_id": call_id,
                "filename": filename,
                "url": f"/recordings/{filename}"
            })

        except Exception as e:

            failed.append({
                "call_id": call_id,
                "error": str(e)
            })

    return {
        "success": True,
        "downloaded": len(downloaded),
        "failed": len(failed),
        "files": downloaded,
        "errors": failed
    }