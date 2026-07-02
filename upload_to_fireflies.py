import os
import requests

FIREFLIES_API_KEY = os.getenv("FIREFLIES_API_KEY")

GRAPHQL_URL = "https://api.fireflies.ai/graphql"

def upload_to_the_fireflies(files):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREFLIES_API_KEY}"
    }

    uploaded = []
    failed = []

    for file in files:

        input_data = {
            "url": file["url"],
            "title": file["filename"]
        }

        data = {
            "query": """
                mutation($input: AudioUploadInput){
                    uploadAudio(input:$input){
                        success
                        title
                        message
                    }
                }
            """,
            "variables": {
                "input": input_data
            }
        }

        try:

            r = requests.post(
                GRAPHQL_URL,
                headers=headers,
                json=data,
                timeout=180
            )

            result = r.json()

            if "errors" in result:

                failed.append({
                    "filename": file["filename"],
                    "error": result["errors"]
                })

            else:

                uploaded.append({
                    "filename": file["filename"],
                    "response": result["data"]["uploadAudio"]
                })

        except Exception as e:

            failed.append({
                "filename": file["filename"],
                "error": str(e)
            })

    return {
        "success": True,
        "uploaded": len(uploaded),
        "failed": len(failed),
        "uploads": uploaded,
        "errors": failed
    }