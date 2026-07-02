import os
import requests

FIREFLIES_API_KEY = os.getenv("FIREFLIES_API_KEY")

GRAPHQL_URL = "https://api.fireflies.ai/graphql"

def upload_to_the_fireflies(files):

    uploaded = []
    failed = []

    headers = {
        "Authorization": f"Bearer {FIREFLIES_API_KEY}",
        "Content-Type": "application/json"
    }

    mutation = """
    mutation UploadAudio($url: String!) {
      uploadAudio(
        input:{
          url:$url
        }
      ){
        success
        title
        transcriptId
      }
    }
    """

    for file in files:

        try:

            payload = {
                "query": mutation,
                "variables": {
                    "url": file["url"]
                }
            }

            r = requests.post(
                GRAPHQL_URL,
                json=payload,
                headers=headers,
                timeout=180
            )

            response = r.json()

            uploaded.append({
                "filename": file["filename"],
                "url": file["url"],
                "fireflies": response
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
        "results": uploaded,
        "errors": failed
    }