import requests

resp = requests.post(
    "http://localhost:5000/predict", files={"file": open("two.png", "rb")}
)
print(resp.text)
