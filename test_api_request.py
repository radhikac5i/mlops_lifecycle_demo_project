import requests

url = "http://127.0.0.1:8000/predict/"

payload = {
    "CreditScore": 650,
    "Age": 30,
    "Balance": 20000,
    "EstimatedSalary": 50000,
    "Tenure": 5,
    "NumOfProducts": 2
}

response = requests.post(url, json=payload)
print(response.json())
