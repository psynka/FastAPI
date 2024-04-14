import requests


url = 'http://127.0.0.1:8000/predict'
#url = 'http://127.0.0.1:8081/predict'

headers = { 
    'Content-Type': 'application/json',
    'Authorization': 'Bearer xgb0fws23'
}

var = {"shop_id": 1,
       "title": "hz",
       "description": "description",
       "price": 999.0,
       "type": "JEANS",
       "wear_degree": "FIRST",
       "sex": "M",
       "status": "IN_SALE",
       "created_at": "2023-07-28",
       "size": "M",
       "brand": "Lewis",
       "color": "Blue",
       "material": "leather",
       "season": "SUMMER"}

response = requests.post(url, json = var, headers = headers)

print(response.status_code)
print(response.json())

'''
Status Code:
- 200 : successfully executed
- 401 : invalid api key
- 422 : validation error
'''
