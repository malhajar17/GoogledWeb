# server_utils.py

import requests

BASE_URL = "http://68.154.41.143:8000"

def register_new_user():
    try:
        response = requests.post(f"{BASE_URL}/new-user")
        response.raise_for_status()  # Raise an error if the request failed
        print(response)
        user_id = response.json().get("user_info")
        return user_id
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to register new user: {e}"}

def delete_user(user_id):
    try:
        response = requests.delete(f"{BASE_URL}/delete-user", params={"user_id": user_id})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to delete user: {e}"}

def submit_query(user_id, query):
    try:
        response = requests.post(f"{BASE_URL}/submit-query", json={"user_id": user_id, "query": query})
        response.raise_for_status()
        return response.json()  # Assuming the server returns some JSON response
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to submit query: {e}"}
    

def retrieve_and_filter(user_id):
    try:
        response = requests.post(f"{BASE_URL}/retrieve-and-filter", json={"user_id": user_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve and filter data: {e}"}

def gen_q(user_id):
    try:
        response = requests.post(f"{BASE_URL}/gen-q", json={"user_id": user_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to generate questions: {e}"}

def gen_a(user_id):
    try:
        response = requests.post(f"{BASE_URL}/gen-a", json={"user_id": user_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to generate answers: {e}"}


def get_complete_result(user_id):
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.get(f"{BASE_URL}/get-complete-result", params={"user_id": user_id}, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve complete results: {e}"}
