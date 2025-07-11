import requests


if __name__ == "__main__":
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    BASE_URL = "https://games-test.datsteam.dev"
    session = requests.Session()
    session.headers.update({'X-Auth-Token': f'{API_KEY}'})
    payload = dict()
    response = session.post(f'{BASE_URL}/api/register', json=payload)
    print(f"Регистрация: {response.status_code} - {response.text}")
