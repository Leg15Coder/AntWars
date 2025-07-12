import random

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/")
async def hello(request: Request):
    return JSONResponse({"msg": "hello, world!"})


@app.get("/api/arena")
async def arena(request: Request):
    playload = {
            "ants": [
                {
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 100,
                    "id": "11111111-2222-3333-4444-555555555555",
                    "lastAttack": {},
                    "lastEnemyAnt": "string",
                    "lastMove": [],
                    "move": [],
                    "q": 10,
                    "r": 20,
                    "type": 0
                },
                {
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 100,
                    "id": "11111111-2222-4333-4444-555555555555",
                    "lastAttack": {},
                    "lastEnemyAnt": "string",
                    "lastMove": [],
                    "move": [],
                    "q": 10,
                    "r": 20,
                    "type": 1
                },
                {
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 100,
                    "id": "11111111-2222-3333-4464-555555555555",
                    "lastAttack": {},
                    "lastEnemyAnt": "string",
                    "lastMove": [],
                    "move": [],
                    "q": 10,
                    "r": 20,
                    "type": 2
                }
            ],
            "enemies": [
                {
                    "attack": 0,
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 10,
                    "q": 12,
                    "r": 20,
                    "type": 1
                },
                {
                    "attack": 0,
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 10,
                    "q": 12,
                    "r": 20,
                    "type": 2
                },
                {
                    "attack": 0,
                    "food": {
                        "amount": 0,
                        "type": 0
                    },
                    "health": 10,
                    "q": 12,
                    "r": 20,
                    "type": 0
                }
            ],
            "food": [
                {
                    "amount": 10,
                    "q": 14,
                    "r": 20,
                    "type": 1
                },
                {
                    "amount": 8,
                    "q": 12,
                    "r": 18,
                    "type": 3
                },
                {
                    "amount": 4,
                    "q": 8,
                    "r": 15,
                    "type": 2
                }
            ],
            "home": [
                {
                    "q": 10,
                    "r": 20
                }
            ],
            "map": [
                {
                    "cost": 1,
                    "q": q,
                    "r": r,
                    "type": random.randint(1, 5)
                } for q in range(4, 16) for r in range(12, 26)
            ],
            "nextTurnIn": 0.666,
            "score": 0,
            "spot": {
                "q": 10,
                "r": 20
            },
            "turnNo": 0
        }

    return JSONResponse(playload)


@app.get("/api/logs")
@app.post("/api/register")
@app.post("/api/move")
async def empty(request: Request):
    return JSONResponse({'lobbyEndsIn': 0})


"""
uvicorn server_simulator:app --reload
http://127.0.0.1:8000
"""
