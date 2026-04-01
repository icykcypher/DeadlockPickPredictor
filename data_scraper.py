import asyncio
import aiohttp
import json
import os

STEAM_ID = "123123123"
BASE_URL = "https://deadlock-api.com" 

endpoints_to_try = [
    f"/api/v1/players/{STEAM_ID}/matches",
    f"/api/v1/players/{STEAM_ID}/match-history",
    f"/api/v1/player/{STEAM_ID}/matches",
    f"/api/v1/players/{STEAM_ID}/history",
    f"/v1/players/{STEAM_ID}/matches",
    f"/players/{STEAM_ID}/matches",
    "/api/v1/matches/75449443",
    "/v1/matches/75449443",
    "/matches/75449443",
]


async def test_endpoints():
    print("Trying endpoint\n")

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints_to_try:
            url = f"{BASE_URL}{endpoint}"
            try:
                print(f"Trying: {url}")
                async with session.get(url) as res:
                    print(f"  Status: {res.status}")

                    if res.status == 200:
                        data = await res.json()

                        filename = f"working_{endpoint.replace('/', '_')}.json"

                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        print(f"  Data was saved in {filename}")
                        print(f"  Response keys: {', '.join(data.keys())}")

                        if isinstance(data, dict) and "players" in data:
                            players = data["players"]
                            print(f"  Found players: {len(players)}")

                            if players:
                                print(f"  Player: {', '.join(players[0].keys())}")

                    print()

            except Exception as err:
                print(f"  Error: {str(err)}\n")


if __name__ == "__main__":
    asyncio.run(test_endpoints())