import json

from tests.helpers import SENIOR_DOG_FOOD


async def stream_events(client, payload):
    events = []
    async with client.stream("POST", "/api/plan", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        async for line in response.aiter_lines():
            if line.strip():
                events.append(json.loads(line))
    return events


class TestPlanStream:
    async def test_streams_stage_events_and_ends_with_done(self, client):
        events = await stream_events(client, {"description": SENIOR_DOG_FOOD})
        assert events[0] == {"stage": "intake", "status": "started"}
        assert events[-1]["stage"] == "done"
        plan = events[-1]["data"]
        assert plan["mode"] == "heuristic"
        assert [p["publisher_name"] for p in plan["publishers"] if p["verdict"] == "recommend"]
        assert all({"stage", "status"} <= e.keys() for e in events)

    async def test_junk_ends_with_stopped(self, client):
        events = await stream_events(client, {"description": "idk just try it"})
        assert events[-1]["stage"] == "stopped"
        assert events[-1]["data"]["clarifying_questions"]


class TestPlanRun:
    async def test_returns_a_plan_document(self, client):
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done" and body["plan"]["config"]["status"] == "draft"

    async def test_empty_description_is_rejected(self, client):
        assert (await client.post("/api/plan/run", json={"description": ""})).status_code == 422

    async def test_unknown_option_is_rejected(self, client):
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD,
                                                            "options": {"nope": True}})
        assert response.status_code == 422

    async def test_llm_mode_request_without_key_falls_back(self, client):
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD,
                                                            "options": {"mode": "llm"}})
        assert response.status_code == 200 and response.json()["plan"]["mode"] == "heuristic"


class TestReadOnly:
    async def test_examples(self, client):
        body = (await client.get("/api/examples")).json()
        assert len(body) == 15 and body[0]["number"] == 1

    async def test_health(self, client):
        body = (await client.get("/health")).json()
        assert body["status"] == "ok" and body["mode"] == "heuristic"
