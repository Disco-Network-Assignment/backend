import json

from app import dependencies
from app.settings import Settings
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
        assert [p["publisher_name"] for p in plan["publishers"] if p["verdict"] == "recommend"]
        assert plan["trace"][0]["agent"] == "brief_writer"
        assert all({"stage", "status"} <= e.keys() for e in events)


class TestPlanRun:
    async def test_returns_a_plan_document(self, client):
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD,
                                                            "options": {"session_id": "s1"}})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done" and body["plan"]["config"]["status"] == "draft"

    async def test_empty_description_is_rejected(self, client):
        assert (await client.post("/api/plan/run", json={"description": ""})).status_code == 422

    async def test_unknown_option_is_rejected(self, client):
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD,
                                                            "options": {"nope": True}})
        assert response.status_code == 422

    async def test_without_an_api_key_the_run_is_refused_clearly(self, client, monkeypatch):
        from app.main import app
        app.dependency_overrides.clear()
        monkeypatch.setattr(dependencies, "settings", lambda: Settings(openai_api_key="", _env_file=None))
        response = await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD})
        assert response.status_code == 503 and "OPENAI_API_KEY" in response.json()["detail"]


class TestReadOnly:
    async def test_examples(self, client):
        body = (await client.get("/api/examples")).json()
        assert len(body) == 15 and body[0]["number"] == 1

    async def test_health_reports_whether_agents_can_run(self, client, monkeypatch):
        from app import main
        monkeypatch.setattr(main, "settings", lambda: Settings(openai_api_key="k", _env_file=None))
        body = (await client.get("/health")).json()
        assert body["status"] == "ok" and body["llm_configured"] is True


class TestRunHistory:
    async def test_runs_are_listed_newest_first_and_loadable(self, client):
        first = (await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD})).json()["plan"]
        second = (await client.post("/api/plan/run", json={"description": SENIOR_DOG_FOOD + " Also cats."})).json()["plan"]

        listing = (await client.get("/api/runs")).json()
        assert [r["run_id"] for r in listing] == [second["run_id"], first["run_id"]]
        assert listing[0]["status"] == "done" and listing[0]["recommended"] == 3 and "plan" not in listing[0]

        record = (await client.get(f"/api/runs/{first['run_id']}")).json()
        assert record["plan"]["run_id"] == first["run_id"] and record["stopped"] is None

    async def test_unknown_run_is_404(self, client):
        assert (await client.get("/api/runs/nope")).status_code == 404

    async def test_limit_is_validated(self, client):
        assert (await client.get("/api/runs?limit=0")).status_code == 422

