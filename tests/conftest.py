import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-falesny-klic")

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import docassist.api.main as main


class FakeAgent:
    def __init__(self):
        self.calls = []         
        self.error = None        
        self.answer = "Máte nárok na 3 dny sick days [1]."

    def invoke(self, input: dict, config: dict) -> dict:
        self.calls.append({"input": input, "config": config})
        if self.error is not None:
            raise self.error

        question = input["messages"][-1].content
       
        return {"messages": [
            HumanMessage(content=question),
            AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": question}, "id": "call_1"}]),
            ToolMessage(content="[1] (strana 3) Zaměstnanec má nárok na 3 dny sick days.", tool_call_id="call_1"),
            AIMessage(content=self.answer)
        ]}


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def client(fake_agent, monkeypatch, tmp_path) -> TestClient:
    """TestClient = "prohlížeč v kódu": posílá HTTP požadavky na app bez spuštěného uvicornu.

    monkeypatch.setattr(objekt, "název", nová_hodnota) dočasně přepíše proměnnou.
    Po skončení testu ji pytest SÁM vrátí zpátky, takže testy se navzájem neovlivní.
    tmp_path = prázdná dočasná složka, pro každý test nová (pytest ji vytvoří i uklidí).
    """
    
    monkeypatch.setattr(main, "agent", fake_agent)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "get_collection", lambda: None)
    monkeypatch.setattr(main, "index_pdf", lambda collection, path: 5)  # index_pdf jen vrátí 5 (jako by vzniklo 5 chunků)  
    return TestClient(main.app)
