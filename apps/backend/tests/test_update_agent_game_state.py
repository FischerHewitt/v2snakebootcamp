import asyncio
import os
import sys
import types


socketio_stub = types.ModuleType("socketio")


class DummyAsyncServer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def attach(self, app) -> None:
        pass

    def event(self, fn):
        return fn

    async def emit(self, *args, **kwargs) -> None:
        pass

    async def save_session(self, *args, **kwargs) -> None:
        pass

    async def get_session(self, *args, **kwargs):
        return {}


socketio_stub.AsyncServer = DummyAsyncServer
sys.modules["socketio"] = socketio_stub


aiohttp_stub = types.ModuleType("aiohttp")
web_stub = types.ModuleType("aiohttp.web")


class DummyApplication:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def router(self):
        return self

    def add_get(self, *args, **kwargs) -> None:
        pass


class DummyAppRunner:
    def __init__(self, app) -> None:
        pass

    async def setup(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass


class DummyTCPSite:
    def __init__(self, runner, host=None, port=None) -> None:
        pass

    async def start(self) -> None:
        pass


def json_response(data):
    return data


web_stub.Application = DummyApplication
web_stub.AppRunner = DummyAppRunner
web_stub.TCPSite = DummyTCPSite
web_stub.json_response = json_response


aiohttp_stub.web = web_stub
sys.modules["aiohttp"] = aiohttp_stub
sys.modules["aiohttp.web"] = web_stub


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import update_agent_game_state
from game import Game


class DummyAgent:
    def __init__(self) -> None:
        self.short_memory_called = False
        self.remember_called = False
        self.long_memory_called = False

    def get_state(self, game: Game):
        return game.to_vector()

    def get_action(self, state):
        return [1, 0, 0]

    def calculate_reward(self, game: Game, done: bool) -> int:
        return 1 if not done else -10

    def train_short_memory(self, state, action, reward, next_state, done):
        self.short_memory_called = True

    def remember(self, state, action, reward, next_state, done):
        self.remember_called = True

    def train_long_memory(self) -> None:
        self.long_memory_called = True


def test_update_agent_game_state_smoke() -> None:
    game = Game()
    agent = DummyAgent()

    asyncio.run(update_agent_game_state(game, agent))

    assert agent.short_memory_called
    assert agent.remember_called
    assert game.running
