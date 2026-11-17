import os
import sys

# repo root (the folder containing call_agent/) must be importable: `python -m pytest call_agent/tests`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio
import inspect

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run this coroutine test on a fresh event loop")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run `async def` tests with asyncio.run (no pytest-asyncio needed in the brain's environment)."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**args))
        return True
