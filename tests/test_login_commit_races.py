"""Offline races using the real private durable credential writer."""
import asyncio
import base64
import json
import pathlib
import tempfile
import threading
import unittest
from unittest.mock import patch

from codex_as_api import auth
from codex_as_api.device_login import POLL_URL, USERCODE_URL, DeviceLoginManager


def jwt(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return "header." + encoded + ".signature"


def tokens(account):
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return {"access_token": jwt(claims), "id_token": jwt(claims), "refresh_token": "refresh-" + account}


async def transport(url, body, headers, timeout):
    del body, headers, timeout
    if url == USERCODE_URL:
        return 200, b'{"device_auth_id":"device","user_code":"user","interval":1}'
    if url == POLL_URL:
        return 200, b'{"authorization_code":"code","code_verifier":"verifier","code_challenge":"challenge"}'
    return 200, json.dumps(tokens("new")).encode()


class LoginCommitRaceTests(unittest.TestCase):
    def test_cancel_before_guard_preserves_existing_file(self):
        async def run(path):
            auth.commit_login(path, tokens("old"))
            entered = threading.Event()
            lock = auth._write_lock(path)
            lock.acquire()
            async def commit(payload, guard):
                def write():
                    entered.set()
                    return auth.commit_login(path, payload, guard)
                return await asyncio.to_thread(write)
            manager = DeviceLoginManager(commit, transport=transport)
            try:
                session = await manager.start()
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                cancel = asyncio.create_task(manager.cancel(session["id"]))
                await asyncio.sleep(0)
                self.assertEqual(manager.get(session["id"])["status"], "pending")
            finally:
                lock.release()
            self.assertEqual((await cancel)["status"], "cancelled")
            self.assertEqual(auth.load_token_data(path).account_id, "old")
            self.assertEqual(list(path.parent.glob(".auth.json.tmp-*")), [])
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(pathlib.Path(directory) / "auth.json"))

    def test_post_replace_fsync_failure_is_ambiguous_not_cancelled(self):
        async def run(path):
            auth.commit_login(path, tokens("old"))
            entered, release = threading.Event(), threading.Event()
            original_fsync = auth.os.fsync
            calls = []

            def fsync(fd):
                calls.append(fd)
                if len(calls) == 2:
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test did not release fsync")
                    raise OSError("fake directory fsync failure")
                return original_fsync(fd)

            async def commit(payload, guard):
                return await asyncio.to_thread(auth.commit_login, path, payload, guard)

            manager = DeviceLoginManager(commit, transport=transport)
            with patch.object(auth.os, "fsync", fsync):
                session = await manager.start()
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    cancel = asyncio.create_task(manager.cancel(session["id"]))
                    await asyncio.sleep(0)
                finally:
                    release.set()
                result = await asyncio.wait_for(cancel, 2)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_code"], "login_persistence_uncertain")
            self.assertEqual(auth.load_token_data(path).account_id, "new")

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(pathlib.Path(directory) / "auth.json"))

    def test_task_cancel_after_claim_waits_for_real_write(self):
        async def run(path):
            auth.commit_login(path, tokens("old"))
            entered, release = threading.Event(), threading.Event()
            original_replace = pathlib.Path.replace
            original_fsync = auth.os.fsync
            operations = []
            def observed_fsync(fd):
                operations.append("fsync")
                return original_fsync(fd)
            def delayed_replace(source, target):
                operations.append("replace")
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release writer")
                return original_replace(source, target)
            async def commit(payload, guard):
                return await asyncio.to_thread(auth.commit_login, path, payload, guard)
            manager = DeviceLoginManager(commit, transport=transport)
            with patch.object(pathlib.Path, "replace", delayed_replace), patch.object(auth.os, "fsync", observed_fsync):
                session = await manager.start()
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    attempt = manager._records[session["id"]]
                    self.assertTrue(attempt.commit_claimed)
                    attempt.task.cancel()
                    await asyncio.sleep(0)
                    self.assertEqual(manager.get(session["id"])["status"], "pending")
                finally:
                    release.set()
                await asyncio.wait_for(attempt.task, 2)
            self.assertEqual(manager.get(session["id"])["status"], "succeeded")
            self.assertEqual(operations, ["fsync", "replace", "fsync"])
            self.assertEqual(auth.load_token_data(path).account_id, "new")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(path.parent.glob(".auth.json.tmp-*")), [])
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(pathlib.Path(directory) / "auth.json"))


if __name__ == "__main__":
    unittest.main()
