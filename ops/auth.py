"""Accounts and sessions for multi-user sora (SORA_AUTH=on).

Each user gets their own sora home, <root>/users/<name>/: memory, lessons, skills,
traces, workspace, permissions, persona. This file only answers "who is this?":

    users       name, a salted scrypt hash of the password, admin or not
    sessions    a random token per login; only its SHA-256 is stored, so a copy of
                auth.db can't be used to log in. 30 days, ended by logging out.
    failures    tries per (name, IP) and per IP, counted before the password is checked
                so parallel guesses can't slip past: after 5 for a name from one IP (or
                30 from one IP, any names) the next try waits, doubling each time, capped
                at 15 minutes. A name alone is never locked: from another IP its owner
                can still log in. At most 4 password checks run at once (32 MB each).

Passwords go in through `sora users add` / `passwd` (a hidden prompt), never through
a file or an environment variable. Only the standard library is used: hashlib.scrypt
is a memory-hard password hash, hmac.compare_digest compares in constant time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{1,31}")
MIN_PASSWORD = 10
SESSION_DAYS = 30
FREE_TRIES = 5                 # per (name, IP)
FREE_TRIES_IP = 30             # per IP, any names: spraying many names
MAX_WAIT = 15 * 60
HASHING = threading.BoundedSemaphore(4)
# scrypt cost: 2^15 x 8 x 1 needs 32 MB and ~50 ms per hash on a laptop.
N, R, P = 2 ** 15, 8, 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    admin INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    name TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS failures (
    key TEXT PRIMARY KEY,                 -- "name:<name>" or "ip:<address>"
    count INTEGER NOT NULL,
    last_at REAL NOT NULL);
"""


class AuthError(Exception):
    """A login that didn't work, worded for the person logging in."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, maxmem=64 * 1024 * 1024)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${N}${R}${P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(got, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password(secrets.token_hex(16))       # unknown names take as long as known ones


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Auth:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(root / "auth.db", check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        try:
            (root / "auth.db").chmod(0o600)              # only this account can read the hashes
        except OSError:
            pass

    def home(self, name: str) -> Path:
        return self.root / "users" / name

    # ---- accounts

    @staticmethod
    def check_name(name: str) -> str:
        if not NAME.fullmatch(name or ""):
            raise ValueError("names are 2-32 characters: lowercase letters, digits, - and _, "
                             "starting with a letter or digit")
        return name

    @staticmethod
    def check_password(password: str) -> str:
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"passwords need at least {MIN_PASSWORD} characters")
        return password

    def add_user(self, name: str, password: str, admin: bool = False) -> Path:
        self.check_name(name)
        self.check_password(password)
        with self.lock:
            if self.conn.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise ValueError(f"there's already a user called {name!r}")
            self.conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                              (name, hash_password(password), int(admin), time.time()))
            self.conn.commit()
        home = self.home(name)
        home.mkdir(parents=True, exist_ok=True)
        return home

    def set_password(self, name: str, password: str) -> None:
        self.check_password(password)
        with self.lock:
            cur = self.conn.execute("UPDATE users SET password = ? WHERE name = ?", (hash_password(password), name))
            self.conn.execute("DELETE FROM sessions WHERE name = ?", (name,))     # log out everywhere
            self.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"no user called {name!r}")

    def remove_user(self, name: str) -> Path:
        """Removes the account and its sessions. Their data folder is left in place."""
        with self.lock:
            cur = self.conn.execute("DELETE FROM users WHERE name = ?", (name,))
            self.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"no user called {name!r}")
        return self.home(name)

    def users(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT name, admin, created_at FROM users ORDER BY name")]

    def is_admin(self, name: str) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT admin FROM users WHERE name = ?", (name,)).fetchone()
        return bool(row and row["admin"])

    # ---- logging in

    def _wait(self, key: str, free: int, now: float) -> float:
        row = self.conn.execute("SELECT count, last_at FROM failures WHERE key = ?", (key,)).fetchone()
        if not row or row["count"] < free:
            return 0.0
        wait = min(MAX_WAIT, 2 ** (row["count"] - free))
        return max(0.0, row["last_at"] + wait - now)

    def login(self, name: str, password: str, ip: str = "") -> str:
        """A new session token for the right password. Raises AuthError otherwise, with
        the same words whether the name or the password was wrong."""
        now = time.time()
        keys = {f"pair:{name}|{ip}": FREE_TRIES, **({f"ip:{ip}": FREE_TRIES_IP} if ip else {})}
        with self.lock:
            wait = max(self._wait(k, free, now) for k, free in keys.items())
            if wait:
                raise AuthError(f"too many wrong tries: wait {int(wait) + 1} seconds")
            for key in keys:          # counted now, cleared on success: parallel tries all count
                self.conn.execute("INSERT INTO failures VALUES (?, 1, ?) ON CONFLICT(key) DO UPDATE "
                                  "SET count = count + 1, last_at = excluded.last_at", (key, now))
            self.conn.commit()
            row = self.conn.execute("SELECT password FROM users WHERE name = ?", (name,)).fetchone()
        if not HASHING.acquire(timeout=10):
            raise AuthError("the server is busy: try again in a moment")
        try:
            ok = verify_password(password or "", row["password"] if row else _DUMMY) and row is not None
        finally:
            HASHING.release()
        with self.lock:
            if not ok:
                raise AuthError("wrong name or password")
            self.conn.execute(f"DELETE FROM failures WHERE key IN ({','.join('?' * len(keys))})", list(keys))
            token = secrets.token_urlsafe(32)
            self.conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                              (_token_hash(token), name, now, now + SESSION_DAYS * 86400))
            self.conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            self.conn.commit()
        return token

    def session(self, token: str | None) -> str | None:
        """The user a session token belongs to, or None (unknown, expired, logged out)."""
        if not token:
            return None
        with self.lock:
            row = self.conn.execute("SELECT name, expires_at FROM sessions WHERE token_hash = ?",
                                    (_token_hash(token),)).fetchone()
        return row["name"] if row and row["expires_at"] > time.time() else None

    def logout(self, token: str | None) -> None:
        if token:
            with self.lock:
                self.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
                self.conn.commit()

    def close(self) -> None:
        self.conn.close()
