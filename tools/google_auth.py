"""Signing in to Google: OAuth 2.0 for an app on your own computer.

sora must never see your Google password. OAuth splits the job:

  1. sora opens Google's sign-in page in your browser, asking for specific
     permissions ("scopes": here, your calendar)
  2. you sign in and click Allow *on Google's page*
  3. Google redirects the browser to a tiny server sora runs on
     http://127.0.0.1:<random port>, carrying a one-time `code`
  4. sora trades the code for tokens:
       access token    proves permission for about an hour
       refresh token   gets new access tokens, until you revoke it
  5. the tokens are saved in <home>/google_token.json, readable only by you

Two protections in that exchange:
  state   a random value sora sends and expects back, so a link from some
          website can't feed sora a code it didn't ask for
  PKCE    sora sends a hash of a random secret (code_challenge) at step 1
          and the secret itself (code_verifier) at step 4; a code stolen
          on the way can't be redeemed without the secret

The "client" (client_id, client_secret) identifies *your* Google Cloud app,
not you. For desktop apps Google treats the secret as not really secret.
Download it from Google Cloud Console > APIs & Services > Credentials >
Create credentials > OAuth client ID > Desktop app, and save it as
<home>/google_client.json.

Standard library only: urllib for HTTP, http.server for the redirect.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly",      # list calendars and events
          "https://www.googleapis.com/auth/calendar.events"]        # create events
CLIENT_FILE = "google_client.json"
TOKEN_FILE = "google_token.json"


class AuthError(Exception):
    pass


def load_client(home: Path) -> dict:
    """client_id and client_secret from the file Google gives you."""
    path = home / CLIENT_FILE
    if not path.is_file():
        raise AuthError(f"Missing {path}. In Google Cloud Console create an OAuth client ID "
                        f"(type: Desktop app), download its JSON, and save it there.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    client = raw.get("installed") or raw.get("web") or raw
    if not client.get("client_id") or not client.get("client_secret"):
        raise AuthError(f"{path} has no client_id/client_secret; download the Desktop app client JSON again")
    return {"client_id": client["client_id"], "client_secret": client["client_secret"]}


def post_form(url: str, fields: dict) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read())
        except ValueError:
            detail = {}
        reason = detail.get("error_description") or detail.get("error") or f"HTTP {exc.code}"
        if detail.get("error") == "invalid_grant":
            reason += (" — sign in again with `uv run main.py google login`. If this happens every 7 days, "
                       "your Google Cloud app is in Testing mode: publish it (OAuth consent screen > Publish app).")
        raise AuthError(reason) from exc


def save_token(home: Path, token: dict) -> None:
    """Written with owner-only permissions (0600) from the first byte."""
    path = home / TOKEN_FILE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(token, f)
    os.chmod(path, 0o600)


def load_token(home: Path) -> dict | None:
    path = home / TOKEN_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _with_expiry(answer: dict) -> dict:
    return {**answer, "expires_at": time.time() + int(answer.get("expires_in", 3600)) - 60}


def access_token(home: Path, post=post_form) -> str:
    """A valid access token, refreshed when the saved one has expired."""
    token = load_token(home)
    if not token:
        raise AuthError("Not signed in to Google. Run: uv run main.py google login")
    if token.get("access_token") and token.get("expires_at", 0) > time.time():
        return token["access_token"]
    client = load_client(home)
    fresh = _with_expiry(post(TOKEN_URL, {"grant_type": "refresh_token", "refresh_token": token["refresh_token"],
                                          **client}))
    save_token(home, {**token, **fresh})            # Google usually keeps the same refresh token
    return fresh["access_token"]


def login(home: Path, open_browser=webbrowser.open, post=post_form, timeout: float = 300) -> str:
    """Run the browser sign-in once and save the tokens. Returns the scopes granted."""
    client = load_client(home)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    result: dict = {}

    class Redirect(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
            if "code" not in query and "error" not in query:
                self.send_response(404)
                self.end_headers()
                return
            result.update(query)
            ok = "code" in query and query.get("state") == state
            body = ("Signed in. You can close this tab and go back to sora." if ok
                    else "Sign-in didn't complete. Go back to the terminal for details.")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)     # port 0: any free port
    server.timeout = 1
    redirect = f"http://127.0.0.1:{server.server_address[1]}"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client["client_id"], "redirect_uri": redirect, "response_type": "code",
        "scope": " ".join(SCOPES), "state": state, "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline", "prompt": "consent"})              # offline + consent: get a refresh token
    print(f"Opening Google sign-in in your browser. If it doesn't open, visit:\n{url}\n", flush=True)
    open_browser(url)
    deadline = time.time() + timeout
    try:
        while not result and time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not result:
        raise AuthError("No answer from the browser within 5 minutes; run the login again")
    if result.get("error"):
        raise AuthError(f"Google said: {result['error']}")
    if result.get("state") != state:
        raise AuthError("The sign-in answer didn't match this login attempt (state mismatch); ignored")
    answer = post(TOKEN_URL, {"grant_type": "authorization_code", "code": result["code"],
                              "redirect_uri": redirect, "code_verifier": verifier, **client})
    if "refresh_token" not in answer:
        raise AuthError("Google sent no refresh token; remove sora's access at "
                        "myaccount.google.com/permissions and log in again")
    save_token(home, _with_expiry(answer))
    return answer.get("scope", "")


def logout(home: Path, post=post_form) -> bool:
    """Revoke the refresh token at Google and delete the local file."""
    token = load_token(home)
    if not token:
        return False
    try:
        post("https://oauth2.googleapis.com/revoke", {"token": token.get("refresh_token", "")})
    except AuthError:
        pass                                          # already revoked; still delete locally
    (home / TOKEN_FILE).unlink(missing_ok=True)
    return True
