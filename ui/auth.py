"""
ui/auth.py — Auth0 (OIDC) authentication + role-based authorization.

Streamlit's native OIDC (st.login / st.user, v1.42+) does authentication; Auth0
carries the user's roles/org as custom claims in the ID token. OIDC gives us
identity, NOT authorization — so authorization (what each role may do) is
enforced here, in app code.

Setup (see the notes at the bottom of this file for the Auth0 Action and
.streamlit/secrets.toml that make the claims below appear).

Usage in ui/dashboard.py, at the VERY TOP before any other UI renders:

    from ui.auth import require_login, has_role, logout_button, session_key
    user = require_login()          # stops the script if not signed in
    logout_button()                 # sidebar "signed in as …" + log out
    can_run = has_role("analyst", "admin")   # gate the Run Analysis button
"""

from __future__ import annotations

import os
import streamlit as st

# Namespaced custom claims (OIDC requires a URI namespace; must match the Auth0
# Action). Override via env if you use a different namespace.
_NS = os.getenv("AUTH_CLAIM_NAMESPACE", "https://signallab.app/")
ROLE_CLAIM = _NS + "roles"
ORG_ID_CLAIM = _NS + "org_id"
ORG_NAME_CLAIM = _NS + "org_name"

# Local-dev escape hatch: set SIGNALLAB_AUTH_DISABLED=1 to bypass login entirely
# and run as a synthetic admin. NEVER set this on a shared/deployed instance —
# it turns off all access control.
_DEV_BYPASS = os.getenv("SIGNALLAB_AUTH_DISABLED", "").strip() in ("1", "true", "yes")
_DEV_USER = {
    "email": "dev@localhost", "name": "Local Dev",
    "roles": ["admin"], "org_id": "dev", "org_name": "Local",
}


def _claim(key: str):
    try:
        return st.user.get(key)          # st.user is dict-like
    except Exception:
        return None


def _roles() -> set[str]:
    raw = _claim(ROLE_CLAIM) or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(r).strip().lower() for r in raw if str(r).strip()}


def is_authenticated() -> bool:
    if _DEV_BYPASS:
        return True
    return bool(getattr(st.user, "is_logged_in", False))


def current_user() -> dict:
    if _DEV_BYPASS:
        return dict(_DEV_USER)
    return {
        "email": _claim("email"),
        "name": _claim("name") or _claim("email"),
        "roles": sorted(_roles()),
        "org_id": _claim(ORG_ID_CLAIM),
        "org_name": _claim(ORG_NAME_CLAIM),
    }


def has_role(*roles: str) -> bool:
    """True if the user holds ANY of the given roles. Admin implies everything."""
    if _DEV_BYPASS:
        return True
    held = _roles()
    return "admin" in held or bool(held & {r.lower() for r in roles})


# Holding ANY of these grants access to the app. A user who authenticates but
# holds NONE of them is signed in but NOT authorized — they get the access-denied
# page, never the dashboard or its data.
ALLOWED_ROLES = {"viewer", "analyst", "admin"}


def is_authorized() -> bool:
    """
    Authentication is not access. Social logins (e.g. Google) let ANY account
    authenticate — Auth0 auto-provisions them on first sign-in — so access must
    be gated on holding a known role, not on the mere fact of signing in.
    """
    if _DEV_BYPASS:
        return True
    return bool(_roles() & ALLOWED_ROLES)


def session_key() -> str:
    """
    Stable per-user id for logging and (if you ever add it) tenant scoping.
    Replaces the random run_session_id — logs become per real user/org:
        logs/<org>:<email>_<TICKER>_<stamp>_<runid>.log
    """
    u = current_user()
    org = u.get("org_id") or "noorg"
    who = u.get("email") or "anon"
    return f"{org}:{who}"


_LOGIN_STYLE = """<style>
#MainMenu, [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
.stAppDeployButton, [data-testid="stStatusWidget"] { display:none !important; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2rem !important; }
.sl-login { max-width: 1040px; margin: 0 auto; padding: 0 1rem;
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
.sl-hero { text-align:center; }
.sl-logo { font-size: 2.3rem; line-height:1; }
.sl-login h1 { font-size: 2.1rem; margin:.15rem 0 .1rem; color:#0f172a; letter-spacing:-.02em; }
.sl-tag { font-size: 1.03rem; font-weight:600; color:#1e293b; margin:.1rem auto .35rem; max-width:none; }
.sl-sub { font-size: .9rem; color:#475569; margin:0 auto; max-width:none; line-height:1.45; }
.sl-cards { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:1.1rem auto .9rem; }
.sl-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:.8rem .85rem; }
.sl-card .sl-ic { font-size:1.3rem; }
.sl-card h3 { margin:.3rem 0 .2rem; font-size:.9rem; color:#0f172a; }
.sl-card p { margin:0; font-size:.78rem; color:#64748b; line-height:1.38; }
.sl-foot { text-align:center; color:#64748b; font-size:.8rem; width:100%;
  margin:.8rem 0 0; line-height:1.45; padding:0 1rem; }
@media (max-width:820px){ .sl-cards{ grid-template-columns:repeat(2,1fr); } }
@media (max-width:520px){ .sl-cards{ grid-template-columns:1fr; } }
</style>"""

_LOGIN_HTML = """<div class="sl-login">
  <div class="sl-hero">
    <div class="sl-logo">📡</div>
    <h1>SignalLab</h1>
    <p class="sl-tag">Know what a company's leadership is really saying — without reading a single earnings call.</p>
    <p class="sl-sub">SignalLab reads the earnings calls and official filings of India-listed
      companies and turns them into clear, plain-English signals — each one backed by the exact
      words management used.</p>
  </div>
  <div class="sl-cards">
    <div class="sl-card"><div class="sl-ic">🎯</div><h3>Management Confidence</h3>
      <p>How sure and upbeat leadership sounds this quarter, on a simple 0–10 scale.</p></div>
    <div class="sl-card"><div class="sl-ic">📈</div><h3>Narrative Shift</h3>
      <p>Which topics are rising or fading in management's story, quarter over quarter.</p></div>
    <div class="sl-card"><div class="sl-ic">✅</div><h3>Guidance Credibility</h3>
      <p>Whether a company actually delivers on the targets it sets for itself.</p></div>
    <div class="sl-card"><div class="sl-ic">⚠️</div><h3>Risk Emergence</h3>
      <p>New risks surfacing in the fine print — often before they reach the headlines.</p></div>
  </div>
</div>"""

_LOGIN_FOOT = ("""<p class="sl-foot">🔒 Sign in with your organisation account to continue. """
               """Every score links back to the exact quote it came from — so you can always """
               """check the source.</p>""")


def _render_denied() -> None:
    """Signed-in-but-not-authorized page, with sign-out to switch accounts."""
    u = current_user()
    who = u.get("email") or u.get("name") or "this account"
    st.markdown(_LOGIN_STYLE, unsafe_allow_html=True)
    st.markdown(
        "<div class='sl-login'><div class='sl-hero'>"
        "<div class='sl-logo'>🔒</div><h1>Access not granted</h1>"
        f"<p class='sl-tag'>You're signed in as {who}, but this account hasn't "
        "been given access to SignalLab.</p>"
        "<p class='sl-sub'>Access is by invitation. Ask an administrator to grant "
        "your account a role (viewer, analyst, or admin), then sign in again. If "
        "you meant to use a different account, sign out and try that one.</p>"
        "</div></div>", unsafe_allow_html=True)
    _l, _c, _r = st.columns([2, 1.3, 2])
    with _c:
        if st.button("Sign out", use_container_width=True, key="_signout_denied"):
            st.logout()


def require_login() -> dict:
    """
    Gate the whole app. BOTH checks are required:
      1. AUTHENTICATION — is the user signed in? If not, show the sign-in page.
      2. AUTHORIZATION  — do they hold a known role? Authentication alone is NOT
         access: a social login lets any account in, so a role-less (uninvited)
         user gets the access-denied page and never reaches the dashboard/data.
    """
    if not is_authenticated():
        st.markdown(_LOGIN_STYLE, unsafe_allow_html=True)
        st.markdown(_LOGIN_HTML, unsafe_allow_html=True)
        _l, _c, _r = st.columns([2, 1.3, 2])
        with _c:
            if st.button("🔓  Sign in", type="primary", use_container_width=True,
                         key="_signin_btn"):
                st.login("auth0")
        st.markdown(_LOGIN_FOOT, unsafe_allow_html=True)
        st.stop()
    if not is_authorized():
        _render_denied()
        st.stop()
    return current_user()


def require_role(*roles: str) -> dict:
    """Gate a page/action: stop with a message if the user lacks the role(s)."""
    user = require_login()
    if not has_role(*roles):
        st.error(
            "You don't have access to this. Your role(s): "
            f"{', '.join(user['roles']) or 'none'}. "
            f"Ask an admin for the {' or '.join(roles)} role."
        )
        st.stop()
    return user


def logout_button() -> None:
    """Sidebar identity line + log-out control."""
    u = current_user()
    with st.sidebar:
        bits = [u.get("name") or "signed in"]
        if u.get("org_name"):
            bits.append(u["org_name"])
        bits.append(", ".join(u["roles"]) or "no role")
        st.caption("👤 " + " · ".join(bits))
        if _DEV_BYPASS:
            st.warning("AUTH DISABLED (dev bypass)")
        elif st.button("Log out", key="_logout_sidebar"):
            st.logout()


def header_bar(title: str | None = None, subtitle: str | None = None) -> None:
    """
    Top bar for the main area: site title + description on the left (pass `title`
    and `subtitle`), identity + Log out on the right. In-flow (scrolls with the
    page). Styling lives in the dashboard CSS (.sl-apptitle / .sl-appsub /
    .sl-userline / .st-key-_logout_top).
    """
    u = current_user()
    name = u.get("name") or u.get("email") or "signed in"   # role removed
    left, right = st.columns([7, 1.3])
    with left:
        if title:
            html = f"<div class='sl-apptitle'>{title}</div>"
            if subtitle:
                html += f"<div class='sl-appsub'>{subtitle}</div>"
            st.markdown(html, unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='sl-userline'>👤 {name}</div>",
                    unsafe_allow_html=True)
        if _DEV_BYPASS:
            st.caption("⚠️ AUTH DISABLED (dev bypass)")
        elif st.button("Log out", key="_logout_top", use_container_width=True):
            st.logout()


# ─────────────────────────────────────────────────────────────────────────────
# Auth0 setup reference (not executed)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1) Auth0 → Applications → create a "Regular Web Application".
#    Allowed Callback URLs:  http://localhost:8501/oauth2callback
#                            https://<your-tailnet-name>.ts.net/oauth2callback
#
# 2) Auth0 → User Management → Roles: create  admin, analyst, viewer.
#    Assign roles to users (or, for tenants, per Organization membership).
#
# 3) (Multi-tenant) Auth0 → Organizations: one Organization per tenant; add
#    users; assign roles within the org. The org id/name flow through below.
#
# 4) Auth0 → Actions → Library → create a Post-Login Action, add the roles and
#    org as NAMESPACED custom claims on the ID token (must match _NS above):
#
#      exports.onExecutePostLogin = async (event, api) => {
#        const ns = "https://signallab.app/";
#        const roles = (event.authorization && event.authorization.roles) || [];
#        api.idToken.setCustomClaim(ns + "roles", roles);
#        if (event.organization) {
#          api.idToken.setCustomClaim(ns + "org_id",   event.organization.id);
#          api.idToken.setCustomClaim(ns + "org_name", event.organization.name);
#        }
#      };
#
#    Then add the Action to your Login flow.
#
# 5) .streamlit/secrets.toml  (gitignored — never commit):
#
#      [auth]
#      redirect_uri = "http://localhost:8501/oauth2callback"
#      cookie_secret = "<run: python -c 'import secrets;print(secrets.token_hex(32))'>"
#
#      [auth.auth0]
#      client_id = "<from the Auth0 app>"
#      client_secret = "<from the Auth0 app>"
#      server_metadata_url = "https://<YOUR_DOMAIN>.auth0.com/.well-known/openid-configuration"
#      # force login each visit; add organization="org_..." here for a fixed tenant
#      client_kwargs = { prompt = "login" }
