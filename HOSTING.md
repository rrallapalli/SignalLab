# 🌐 Hosting SignalLab on your Mac, shared over Tailscale

This is the **small-team setup**: SignalLab runs on your MacBook, and a handful
of people you trust reach it from their own laptops and phones over an
encrypted private network. No public URL, no ports opened on your router, no
passwords to invent, no cloud bill.

If you need this available to strangers, or reliably at 3am, this is the wrong
page — you want a VPS. See [the limits](#-what-this-setup-is-not) below and be
honest with yourself about which one you need.

---

## 🤔 Why Tailscale rather than putting it on the internet

SignalLab has **no login screen**. None. Anyone who can reach the URL can run
analyses, and every run spends your OpenAI credit. That's completely fine on
your own laptop, and completely unacceptable on a public URL.

Tailscale sidesteps the whole problem instead of solving it:

- Your Mac and your colleagues' devices join a private network (a **tailnet**)
- Only devices you've explicitly approved can connect — everyone else can't
  even see it exists
- Traffic is end-to-end encrypted, and you get a real HTTPS certificate
- **No inbound ports.** Nothing on your router changes. Works from any wifi.

The auth you'd otherwise have to build is replaced by "is this device on my
tailnet". For 2–5 known people, that's the right trade.

---

## 1 — Install Tailscale on your Mac

Download from **[tailscale.com/download](https://tailscale.com/download)**, or:

```bash
brew install --cask tailscale
```

Open it, sign in (Google/Microsoft/GitHub account), and leave it running.
Confirm it's up:

```bash
tailscale status
```

> **`command not found: tailscale`?** The Mac App Store build doesn't put the
> CLI on your PATH. Either add it:
> ```bash
> alias tailscale="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
> ```
> …or ignore it — SignalLab finds the binary inside the app bundle by itself.

## 2 — Start SignalLab in tailnet mode

```bash
cd ~/SignalLab
./start.command --serve
```

You'll see:

```
🚀  SignalLab is starting…
  On your tailnet:  https://rashmis-macbook-air.tail1234.ts.net
  On this Mac:      http://localhost:8501
```

That HTTPS URL is your dashboard. It works from any device on your tailnet,
anywhere in the world, with a valid certificate.

**What `--serve` changes:**

| | Without `--serve` | With `--serve` |
|---|---|---|
| Streamlit binds to | `127.0.0.1` | `127.0.0.1` |
| Reachable from | this Mac only | this Mac + your tailnet |
| Opens a browser | yes | no (it's a host now) |
| TLS | n/a | yes, real cert from Tailscale |

Note the binding is loopback **either way**. Streamlit's own default is to
listen on every interface — meaning the coffee shop's network — so SignalLab
overrides it. Tailscale Serve proxies inward from the tailnet, which is also
why the identity headers below can be trusted: nobody on the LAN can reach
around Serve and forge them.

Serve persists across reboots. To stop sharing:

```bash
tailscale serve --bg 8501 off
```

## 3 — Add the people you want

In the [Tailscale admin console](https://login.tailscale.com/admin/machines):

- **Their own device:** *Users → Invite* — they install Tailscale, sign in, done.
- Their laptop and phone both work. Nothing to configure on their side.

Anyone not on the tailnet gets nothing — the DNS name doesn't even resolve.

> **Who did what?** Serve adds identity headers (`Tailscale-User-Login`) to
> every request, so you can see which tailnet user is calling. SignalLab
> doesn't read them today, but it's there if you later want per-user
> attribution on OpenAI spend.

## 4 — Keep it awake

**This is the part that actually bites.**

A MacBook sleeps. When it sleeps, your "server" disappears mid-run and everyone
gets a dead URL. `--serve` doesn't change that.

```bash
# Simplest: run in a Terminal you leave open
caffeinate -ims ./start.command --serve
```

`-i` no idle sleep · `-m` no disk sleep · `-s` no system sleep on AC power.

> **Closing the lid still sleeps the Mac** — `caffeinate` doesn't override that
> on Apple Silicon. For an always-on host you need the **lid open and plugged
> in**, or an external display attached (clamshell mode), or a tool like
> Amphetamine. There's no clean CLI trick here. If "lid open on a desk" isn't
> acceptable, that's your signal to move to a VPS.

Also worth turning off: System Settings → Battery → Options → **Wake for
network access** helps, but it isn't a substitute for the above.

## 5 — (Optional) Start automatically at login

A LaunchAgent template lives in [`deploy/com.signallab.dashboard.plist`](deploy/com.signallab.dashboard.plist).

```bash
mkdir -p ~/SignalLab/logs

# Fill in your username and Python path
sed -e "s|__USERNAME__|$(whoami)|g" \
    -e "s|__PYTHON__|$(which python3)|g" \
    deploy/com.signallab.dashboard.plist > ~/Library/LaunchAgents/com.signallab.dashboard.plist

launchctl load ~/Library/LaunchAgents/com.signallab.dashboard.plist
launchctl list | grep signallab      # should show a PID
```

Logs: `~/SignalLab/logs/dashboard.log` and `dashboard.error.log`.

To stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.signallab.dashboard.plist
```

> It's a **LaunchAgent**, not a LaunchDaemon — it runs as you, after you log
> in. That's deliberate: it needs your Tailscale session. It won't come back on
> its own after a reboot until you log in.

---

## 🛡️ Before you share the URL

**Set a spend limit on your OpenAI key.** Everyone on the tailnet shares your
key with no per-user cap. A colleague exploring 30 tickers costs real money,
and there's nothing in SignalLab to stop them.
→ [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits)

**Use a separate key for this**, not your dev key, so you can revoke it without
breaking your own laptop.

**Back up your data.** Signals cost money to generate; regenerating them isn't
free.

```bash
# Whole state: signals DB + vector store
tar czf ~/signallab-backup-$(date +%F).tgz -C ~/SignalLab data/
```

---

## ⚠️ What this setup is *not*

Be clear-eyed:

- **Not highly available.** Laptop sleeps, closes, goes to a café → service
  gone. Fine for a team that knows your schedule; not fine for anything
  automated.
- **Not multi-tenant.** One shared OpenAI key, no per-user limits, no
  attribution. Everyone spends your money.
- **Not concurrent-safe beyond a handful of users.** Same-ticker runs are
  guarded, but the pipeline still runs inside the web process.
- **Not public.** By design. If a stranger needs access, don't reach for
  `tailscale funnel` — that publishes it to the entire internet with **no
  authentication at all**, which is precisely the thing this page exists to
  avoid. Build real auth first.

When you outgrow this, the move is a small always-on box (ideally in India, so
NSE/BSE will talk to it) with Cloudflare Tunnel + Access in front. Same shape,
none of the sleep problems.

---

## 🧯 Troubleshooting

**URL doesn't resolve on another device**
That device isn't on the tailnet, or Tailscale is off there. `tailscale status`
on both ends.

**"Tailscale is installed but not logged in / running"**
`tailscale up`, or open the app and sign in.

**Hangs at `Enabling Tailscale Serve…`, or times out there**
Your tailnet doesn't have HTTPS certificates enabled yet. `tailscale serve`
prints a one-time link and waits for you to click it. Run it by hand to see
the link:

```bash
tailscale serve --bg 8501
# or, if the CLI isn't on PATH:
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg 8501
```

Follow the link, approve it, then `./start.command --serve` again. One-time
only — Serve persists after that.

Meanwhile `./start.command` with no flag still works on localhost.

**Works on the Mac, not from the tailnet**
Check Serve is actually configured: `tailscale serve status`

**Dashboard loads but hangs / websocket errors**
Reload once. If it persists, confirm nothing else holds port 8501:
`lsof -i :8501`

**It died overnight**
The Mac slept. See [step 4](#4--keep-it-awake).
