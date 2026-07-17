# Live GPU loop: Claude ↔ Colab A100 via Jupyter MCP

Goal: let Claude execute/iterate on cells directly in a GPU notebook (ends the relay).
Uses **Datalayer's jupyter-mcp-server** (https://jupyter-mcp-server.datalayer.tech).

Architecture:
```
[Claude Desktop] --uvx jupyter-mcp-server (on your Mac)--> [cloudflared tunnel] --> [JupyterLab on the Colab A100]
```

You do steps 1–3 (I can't touch the connector or your Colab UI). After step 3, start a
NEW Cowork conversation and I drive the notebook.

---

## STEP 1 — Colab (GPU runtime): start JupyterLab + a public tunnel

New Colab notebook → Runtime → Change runtime type → **A100 GPU**. Then run this one cell:

```python
!pip install -q jupyterlab==4.4.1 jupyter-collaboration==4.0.2 "jupyter-mcp-tools>=0.1.4" ipykernel pycrdt
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared

import secrets, subprocess, time, threading, re, os
TOKEN = secrets.token_hex(16)
os.makedirs("/content/work", exist_ok=True)

subprocess.Popen(
    ["jupyter","lab","--port","8888","--ip","0.0.0.0","--no-browser","--allow-root",
     "--IdentityProvider.token", TOKEN, "--ServerApp.disable_check_xsrf=True",
     "--notebook-dir","/content/work"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(10)

url = {}
def tunnel():
    p = subprocess.Popen(["cloudflared","tunnel","--url","http://localhost:8888","--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
        if m and "u" not in url:
            url["u"] = m.group(0); print("TUNNEL:", m.group(0))
threading.Thread(target=tunnel, daemon=True).start()
time.sleep(12)

print("\n==== COPY THESE TWO VALUES ====")
print("JUPYTER_URL   :", url.get("u", "(re-run this print cell in a few seconds)"))
print("JUPYTER_TOKEN :", TOKEN)
```

Copy the printed **JUPYTER_URL** (the `https://….trycloudflare.com`) and **JUPYTER_TOKEN**.
Leave this Colab tab running — if the runtime resets, the URL + token change.

---

## STEP 2 — Your Mac: make sure `uvx` exists

```bash
pip install uv        # or: brew install uv
uv --version          # 0.6.14 or higher
```

---

## STEP 3 — Claude Desktop: add the connector

Claude Desktop → **Settings → Developer → Edit Config** (opens
`~/Library/Application Support/Claude/claude_desktop_config.json`). Add:

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "uvx",
      "args": ["jupyter-mcp-server@latest"],
      "env": {
        "JUPYTER_URL": "PASTE_THE_TRYCLOUDFLARE_URL",
        "JUPYTER_TOKEN": "PASTE_THE_TOKEN",
        "ALLOW_IMG_OUTPUT": "true"
      }
    }
  }
}
```

Save, then **fully quit and reopen Claude Desktop**.

> If it errors about `MCP_TOKEN` (a v1.0.0 change), add `"MCP_TOKEN": "any-string-you-pick"`
> to that `env` block, or pin `"jupyter-mcp-server==0.14.0"` instead of `@latest`.

---

## STEP 4 — New Cowork conversation

Open a **new** Cowork conversation (the connector's tools only appear in a fresh session)
and say: **"the Jupyter MCP is connected."** I'll verify by listing kernels/notebooks,
then create the protein notebook on the Colab A100, get ProGen2 loading faithfully
(validation-1 perplexity ≈ 15–20), and run it — iterating live instead of relaying.

Everything is saved in your `lrtia` folder, so I pick up exactly where we are: DNA done,
protein prereg committed, manuscript finished. We just need the live kernel.

---

## Notes
- Keep the Colab tab awake; idle runtimes disconnect and the URL/token rotate.
- The `jupyter-collaboration` + `pycrdt` packages are required (real-time doc model the MCP uses).
- For a sturdier setup than Colab+tunnel, a cloud GPU (RunPod/Lambda) with JupyterLab
  exposes a URL+token directly — same steps 2–4, no tunnel.
