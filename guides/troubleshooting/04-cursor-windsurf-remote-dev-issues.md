# Practical fallback options while Cursor / WindSurf Remote-SSH is broken

*Try this guide when Cursor/WindSurf Remote-SSH is broken. Setting WSL as the default shell solved the issue for me, but I can't guarantee it will work for you: [How to set WSL as the default shell when SSHing into a Windows system](05-default-wsl-shell.md)*

_Possible reasons this helps (your mileage may vary):_
- The **SSH session lands in a real Linux environment** immediately, so tools that expect `/bin/bash` or a Unix-style `$PATH` work without translating Windows paths.
- **Service-spawned sessions see the correct `wsl.exe`** (the one in `C:\Program Files\WSL`) instead of the thin System32 alias, preventing registry-lookup failures.
- **Cursor / WindSurf use the same Remote-SSH core as VS Code**, which negotiates better with a Unix shell than with PowerShell or `cmd.exe`.
- **Path ordering quirks auto-resolve** when the wrapper forces WSL, sidestepping edge cases where environment variables differ between interactive and service contexts.

---

| Option | What you do | Pros | Cons |
|--------|-------------|------|------|
| **1. Use vanilla VS Code Remote-SSH** | Keep editing in Cursor/WindSurf if you like, but open VS Code when you need the remote file tree or IntelliSense. | Uses the *new* server layout, auto-installs once, rock-solid. | Two editors open; small mental context-switch. |
| **2. Windows App (new Microsoft Remote Desktop)** | Launch the Windows App on macOS → fullscreen into the 4090 box → compile, run, profile as if you were sitting there. Copy/paste text or files via RDP clipboard. | 100 % control of the remote desktop; no server mismatch to worry about. | Requires good bandwidth; GUI latency vs. pure SSH; can't script easily. |
| **3. Plain Dropbox + SSH terminal** *(Cursor/WindSurf terminal)* | Let Dropbox sync code, then `ssh win4090` inside Cursor's integrated terminal to build/run (`nvcc`, `nsys`, etc.). | Zero moving parts, no GUI lag; works even if Remote-SSH never gets fixed. | You lose remote file-explorer UX and IntelliSense for headers on the Windows side. |

> **Why we don't recommend complex shims right now**  
> You *can* patch `.cursor-server` with junctions or scripts, but every daily auto-update in Cursor or WindSurf spawns a new commit folder and breaks the patch. The time you save with a one-click RDP session (Option 2) or a rock-solid VS Code connection (Option 1) outweighs the maintenance hassle.

Pick whichever fallback matches your workflow speed:

* **Need full desktop apps (Nsight GUI, Visual Profiler)** → **Windows App (RDP)**.  
* **Want code navigation & remote IntelliSense** → **VS Code Remote-SSH**.  
* **Just need to compile and run quickly** → **Dropbox + SSH terminal**.

Once Cursor/WindSurf ship the updated Remote-SSH core, you can switch back in a single click.

**Remember this simple engineering (in fact, life) principle:** if you spend more time implementing a "reliable fix" than you would using a straightforward alternative, it's not a fix—it's a workaround. And if that workaround is both time-consuming and unreliable? Then it's just a kludge that will cost you twice: once to implement and again every time it breaks.

Your time is valuable. Choose solutions that maximize productivity, not ones that satisfy the urge to "fix everything perfectly."