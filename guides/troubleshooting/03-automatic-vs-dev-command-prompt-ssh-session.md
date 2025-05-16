# One-Time Setup: Automatically Load “Developer Command Prompt for VS 2022” in Every SSH Session

> **Goal** — When you `ssh` from Cursor (Mac) into your Windows RTX 4090 box, you land in a shell where  
> `cl.exe`, `link.exe`, the Windows SDK paths, and `nvcc` **“just work.”**  
> No more manual `vcvars64.bat`, no more “compiler not found” errors.

---

## 0 · What you’ll do (overview)

1. **Create one wrapper batch** that calls `vcvars64.bat` for VS 2022 Community.  
2. Place that wrapper under your user profile.  
3. Add a tiny **`%USERPROFILE%\.ssh\rc`** file that calls the wrapper on every interactive SSH login.

Total time ≈ 2 minutes.

---

## 1 · Locate `vcvars64.bat`

Default path for Build-Tools 2022 Community:

```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
```

*(If you’re unsure, run  
`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`
and append `\VC\Auxiliary\Build\vcvars64.bat`.)*

---

## 2 · Create the wrapper batch

`C:\Users\<YOU>\vcdev22.cmd`

```bat
@echo off
rem --- Load Visual C++ 2022 environment ----------------------
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul

rem --- Drop into an interactive cmd shell --------------------
cmd /k
```

*(Switch to `powershell -NoExit` if you prefer PowerShell instead of cmd.)*

---

## 3 · Create the per-user **.ssh** folder & rc script

```powershell
# 1) make the folder (if it doesn't exist yet)
mkdir $HOME\.ssh               # -> C:\Users\<YOU>\.ssh

# 2) create rc file
notepad $HOME\.ssh\rc
```

Paste in **two lines**:

```bat
@echo off
call "%USERPROFILE%\vcdev22.cmd"
```

Save → exit.

*(Windows hides dot-folders; use *Show Hidden Items* if browsing in Explorer.)*

---

## 4 · Test from Cursor (Mac)

```bash
ssh win4090                        # connects to Windows box
where cl                           # shows VS 2022 path
nvcc -v                            # no “cl.exe not found” warning
```

Compile a sample:

```cmd
nvcc -arch=sm_89 -O3 src\wmma_gemm.cu -o build\gemm.exe && build\gemm.exe
```

If it runs, the environment is loaded automatically.

---

## 5 · Troubleshooting quick table

| Symptom | Check | Fix |
|---------|-------|-----|
| `cl` “not recognized” | Did `vcdev22.cmd` path match your VS install? | Correct the path; reconnect. |
| `.ssh\rc` ignored | Folder/file typo? (`.ssh`, not `ssh`) | `dir %USERPROFILE%\.ssh` — ensure `rc` exists. |
| Want **plain** shell sometimes | SSH with another host alias:<br>`Host win4090_plain`<br>`RemoteCommand ""` | Skips `rc`; no VC env. |

---

## 6 · Why this beats global hacks

* **Per-user scope** — No risk of breaking other accounts or services.  
* **No PATH pollution** — Variables live only for the SSH session.  
* **VS upgrades** — When you update VS, just edit the wrapper’s path once.

---

You’re done — every time you open Cursor’s integrated terminal and type `ssh win4090`, you’re already inside a fully armed **VS 2022 Developer Command Prompt**. Happy compiling! 🚀