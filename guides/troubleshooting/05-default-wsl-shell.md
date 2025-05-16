# How to set WSL as the default shell when SSHing into a Windows system

*This guide assumes you can access your Windows system via the Windows App (which replaces RDP) on Mac and run commands in an elevated PowerShell terminal (with administrator privileges).*

## Part 1 · General recipe — log straight into WSL every time you SSH into Windows 11

### Phase A · Make sure the SSH session can *find* the real WSL engine

**Why it matters**: OpenSSH runs as a service and ignores *user* PATH tweaks. It will grab the first `wsl.exe` it sees; you want that to be the real one in *Program Files*, not the lightweight alias in System32 that can't locate distros in a service context.

**Steps**:
1. Verify the real binary exists:
   ```powershell
   dir "C:\Program Files\WSL\wsl.exe"
   ```

2. Put that folder at the very front of the **system** PATH (one-time, admin PowerShell):
   ```powershell
   $sys = [Environment]::GetEnvironmentVariable('Path','Machine')
   if (-not $sys.ToLower().StartsWith('c:\program files\wsl')) {
       [Environment]::SetEnvironmentVariable('Path', "C:\Program Files\WSL;$sys", 'Machine')
   }
   ```

3. Toggle **Settings ▶ Apps ▶ App-execution aliases ▶ wsl.exe → OFF** to prevent the thin System32 stub from jumping ahead.

### Phase B · Register (or clone) a Linux distro for the new Windows account

**Option 1 - Simplest**:
```powershell
wsl --install -d Ubuntu
```

**Option 2 - Zero-download clone** (from another account that already has Ubuntu):
```powershell
# On the old account
wsl --export Ubuntu C:\Temp\ubuntu.tar

# On the new account
mkdir C:\WSL\Ubuntu
wsl --import Ubuntu C:\WSL\Ubuntu C:\Temp\ubuntu.tar
```

**Note**: A distro is only "visible" to the Windows user whose registry contains an `HKCU\…\Lxss\{GUID}` entry; export/import writes that key for you without redownloading.

### Phase C · Tell OpenSSH to drop you inside WSL, not PowerShell

**Why it matters**: `ForceCommand` overrides the default Windows shell for just that account. A couple of environment variables are injected so `wsl.exe` can locate its per-user registry hive even though the session is service-spawned.

**Steps**:
1. Create `C:\Users\<user>\wsl-shell.cmd`:
   ```bat
   @echo off
   set "LOCALAPPDATA=C:\Users\<user>\AppData\Local"
   set "APPDATA=C:\Users\<user>\AppData\Roaming"
   "C:\Program Files\WSL\wsl.exe" -d Ubuntu
   ```

2. Edit `C:\ProgramData\ssh\sshd_config`:
   ```text
   Match User <user>
       ForceCommand C:\Users\<user>\wsl-shell.cmd
   ```

3. Restart the SSH service:
   ```powershell
   Restart-Service sshd
   ```

After those three phases you can:

```bash
ssh <user>@windows-host
#   ➜   user@machine:~$   (already in Linux)
```

---

## Part 2 · PowerShell Only Recipe

Below is a **copy-/-paste-ready, PowerShell-only playbook** that turns a brand-new Windows account (`wankyuchoi` in the examples) into a user that drops **directly into Ubuntu inside WSL** whenever you SSH to the machine.

> *Run every block in an **elevated PowerShell** window (Administrator).
> Comments (`# …`) explain what each line does; they are safe to leave in or delete.*

---

### 1 · Make sure the real WSL engine is installed and first in PATH

```powershell
# 1-a  Update / install the published Store engine (idempotent)
wsl --update

# 1-b  Verify the real binary exists
dir "C:\Program Files\WSL\wsl.exe"  # ⇦ must show the file

# 1-c  Pre-pend Program Files\WSL to the **system** PATH (service accounts read this)
$sys = [Environment]::GetEnvironmentVariable('Path','Machine')
if (-not $sys.ToLower().StartsWith('c:\program files\wsl')) {
    [Environment]::SetEnvironmentVariable('Path',"C:\Program Files\WSL;$sys",'Machine')
}

```

*(WSL's app-execution alias can confuse service sessions. Disabling it ensures `sshd` finds the real engine.)* 

---

### 2 · Register an Ubuntu distro for **this** Windows user

*(Pick **one** of the two options)*

#### Option A – fresh download

```powershell
runas /user:wankyuchoi `
      "powershell -NoLogo -NoExit -Command wsl --install -d Ubuntu"
```

Follow the on-screen prompt once to set the Unix password.

#### Option B – clone an existing Ubuntu without re-downloading

```powershell
# Run in the Windows account that ALREADY has Ubuntu:
wsl --export Ubuntu C:\Temp\ubuntu.tar

# Back in the admin PowerShell, impersonate the new account and import:
runas /user:wankyuchoi `
      "powershell -NoLogo -NoExit -Command ^
       mkdir C:\WSL\Ubuntu; ^
       wsl --import Ubuntu C:\WSL\Ubuntu C:\Temp\ubuntu.tar"
```

A distro becomes visible to a Windows user the moment a matching **HKCU\…\Lxss** registry entry is created – `--install` or `--import` both do that automatically.

Verify:

```powershell
runas /user:wankyuchoi "wsl -l -v"   # ⇦ should list Ubuntu, state = Stopped/Running
```

---

### 3 · Create a wrapper script that launches WSL inside an SSH TTY

```powershell
Set-Content -Path "C:\Users\wankyuchoi\wsl-shell.cmd" -Value @'
@echo off
rem Minimal env so wsl.exe can find HKCU during a service spawn
set "LOCALAPPDATA=C:\Users\wankyuchoi\AppData\Local"
set  "APPDATA=C:\Users\wankyuchoi\AppData\Roaming"
"C:\Program Files\WSL\wsl.exe" -d Ubuntu
'@
```

---

### 4 · Tell OpenSSH to run that script instead of PowerShell

```powershell
Add-Content -Path "C:\ProgramData\ssh\sshd_config" -Value @'
Match User wankyuchoi
    ForceCommand C:\Users\wankyuchoi\wsl-shell.cmd
'@

Restart-Service sshd
```

---

### 5 · Test

```powershell
ssh wankyuchoi@<windows-IP>
# You should land directly at:
#    wankyuchoi@<hostname>:~$
```

If the prompt appears, the path, distro registration and `ForceCommand` are all good.

---

## Part 3 · Comprehensive Troubleshooting

### Common Issues and Solutions

| Symptom                                                            | Root cause                                                                                 | One-step fix                                                                                               |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `WSL_E_DISTRO_NOT_FOUND` even though Ubuntu works for another user | The new Windows account has **no Lxss registry entry**                                     | Run `wsl --install -d Ubuntu` **or** use `--export/--import` to copy the distro ([learn.microsoft.com][2]) |
| `where wsl` still shows **System32** first                         | System PATH wasn't updated—or you never opened a new terminal                              | Re-run the Phase A snippet, reboot, then re-check                                                          |
| SSH connection freezes (blank screen)                              | Wrapper used `start "" …` which detaches from the TTY                                      | Call `wsl.exe` directly as in Phase C                                                                      |
| You want to undo everything                                        | Delete the two lines you added to **sshd\_config**, remove `wsl-shell.cmd`, restart `sshd` |                                                                                                            |

### Quick Health Check

```powershell
# Check which wsl.exe is found first in PATH
where wsl | Select-Object -First 1     # => C:\Program Files\WSL\wsl.exe  (good)
# Verify distro is registered for this user
wsl -l -v                              # => Ubuntu listed (good)

# Common Fixes #
# 1. "WSL_E_DISTRO_NOT_FOUND": Register the distro for this user
wsl --install -d Ubuntu                # or the export/import sequence above

# 2. Restart the SSH service
Restart-Service sshd                   # or simply reboot

# 3. SSH session freezes (blank screen)
#    Remove any `start "" …` lines and call wsl.exe directly in wsl-shell.cmd
```

Both checks correct? Your setup is solid. SSH away and enjoy a friction-free WSL landing zone!

## Part 4 · VS Code missing Workspaces - when "SSH-into-WSL" breaks VS Code Remote-SSH (or other tools)

| Symptom                                                                                  | Root Cause                                                                                | PowerShell-only Fix                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **VS Code opens "/" (Linux root) and can't find `E:\dropbox`**                           | The VS Code server is running *inside* WSL, so it sees Windows drives at `/mnt/<drive>`.  | 1. Hit **Ctrl + K O** → type the Linux path: <br>`/mnt/e/dropbox` <br>2. *(Optional)* Create a friendly link once:<br>`bash ln -s /mnt/e/dropbox ~/dropbox` <br>then open `~/dropbox` in the future.                                                                                                                                             |
| **You'd rather keep VS Code on Windows paths but still want terminals to drop into WSL** | Your `ForceCommand` wrapper always launches WSL—even for automated VS Code sessions.      | Add a second SSH host entry that forces a Windows shell:  `powershell echo @' Host win4090-code   HostName <ip-address>   User wankyuchoi   RemoteCommand powershell.exe -NoLogo '@ >> $HOME\.ssh\config ` Connect VS Code to **win4090-code**; interactive terminals can keep using **win4090** (no RemoteCommand) and will still land in Ubuntu. |
| **VS Code still hits WSL even with the new host entry**                                  | The smart wrapper isn't detecting the RemoteCommand.                                      | Make sure `wsl-shell.cmd` starts with:  `bat IF NOT "%SSH_ORIGINAL_COMMAND%"=="" (   powershell.exe -NoLogo -NoProfile -Command "%SSH_ORIGINAL_COMMAND%"   GOTO :EOF ) ` Restart `sshd` after edits: `Restart-Service sshd`.                                                                                                                     |
| **Wrapper throws `'M'`, `'NOT'` or other random tokens and the SSH session closes**      | The CMD file was saved with UTF-8 BOM or broken line endings, so `cmd.exe` mis-parses it. | Regenerate the file in pure ASCII:  `powershell Set-Content -Path C:\Users\wankyuchoi\wsl-shell.cmd -Encoding ASCII -Value @' @echo off … (full script) '@ ` Re-test with `cmd /c "C:\Users\wankyuchoi\wsl-shell.cmd"`.                                                                                                                          |
| **`WSL_E_DISTRO_NOT_FOUND` reappears for this user**                                     | PATH is fine, but the *distro isn't registered* for this Windows account.                 | Register it:  `powershell wsl --install -d Ubuntu` **or** clone:  `powershell wsl --export Ubuntu C:\Temp\ubuntu.tar mkdir C:\WSL\Ubuntu wsl --import Ubuntu C:\WSL\Ubuntu C:\Temp\ubuntu.tar`                                                                                                                                                   |
| **`where wsl` still shows System32 first in a new session**                              | System PATH wasn't updated or alias is overriding.                                        | 1. Ensure step 1 in the main guide (prepend `C:\Program Files\WSL\`) really ran as admin. <br>2. Disable alias:  `powershell reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModel\ExecAlias\wsl.exe /v Deleted /t REG_DWORD /d 1 /f` <br>3. Reboot or `Restart-Service sshd`.                                                         |

**Quick sanity check**

```powershell
# 1️⃣ Which wsl.exe does the service see?
where wsl | Select-Object -First 1      # → C:\Program Files\WSL\wsl.exe

# 2️⃣ Does this user have the distro?
wsl -l -v                               # → Ubuntu listed

# 3️⃣ Can VS Code open Windows paths?
ssh win4090-code "echo OK"              # should print OK from Windows PowerShell
```

If all three pass, VS Code and plain SSH will each land exactly where you expect—Windows paths for the editor, Ubuntu for your interactive terminal. 

---

Now fire up `tmux` and do your thing!
