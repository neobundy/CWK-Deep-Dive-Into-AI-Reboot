# SSH Guide for the CWK Mac‑Studio AI Cluster

> **Scope:** Key‑based, passwordless administration across all Macs (M3 Ultra heads, M2 Ultra workers, laptops) running macOS 15.4.1.
>
> **Admin user:** `wankyuchoi`

---

## 1 · Generate a Master Key (once on your admin Mac)
```bash
ssh-keygen -t ed25519 -C "wankyuchoi main key" -f ~/.ssh/id_ed25519
```
* **Empty passphrase** → no Touch ID prompts **or** set a passphrase and let macOS keychain handle unlocks.

## 2 · Distribute the Key to Every Node (first‑time only)
```bash
brew install ssh-copy-id        # on admin Mac if not present
ssh-copy-id wankyuchoi@cwk-h-m3u-01.local   # head‑primary
ssh-copy-id wankyuchoi@cwk-h-m3u-02.local   # head‑secondary
ssh-copy-id wankyuchoi@cwkmusicstudiom2u.local  # studio node
```
If `ssh-copy-id` is unavailable:
```bash
cat ~/.ssh/id_ed25519.pub | ssh wankyuchoi@node "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Permissions sanity on each server (run once)
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chown wankyuchoi ~/.ssh ~/.ssh/authorized_keys
```

## 3 · Create Convenient Aliases (`~/.ssh/config` on admin Mac)
```config
Host m3-head
  HostName cwk-h-m3u-01.local
  User wankyuchoi
  IdentityFile ~/.ssh/id_ed25519

Host m3-sec
  HostName cwk-h-m3u-02.local
  User wankyuchoi

Host studio
  HostName cwkmusicstudiom2u.local
  User wankyuchoi
```
Usage:
```bash
ssh m3-head      # no password prompt
tmux a -t cluster  # resume tmux on head node
```

## 4 · Auto‑start `ssh-agent` & Load Key (macOS zsh)
Add to `~/.zshrc` **on admin Mac**:
```zsh
# Start agent if not running
if ! pgrep -u "$USER" ssh-agent > /dev/null; then
  eval "$(ssh-agent -s)" >/dev/null
fi
# Load key once per boot
ssh-add --apple-use-keychain ~/.ssh/id_ed25519 2>/dev/null
```
`source ~/.zshrc` or open a new terminal—no more `ssh-add` hassles.

## 5 · Lock Down Password Auth (after verifying keys)
```bash
sudo sed -i '' 's/^#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo launchctl kickstart -k system/com.openssh.sshd
```
Repeat on each server.

## 6 · Quality‑of‑Life Tips
| Need | Command / Config |
|------|------------------|
| **Persistent sessions** | `tmux new -s cluster` then `Ctrl‑b d` to detach. |
| **Port‑forward Ray dashboard** | `ssh -L 8265:localhost:8265 m3-head` then open `http://localhost:8265` on admin Mac. |
| **Off‑site access** | Install **Tailscale** → SSH to `100.x.x.x` addresses, same keys. |
| **Copy file to head** | `scp model.gguf m3-head:/opt/gguf/` |
| **Run same cmd on all nodes** | `parallel -a hosts.txt "ssh {} 'uptime'"` (requires `parallel`). |

## 7 · Troubleshooting Checklist
1. **Key not used?** `ssh -i ~/.ssh/id_ed25519 -v m3-head` shows why.
2. **Permission denied** on server → re‑check `chmod 700 ~/.ssh` and 600 authorized_keys.
3. **Agent missing** → `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519`.
4. **Wrong hostnames** after renaming Macs → `dscacheutil -flushcache; killall -HUP mDNSResponder`.

---

## Copying Models to the Cluster (TB)

Create folders on the head node:

```bash
ssh head-primary.tb "mkdir -p ~/.cache/huggingface ~/.ollama"
```

Copy models to the head node:

```bash
rsync -avP ~/.cache/huggingface/* head-primary.tb:~/.cache/huggingface/
```

Use folder names to suit your needs. For example, you might want to sync specific model folders instead of the entire cache. To copy Ollama models:

```bash
rsync -avP ~/.ollama head-primary.tb:~/.ollama
```
---

[⇧ Back&nbsp;to&nbsp;README](../../README.md)