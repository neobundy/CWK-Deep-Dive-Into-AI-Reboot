# 🧩 Ultimate tmux Guide for AI Server Ops on Mac Studios

**tmux** is a terminal multiplexer inspired by the legendary vi editor, bringing modal editing and lightning-fast, vi-style navigation to your terminal. If you love efficiency and keyboard-driven workflows, tmux will feel like home: you can split, move, and manage panes with familiar vi commands.

But if you're not from the vi era, tmux can feel a bit arcane at first. That's why I **strongly recommend starting with tmux**—especially with mouse support and iTerm2's copy mode. It's much more approachable for most users, and you can always graduate to vi-style tools later.

Still, if you're serious about terminal productivity, learning vi-style operations is a superpower. They're everywhere in Unix tools, and once you're comfortable, you'll fly through your workflow.

**How tmux works:**
- **Normal Mode:** Navigate between panes, switch windows, and execute commands—all from the keyboard.
- **Command Mode:** Issue tmux-specific commands with a vi-like prompt.
- **Insert Mode:** Type and interact with terminal apps as usual.

**Common vi-style ops in tmux:**
- `h`, `j`, `k`, `l`: Move focus left, down, up, or right between panes.
- `:split`, `:vsplit`: Split the current pane horizontally or vertically.
- `:q`, `:wq`: Close panes or save layouts, just like in vi.
- `/` and `?`: Search for windows or panes.
- `d`, `y`, `p`: Delete, yank (copy), and put (paste) panes or window arrangements.

By embracing modal, vi-style operations, tmux lets you harness muscle memory for a seamless, efficient terminal experience—no mouse required.

But, let’s be real—most of us use the mouse these days. You can easily enable mouse support in tmux, making navigation and resizing simple and intuitive. With iTerm2, even vi-style operations feel much more approachable, so you don’t need to be a vi expert to work efficiently.

If you're ready to dive in, let's get started with tmux. First, install it:

```bash
brew install tmux
```

Install tmux on every machine you'd like to use it on.

### 1️⃣ · Start (or attach to) a tmux session

```bash
tmux new -s studio        # create a new session named "studio"
```

*(If you later run `tmux a -t studio` the command will re‑attach instead of creating a duplicate.)*

#### 🏷️ Session Naming Best Practice

Use descriptive session names to keep things organized, especially if you run multiple servers or projects:

```bash
tmux new -s ray-prod      # for production Ray cluster
tmux new -s devbox        # for development
tmux new -s ollama-lab    # for Ollama experiments
```

List sessions with:
```bash
tmux ls
```

### 2️⃣ · Basic pane & window management

| Action | Key sequence |
|--------|--------------|
| **Vertical split** (side‑by‑side) | `Ctrl‑b` then `%` |
| **Horizontal split** (top/bottom) | `Ctrl‑b` then `\"` |
| **Move between panes** | `Ctrl‑b` + arrow keys |
| **Resize pane** | `Ctrl‑b` then hold `Alt` + arrows |
| **Open a new window** (tab) | `Ctrl‑b` then `c` |
| **Switch windows** | `Ctrl‑b` then `n` (next) or `p` (previous) |

#### 🗂️ Window Naming Best Practice

Give each window a meaningful name for quick navigation:

```bash
Ctrl-b ,   # then type a name like: logs, api, train, monitor
```

This helps you know at a glance what each window is for.

### 3️⃣ · Detach and keep everything running

```bash
# Press:
Ctrl-b  d
# (that's Control + b, release, then d)
```

The SSH prompt returns; your Ray logs, training scripts, etc. keep running in the background.

### 4️⃣ · Re‑attach later (after laptop sleep or network drop)

```bash
tmux attach -t studio     # or just: tmux a -t studio
```

Your panes and commands reappear exactly as you left them.

*(If you forget the session name, use `tmux ls` to list them.)*

### 5️⃣ · End the session when all work is done

Inside the tmux shell:

```bash
exit                      # in each pane, or:
Ctrl-b  :kill-session
```

No leftover processes remain.

### 6️⃣ · 🔥 Kill a specific tmux session manually

```bash
tmux kill-session -t studio
```

Useful when a session hangs or you want to clean it up explicitly.

### 7️⃣ · ✏️ Rename an existing session

```bash
tmux rename-session -t oldname newname
```

Helps when you created a session too quickly and want a more descriptive name later.

### 8️⃣ · 📋 View or save tmux buffers

If you've copied text in copy mode, you can inspect or save it:

```bash
tmux show-buffer           # View copied text
tmux save-buffer ~/clip.txt   # Save to file
```

This is especially helpful if you missed the chance to paste right away.

### 9️⃣ · 🛠 View all current key bindings

To check what keybindings are active (great for debugging custom configs):

```bash
tmux list-keys
```

### 🔟 · 🌐 Sync your tmux config across nodes

If you're managing multiple Macs in a cluster, make sure all nodes use the same `.tmux.conf`:

```bash
rsync ~/.tmux.conf user@mac-node-02:~
```

Or automate it via your dotfiles manager (e.g., chezmoi, yadm, or just a Git bare repo).

### 1️⃣1️⃣ · Quick starter workflow

1. **Log tail pane**

   ```bash
   tail -f /var/log/ray/monitor.log
   ```

2. `Ctrl‑b %` → **Second pane** for interactive commands:

   ```bash
   python benchmark_lm.py --model deepseek-70b
   ```

3. `Ctrl‑b \"` → **Third pane** for system monitoring:

   ```bash
   top -o cpu
   ```

Detach (`Ctrl‑b d`), close laptop, come back later, `tmux a -t studio`—everything continues.

### 🛠️ Optional quality‑of‑life tweaks

Create `~/.tmux.conf` on each node (or sync it via dotfiles):

```conf
# Use Ctrl‑a as prefix instead of Ctrl‑b (muscle‑memory from GNU screen)
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# Enable mouse pane selection & scrolling
set -g mouse on

# Vi‑style copy mode
setw -g mode-keys vi

# Status bar with host
set -g status-left "#[fg=cyan]#H "
```

Make sure you export $HOSTNAME in your shell:

```bash
export HOSTNAME=$(hostname)
```

Without this step, the hostname won't show up in the status bar.

Reload without restarting tmux:

```bash
Ctrl-b :source-file ~/.tmux.conf
```

(Use original `Ctrl‑b` for the reload if you change the prefix.)

Or in session:

```bash
tmux source-file ~/.tmux.conf
```

Settings like status-left, status-right, mouse, and key bindings take effect immediately when you reload .tmux.conf. However, settings related to windows or plugin execution may require restarting tmux to take effect.

### 💡 Quick workflow idea

1. **Left pane** → run `ray status` periodically.  
2. **Right‑top pane** → `tail -f /var/log/ray/monitor.log`.  
3. **Right‑bottom pane** → `mactop` to watch CPU/GPU on the M3.

Detach; reconnect tomorrow; everything still rolling.

### 🖱️ Using tmux with mouse and iTerm2

#### 🧑‍💻 iTerm2 + tmux: Copying Text is Effortless

If you use **iTerm2** on macOS, copying text from tmux is seamless:

- Just **select text with your mouse**—it's instantly copied to your clipboard, even if the highlight disappears.
- You do **not** need to enter tmux copy mode for this.
- This works out of the box if you have "Copy to pasteboard on selection" enabled in iTerm2's Preferences (General → Selection).

If you want to use tmux's vi copy mode for precise, pane-aware selection (or when SSH'd from a non-GUI terminal), use `Ctrl-b [` and follow the vi-style selection/copy commands.

**Summary:**
> For most users on iTerm2, just select text as usual—copying "just works." Use tmux copy mode only for advanced needs or when not in iTerm2.

## Appendix A. 🖱️ Using tmux with mouse and iTerm2

Using tmux with mouse is sometimes a pain especially when copying text. 

Here's a quick hack to make it work: 

```plaintext
# ~/.tmux.conf
set -g mouse on

# Set Ctrl-a as prefix (Screen muscle memory)
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# Enable mouse support (pane selection, resize, scroll)
set -g mouse on

# Use Vi keybindings in copy mode
setw -g mode-keys vi

# Copy selected text to macOS clipboard (pbcopy)
# Works when in copy mode (prefix + [) and pressing 'y'
bind -T copy-mode-vi y send -X copy-pipe-and-cancel "pbcopy"

# Quickly toggle mouse mode (ON: prefix + m, OFF: prefix + M)
bind m set -g mouse on \; display-message "Mouse: ON"
bind M set -g mouse off \; display-message "Mouse: OFF"

# Status bar with hostname and battery info
set -g status-left "#[fg=cyan]#(hostname) "

# Optional: refresh tmux env on reload
set-environment -g HOSTNAME "$(hostname)"
```

| Action                                   | Shortcut                                              |
|-------------------------------------------|-------------------------------------------------------|
| Copy text (pane-aware, to clipboard)      | prefix + [ → drag → y                                 |
| Toggle mouse off for normal drag-copy     | prefix + M                                            |
| Toggle mouse back on                      | prefix + m                                            |
| Paste inside tmux                         | prefix + ] (for internal buffer) or just ⌘ + V if used pbcopy |

## Appendix B. 🧠 Ultimate tmux Cheat Sheet (for AI Server Ops on Mac Studios)

| 🧩 **Action** | ⌨️ **Command / Shortcut** | 📝 **Notes** |
|---------------|----------------------------|----------------|
| Start session | `tmux new -s name` | Create a named session |
| Attach session | `tmux a -t name` | Resume session |
| List sessions | `tmux ls` | View all running sessions |
| Kill session | `tmux kill-session -t name` | Destroys session |
| Rename session | `tmux rename-session -t old new` | Renames session |
| Detach session | `Ctrl-b d` | Keep it running in background |
| Split pane (vertical) | `Ctrl-b %` | Side-by-side layout |
| Split pane (horizontal) | `Ctrl-b "` | Top-bottom layout |
| Move between panes | `Ctrl-b` + arrow keys | Switch focus |
| Resize panes | `Ctrl-b` then `Alt` + arrows | Mac-friendly resizing |
| Create new window (tab) | `Ctrl-b c` | |
| Rename window | `Ctrl-b ,` | Name it meaningfully (e.g., ray/logs) |
| Switch window | `Ctrl-b n / p` | Next / Previous window |
| Copy mode (vi) | `Ctrl-b [` | Use `Space` to start selection, `y` to copy |
| Paste inside tmux | `Ctrl-b ]` | |
| Reload config | `tmux source-file ~/.tmux.conf` | No restart needed |
| Copy to system clipboard | Select → `y` (with pbcopy) | Requires pbcopy binding |
| Show current keybindings | `tmux list-keys` | Explore or debug configs |
| View buffer | `tmux show-buffer` | See copied text |
| Save buffer to file | `tmux save-buffer ~/copy.txt` | Dump last copied content |
| Toggle mouse on/off | `prefix + m` / `M` | Fast switch for selection UX |
| Copy with mouse (iTerm2) | Drag (if `copy on selection` enabled) | Native clipboard integration |
| Check clipboard content | `pbpaste` | Verify copy success |
| Sync config to all nodes | Use dotfiles or `rsync ~/.tmux.conf user@host:` | Keep nodes identical |

## Appendix C. 🛠️ Common tmux Troubleshooting & Gotchas

| 🐞 **Issue** | 💡 **Tip / Fix** |
|--------------|-----------------|
| "lost server" errors | Try `tmux kill-server` to reset all sessions |
| Colors look weird | Ensure your `$TERM` is set to `screen-256color` |
| Can't copy with mouse | Check iTerm2 "Copy to pasteboard on selection" is enabled |
| Session won't attach | Use `tmux ls` to list, then `tmux attach -t <name>` |
| Pane resizing not working | Hold `Alt` while using arrow keys |
| Status bar missing hostname | Make sure you export `$HOSTNAME` in your shell |

[⇧ Back&nbsp;to&nbsp;README](../../README.md)