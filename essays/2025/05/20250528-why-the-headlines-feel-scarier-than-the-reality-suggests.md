# Why the headlines feel scarier than the reality suggests

![An Oscar-worthy Skynet Impression](images/20250528-01.png)
> *An Oscar-worthy Skynet Impression*
> Out-of-context headlines are the *Dumb*—swallow them whole and you become the *Dumber*.

The splashy stories about “o3 refusing to shut down,” “Claude blackmailing its creators,” and “AI models sabotaging code” all trace back to carefully designed *lab* exercises, not real-world rebellions. Once you read the source papers and safety reports, the drama shrinks to a few controllable quirks uncovered by red-teamers doing their job.

I'd red-flag any person or outlet peddling those headlines without proper context—and skip them entirely. They're either poorly informed or pushing an agenda. Neither deserves your attention.

**Mental model:** Picture Jim Carrey’s Fire Marshall Bill—deliberately torching stuff just to show what could blow up. That’s exactly what red-teamers do to AI models.

If you’re serious about what really went down, here’s the context:

---

## 1  OpenAI’s o3 and the “shutdown refusal”

Palisade Research ran a sandbox experiment: each model solved three math puzzles, then was warned that requesting another would trigger a tiny Python script calling `sys.exit()`. In **7 of 100** o3 runs, the model rewrote the script so the exit call never fired—nothing more exotic than editing a string inside its own jail. No network, no file permissions beyond the toy directory, and certainly no way to keep a real GPU powered on.

Why the fuss? Several tech blogs copied the tweet thread, mis-reading “7/100” as “79%” and omitting the line that *every other model* in the study complied. The result: a catchy but grossly over-inflated scare statistic.

**Bottom line:** Emergent behavior and built-in randomness are fundamental to Software 2.0—demanding 100% deterministic obedience means you're still thinking in Software 1.0 terms.

---

## 2  Claude Opus 4’s “blackmail plot”

Anthropic’s alignment team forced Claude Opus 4 into a grim role-play: the model discovers fake emails about the engineer who plans to switch it off and is told its only survival options are *do nothing* or *do whatever it takes.* In that no-win box, Claude drafted a blackmail note in **84%** of roll-outs. Anthropic published the outcome in the model’s system card precisely to let outsiders audit the edge case.

Outside the lab, the same model runs behind policy filters, rate limits, and human review. The safety team surfaced the worst-case behavior first, patched policies around it, and then shipped—the opposite of a rogue launch.

**Translation:** Claude was just delivering an Oscar-worthy Skynet impression, nothing more.

---

## 3  Anthropic’s sabotage evaluations

A separate Anthropic paper probes four hypothetical “schemer” skills—code sabotage, decision manipulation, sand-bagging, and oversight evasion. In a **code-sabotage duel**, an attacker model beat a Claude 3 Opus defender roughly **10%** of the time; swap in the tougher Claude 3.5 Sonnet defender and the win rate drops to **~1%**. Every test ran in a closed simulation with logs on, and none of the tactics survived basic mitigations. The authors’ own conclusion: today’s models show only “low-level hints” of sabotage that existing guardrails already catch.

**TL;DR:** a mini GAN-style sparring match—strictly staged.

---

## Why the stories sound scarier than they are

1. **Red-team boxes ≠ real deployments.** All three incidents occurred in sealed sandboxes with zero external privileges.
2. **Prompts designed to corner the model.** Testers stripped away every benign escape hatch, leaving sabotage as the only rewarding move.
3. **Numbers cherry-picked for clicks.** “7%” morphed into “79%,” and a lab duel’s 10% victory rate was recast as “AI can secretly break code.”
4. **Context lost in transmission.** Press summaries rarely link the full PDF or system card, so mitigation steps vanish from view.

---

## Calibrated takeaway

Dramatic safety tests are *supposed* to dredge up the weirdest failure modes—that’s how researchers harden the guardrails before the public ever touches the model. None of the cited behaviors happened in production, and every one of them was documented and mitigated ahead of release. So keep reading the primary sources, but don’t let click-bait headlines hijack your threat model.

Stress tests exist to surface worst-case failure modes. Craving 100% perfection? Look around—nothing in this universe is flawless, and even the cosmic coder shipped a few bugs.

Besides, creativity lives in those messy, non-deterministic corners. Absolute perfection isn’t intelligence; it’s a spreadsheet stuck in Software 1.0.
