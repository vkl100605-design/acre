# When Three Bosses Walk Into a Server Room: the ACRE++ Story

Reinforcement Learning · Cloud Ops · Open Source · 
April 26, 2026

---

It's 2 a.m. The dashboard looks fine — until it doesn't.
Traffic spikes, errors creep up, latency starts eating SLIs, and
someone's already asking why the cloud bill jumped.
ACRE++ was built for exactly this moment.

Most RL environments are polite. Single scalar reward, one agent,
well-behaved state space. The real world is none of those things.
In production you're simultaneously trying to keep the service alive,
hold down costs, and not let users notice anything's wrong.
These three goals don't cooperate — and ACRE++ doesn't pretend they do.

"It's one system squeezed by three masters — and a single controller still has to pick the next move."

---

## What ACRE++ actually is ?

A compact RL environment simulating cloud-style ops under realistic
constraints: delayed capacity effects, noisy telemetry, and three
competing agents who can disagree on every single step.

| Agent       |                           Role                                         |
|------------ |------------------------------------------------------------------------|
|RL agent     |Tabular Q-learning proposes actions based on learned long-horizon value |
|Cost agent   | Rule-based FinOps guardrails — blocks wasteful scaling                 |
|Latency agent| UX guardrail — escalates when latency crosses thresholds               |

When these agents disagree, overrides fire.
The final action may not be what the RL policy proposed — and that's
by design. It mirrors how real SRE teams work: the on-call engineer
makes a call, but the FinOps ticket and the SLO dashboard both have
veto power.

---

## The numbers that matter

Same environment, same agents, same override rules — only the RL
policy changes. 10-episode evaluation on incident_recovery.

| Metric         | Random | Trained | Change        |
|----------------|--------|---------|---------------|
| Episode Reward | -30.66 | -25.10  | +5.56 (~+18%) |
| Avg Latency(ms)| 194.3  | 187.3   | -7.0          |
| Uptime         | 42.7%  | 45.0%   | +2.27 pts     |
| Total Cost     | 114.48 | 112.42  | -2.06         |

What makes these numbers meaningful isn't any single metric —
it's that all four move in the right direction simultaneously.

A naive approach might squeeze latency at the cost of a bigger bill,
or cut costs by letting uptime suffer.
The trained policy earns better outcomes across every dimension at once.

---

## Why "emergent behavior" is more than a buzzword ?

In most single-agent RL, the agent learns to exploit whatever the
reward function rewards. Tweak the reward, get a different agent.

ACRE++ adds friction: the cost and latency agents can override
the RL policy's proposals, so raw reward maximization runs into
real resistance.

Outcomes aren't scripted. They emerge from who wins each
negotiation at each timestep — richer learning signal, more
robust policy. You don't get a policy tuned for the eval; you get
one that has genuinely learned to navigate disagreement.

---

## What's inside ?

State: traffic, health, errors, latency (ms), step cost,
capacity, risk — delayed effects and noisy signals.

Actions: hold · scale_up · scale_down · restart · rollback

Reward: blends uptime/recovery with penalties for overload,
waste, cost, errors, latency — winning means balancing, not
maxing one dial.

3 task presets:
- stability_spike — sudden traffic pressure
- incident_recovery — messy aftermath of a bad deploy
- budget_guardrail — cost discipline vs reliability

---

## Run it yourself

```bash
python train_clean.py --task-id incident_recovery --episodes 80 --seed 42
```

Live demo: https://huggingface.co/spaces/vkl1006/acre

Source: https://github.com/vkl100605-design/acre

Colab: https://colab.research.google.com/github/vkl100605-design/acre/blob/main/ACRExx_Training_Colab.ipynb

Apache 2.0 — open source, run it, fork it, break it.

---

> "If your RL can't survive this room, it won't survive the pager —
> ACRE++ is the room."