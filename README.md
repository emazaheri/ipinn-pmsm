# ipinn-pmsm

An inverse physics-informed neural network that recovers a permanent-magnet
synchronous machine's magnet flux linkage from nothing but the terminal
voltages and currents of a field-oriented drive.

The parameter matters because losing it *is* demagnetization. A magnet that has
been overheated, over-driven or simply aged carries less flux, and the machine's
torque per ampere falls with it. Measuring that without taking the motor apart
means inferring it from what the drive already logs.

This repository is the reference implementation behind the interactive demo at
[emazaheri.com](https://emazaheri.com), and it is the source of the golden
vectors that the browser port is tested against.

## The idea

A physics-informed network is fitted to a short window of measured signals. It
takes `[t, vd, vq]` and predicts `[id, iq, we]`, and it is trained on two terms
at once:

- a **supervised** term, the mean squared error against the measured currents
  and speed;
- a **physics** term, the residual of the machine's own differential equations,
  evaluated using the network's derivative with respect to its time input.

The flux linkage is not an output. It is a *trainable scalar inside the physics
term*, initialised deliberately wrong at 0.1 Wb. Minimising the residual is what
drags it to the truth. Nothing ever tells the network the answer.

```
r_d = Ld·(∂id/∂t) - vd + Rs·id - we·Lq·iq
r_q = Lq·(∂iq/∂t) - vq + Rs·iq + we·(Ld·id + λm)
                                          ^^^^
                                          the unknown

L = α·L_supervised + β·(mean r_d² + mean r_q²)
```

The window is short (15 ms, 14 samples at 1 kHz), a fresh network is fitted to
each one, and the two second record yields 142 independent estimates. That is
the whole method: no pretrained weights, no labels for the parameter, and a
model small enough that all of it runs in a browser tab.

## Identifiability, which is the interesting part

`λm` appears in exactly one place, multiplied by the electrical angular
velocity. So the sensitivity of the measurement to the parameter is

```
∂vq/∂λm = we = pole_pairs · omega_m
```

At 500 rpm with four pole pairs that is 209 V/Wb. At 1000 rpm it is 419. The
same voltage error therefore maps to **twice** the flux error at the lower
speed, and **at standstill the parameter cannot be identified at all**.

The reference scenario steps the dynamometer from 500 to 1000 rpm at `t = 1.0 s`
precisely so this is visible: the scatter in the estimate should narrow by about
a factor of two, in the middle of a run that is otherwise unchanged. It is a
prediction the plots confirm or refute, not a claim in a caption.

## What is here

| Module | Does |
| --- | --- |
| `params.py` | Frozen configuration records for the machine, inverter, simulation, windowing and estimator |
| `transforms.py` | Clarke and Park, amplitude invariant |
| `machine.py` | The dq state equations, with current-dependent saturation |
| `inverter.py` | Magnitude limit, conduction drop, dead-time projection |
| `control.py` | Two field-oriented PI current loops |
| `simulate.py` | Fixed-step RK4 over the closed loop |
| `scenarios.py` | The eight-combination excitation and the speed step |
| `windows.py` | Cutting a record into estimation windows |
| `pinn/` | The network, the physics loss, and the rprop training loop |
| `fixtures.py` | Golden-vector export for the TypeScript twin |

## Run

```bash
uv sync
uv run ipinn-pmsm run --case a        # clean
uv run ipinn-pmsm run --case b        # magnetic saturation
uv run ipinn-pmsm run --case c        # hot motor: weak magnet and wrong Rs
uv run ipinn-pmsm run --case d        # saturation plus measurement noise
uv run ipinn-pmsm run --case all --check   # assert every published figure
uv run ipinn-pmsm describe --case c        # print what a case gets wrong
uv run ipinn-pmsm export-fixtures --out ../../components/sections/observer/fixtures
```

The four cases are TOML files in `configs/`, not four copies of a script.

## Two presets

`reference` reproduces the original research code, quirks included. `tuned`
corrects them. Both ship, because the published numbers have to stay
reproducible and the corrections have to be attributable.

What `reference` preserves, all four measured rather than inferred:

1. **No input or output scaling.** The network is fed absolute time, so within
   a window spanning 1.715 to 1.728 s the time input varies by one part in a
   hundred. Meanwhile `vq` is hundreds of volts and `id` is about one ampere.
   This is badly conditioned, and it is the likeliest cause of the roughly five
   percent per-window spread.

2. **The optimiser schedule is a constant.** The original calls
   `cosine_onecycle_schedule(1e-3, epochs, 1e-4, 0.1)` positionally against a
   signature of `(transition_steps, peak_value, pct_start, div_factor,
   final_div_factor)`, so the arguments do not mean what the call site implies.
   Measured against optax 0.2.8 it returns `epochs / 100` at **every** step:
   one unique value across the whole run. It is not a schedule. Two
   consequences follow. The published runs used a flat 1.5x multiplier on
   every rprop update, and epoch count and step size are the same knob in the
   original, so any epoch sweep there conflated the two. `tuned` uses a flat
   one and breaks the coupling. A test pins this so a future optax cannot
   change it silently.

3. **Windows that do not slide.** `start = index · length` tiles them without
   overlap, despite the name in the original filenames.

4. **A time axis 50 ppm wider than the integration step.** The original built
   its sample times with `linspace(0, 2.0, 20000)`, which spaces them by
   `2.0 / 19999` rather than by the `Ts = 1e-4` it actually integrates with.
   The drift is irrelevant to the physics, but it turns `int(0.015 / dt)` from
   15 into 14, and that is the entire reason the published run reports 142
   windows of 14 samples instead of 133 of 15. `legacy_time_axis` reproduces
   it.

`tuned` normalises the time input to `[0, 1]` within each window (rescaling the
residual's derivative by `1/T`, where `T = (B-1)·dt` and not the nominal window
length), standardises the outputs, re-sweeps `β`, and exposes a real stride.

A caution on what to expect: the estimator deliberately uses **unsaturated**
`Ld` and `Lq` while the plant saturates. That is a *bias*, and normalisation
addresses *conditioning*, which is *variance*. The honest expectation is that
`tuned` tightens the spread substantially and moves the mean error very little.
Both numbers are reported.

## Results

Mean absolute error on `λm`, 142 windows, reference preset:

| Case | Plant | Estimator is told | True λm | Published | Reproduced |
| --- | --- | --- | --- | --- | --- |
| A | ideal | the truth | 0.206 | 0.81% | **0.81%** |
| B | saturation | unsaturated Ld, Lq | 0.206 | 0.95% | **0.95%** |
| C | saturation, weak magnet, Rs = 0.52 | Rs = 0.46 | 0.191 | 2.13% | **2.13%** |
| D | saturation, 10% current and 5% voltage noise | unsaturated Ld, Lq | 0.206 | 2.11% | **2.16%** |

Cases A, B and C land on the published figures to two decimal places, and case
C reproduces the reference's *biased* mean estimate of 0.193996 as 0.193999.
Case D is within 0.05 points, which is the residue of a different noise draw
and of fixed-step RK4 in place of adaptive LSODA. `ipinn-pmsm run --case all
--check` asserts all four, and CI runs it.

### What case C actually is

It is worth naming, because the original filenames call it a demagnetization
case and that is only half of it. Case C changes **two** things: the magnet
loses 7% of its flux *and* the stator resistance rises 13%, from 0.46 to 0.52,
while the estimator keeps using 0.46.

Both changes are thermal and they happen together in a real machine, so this
is the most physically honest of the four: a hot motor, diagnosed from a cold
commissioning sheet. It is also where the difficulty comes from. Correct the
resistance and the same weakened magnet is recovered at about 1.0% error, less
than half of 2.13%. **Demagnetization is not what makes case C hard. The
resistance the estimator has wrong is.**

### Where the spread comes from

The roughly 5% standard deviation is not spread evenly. Two things dominate it.

**Speed.** Every case shows the first half of the record, at 500 rpm, doing
measurably worse than the second half at 1000 rpm, exactly as
`∂vq/∂λm = ωe` predicts:

| Case | 500 rpm | 1000 rpm | ratio |
| --- | --- | --- | --- |
| A | 0.89% | 0.73% | 1.2 |
| B | 1.02% | 0.87% | 1.2 |
| C | 2.61% | 1.66% | 1.6 |
| D | 2.69% | 1.63% | 1.7 |

**Transitions.** One single window out of 142 straddles the speed step at
t = 1.0 s, so 209 rad/s of change lands inside 13 ms. It comes back 39% wrong
while both its neighbours are correct to better than 0.01%. Medians are
therefore far below means in every case: case A's median error is 0.00%
against a mean of 0.81%.

## The browser port

`components/sections/observer/` on the website carries a TypeScript twin: the
same simulator, the same network, and hand-derived forward-over-reverse
gradients in place of JAX. `export-fixtures` writes the golden vectors its test
suite asserts against, so the two cannot drift silently.

Exact parity is not the goal and is not achievable: JAX runs float32 and
reassociates, the port runs float64, and JAX's threefry PRNG has no TypeScript
equivalent, so the shipped run initialises differently and matches in aggregate
rather than window for window. What *is* asserted exactly is the simulator
trace, the forward pass, the analytic gradients against finite differences, and
the optimiser's step-size ladder.

## Provenance

Consolidated from four years of research code: a field-oriented drive simulator
and a three-state inverse PINN, previously spread over seventeen
copy-and-edited forks of one engine and four near-identical case scripts
differing in two numbers each.

## Licence

MIT. See `LICENSE`.
