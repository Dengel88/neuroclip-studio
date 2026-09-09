# Neuroclip Studio

A multi-agent pipeline that turns a one-line video brief into a storyboard and a
set of technical prompts for a video generation model.

Four agents run in sequence — scriptwriter, storyboarder, prompt engineer, VFX
supervisor — behind a FastAPI backend and a single-page UI. The interesting part
is not the agent chain; it is that **the pipeline checks its own output in code**
and repairs it before the user ever sees it.

---

## What is actually enforced

An LLM asked for a 30-second video in 4/6/8-second clips will occasionally hand
back scenes that add up to 28. Asking nicely in the prompt is not a control. So
every rule lives twice: once in the prompt (rendered from `config.yaml`) and once
in [`domain.py`](domain.py), which decides whether an answer is acceptable.

| Rule | Enforced in |
|---|---|
| Scene durations sum to exactly the requested total | `domain.validate_storyboard` |
| Every clip length is one the renderer can produce (`4/6/8s`) | `domain.validate_storyboard` |
| Scenes numbered `1..n` with no gaps, count under the cap | `domain.validate_storyboard` |
| One prompt per scene, `omni_duration` matches the storyboard | `domain.validate_prompts` |
| Every prompt carries all six template dimensions | `domain.validate_prompts` |
| `image-to-video` scenes actually carry a reference-frame prompt | `domain.validate_prompts` |
| The brief is satisfiable at all before any model call | `domain.assert_brief_feasible` |

The last one is worth a sentence: a 15-second video cannot be built from 4/6/8
second clips, whatever the model says. That is a coin-problem check, so the
request is refused in milliseconds for zero API spend rather than after three
failed repair attempts.

When a check fails, `PipelineOrchestrator._generate_validated` asks again with
the failure reason appended to the prompt (`repair.max_attempts` in the config).
When the attempts run out the endpoint returns **502 with the actual reason**,
not a generic error.

## Architecture

```
index.html            single-page UI; holds a session id, nothing else
  |
main.py               auth, rate limiting, sessions, HTTP <-> domain error mapping
  |
pipeline.py           PipelineState + stages + repair loop + regenerate_scene
  |
domain.py             the rules, in code
  |
llm/                  base.py (interface) | gemini.py | mock.py | factory.py
  |
config.yaml           models, temperatures, retries, domain constants
prompts/*.md          system instructions with $placeholders filled from config
```

**State is server-side.** `PipelineState` holds the brief, the concepts, the
chosen one, the storyboard, the prompts and an audit trail. That is what makes
`regenerate_scene(state, n)` possible: one shot is re-generated, its duration is
held fixed so the total stays valid, the stale prompt for that scene is dropped,
and every other scene is untouched. There is a test that asserts exactly that.

**Transport failures and validation failures are different things.**
`GeminiProvider` retries and rotates keys on 429/503/timeouts with exponential
backoff and jitter; a `ValidationError` is raised straight through to the repair
loop instead. Conflating them means one malformed JSON answer burns every API key
in the pool and arrives as an error the repair loop cannot recognise —
`tests/test_provider.py::test_validation_error_does_not_consume_keys` pins this.

Error classification is explicit: `429/503/…` → retry, `401/403` → next key,
`404` → next model in the chain, `400` → abort immediately.

## Setup

```bash
cp .env.example .env       # fill in BASIC_AUTH_* and at least GEMINI_API_KEY_1
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The app **refuses to start** without `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD`.
There are no default credentials. See [`.env.example`](.env.example).

Run it offline, with no API key at all:

```bash
LLM_PROVIDER=mock python main.py
```

## Swapping the model

`config.yaml` is the only place a model name appears. To move the whole pipeline
onto a different model, edit the chain:

```yaml
models:
  chains:
    reasoning:
      - gemini-3.1-pro-preview   # primary
      - gemini-3.5-flash         # fallback
```

No Python file mentions a model name. Temperatures, retry policy, repair
attempts, session TTL, rate limits and the domain constants live in the same
file — and `tests/test_config_and_prompts.py::test_changing_the_config_changes_the_prompt`
verifies that a change there actually reaches the rendered prompt.

To add a provider: implement `llm/base.py::LLMProvider`, add one line to
`llm/factory.py`. Agents never import `google.genai`.

## Tests

```bash
pytest
```

100 tests, no network. Coverage is aimed at the things that broke before:
placeholder substitution, the duration validator, key rotation against faked
429/503/404/400 responses, the transport-vs-validation split, repair-loop
recovery and exhaustion, targeted retakes, session expiry and eviction, rate
limiting, and the HTTP contract the frontend depends on.

## Evals

```bash
python evals/run.py                 # offline, deterministic, free
python evals/run.py --provider gemini   # spends real API quota
```

11 briefs: 8 that must succeed (10s reel through 120s long-form, plus `14s`
which forces awkward arithmetic) and 3 that must be **refused** — an impossible
15s total, an unsupported aspect ratio, and a sub-minimum duration. A refusal is
only counted as a pass if it cost zero model calls.

Latest run on the mock provider:

```
CASE                       EXPECT   RESULT   DETAIL
short_reel                 pass     PASS     duration_sum=10s == 10s, scene_count=2, dimensions=all present
minute_explainer           pass     PASS     duration_sum=60s == 60s, scene_count=8, dimensions=all present
character_scene            pass     PASS     duration_sum=30s == 30s, scene_count=4, dimensions=all present
square_social              pass     PASS     duration_sum=24s == 24s, scene_count=3, dimensions=all present
minimum_length             pass     PASS     duration_sum=4s == 4s,   scene_count=1, dimensions=all present
awkward_arithmetic         pass     PASS     duration_sum=14s == 14s, scene_count=2, dimensions=all present
long_form                  pass     PASS     duration_sum=120s == 120s, scene_count=15, dimensions=all present
text_heavy_brief           pass     PASS     duration_sum=20s == 20s, scene_count=3, dimensions=all present
impossible_odd_duration    reject   PASS     rejected: 15s cannot be composed from [4, 6, 8]; cost=0 model calls
unsupported_aspect_ratio   reject   PASS     rejected: 21:9 not supported; cost=0 model calls
below_minimum_duration     reject   PASS     rejected: outside 4-300s; cost=0 model calls

Pass rate: 11/11 (100.0%)
```

Reports are written to `evals/results/` (gitignored). The runner exits non-zero
below 100%, so it works as a CI gate. Numbers above are from the mock provider —
they verify the pipeline's own guarantees, not Gemini's creative quality.

## Security notes

- No credentials in code or in this README; the app refuses to start without them.
- `secrets.compare_digest` for the auth comparison.
- Input length and range limits come from `config.yaml` (`limits.*`, `domain.*`)
  and are enforced by the request models, so the advertised bound is the enforced
  bound (`tests/test_api.py::test_oversized_input_is_rejected`).
- Per-user + per-IP fixed-window rate limiting, with a tighter window on the
  expensive generation endpoints.
- Sessions are TTL-bounded *and* capacity-bounded — an unbounded session dict is
  a memory-exhaustion vector.
- API keys never reach the logs, not even partially; log lines carry a key index.
- Model output is HTML-escaped before it is interpolated into the page.

## Limitations

These are real and deliberate, not oversights:

- **In-memory sessions.** Restarting the process drops every in-flight pipeline,
  and the store is per-worker. Production wants Redis.
- **In-process rate limiting.** Same caveat: it protects one instance's key
  budget, not a fleet.
- **Ephemeral image storage.** `images.storage: static` writes JPEGs to `static/`
  and cleans them up after `retention_seconds`. On a host with an ephemeral or
  read-only filesystem, switch to `images.storage: base64` — the bytes come back
  inline and nothing touches the disk.
- **The final render is a showcase.** Full video synthesis is not wired up; the
  "Render" button plays a pre-rendered clip. The pipeline produces prompts, not
  video.
- **Eval numbers measure the pipeline, not the model.** They prove the contracts
  hold; they say nothing about whether the storyboard is any good.
