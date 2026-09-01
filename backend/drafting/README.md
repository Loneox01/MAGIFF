# Draft advisor

The draft advisor is a separate, read-only agent loop. It does not use the
general request router or the full structured-tool catalog. Application code
first creates a verified draft snapshot from stored ECR plus completed picks;
the model receives only that compact state and may make at most two maintained
report searches for material current uncertainty.

This separation guarantees that an already-drafted player cannot enter the
recommendation shortlist merely because the model remembers or guesses that the
player is available. It also avoids spending a model round to discover basic
draft state.

## Test without a live draft

From `backend/`, inspect a simulated 12-team snake draft at slot 10's fifth
round selection without spending model tokens:

```bash
python -m drafting.cli simulate \
  --teams 12 \
  --draft-slot 10 \
  --round 5 \
  --board-only
```

The simulator drafts every earlier pick in ECR order. It is intentionally
predictable rather than realistic, making regressions easy to reproduce. Run
the dedicated advisor on the same state by omitting `--board-only`:

```bash
python -m drafting.cli simulate \
  --teams 12 \
  --draft-slot 10 \
  --round 5 \
  --question "I started WR-WR. Should I take an RB or keep taking value?"
```

Vary `--draft-slot` and `--round` to cover early, middle, turn, and late-round
decisions. Use `--as-of-date YYYY-MM-DD` to replay a stored ECR snapshot.

## Test with Sleeper

Sleeper's draft endpoints are public and read-only; no Sleeper credential is
needed. Read one current snapshot with either the Sleeper user ID in that draft
or an explicit slot:

```bash
python -m drafting.cli live \
  --draft-id YOUR_DRAFT_ID \
  --user-id YOUR_SLEEPER_USER_ID \
  --board-only

python -m drafting.cli live \
  --draft-id YOUR_DRAFT_ID \
  --draft-slot 7 \
  --question "Who are my pick and two backups?"
```

Rerun the command after the board changes. This first version deliberately does
not poll continuously or submit picks. A later Discord/web adapter can call the
same context builder when the target roster is on the clock.

Live reads add a unique, ignored query value to Sleeper's public draft and picks
requests. Sleeper otherwise permits shared CDN responses to remain stale while
revalidating, which can make a first request show the preceding pick and a
second request show the current board.
