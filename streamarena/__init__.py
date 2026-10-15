"""StreamArena — deployment-oriented streaming ML benchmark.

Two kinds of stream source live here:

* ``streamarena.synth``   — synthetic generators written from scratch (classic
  MOA/River concepts plus new API-like simulators). Every generator is both a
  finite dataset writer and an infinite timestamped *live* source.
* ``streamarena.sources`` — real-world streams built from public open APIs.
  ``history`` mode downloads the multi-year archive that the benchmark runs
  on; ``live`` mode polls the same API for deployment tests.

Both produce the same record shape: a dict of features plus a target, so the
benchmark harness does not care where a stream came from.
"""

__version__ = "2.0.0a0"
