# Divergences

## Lowering

rune lowers each nx operation to tolk's UOps (`packages/rune/next/lib/`). Where
tinygrad's `Tensor` builds the same operation, the lowering builds tinygrad's
decomposition if it computes what nx documents. This section lists every place
where it builds something else, or builds what tinygrad has no source for.

An entry is admitted for one of three reasons only:

- **(a) an OCaml constraint;**
- **(b) nx's documented meaning:** the operation's contract in `nx.mli` or
  `nx_backend.mli`, which nx.cpu computes;
- **(c) the engine's contract:** nx.device's submission protocol.

Each entry gives the reference (tinygrad `79af1ca70`, or none), the raven
lines, what differs, nx's meaning, the agreement class (RFC 0012: exact,
rounded sum, ulp per target, measured bound), the reason, and the test that
pins it. An entry without its test is rejected at review. An entry goes when
its reason goes.

A test is named by its module's suite (`packages/rune/next/test/<module>/`)
and its path in it. An ulp row is measured against correctly rounded results
over its sweep, and the measured maxima per target are recorded as each
target's run lands.
