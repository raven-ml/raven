A toplevel that loads rig a second time, as one does when [#require]
loaded it and the directives of [dune ocaml top] [#load] it again, fails
at the second load and names the cause.

  $ cat >once.ml <<'ML'
  > #load "rig_edge.cma";;
  > #load "rig_elf.cma";;
  > #load "rig_pool.cma";;
  > #load "rig.cma";;
  > ML
  $ cat once.ml once.ml >twice.ml
  $ ocaml -noinit -I ../lib/edge -I ../lib/elf -I ../lib/pool -I ../lib once.ml
  $ ocaml -noinit -I ../lib/edge -I ../lib/elf -I ../lib/pool -I ../lib twice.ml
  Exception:
  Failure
   "rig is already loaded in this process: a toplevel loads it once, by #require or by the #load directives of dune ocaml top, not both; restart the toplevel".
  [125]

An interactive toplevel, which reads its phrases from standard input,
goes on after the failure, and the collector runs to its end.

  $ cat twice.ml - >session.ml <<'ML'
  > Gc.full_major ();;
  > ML
  $ ocaml -noinit -no-version -noprompt -I ../lib/edge -I ../lib/elf -I ../lib/pool -I ../lib <session.ml
  Exception:
  Failure
   "rig is already loaded in this process: a toplevel loads it once, by #require or by the #load directives of dune ocaml top, not both; restart the toplevel".
  - : unit = ()
  
