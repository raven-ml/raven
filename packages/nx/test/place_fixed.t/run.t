place_fixed takes a walk of fixed arrays only. One compiles to a function of
any placement; a walk that reaches positions of a parameter fails to compile.

  $ for d in $(echo "$OCAMLPATH" | tr ':' ' '); do
  >   if [ -f "$d/nx/META" ]; then nx=$d/nx; fi
  > done
  $ ocamlc_nx () {
  >   ocamlc -I $nx -I $nx/array -I $nx/kernel "$@"
  > }
  $ ocamlc_nx -c walk.ml

A walk of fixed arrays:

  $ ocamlc_nx -i accepted.ml
  val steps : Nx.host Walk.steps
  val on : 'a Nx.Placement.t -> 'b Walk.steps -> 'a Walk.steps
  val placed : Nx.Host.d Walk.steps

A walk that reaches a parameter's positions:

  $ ocamlc_nx -i positions.ml 2>&1 | grep -o 'less general'
  less general
