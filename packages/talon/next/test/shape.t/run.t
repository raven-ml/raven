An expression's shape is in its type: each mistake below fails to compile
with the message the RFC promises, and the forms that should compile do.

  $ for d in $(echo "$OCAMLPATH" | tr ':' ' '); do
  >   if [ -f "$d/nx/META" ]; then nx=$d/nx; fi
  >   if [ -f "$d/talon/META" ]; then talon=$d/talon; fi
  > done
  $ ocamlc_talon () {
  >   ocamlc -I $talon/next -I $nx -I $nx/effect -I $nx/array -I $nx/backend \
  >     -I $nx/cpu -I $nx/device -I $nx/dtype -c "$1"
  > }
  $ compile () { ocamlc_talon "$1" 2>&1 | grep -A3 Error; }

A row expression where a reduction belongs:

  $ compile agg_out.ml
  Error: This expression has type Talon_next.Expr.row Talon_next.Expr.out
         but an expression was expected of type
           Talon_next.Expr.agg Talon_next.Expr.out
         Type Talon_next.Expr.row is not compatible with type

A reduction of a reduction:

  $ compile mean_mean.ml
  Error: This expression has type
           (float, Talon_next.Expr.agg) Talon_next.Expr.t
         but an expression was expected of type
           (float, Talon_next.Expr.row) Talon_next.Expr.t

A reduction combined with a row expression:

  $ compile mixed.ml
  Error: This expression has type (int, Talon_next.Expr.row) Talon_next.Expr.t
         but an expression was expected of type
           (int, Talon_next.Expr.agg) Talon_next.Expr.t
         Type Talon_next.Expr.row is not compatible with type

A cast read at another kind:

  $ compile cast_kind.ml
  Error: This expression has type
           (float, Talon_next.Expr.row) Talon_next.Expr.t
         but an expression was expected of type
           (string, Talon_next.Expr.row) Talon_next.Expr.t

An integer where a float belongs:

  $ compile int_float.ml
  Error: This expression has type (int, 'a) Talon_next.Expr.t
         but an expression was expected of type (float, 'b) Talon_next.Expr.t
         Type int is not compatible with type float

Literals generalize over shapes, a datetime has an hour and a date a year, and
[each] takes a function of every type:

  $ ocamlc_talon accepted.ml && echo compiled
  compiled
