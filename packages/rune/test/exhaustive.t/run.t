An interpreter names every operation, which warnings 4 and 8 check as errors.
One that forgets an operation does not compile, and neither does one that
covers what it forgets with a wildcard.

  $ for d in $(echo "$OCAMLPATH" | tr ':' ' '); do
  >   if [ -d "$d/nx/effect" ]; then lib=$d/nx; fi
  > done
  $ compile () {
  >   ocamlc -I $lib/effect -I $lib/array -I $lib/backend -I $lib/cpu \
  >     -I $lib/device -I $lib/dtype -c "$1" 2>&1 | grep Error
  > }
  $ compile forgets.ml
  Error (warning 8 [partial-match]): this pattern-matching is not exhaustive.
  $ compile wildcard.ml
  Error (warning 4 [fragile-match]): this pattern-matching is fragile.
