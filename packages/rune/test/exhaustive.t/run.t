An interpreter names every construct, which warnings 4 and 8 check as errors.
One that names them all compiles; one that forgets a construct does not, and
neither does one that covers what it forgets with a wildcard.

  $ for d in $(echo "$OCAMLPATH" | tr ':' ' '); do
  >   if [ -f "$d/nx/META" ]; then nx=$d/nx; fi
  > done
  $ internals=$PWD/../internals/.rune_internals.objs/byte
  $ compile () {
  >   ocamlc -I $nx -I $nx/effect -I $nx/array -I $nx/backend -I $nx/cpu \
  >     -I $nx/device -I $nx/dtype -I $internals -c "$1" 2>&1 | grep Error
  >   true
  > }
  $ compile names.ml
  $ compile forgets.ml
  Error (warning 8 [partial-match]): this pattern-matching is not exhaustive.
  $ compile wildcard.ml
  Error (warning 4 [fragile-match]): this pattern-matching is fragile.
