An interpreter names every operation of Nx.Prim, which warnings 4 and 8 check
as errors. One that names them all compiles; one that forgets an operation
does not, and neither does one that covers what it forgets with a wildcard.

  $ lib=$PWD/../../lib
  $ compile () {
  >   ocamlc -I $lib/.nx.objs/byte -I $lib/array/.nx_array.objs/byte \
  >     -I $lib/kernel/.nx_kernel.objs/byte \
  >     -I $PWD/../../../rig/lib/.rig.objs/byte -c "$1" 2>&1 | grep Error
  >   true
  > }
  $ compile names.ml
  $ compile forgets.ml
  Error (warning 8 [partial-match]): this pattern-matching is not exhaustive.
  $ compile wildcard.ml
  Error (warning 4 [fragile-match]): this pattern-matching is fragile.
