An interpreter names every construct, which warnings 4 and 8 check as errors.
One that names them all compiles; one that forgets a construct does not, and
neither does one that covers what it forgets with a wildcard.

nx is found on OCAMLPATH, whose separator is ';' on Windows, where a child's
output ends its lines with CRLF, and the internals by a relative path, which
the native Windows compiler reads where it would not read the shell's $PWD.

  $ sep=:
  $ case "$(ocamlc -config-var os_type)" in Win32*) sep=';' ;; esac
  $ IFS=$sep
  $ for d in $OCAMLPATH; do
  >   if [ -f "$d/nx/META" ]; then nx=$d/nx; fi
  > done
  $ unset IFS
  $ internals=../internals/.rune_internals.objs/byte
  $ compile () {
  >   ocamlc -I "$nx" -I "$nx/effect" -I "$nx/array" -I "$nx/backend" \
  >     -I "$nx/cpu" -I "$nx/device" -I "$nx/dtype" -I "$internals" -c "$1" \
  >     2>&1 | grep Error
  >   true
  > }
  $ compile names.ml
  $ compile forgets.ml
  Error (warning 8 [partial-match]): this pattern-matching is not exhaustive.
  $ compile wildcard.ml
  Error (warning 4 [fragile-match]): this pattern-matching is fragile.
