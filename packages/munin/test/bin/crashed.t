A run whose writer exited without an end reads crashed to a reader under
the writer's boot.

  $ crashed=$(./make_store.exe crash store)
  $ munin runs --root store --status crashed | sed -e "s/$crashed/CRASHED/" \
  >   -e 's/[0-9]\{4\}-[0-9][0-9]-[0-9][0-9]T[0-9:.]*Z/TIME/'
  CRASHED	crashed	TIME	mnist	resumed	-
