(** Goldens: outputs of tinygrad, recorded by [test/gen/generate.py].

    A golden is a file [<name>.golden] beside the suite that reads it: its
    body and a final newline. A suite lists its goldens
    and their {{!tolk}[.tolk]} files in its stanza's [deps], which copies them
    beside its executable, and names
    them by file name. A relative file name is read from there, whatever the
    working directory.

    Each function here reads its golden when the test runs, except {!cases},
    which reads it when the suite is declared, to make its rows. A file that
    cannot be read raises [Sys_error], and one that is not a golden raises
    [Failure]. *)

(** {1:tolk Tolk's output}

    Where tolk differs from upstream on purpose, [<name>.tolk] sits beside
    [<name>.golden]. Every function here, given a file [<name>.golden], reads
    [<name>.tolk] when it exists. Its first line is

    {v # tolk D24 D74 golden=<hash> v}

    and the rest is the golden's body format. The [D]-numbers are the entries of
    [DIVERGENCES.md] the difference belongs to, and [golden=] is the hash of the
    body of [<name>.golden] the file was recorded against, so a golden that
    moves fails its test without running the generator. A [<name>.tolk] with no
    [<name>.golden] is an output with no upstream answer and carries no hash.

    Reading a [.tolk] raises [Failure] naming the file and what is wrong if its
    header names no entry or is malformed, if the golden's hash is missing or
    not the current golden's, if it carries a hash and the golden is gone, or if
    its body equals the golden's. Under windtrap's [-u], {!text} and {!graph}
    write an output that differs from its golden as [<name>.tolk], keeping the
    entries of the file it replaces; a new file names none and fails until a
    person adds one. [-u] writes nothing for an output equal to its golden. A
    table or an input has no output to write: when its golden moves, the failure
    prints the new hash, for the header. *)

val updating : bool
(** [updating] is whether this run has windtrap's [-u], which writes each
    output that differs from its golden as a [.tolk]. A suite whose golden is
    compared in parts records the whole output only then. *)

val body : string -> string
(** [body file] is the body of [file]'s [.tolk] when one exists, else of the
    golden [file]. *)

val text : string -> (unit -> string) -> Windtrap.test
(** [text file actual] is the test, named [file], that [actual ()] is the body
    of the golden [file]. Its failure prints their diff. *)

(** {1:graphs Graphs}

    The body of a graph golden is a UOp graph in the text of {!Graph}. *)

val graph : string -> (unit -> Tolk.Ops.t) -> Windtrap.test
(** [graph file sink] is the test, named [file], that the graph under [sink ()]
    is the graph golden [file]. Its failure prints the diff of their texts. *)

val sink : string -> Tolk.Ops.t
(** [sink file] is the sink of the graph golden [file] ({!Graph.of_string}).
    Raises [Failure] naming [file] if its body is not a graph. Under [-u] a
    [.tolk] whose golden moved is read without failing, since the run rewrites
    it. *)

val upstream : string -> Tolk.Ops.t
(** [upstream file] is the sink of the graph golden [file] itself, whatever
    [.tolk] sits beside it. Raises [Failure] as {!sink} does. *)

(** {1:tables Tables}

    The body of a table golden is lines of cells separated by tabs: the column
    names, then one line per row, and at least one row. A cell is the text
    tinygrad prints for its value, such as [dtypes.float], [True] or [inf].

    A row is given as its [cell] function: [cell column] is the row's cell in
    [column], and raises [Invalid_argument] if the table has no column [column].
*)

val cases :
  ?key:string list ->
  ?differs:(string * string list) list ->
  string ->
  ((string -> string) -> unit) ->
  Windtrap.test
(** [cases file check] is the table, named [file], that runs [check cell] on
    each row of the table golden [file]. A row is identified by its cells in
    [key], as [a=dtypes.half b=dtypes.char]; [key] defaults to the first column.
    A table that cannot be read as a golden is one failing test named [file],
    so a stale [.tolk] does not stop the suite.

    [differs] states which rows of [file]'s {{!tolk}[.tolk]} differ from the
    golden, as [("D24", ["op=Ops.ADD dtype=dtypes.f32"])]. A pattern is pairs
    [column=value] separated by spaces, all of which a row must hold; a word
    without [=] extends the value before it, as in [name=kernel with opts]. The
    [fst] is an entry of the [.tolk]'s header. [check] runs on the rows of the
    [.tolk] as usual, so a listed row is held to tolk's value. The list must be
    exact: the table is one failing test if a row that differs is not listed,
    if a listed row equals the golden, if a pattern holds on no row, if an
    entry is not in the header, if the two tables have different numbers of
    rows, or if the [.tolk] or the golden is missing. *)

val columns : string -> string list
(** [columns file] is the column names of the table golden [file], in order. *)

val rows : string -> (string -> string) list
(** [rows file] is the rows of the table golden [file], in order. *)
