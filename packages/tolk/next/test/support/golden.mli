(** Goldens: outputs of tinygrad, recorded by [test/gen/generate.py].

    A golden is a file [<name>.golden] beside the suite that reads it. Its first
    line names the tinygrad commit it was recorded from, as
    [# tinygrad <commit>], and the rest is its body. A suite lists its goldens
    in its stanza's [deps], which copies them beside its executable, and names
    them by file name. A relative file name is read from there, whatever the
    working directory.

    Each function here reads its golden when the test runs, except {!cases},
    which reads it when the suite is declared, to make a test per row. A file
    that cannot be read raises [Sys_error], and one that is not a golden raises
    [Failure]. *)

val text : string -> (unit -> string) -> Windtrap.test
(** [text file actual] is the test, named [file], that [actual ()] is the body
    of the golden [file]. Its failure prints their diff. *)

(** {1:tables Tables}

    The body of a table golden is lines of cells separated by tabs: the column
    names, then one line per row, and at least one row. A cell is the text
    tinygrad prints for its value, such as [dtypes.float], [True] or [inf].

    A row is given as its [cell] function: [cell column] is the row's cell in
    [column], and raises [Invalid_argument] if the table has no column [column].
*)

val cases :
  ?key:string list -> string -> ((string -> string) -> unit) -> Windtrap.test
(** [cases file check] is the group, named [file], of one test per row of the
    table golden [file], each running [check cell] on its row. A test is named
    by the row's cells in [key], as [a=dtypes.half b=dtypes.char]; [key]
    defaults to the first column. *)

val columns : string -> string list
(** [columns file] is the column names of the table golden [file], in order. *)

val rows : string -> (string -> string) list
(** [rows file] is the rows of the table golden [file], in order. *)
