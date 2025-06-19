(** Evaluate markdown files with code blocks *)

val eval_file : string -> (string, string) result
(** Evaluate a markdown file and return the evaluated markdown *)

val eval_stdin : unit -> (string, string) result
(** Evaluate markdown from stdin and return the evaluated markdown *)
