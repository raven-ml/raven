(** Quill top-level interface *)

type execution_result = {
  output : string;
  error : string option;
  status : [ `Error | `Success ];
}
(** The result of executing a phrase *)

val eval : id:string -> string -> execution_result
(** Evaluate a code phrase in the top-level identified by [id] *)

val initialize_toplevel : ?libraries:string list -> string -> unit
(** Initialize a new top-level identified by [id] *)
