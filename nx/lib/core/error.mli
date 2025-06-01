(** Structured error reporting for tensor operations. *)

exception Shape_mismatch of string

(** {2 Error Construction} *)

val cannot :
  op:string ->
  what:string ->
  from:string ->
  to_:string ->
  ?reason:string ->
  ?hint:string ->
  unit ->
  'a
(** For transformation errors: "op: cannot <what> <from> to <to_>" *)

val invalid :
  op:string -> what:string -> ?reason:string -> ?hint:string -> unit -> 'a
(** For invalid inputs: "op: invalid <what>" *)

val failed :
  op:string -> what:string -> ?reason:string -> ?hint:string -> unit -> 'a
(** For operation failures: "op: <what>" *)

(** {2 Specialized Error Functions} *)

val shape_mismatch :
  op:string ->
  expected:int array ->
  actual:int array ->
  ?hint:string ->
  unit ->
  'a
(** Common pattern for shape mismatches. Automatically formats shapes and
    calculates element counts when relevant. *)

val broadcast_incompatible :
  op:string ->
  shape1:int array ->
  shape2:int array ->
  ?hint:string ->
  unit ->
  'a
(** For broadcasting errors. Shows which specific dimensions are incompatible
    with detailed comparison (e.g., "dim 0: 2≠4, dim 1: 3≠5"). *)

val dtype_mismatch :
  op:string -> expected:string -> actual:string -> ?hint:string -> unit -> 'a
(** For type mismatches. Formats as "cannot <op> <expected> with <actual>". *)

val axis_out_of_bounds :
  op:string -> axis:int -> ndim:int -> ?hint:string -> unit -> 'a
(** For single axis errors. Shows "axis <n> out of bounds for <m>D array". *)

val invalid_shape :
  op:string -> shape:int array -> reason:string -> ?hint:string -> unit -> 'a
(** For shape validation errors like negative dimensions or incompatible sizes.
*)

val empty_input : op:string -> what:string -> 'a
(** For operations on empty inputs. *)

(** {2 Common Checks} *)

val check_bounds :
  op:string -> name:string -> value:int -> ?min:int -> ?max:int -> unit -> unit
(** Checks if min <= value <= max (inclusive) *)

val check_axes : op:string -> axes:int array -> ndim:int -> unit
(** Checks all axes are valid for given ndim *)

(** {2 Multi-issue Errors} *)

val multi_issue : op:string -> (string * string) list -> 'a
(** For errors with multiple problems *)
