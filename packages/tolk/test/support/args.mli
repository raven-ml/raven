(** Nodes of every kind of argument, for the laws of their text and their key.

    Each field of an argument is drawn from few values, among them values that a
    repr prints alike: NaNs of different bits, a stage that keeps [Argument]
    beside one that keeps [Whole], targets whose fields split one text two ways,
    and arguments that hold nodes differing only in such a NaN. *)

open Tolk

(** {1:generators Generators} *)

val node : Ops.t Windtrap.Gen.t
(** [node] draws a node of any kind of argument. *)

val field_names : string list
(** [field_names] are the names of every field {!neighbours} redraws, as
    ["call.write"]. *)

val neighbours : (string * Ops.t * Ops.t) Windtrap.Gen.t
(** [neighbours] draws [(field, u, v)]: two nodes of one kind whose arguments
    differ at most in [field], redrawn. *)

val redrawn : (string * Ops.t * Ops.t * bool) Windtrap.Gen.t
(** [redrawn] draws [(field, u, v, same)] as {!neighbours} draws
    [(field, u, v)], with [same] whether [field]'s two values are equal: by
    their bits for floats, physically for nodes, and structurally otherwise,
    without the arguments' own equality. *)

val pairs : (Ops.t * Ops.t) Windtrap.Gen.t
(** [pairs] draws two nodes of one kind, each argument drawn on its own. *)

(** {1:nodes Nodes} *)

val x : Ops.t
(** [x] is the constant [1] cast to {!Dtype.Float32}. *)

val buffer : Ops.t
(** [buffer] is a buffer of 16 {!Dtype.Float32} elements on ["CPU"], in slot 0.
*)

val quiet : float
(** [quiet] is the quiet NaN of bits [0x7ff8000000000000]. *)

val signalling : float
(** [signalling] is the signalling NaN of bits [0x7ff8000000000001]. *)

val bounded : float -> Ops.t
(** [bounded v] is a {!Dtype.Float32} scalar parameter in slot 1 whose bounds
    are [0.] and [v] and whose bound value is [v]: a repr prints it as [nan] for
    {!quiet} and {!signalling} alike. *)
