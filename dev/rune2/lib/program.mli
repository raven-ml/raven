(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs: functions traced once into equations, evaluated again.

    A {e program} has parameters, equations and outputs. Its {e variables} are
    the traced values of the tracer that made it ({!Trace}), each numbered by
    its payload: the parameters first, then each equation's results. An equation
    is an [Nx.Prim] operation over variables and constants, or a {e construct}
    over them: a call that a transformation around it may answer, such as a loop
    carrying its step's program. A constant is a value the tracer received and
    did not own: an array, a formula, a donated handle read as its arrays.

    {!eval} issues each equation again, an operation through [Nx.Prim.eval] and
    a construct through {!perform}, under whatever interpretations are around
    the call. A program's last outputs are its {e additions}: values it adds to
    totals, which its construct sums and adds once it has run ({!run}), so that
    evaluating a program adds nothing.

    Constructs are performed as one effect. A handler of a transformation or a
    tracer answers the constructs over its own values; one no handler answers
    runs as its {e default}: a loop evaluates its programs, a call its body,
    [Detach] is its operand, an addition does nothing, and [Lanes] and
    [Lane_index] raise. The default runs where the construct was performed,
    inside every interpretation around that point. *)

(** {1:values Values} *)

type value =
  | V : ('v, 's, 'd) Nx.t -> value  (** A value of any dtype and set. *)

type form =
  | F : ('v, 's, 'd) Nx.Prim.form -> form
      (** A value's form, of any dtype and set. *)

val form : value -> form
(** [form v] is [v]'s form. *)

val cast : ('v, 's, 'd) Nx.Prim.form -> value -> ('v, 's, 'd) Nx.t
(** [cast f v] is [v] at [f]'s type, as {!Nx.Prim.coerce} checks it.

    Raises [Invalid_argument] if [v]'s dtype or set is not [f]'s. *)

val like : ('v, 's, 'd) Nx.t -> value -> ('v, 's, 'd) Nx.t
(** [like x v] is [cast (Nx.Prim.form x) v]. *)

val same : value -> value -> bool
(** [same u v] is [true] iff [u] and [v] are one value. *)

(** {1:names Names} *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name. *)

val same_axis : axis -> axis -> bool
(** [same_axis a b] is [true] iff [a] and [b] are one name. *)

type ('v, 's, 'd) total
(** The type for totals of [('v, 's, 'd) Nx.t] values. *)

val total : unit -> ('v, 's, 'd) total
(** [total ()] is a fresh total. *)

val same_total :
  ('v, 's, 'd) total ->
  ('w, 'r, 'e) total ->
  (('v, 's, 'd) Nx.t, ('w, 'r, 'e) Nx.t) Type.eq option
(** [same_total t u] is [Some Equal] iff [t] and [u] are one total. *)

(** {1:programs Programs} *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Var : int -> ('v, 's, 'd) Nx.Prim.payload  (** A variable's number. *)

type addition =
  | Addition : ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form -> addition
      (** The total an addition output adds to, and the output's form. *)

type t = private {
  owner : Nx.Prim.interpretation;  (** The tracer whose variables it has. *)
  params : int array;  (** The variables its arguments bind, in order. *)
  eqns : eqn array;  (** In the order they are issued. *)
  outputs : value array;  (** Variables or constants, additions last. *)
  adds : addition array;  (** The totals of the last outputs, in order. *)
  slots : int;  (** The number of variables. *)
}

and eqn =
  | Op : { by : string; op : 'r Nx.Prim.t; base : int } -> eqn
      (** [op], applied by [by], its result at position [k] the variable
          [base + k]. *)
  | Do : { c : construct; base : int } -> eqn
      (** [c], its result [k] the variable [base + k]. *)

and construct =
  | Detach : ('v, 's, 'd) Nx.t -> construct
      (** The operand, with a zero derivative; one result. *)
  | Lanes : axis * ('v, 's, 'd) Nx.t -> construct
      (** Every lane's operand stacked along a new axis 0, in the map named by
          the axis; one result. *)
  | Lane_index : axis -> construct
      (** The calling lane's index in the map named by the axis: one 0-d [int32]
          result. *)
  | Add : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> construct
      (** The operand added to the total; no result. *)
  | Scan : {
      step : t;
          (** Over the carry, a row and the captures; to the carry, the outputs
              and the additions. *)
      length : int;  (** The number of rows. *)
      init : value array;
      xs : value array;  (** Rows along axis 0. *)
      captures : value array;
    }
      -> construct
  | Iterate : {
      name : string;  (** The entry point, in messages. *)
      step : t;  (** Over the carry and its captures, to the carry. *)
      until : t;  (** Over the carry and its captures, to one boolean. *)
      max : int;
      init : value array;
      step_captures : value array;
      until_captures : value array;
    }
      -> construct
  | Call : { body : t; args : value array } -> construct
      (** [body] applied to [args]. *)

val make :
  owner:Nx.Prim.interpretation ->
  params:int array ->
  eqn list ->
  outputs:value array ->
  adds:addition array ->
  slots:int ->
  t
(** [make ~owner ~params eqns ~outputs ~adds ~slots] is the program of the
    equations [eqns], in order. *)

val eval : t -> value array -> value array
(** [eval p args] is [p]'s outputs with its parameters bound to [args], in
    order. Raises what an equation raises. *)

(** {1:constructs Constructs} *)

val perform : construct -> value array
(** [perform c] is [c]'s results, as the innermost handler answering [c] gives
    them, or [c]'s default. A loop's and a call's results end with the sums of
    their programs' additions. *)

val run : construct -> value array
(** [run c] is [perform c] after which each addition of [c]'s results is added
    to its total, as {!perform} of [Add]. *)

type _ Effect.t +=
  | Construct : construct -> value array Effect.t
        (** The effect {!perform} performs: a handler that answers it continues
            with the construct's results, of the forms {!forms} gives. *)

exception Evaluate
(** A handler that answers {!Construct} by discontinuing it with [Evaluate] has
    the construct run its default where it was performed, inside every
    interpretation around that point, and no handler outside it sees the
    construct. *)

val operands : construct -> value list
(** [operands c] is the values [c] reads, captures included. *)

type map = { f : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }
(** A function on values of every dtype and set. *)

val map_operands : map -> construct -> construct
(** [map_operands m c] is [c] with each operand [x] replaced by [m.f x]. *)

val map_programs : (t -> t) -> construct -> construct
(** [map_programs f c] is [c] with each program [p] it carries replaced by
    [f p]. *)

val forms : construct -> form array
(** [forms c] is the forms of [c]'s results, as its default gives them. Raises
    [Invalid_argument] as {!perform} would for [Lanes] outside its map. *)

type _ Effect.t +=
  | Bound : axis -> int option Effect.t
        (** Answered by each map with its number of lanes, when the axis names
            it. *)

val bound : axis -> int option
(** [bound a] is [Some n] inside the map named [a] of [n] lanes, and [None] with
    none around. *)
