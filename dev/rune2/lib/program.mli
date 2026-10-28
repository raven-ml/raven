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
    did not own: an array, or a donated handle read as its arrays.

    {!eval} issues each equation again, an operation through [Nx.Prim.eval] and
    a construct through {!perform}, under whatever interpretations are around
    the call. A program's last outputs are its {e additions}: values it adds to
    totals, which its construct sums and adds once it has run ({!run}), so that
    evaluating a program adds nothing.

    Constructs are performed as one effect. A handler of a transformation or a
    tracer answers the constructs over its own values; one no handler answers
    runs as its {e default}: a loop evaluates its programs, a call and a remat
    their body, [Detach] is its operand, an addition does nothing, and [Lanes]
    and [Lane_index] raise. The default runs where the construct was performed,
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
  | Lane_index : axis * 'd Nx.Placement.t -> construct
      (** The calling lane's index in the map named by the axis: one 0-d [int32]
          result at the placement, which must cut no axis. *)
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
      seeds : value array;
          (** Where each sum starts, or none: each starts at the identity. *)
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
      seeds : value array;  (** As a scan's. *)
    }
      -> construct
  | Call : { body : t; args : value array; site : site } -> construct
      (** [body] applied to [args], [site] its compiled form and the programs
          derived from it. Its default runs the compiled form where no operand
          is traced and no tracer stages the operations around the call
          ({!staging}), and evaluates [body] otherwise. *)
  | Custom : {
      name : string;  (** The entry point, in messages. *)
      primal : t;
          (** Over the arguments and the captures; to the [ny] results, the
              residuals and the additions. *)
      derivative : derivative;
      ny : int;
      args : value array;
      captures : value array;
      residuals : bool;
          (** Whether the residuals among the results are used: then they carry
              the derivatives of every differentiation around the call. A
              differentiation applying the rule asks for them, unless it is
              asked for its own, which it derives again. *)
      applied : bool;
          (** Whether a reverse mode inside the call applies its {!Pullback},
              which a forward mode around then has no rule to answer. *)
    }
      -> construct
      (** A call of a function with a custom derivative: the results, the
          residuals (unused where [residuals] is [false]), [primal]'s additions,
          then a tangent map's. The default evaluates [primal] without what only
          the residuals read, and adds nothing for a tangent map. *)
  | Linear : {
      name : string;
      apply : t option;
          (** Over the arguments and the captures; to the results and the
              additions. [None] where only the transpose is known, as for a
              {!Pullback} rule's tangents. *)
      transpose : t;
          (** Over a cotangent per result and the residuals; to a cotangent per
              argument and the additions. *)
      args : value array;
      captures : value array;
      residuals : value array;
      forms : form array;  (** The results'. *)
    }
      -> construct
      (** A function linear in its arguments with the transpose [transpose]:
          [apply]'s results, then its additions. Without [apply], it is the
          tangents of a {!Pullback} rule's results, which only a tape takes: its
          default raises [Invalid_argument]. *)
  | Remat : { body : t; args : value array } -> construct
      (** [body] applied to [args], the function's arguments then its captures,
          whose reverse mode keeps [args] alone and computes [body] again. *)

and derivative =
  | Tangent of t
      (** A tangent map: over a tangent per argument and the residuals; to the
          results' tangents and the additions. *)
  | Pullback of t
      (** A pullback: over a cotangent per result and the residuals; to a
          cotangent per argument and the additions. *)

and site
(** The type for a call's {e site}: its program compiled, made at its first
    run, and the programs a transformation derives from it ({!derive}). *)

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

val sums : t list -> addition array * form array * int array list
(** [sums ps] is the sums of a loop of the programs [ps], as {!feeds} gives
    them, with the sum each addition of each program feeds, in order. *)

val feeds : t -> addition array * form array * int array
(** [feeds step] is the sums of a loop of the step [step]: one per total and
    shape its additions add to, in the order of their first additions, their
    forms, and the sum each of [step]'s additions feeds, in order. *)

val eval : t -> value array -> value array
(** [eval p args] is [p]'s outputs with its parameters bound to [args], in
    order. Raises what an equation raises. *)

val prune : t -> outputs:int array -> t
(** [prune p ~outputs] is [p] with its outputs [outputs], in order, and no
    addition, keeping the equations they read and every check. *)

(** {1:sites Sites}

    A transformation of a call derives a program from the call's ({!derive}):
    its forward-mode derivative, its map, its parts known and unknown to a
    tape, or its transpose. A derivation is a value, and a site keeps one
    program per derivation, so that two calls share a derived program only
    when their derivations are equal, and a derived program compiles once. *)

val site :
  name:string ->
  compile:(t -> form array -> value array -> value array) ->
  t ->
  form array ->
  site
(** [site ~name ~compile p forms] is the site of [p] over arguments of [forms],
    for the entry point [name]: its compiled form is [compile p forms], applied
    to arguments of [forms], and each program derived from [p] is compiled by
    [compile] too. *)

val axes : site -> axis list
(** [axes s] is the maps that [s]'s program or a program it carries names
    through [Lanes] or [Lane_index], in the order first named. *)

val iterates : site -> bool
(** [iterates s] is [true] iff [s]'s program or a program it carries holds an
    [Iterate]. *)

val customs : site -> bool
(** [customs s] is [true] iff [s]'s program or a program it carries holds a
    custom rule or a linear call, the constructs whose derivatives depend on
    whether a tape holds the tangents. *)

type derivation =
  | Jvp of { tangents : bool array; reverse : bool }
      (** Forward mode, [tangents.(k)] whether argument [k] has a tangent;
          [reverse] whether the tangents are a tape's, which only a program
          holding a custom rule tells apart ({!customs}). Over the primals, then
          the tangents there are; to the results, the tangents there are, then
          the additions. *)
  | Vmap of { lanes : bool array; n : int; named : bool }
      (** A map of [n] lanes, [lanes.(k)] whether argument [k] has them;
          [named] whether the program names the map. Over the batches and the
          shared arguments; to the results, a batch where it has lanes, then
          the additions, every lane's. *)
  | Known of { unknown : bool array }
      (** The part of the program that reads no argument where [unknown] holds,
          computed at the call: over the other arguments; to its results, the
          residuals, then the additions. Its linear part: over the unknown
          arguments then the residuals, to the unknown results. *)
  | Transpose of { linear : bool array; cts : bool array }
      (** The transpose of a linear program over the arguments where [linear]
          holds, given a cotangent of each result where [cts] holds: over the
          other arguments and the cotangents; to the cotangents of the linear
          arguments that receive one, then the additions. *)

type derived = {
  program : t;
  at : site;  (** [program]'s site. *)
  flags : bool array;
      (** By derivation: which results have a tangent, which have lanes, which
          results are unknown, or which linear arguments receive a
          cotangent. *)
  linear : (t * site) option;  (** A [Known] derivation's linear part. *)
}
(** The type for derived programs. *)

val derive :
  ?map:axis * int ->
  site ->
  derivation ->
  form array ->
  (unit -> (t * form array) * bool array * (t * form array) option) ->
  derived
(** [derive ~map s d forms make] is the program of [s] derived by [d] from
    arguments of [forms], made by [make ()] the first time [s] meets [d] with
    arguments of [forms]' dtypes, shapes and placements and with the maps that
    [s]'s program names of the same lanes. [make ()] gives the program and its
    arguments' forms, its flags, and a linear part with its arguments' forms.
    [map] is a map that the derivation runs in, [(axis, n)], whose lanes no
    handler around answers. Calls on several domains share the derived
    programs, as {!Jit} shares its programs. *)

val derivations : unit -> int
(** [derivations ()] is the number of programs [derive] made on the calling
    domain. *)

(** {1:constructs Constructs} *)

val perform : construct -> value array
(** [perform c] is [c]'s results, as the innermost handler answering [c] gives
    them, or [c]'s default. A loop's and a call's results end with the sums of
    their programs' additions. *)

val run : construct -> value array
(** [run c] is [perform c] after which each addition of [c]'s results is added
    to its total, as {!perform} of [Add]. A loop first asks each total it adds
    to for its running value ({!Running}): where one answers, the loop's sum
    starts from it and its end replaces it ({!Replace}), so that the total is
    the one its additions give made in turn. *)

type _ Effect.t +=
  | Running :
      ('v, 's, 'd) total * int array
      -> ('v, 's, 'd) Nx.t option Effect.t
        (** Answered by the innermost collect of the total around the call with
            its running value, if it has the shape given. A tracer or a map
            between answers [None], so that a loop whose additions reach a
            collect through it adds its sum. *)
  | Replace : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> unit Effect.t
        (** Answered by the collect that answered {!Running}, whose running
            value becomes the value given. *)

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

val carried : construct -> t list
(** [carried c] is the programs [c] carries. *)

type map = { f : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }
(** A function on values of every dtype and set. *)

val map_operands : map -> construct -> construct
(** [map_operands m c] is [c] with each operand [x] replaced by [m.f x]. *)

val map_programs : (t -> t) -> construct -> construct
(** [map_programs f c] is [c] with each program [p] it carries replaced by
    [f p], and a call with a site of no compiled form and no derived
    program. *)

val stacked : int -> form -> form
(** [stacked n f] is the form of a value of [f]'s with a leading axis of [n], in
    C order. *)

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

type _ Effect.t +=
  | Staging : bool Effect.t
        (** Answered [true] by each tracer that stages every operation of its
            extent, as the jit tracer does. *)

val staging : unit -> bool
(** [staging ()] is [true] iff a tracer around the call stages the operations
    applied there, so that a call there is inlined, not run compiled. *)

(** {1:printing Printing} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a program: first each constant it or a program it carries
    reads, as [cK : dtype shape at placement], numbered in the order they are
    first read; then [program (%0, %1, …)], its parameters,
    each equation, and [return] with its outputs, then [adds] with its
    additions. Each program's variables are numbered from where the previous
    one's end, so no two programs share a name. An operation is its results,
    each [%n : dtype shape at placement], [=], the operation as {!Nx.Prim.pp}
    formats it, and on the next line [with x0 = …] for each operand; a construct
    is its results, its operands, and each program it carries under a label,
    indented; a map is numbered in the order the program first names it. *)
