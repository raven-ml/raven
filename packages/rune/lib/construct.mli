(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs, and the one frame that installs their interpreters.

    A {e construct} is an operation rune adds to nx's: a scan, a compiled call,
    a remat, a custom rule, a collective of a map, an addition to a total, a
    detach. Its performer asks the installations around it with {!perform}; an
    installation ({!install}) is one application of a transformation, an
    interpreter of nx's operations and of these constructs.

    Every interpreter matches {!type-t} exhaustively, with no wildcard, written
    [match[@warning "@4@8"] c with], so a construct added here is a compile
    error in each interpreter until it says what it does with it. *)

(** {1:names Names} *)

type axis = unit Type.Id.t
(** The type for names of maps. Each {!Type.Id.make} is a name distinct from
    every other; two are the same name iff their {!Type.Id.uid}s are equal. *)

type ('a, 'b) total = ('a, 'b) Nx.t Type.Id.t
(** The type for totals of [('a, 'b) Nx.t] values. Each {!Type.Id.make} is a
    total distinct from every other, and {!Type.Id.provably_equal} tells two
    apart. *)

(** {1:constructs Constructs} *)

(** The type for custom rules of results of type ['q]: the structures [p] of the
    arguments and [q] of the result, the rule, and the arguments. *)
type 'q rule =
  | Jvp_rule : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      rule : 'p -> 'q * ('p -> 'q);
          (** [rule args] is the result and its tangent map, linear in the
              arguments' tangents. *)
      args : 'p;
      value : 'q option;
          (** The result, when an inner installation computed it. *)
    }
      -> 'q rule
  | Vjp_rule : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      rule : 'p -> 'q * ('q -> 'p);
          (** [rule args] is the result and its pullback. *)
      args : 'p;
    }
      -> 'q rule

(** The type for the transformations a compiled function's programs derive from:
    a step turns a function from ['p] to ['q] into one from ['p2] to ['q2]. *)
type (_, _, _, _) step =
  | Forward : bool list -> ('p, 'q, 'p, 'q * Nx.packed list) step
      (** [Forward tracked] is the forward half of the function's reverse mode,
          the leaves [tracked] marks being differentiated: from the arguments to
          the results and the residuals the forward computes ({!split}). *)
  | Backward :
      bool list
      -> ('p, 'q, Nx.packed list * Nx.packed list, Nx.packed list) step
      (** [Backward tracked] is the backward half: from the residuals and the
          cotangents of the results that are real or complex to the cotangents
          of the leaves [tracked] marks. *)
  | Jvp : bool list -> ('p, 'q, 'p * 'p, 'q * 'q) step
      (** [Jvp tracked] is the function's forward derivative, from the arguments
          and their tangents to the results and theirs, the leaves [tracked]
          marks being differentiated. *)
  | Vmap : {
      lanes : bool list;
      size : int;
      axis : axis option;
    }
      -> ('p, 'q, 'p, 'q) step
      (** The function mapped over [size] lanes, the leaves [lanes] marks
          carrying them on a leading axis, under the map named [axis]. The
          results all carry the lanes. *)
  | Totals :
      ('a, 'b) total
      -> ('p, 'q, 'p * ('a, 'b) Nx.t, 'q * ('a, 'b) Nx.t) step
      (** [Totals t] is the function that also returns the sum of its additions
          to [t], from the arguments and a zero. *)
  | Discarding : ('p, 'q, 'p, 'q) step
      (** The function with its additions to totals dropped. *)

val same_step :
  ('p, 'q, 'a, 'b) step ->
  ('p, 'q, 'c, 'd) step ->
  ('a * 'b, 'c * 'd) Type.eq option
(** [same_step s s'] is [Some Equal] iff [s] and [s'] derive the same function.
*)

type ('p, 'q) split = {
  forward : 'p -> 'q * Nx.packed list;
      (** [forward args] is the function's results at [args] and the residuals
          its forward pass computes. *)
  backward : Nx.packed list * Nx.packed list -> Nx.packed list;
      (** [backward (residuals, cts)] is the cotangents of the tracked
          arguments, from the residuals and the cotangents of the results that
          are real or complex. It does not run the function. *)
  residuals : 'p -> Nx.packed list -> Nx.packed list;
      (** [residuals args computed] is the residuals [backward] takes, from the
          arguments and the residuals [forward] computed: an argument the
          backward pass reads is a residual as it is. *)
}
(** The type for a function's reverse mode split in two plain functions. *)

type ('p, 'q) vjp = 'p -> 'q * (Nx.packed list -> Nx.packed list)
(** The type for a function's reverse mode: [vjp args] is the results at [args]
    and the transpose of the derivative, from the cotangents of the results that
    are real or complex to those of the tracked arguments. *)

type ('p, 'q) compiler = {
  run : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q;
      (** [run p q f args] is [f args] computed by a program compiled for the
          devices of [args], which [f] traces the first time its key is met. *)
  derive : 'p2 'q2. ('p, 'q, 'p2, 'q2) step -> ('p2, 'q2) compiler;
      (** [derive s] is the compiler of the functions [s] derives, which keeps
          their programs. *)
  split :
    bool list ->
    'p Nx.Ptree.t ->
    'q Nx.Ptree.t ->
    ('p, 'q) vjp ->
    'p ->
    ('p, 'q) split;
      (** [split tracked p q vjp args] is the function's reverse mode, the
          leaves [tracked] marks being differentiated, split in two at residuals
          that tracing [vjp] fixes once per [tracked] and per dtype, shape and
          placement of [args]' leaves. *)
}
(** The type for compilers of a compiled function and of the functions
    transformations derive from it. *)

val packed : Nx.packed list Nx.Ptree.t
(** [packed] is the structure of a list of tensors of any dtypes. *)

(** The type for constructs whose answer is ['r]. Each has a {e default}, its
    answer when no installation takes it. *)
type _ t =
  | Scan : Scan.request -> Scan.result t
      (** A scan. Default: raises {!Scan.Not_staged}. *)
  | Compiled : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      args : 'p;
      compiler : ('p, 'q) compiler;
    }
      -> 'q t
      (** [f args], compiled. A transformation passes on the call of the
          function it derives from [f], with the compiler [compiler.derive]
          gives that derivation. Default: [compiler.run p q f args]. *)
  | Remat : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      args : 'p;
      recomputed : bool;
          (** Whether a backward pass runs [f] again at [args]. *)
    }
      -> 'q t
      (** [f args], whose intermediates reverse mode recomputes. Default:
          [f args]. *)
  | Barrier : {
      values : Nx.packed list;
      after : Nx.packed list;
    }
      -> Nx.packed list t
      (** [values], read only once [after] exist. Default: [values]. *)
  | Custom : 'q rule -> 'q t
      (** A call of a function with a custom rule. Default: the rule's [value]
          when it has one, and the first component of [rule args] otherwise. *)
  | Lanes : axis * ('a, 'b) Nx.t -> ('a, 'b) Nx.t t
      (** Every lane's value of the map named [axis], stacked on a new leading
          axis. Default: [Nx.unsqueeze ~axes:[0] x]. *)
  | Lane_index : axis option -> (int32, Nx.int32_elt) Nx.t t
      (** The calling lane's index in the map named [axis], or in the innermost
          anonymous map. Default: the [int32] scalar [0]. *)
  | Lane_count : axis -> int t
      (** The number of lanes of the map named [axis]. Default: [1]. *)
  | Add : ('a, 'b) total * ('a, 'b) Nx.t -> unit t
      (** An addition to a total. Default: [()], the addition dropped. *)
  | Detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t t
      (** The value with a zero derivative under every differentiation. Default:
          the value. *)

(** {1:interpreting Interpreting} *)

type interpreter = {
  op : Nx.Op.interpreter option;
      (** The interpreter of nx's operations, if any. *)
  call : 'r. 'r t -> (unit -> 'r) option;
      (** [call c] is [Some answer] if the installation takes [c], whose answer
          is [answer ()], and [None] if it passes [c] outward. *)
}
(** The type for interpreters of constructs and operations. *)

type owner = { owns : 'a 'b. ('a, 'b) Nx.t -> bool }
(** The type for an installation's test of the traced values it owns. *)

val claims : owner -> 'r Nx.Op.t -> bool
(** [claims o op] is [true] iff an operand of [op] is one [o] owns: the
    operations an installation that owns traced values interprets. *)

val install : interpreter -> (unit -> 'a) -> 'a
(** [install i f] is [f ()] with every construct of its extent delivered to
    [i.call] and, when [i.op] is [Some o], every operation to [o] through
    {!Nx.Op.intercept}, inside the construct handler. [None] passes a construct
    outward. An answer, a value or an exception, resumes the performer; an
    exception [i.call c] raises is its answer. With [i.op = None] nothing is
    intercepted and no gate is raised.

    [i.call] and its answers run in the handler, outside [f]'s extent: a
    construct they perform reaches the installations around [install], and an
    operation they issue the interpretation around it. [o] runs inside the
    construct handler: a construct it performs reaches [i.call] first, and an
    operation it issues the interpretation around [install]. *)

val default : 'r t -> 'r
(** [default c] is [c]'s default: its answer when no installation takes it. *)

val perform : 'r t -> 'r
(** [perform c] is the answer of the nearest installation that takes [c], or
    [c]'s default, computed at the call, when none does. An exception an
    installation answers with is raised here. *)

val scan : Scan.request -> Scan.result
(** [scan r] is [perform (Scan r)], or, when [r] is declined with
    {!Scan.Not_staged}, [Scan.fold r] at the call, inside every installation
    around it. *)
