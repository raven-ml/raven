(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs, and the one frame that installs their interpreters.

    A {e construct} is an operation rune adds to nx's: a loop, a compiled call,
    a remat, a custom rule, a root, a collective of a map, a call at a map's
    level, an addition to a total, a detach. Its performer asks the
    installations around it with {!perform}; an installation ({!install}) is one
    application of a transformation, an interpreter of nx's operations and of
    these constructs.

    Every interpreter matches {!type-t} exhaustively, with no wildcard, written
    [match[@warning "@4@8"] c with], so a construct added here is a compile
    error in each interpreter until it says what it does with it. *)

(** {1:names Names} *)

type axis = unit Type.Id.t
(** The type for names of maps. Each {!Type.Id.make} is a name distinct from
    every other; two are the same name iff their {!Type.Id.uid}s are equal. *)

type map
(** The type for the identities of map installations: each {!fresh_map} is
    distinct from every other, whatever the map's name. *)

val fresh_map : unit -> map
(** [fresh_map ()] is a map identity distinct from every other. *)

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
  | Loop : Trips.request -> Trips.result t
      (** A loop. Default: raises {!Trips.Not_staged}. *)
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
          (** Whether a backward pass replays [f]'s record at [args]. *)
    }
      -> 'q t
      (** [f args], whose intermediates reverse mode replays. Default: [f args].
      *)
  | Barrier : {
      values : Nx.packed list;
      after : Nx.packed list;
    }
      -> Nx.packed list t
      (** [values], read only once [after] exist. Default: [values]. *)
  | Custom : 'q rule -> 'q t
      (** A call of a function with a custom rule. Default: the rule's [value]
          when it has one, and the first component of [rule args] otherwise. *)
  | Root : {
      x : 'x Nx.Ptree.t;  (** The structure of the solution. *)
      residual : 'x -> 'x;
          (** [residual x] has [x]'s structure, and vanishes at the solution. *)
      solve : unit -> 'x;  (** [solve ()] is the solution. *)
      linear_solve : ('x -> 'x) -> 'x -> 'x;
          (** [linear_solve op b] is a [v] with [op v = b] for a linear [op]. *)
    }
      -> 'x t
      (** A value stated to be a zero of [residual]. Default: [solve ()]. *)
  | At_map : {
      map : map;
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      x : 'p;
    }
      -> 'q t
      (** [f], a linear function, applied at the level of [map]: each lane's row
          of [f] applied to [x]'s values batched over [map], every lane's value
          of each leaf stacked on a leading axis. A level of {!operator} has one
          lane. [f] is code of the level around [map]; it runs where [map]
          answers, outside its extent. A differentiation inside that level
          applies [f] to tangents and transposes it. Default: raises
          [Invalid_argument]. *)
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

type 'r answer
(** The type for an installation's answers to constructs of result ['r]. *)

val value : (unit -> 'r) -> 'r answer
(** [value f] answers with [f ()], run in the handler, outside the extent of the
    installation: a construct [f] performs reaches the installations around it,
    and an operation it issues the interpretation around it. A construct whose
    answer computes from its operands alone takes it. *)

val here : (unit -> 'r) -> 'r answer
(** [here f] answers with [f ()], run at the call: its operations meet the
    interpreters at the call and its effects the handlers there. A construct [f]
    performs that no installation [f] opens takes is offered to the
    installations around the one that answered, as if its handler had performed
    it: the installations between the call and that one met the construct it
    answered, and meet the body through their installations again ({!install}).
    A construct that carries a function takes it, so that the function runs
    where its arguments exist. *)

type interpreter = {
  op : Nx.Op.interpreter option;
      (** The interpreter of nx's operations, if any. *)
  call : 'r. 'r t -> 'r answer option;
      (** [call c] is [Some a] if the installation takes [c], answering [a], and
          [None] if it passes [c] outward. *)
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
    {!Nx.Op.intercept}, inside the construct handler. An answer, a value or an
    exception, resumes the performer; an exception [i.call c] or a {!value}
    answer raises is raised at the call. With [i.op = None] nothing is
    intercepted and no gate is raised.

    [i.call c = None] passes [c] outward. When [c] carries a function
    ({!carries}), the installation is installed again around that function, so
    that it meets the function's constructs and operations at its own level,
    inside every installation that answers [c] further out. A compiled call is
    the exception: its function runs inside the installations its compiler
    derives, so an installation that takes constructs by name or by kind derives
    every compiled call it meets, and one that passes it takes only the values
    it owns, which a trace refuses to capture.

    [i.call] runs in the handler, and so does a {!value} answer; a {!here}
    answer runs at the call. [o] runs inside the construct handler: a construct
    it performs reaches [i.call] first, and an operation it issues the
    interpretation around [install]. *)

val default : 'r t -> 'r
(** [default c] is [c]'s default: its answer when no installation takes it. *)

val perform : 'r t -> 'r
(** [perform c] is the answer of the nearest installation that takes [c], or
    [c]'s default, computed at the call, when none does. An exception an
    installation answers with is raised here. *)

val carries : 'r t -> bool
(** [carries c] is [true] iff [c] carries a function that runs where its
    arguments exist: a loop, a compiled call, a remat, a custom rule or a root.
    An installation answers such a construct {!here}; a map's call of a function
    at its own level ({!At_map}) runs where the map answers. *)

val loop : Trips.request -> Trips.result
(** [loop r] is [perform (Loop r)], or, when [r] is declined with
    {!Trips.Not_staged}, [Trips.fold r] at the call, inside every installation
    around it. *)

val operator :
  'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> (('p -> 'q) -> 'r) -> 'r
(** [operator p q f k] is [k op], where [op], applied during [k]'s extent, is
    [f] applied at a level of [k]'s call through {!At_map}: every transformation
    [k] opens sees [op] as one linear function. Applied after [k] returned, or
    inside a compiled call [k] makes, it raises [Invalid_argument]. *)

val substituting : owner -> Nx.Op.mapper -> (unit -> 'a) -> 'a
(** [substituting o s f] is [f ()] with each value [o] owns that an operation or
    a construct of its extent reads replaced by [s]'s for it, in the functions
    the constructs carry too. A construct passes outward, with its values
    replaced. *)
