(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs, and the one frame that installs their interpreters.

    A {e construct} is an operation rune adds to nx's: a scan, a remat, a custom
    rule, a collective of a map, an addition to a total, a detach. Its performer
    asks the installations around it with {!perform}; an installation
    ({!install}) is one application of a transformation, an interpreter of nx's
    operations and of these constructs.

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

(** The type for constructs whose answer is ['r]. Each has a {e default}, its
    answer when no installation takes it. *)
type _ t =
  | Scan : Scan.request -> Scan.result t
      (** A scan. Default: raises {!Scan.Not_staged}. *)
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

val perform : 'r t -> 'r
(** [perform c] is the answer of the nearest installation that takes [c], or
    [c]'s default, computed at the call, when none does. An exception an
    installation answers with is raised here. *)
