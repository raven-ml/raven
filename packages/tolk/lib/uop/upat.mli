(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Pattern DSL for matching and rewriting {!Uop} nodes.

    A [Upat.t] describes a structural shape over the uop DAG. Patterns are
    built with smart constructors and infix operators, paired with a callback
    through {!(=>)}, and collected into a {!Pattern_matcher.t} that applies
    the first matching rule whose callback returns [Some _].

    Patterns constrain any subset of a node's [op], [dtype], [src], and [arg].
    They can also constrain {!Uop.node_tag} through {!tag}. A pattern with no
    constraints is a wildcard. Named patterns capture the matched node into a
    {!bindings} table; the same name used twice in one pattern requires the two
    captures to be physically equal.

    The module is designed to be locally opened — `Upat.(...)` or
    `let open Upat in ...` — which brings pattern constructors, operators,
    and [=>] all into scope at once.

    {1:example Example}

    {[
      let open Upat in
      Pattern_matcher.make [
        (* x + 0 -> x *)
        var "x" + zero => (fun b -> Some (b $ "x"));

        (* x * 1 -> x *)
        rewrite1 (fun x -> x * one) (fun x -> Some x);
      ]
    ]} *)

(** {1:types Types} *)

type t
(** The type for patterns. *)

type dtype_pat
(** The type for dtype constraints. Constructors that accept [?dtype:Dtype.t]
    follow tinygrad matching: exact dtype or matching scalar identity. Use
    {!exact_dtype} when vector width and pointer metadata must also match. *)

type src_pat
(** The type for source-child constraints. *)

type arg_pat
(** The type for constraints on a uop's {!Uop.arg} field. Build with
    {!arg_any}, {!arg_eq}, {!has_int}, {!has_const}, {!has_op},
    {!has_op_in}, {!has_reduce_op}, or {!has_reduce_op_in}. *)

type pos = string * int * int * int
(** The type of OCaml source positions such as [__POS__]. *)

type bindings
(** The table of name-to-node captures produced by a successful match.
    A name appearing multiple times in a pattern binds to the same node
    in all positions. *)

type 'ctx rule_with_ctx
(** The type for a pattern paired with a context-sensitive rewrite callback.
    The context is supplied by {!Pattern_matcher.rewrite_with_ctx}. *)

(** {1:ctors Pattern constructors} *)

val any : t
(** [any] is the wildcard: matches any node and captures nothing. *)

val var : string -> t
(** [var name] is a wildcard that captures the matched node under [name].
    If [name] already appears earlier in the pattern, the two matched nodes
    must be equal (via {!Uop.equal}) for the match to succeed.

    Raises [Invalid_argument] if [name] is ["ctx"], which is reserved for
    rewrite callback context. *)

val exact_dtype : Dtype.t -> dtype_pat
(** [exact_dtype dt] matches exactly [dt]. *)

val scalar_dtype : Dtype.t -> dtype_pat
(** [scalar_dtype s] matches exactly [s]. Dtypes are scalar, so this is
    equivalent to {!exact_dtype}. *)

val any_dtype : dtype_pat list -> dtype_pat
(** [any_dtype ps] matches when any dtype pattern in [ps] matches. *)

val fixed : t list -> src_pat
(** [fixed ps] matches exactly the source list [ps]. *)

val prefix : t list -> src_pat
(** [prefix ps] matches [ps] at the beginning of the source list and accepts
    any trailing sources. *)

val perms : t list -> src_pat
(** [perms ps] matches the source list as any permutation of [ps].
    Duplicate binding sets are collapsed. *)

val repeat : t -> src_pat
(** [repeat p] matches every source child with [p], accepting any arity. *)

val is_any : t list -> t
(** [is_any ps] matches when any pattern in [ps] matches. *)

val var_dtype : string -> dtype_pat -> t
(** [var_dtype name dt] is {!var} restricted by the explicit dtype pattern
    [dt]. *)

val var_scalar : string -> Dtype.t -> t
(** [var_scalar name s] is [var_dtype name (scalar_dtype s)]. *)

val op :
  ?loc:pos ->
  ?dtype:Dtype.t ->
  ?src:t list ->
  ?arg:arg_pat ->
  ?name:string ->
  ?allow_any_len:bool ->
  Ops.t -> t
(** [op ?dtype ?src ?arg ?name ?allow_any_len o] matches a node whose op is
    [o].

    [src] is a positional match: each pattern in the list is applied to the
    child at the same index. By default the pattern requires the exact
    same number of children as the list length; with [~allow_any_len:true]
    extra trailing children are accepted (but not fewer). Omitting [src]
    imposes no constraint on children.

    [arg] defaults to {!arg_any}. Use {!arg_eq}, {!has_int}, {!has_const},
    {!has_op}, {!has_op_in}, {!has_reduce_op}, or {!has_reduce_op_in} to
    constrain the payload.

    [dtype] pins the node's dtype when given.

    [name] captures the matching node under that key, with the uniqueness
    rule of {!var}.

    This constructor does not canonicalise commutative operands; use {!alu}
    for that. *)

val ops :
  ?loc:pos ->
  ?dtype:Dtype.t ->
  ?src:t list ->
  ?arg:arg_pat ->
  ?name:string ->
  ?allow_any_len:bool ->
  Ops.t list -> t
(** [ops os] is like {!op} but matches any node whose op is in [os]. *)

val op_src :
  ?loc:pos ->
  ?dtype:dtype_pat ->
  ?src:src_pat ->
  ?arg:arg_pat ->
  ?name:string ->
  Ops.t -> t
(** [op_src] is the explicit-source-pattern variant of {!op}. Use
    {!fixed}, {!prefix}, {!perms}, or {!repeat} to describe child matching,
    and {!exact_dtype}, {!scalar_dtype}, or {!any_dtype} for dtype matching. *)

val ops_src :
  ?loc:pos ->
  ?dtype:dtype_pat ->
  ?src:src_pat ->
  ?arg:arg_pat ->
  ?name:string ->
  Ops.t list -> t
(** [ops_src] is like {!op_src} but matches any node whose op is in the
    supplied list. *)

val const :
  ?loc:pos -> ?dtype:Dtype.t -> ?name:string ->
  Const.t -> t
(** [const c] matches an {!Ops.Const} node whose value equals [c]. *)

val const_int : int -> t
(** [const_int n] matches a {!Ops.Const} of value [n] with dtype
    {!Dtype.index} (the dtype {!Uop.const_int} builds). *)

val const_float : float -> t
(** [const_float x] matches a {!Ops.Const} of value [x] with dtype
    {!Dtype.weakfloat} (the dtype {!Uop.const_float} builds). *)

val const_bool : bool -> t
(** [const_bool b] matches a boolean {!Ops.Const} of value [b]. *)

val cvar :
  ?loc:pos -> ?name:string -> ?dtype:Dtype.t -> ?arg:Const.t -> unit -> t
(** [cvar ?arg ()] matches an {!Ops.Const}, optionally restricted to [arg]. *)

val tag : string -> t -> t
(** [tag s p] is [p] restricted to nodes whose {!Uop.node_tag} is [Some s]. *)

val tags : string list -> t -> t
(** [tags ss p] is [p] restricted to nodes whose {!Uop.node_tag} is one of
    [ss]. An empty [ss] never matches a tagged node. *)

val early_reject : Ops.t list -> t -> t
(** [early_reject ops p] records a custom source-op precondition for [p].
    A matcher can reject [p] before structural matching unless every op in
    [ops] appears among the candidate node's immediate sources. *)

val located : file:string -> line:int -> t -> t
(** [located ~file ~line p] is [p] annotated with source-location metadata
    for diagnostics. The annotation does not affect matching. *)

val located_pos : loc:pos -> t -> t
(** [located_pos ~loc p] is [p] annotated with the file and line in [loc].
    Pass [~loc:__POS__] at rule construction sites for diagnostic
    locations. *)

val location : t -> (string * int) option
(** [location p] is [p]'s diagnostic location, if any. *)

(** {2:literals Constant literals}

    Shorthands for the most common constants in rewrite rules. *)

val zero : t
(** [zero] is [const_int 0]. *)

val one : t
(** [one] is [const_int 1]. *)

val neg_one : t
(** [neg_one] is [const_int (-1)]. *)

val true_ : t
(** [true_] is [const_bool true]. *)

val false_ : t
(** [false_] is [const_bool false]. *)

(** {2:chain Chainable builders}

    Mirror the per-op [src] contracts of {!Uop}. Each takes the principal
    child last so that pipelines read left-to-right. *)

val load :
  ?loc:pos -> ?alt:t -> ?gate:t ->
  ?name:string -> t -> t
(** [load ?alt ?gate idx] matches {!Ops.Load} with [src = (idx,)], or
    [src = (idx, alt, gate)] when [alt] and [gate] are given.

    Raises [Invalid_argument] if only one of [alt] or [gate] is supplied. *)

val store :
  ?loc:pos -> ?gate:t -> ?name:string -> t -> t -> t
(** [store ?gate dst value] matches {!Ops.Store} with
    [src = (dst, value)] or [src = (dst, value, gate)]. *)

val index : ?loc:pos -> ?name:string -> t -> t -> t
(** [index idx ptr] matches scalar {!Ops.Index} with [src = (ptr, idx)].
    Use a raw pattern with [allow_any_len] to match variadic indexes. *)

val cast :
  ?loc:pos -> ?dtype:Dtype.t -> ?name:string -> t -> t
(** [cast p] matches {!Ops.Cast} of the child [p], optionally pinning the
    result dtype. *)

val bitcast :
  ?loc:pos -> ?dtype:Dtype.t -> ?name:string ->
  t -> t
(** [bitcast p] matches {!Ops.Bitcast} of the child [p], optionally pinning
    the result dtype. *)

val gep :
  ?loc:pos -> ?idx:int -> ?name:string -> t -> t
(** [gep ?idx p] matches value-lane {!Ops.Index} on [p]. When [idx] is
    given the index child is constrained to the weak-int constant [idx];
    without [idx] any constant index is accepted. *)

val sink :
  ?loc:pos -> ?name:string -> t list -> t
(** [sink srcs] matches {!Ops.Sink} whose first children start with [srcs].
    Extra trailing children are accepted. *)

val where :
  ?loc:pos -> ?name:string -> t -> t -> t -> t
(** [where cond then_ else_] matches {!Ops.Where} with
    [src = (cond, then_, else_)]. *)

val alu :
  ?loc:pos -> ?name:string -> t list -> Ops.t -> t
(** [alu args o] matches an ALU node with op [o] and operands [args]. When
    [o] is binary and commutative (see {!Ops.Group.is_commutative}), the
    operand order is canonicalised: the pattern succeeds on either
    permutation of [args]. Otherwise the match is positional. *)

(** {1:args Arg patterns} *)

val arg_any : arg_pat
(** [arg_any] accepts any arg payload. *)

val arg_eq : Uop.Arg.t -> arg_pat
(** [arg_eq a] requires the arg payload to equal [a] structurally. *)

val has_int : int -> arg_pat
(** [has_int n] matches [Uop.Arg.Int n]. *)

val has_const : Const.t -> arg_pat
(** [has_const v] matches [Uop.Arg.Value v'] with [Const.equal v v']. *)

val has_op : Ops.t -> arg_pat
(** [has_op o] matches [Uop.Arg.Op o]. *)

val has_op_in : Ops.t list -> arg_pat
(** [has_op_in os] matches [Uop.Arg.Op o] for any [o] in [os]. *)

val has_reduce_op : Ops.t -> arg_pat
(** [has_reduce_op o] matches [Uop.Arg.Reduce_arg { op = o; _ }]. *)

val has_reduce_op_in : Ops.t list -> arg_pat
(** [has_reduce_op_in os] matches [Uop.Arg.Reduce_arg { op; _ }] when
    [op] is one of [os]. *)

(** {1:operators Operators}

    Infix shorthands for {!alu}-built binary patterns. Kept in a
    sub-module so that [let open Upat in ...] does not shadow int
    arithmetic inside rule bodies. Open locally with [O.(...)] for
    pattern expressions:

    {[
      let rule =
        let open Upat in
        O.(var "x" + zero) => fun b -> Some (b $ "x")
    ]} *)

module O : sig
  val ( + ) : t -> t -> t
  (** [a + b] is [alu [a; b] Ops.Add]. Commutative. *)

  val ( * ) : t -> t -> t
  (** [a * b] is [alu [a; b] Ops.Mul]. Commutative. *)

  val ( - ) : t -> t -> t
  (** [a - b] is [alu [a; b] Ops.Sub]. Positional. *)

  val ( / ) : t -> t -> t
  (** [a / b] is [alu [a; b] Ops.Fdiv]. Positional. *)

  val ( // ) : t -> t -> t
  (** [a // b] is [alu [a; b] Ops.Floordiv]. Positional. *)

  val ( mod ) : t -> t -> t
  (** [a mod b] is [alu [a; b] Ops.Floormod]. Positional. *)

  val ( < ) : t -> t -> t
  (** [a < b] is [alu [a; b] Ops.Cmplt]. Positional. *)

  val cdiv : t -> t -> t
  (** [cdiv a b] is [alu [a; b] Ops.Cdiv]. Positional. *)

  val cmod : t -> t -> t
  (** [cmod a b] is [alu [a; b] Ops.Cmod]. Positional. *)

  val floordiv : t -> t -> t
  (** [floordiv a b] is [alu [a; b] Ops.Floordiv]. Positional. *)

  val floormod : t -> t -> t
  (** [floormod a b] is [alu [a; b] Ops.Floormod]. Positional. *)

  val ne : t -> t -> t
  (** [ne a b] is [alu [a; b] Ops.Cmpne]. Commutative. *)
end

(** {1:bindings Bindings access} *)

val ( $ ) : bindings -> string -> Uop.t
(** [b $ name] is the node captured under [name].

    Raises [Not_found] if [name] is not bound. *)

val find : bindings -> string -> Uop.t option
(** [find b name] is [Some u] when [name] is bound to [u], [None]
    otherwise. *)

val mem : bindings -> string -> bool
(** [mem b name] is [true] iff [name] is bound. *)

val pp_bindings : Format.formatter -> bindings -> unit
(** [pp_bindings ppf b] formats [b] as [{name0=%tag0, name1=%tag1, ...}]
    for debugging. *)

(** {1:matching Matching} *)

val match_ : t -> Uop.t -> bindings list
(** [match_ p u] is the list of bindings under which [p] matches [u].
    Empty means no match; commutative and [Perms]-style patterns may produce
    more than one successful binding set, but duplicate binding sets are
    collapsed. Matching is purely local: children of [u] are inspected only as
    far as [p] requires. *)

(** {1:matcher Pattern matcher} *)

type rule
(** A pattern paired with its rewrite callback. *)

val ( => ) : t -> (bindings -> Uop.t option) -> rule
(** [pat => f] is a rule pairing [pat] with callback [f]. The callback is
    invoked with each capture-binding set produced by a match; it returns
    [Some u'] to rewrite to [u'], or [None] to reject the match and let
    the matcher try the next binding or rule. *)

val with_ctx : t -> ('ctx -> bindings -> Uop.t option) -> 'ctx rule_with_ctx
(** [with_ctx pat f] is a context-sensitive rule. The callback receives the
    context passed to {!Pattern_matcher.rewrite_with_ctx}. *)

(** {2:variadic Variadic capture combinators}

    Shortcut rule builders when the only captures are positional [var]
    wildcards. The pattern-builder and callback both take the captures as
    direct arguments, eliminating name strings and the bindings lookup. *)

val rewrite1 : (t -> t) -> (Uop.t -> Uop.t option) -> rule
(** [rewrite1 p k] builds the pattern by applying [p] to a single fresh
    [var], and invokes [k] with the matched node. Using the same argument
    twice in [p] imposes an equality constraint (as with {!var}). *)

val rewrite2 : (t -> t -> t) -> (Uop.t -> Uop.t -> Uop.t option) -> rule
(** Two-capture variant of {!rewrite1}. *)

val rewrite3 :
  (t -> t -> t -> t) ->
  (Uop.t -> Uop.t -> Uop.t -> Uop.t option) -> rule
(** Three-capture variant of {!rewrite1}. *)

val rewrite4 :
  (t -> t -> t -> t -> t) ->
  (Uop.t -> Uop.t -> Uop.t -> Uop.t -> Uop.t option) -> rule
(** Four-capture variant of {!rewrite1}. *)

module Pattern_matcher : sig
  type t
  (** An ordered collection of rules. *)

  type 'ctx with_ctx
  (** An ordered collection of context-sensitive rules. *)

  val make : rule list -> t
  (** [make rs] is the pattern matcher that tries [rs] in order. Rules are
      dispatched by the candidate node's root op before structural matching.

      Raises [Invalid_argument] if a top-level pattern has no root op. *)

  val make_with_ctx : 'ctx rule_with_ctx list -> 'ctx with_ctx
  (** [make_with_ctx rs] is the context-sensitive variant of {!make}.

      Raises [Invalid_argument] if a top-level pattern has no root op. *)

  val rewrite : t -> Uop.t -> Uop.t option
  (** [rewrite pm u] tries each rule of [pm] in order. For each rule whose
      pattern matches [u], the callback is invoked with every binding set
      produced by {!match_}; the first [Some _] return is the result.
      A callback result physically equal to the candidate node is ignored.
      [None] means no rule (with any binding) fired. *)

  val rewrite_with_ctx : 'ctx with_ctx -> ctx:'ctx -> Uop.t -> Uop.t option
  (** [rewrite_with_ctx pm ~ctx u] is like {!rewrite} except callbacks receive
      [ctx]. *)

  val ( ++ ) : t -> t -> t
  (** [a ++ b] concatenates rules: [a]'s rules are tried before [b]'s. *)

  val compose : t list -> t
  (** [compose pms] concatenates the rule lists of [pms] in order. *)
end
