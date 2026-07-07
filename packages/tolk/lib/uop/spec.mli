(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Structural validators for {!Uop} DAGs.

    A spec is an ordered list of [(pattern, predicate)] rules that
    characterises the well-formed nodes at a given compilation stage. A
    node [u] is accepted iff the first rule whose {!Upat} pattern matches
    [u] has a predicate that returns [true] on some binding produced by
    the match. Later rules are not consulted once a pattern has matched,
    so specific rules must come before permissive ones.

    Predicates only inspect a node; they never rewrite it. This keeps
    specs cheap and callable from the inside of passes as an integrity
    check, distinct from the rewrite machinery in {!Upat.Pattern_matcher}.

    {1:example Example}

    {[
      let open Upat in
      let shared_spec = Spec.make [
        (* SINK has void dtype. *)
        Spec.(op ~dtype:Dtype.void Ops.Sink
              =?> fun _ _ -> true);

        (* INDEX offsets are integer-valued. *)
        Spec.(op ~allow_any_len:true ~src:[any] Ops.Index
              =?> fun u _ ->
                Array.for_all Dtype.is_int
                  (Array.map Uop.dtype (Uop.src u)));
      ]
    ]} *)

(** {1:types Types} *)

type rule
(** A pair of a {!Upat} pattern and a [bool]-returning predicate over the
    matched node and its capture bindings. *)

type t
(** An ordered list of rules. Matching semantics are first-match: see the
    module preamble. *)

(** {1:ctors Constructors} *)

val ( =?> ) : Upat.t -> (Uop.t -> Upat.bindings -> bool) -> rule
(** [pat =?> pred] is the rule "whenever [pat] matches a node [u] with
    bindings [bs], accept [u] iff [pred u bs]". Several bindings may be
    produced by a single match (e.g. through commutative permutation in
    {!Upat.alu}); the rule accepts if [pred] returns [true] for at least
    one of them.

    The [?] in the operator signals the predicate (boolean-returning)
    nature of the callback, distinguishing it from {!Upat.(=>)} which
    binds a rewrite (option-returning) callback. The distinction matters
    because [Upat] is typically opened locally. *)

val ( =??> ) : Upat.t -> (Uop.t -> Upat.bindings -> bool option) -> rule
(** [pat =??> pred] is like {!(=?>)} but uses a three-valued predicate.

    On a match, [pred u bs] returns [Some true] to accept, [Some false]
    to reject, or [None] to defer to the next matching rule.

    Typical use: an invariant that rejects a subset of nodes without
    deciding the rest. For example, "no tag on tensor-graph ops" rejects
    tagged nodes ([Some false]) and defers for untagged ones ([None]). *)

val make : rule list -> t
(** [make rs] is the spec consisting of rules [rs] in order. *)

val ( ++ ) : t -> t -> t
(** [a ++ b] concatenates [a] and [b], preserving order. Rules of [a] are
    tried first. *)

(** {1:specs Stage specs}

    Predefined specs for each compilation stage. Each stage's spec is
    the concatenation of stage-specific rules and the shared rules
    valid across stages. *)

val shared_spec : t
(** Rules that hold at every stage: {!Ops.Sink}, {!Ops.Noop},
    {!Ops.Const}, local/register {!Ops.Buffer}, {!Ops.Stack}, ALU and casts,
    {!Ops.Range}, {!Ops.Index}, {!Ops.End}, grouped side effects, ordering
    {!Ops.After}, backend escapes, specials, machine instructions, memory
    access, and {!Ops.Wmma}. {!Ops.Special} follows tinygrad's shared rule:
    the result and size child have the same dtype, which must be [weakint] or
    [int32].

    {!Ops.Load} and {!Ops.Store} follow the current tinygrad gate layout:
    loads are [(idx)] or [(idx, alt, gate)] and stores are [(idx, value)]
    or [(idx, value, gate)]. Gates must be bool values on the load/store,
    not on {!Ops.Index}. With [CHECK_OOB] set in the process environment,
    memory access validation is delegated to the UOp validation layer: it uses
    deterministic {!Uop.vmin}/{!Uop.vmax} bounds, explicit shape-derived buffer
    sizes, image-dtype and hard-to-model bypasses, invalid-index sentinels,
    statically false gates, and simple boolean gate refinements. Memory sources
    may be {!Ops.Index}, {!Ops.Shrink}, or one {!Ops.Cast} over those sources;
    bitcasts over indexes are not memory sources. There is no general z3
    fallback in this OCaml layer, so masked accesses that need solver-strength
    arithmetic reasoning remain unproven. *)

val tensor_spec : t
(** Tensor-graph spec. Accepts tensor-level devices, global buffers,
    calls/functions/tuples, scalar constant binds,
    copy/allreduce/multi-device ops, movement ops, reductions with tensor axes
    or lowered integer tail sources, staging, and program packaging on top of
    {!shared_spec}. {!Ops.Slice} is an explicit {!full_spec} intermediate,
    not a tensor-stage node. Concrete device payloads reject positional
    selectors and empty multi-device groups; sharding axes must point into the
    source shape of a multi-device value. *)

val kernel_spec : t
(** Kernel-AST spec. Accepts kernel-level structural ops
    ({!Ops.Shaped_wmma},
    lowered {!Ops.Reduce} nodes with empty axes and range sources), all
    movement ops, and {!shared_spec}. Control-flow {!Ops.If}/{!Ops.Endif}
    nodes are program-stage only. *)

val program_spec : t
(** Linearized-program spec. Accepts {!shared_spec} plus program-only
    rejection rules: no weakint values, movement ops only for
    the special index-like {!Ops.Shrink} form, only local/register
    {!Ops.Buffer} nodes, no invalid constants, no vector dtype whose shape
    does not match its lane count, and {!Ops.If}/{!Ops.Endif} with bool
    conditions and {!Ops.Cast}, {!Ops.Index}, or {!Ops.Shrink} dedup
    sources. Two-source alternate loads are rejected before shared memory
    rules so gated loads must use [(idx, alt, gate)]. {!Ops.Wait} is not
    accepted by the current tinygrad parity specs. *)

val full_spec : t
(** [full_spec] is the explicit intermediate validator formed from
    tinygrad's transitional full-spec forms plus {!tensor_spec} and
    {!program_spec}. It accepts known intermediate {!Ops.Slice},
    {!Ops.Call} over slice bodies, loose {!Ops.After}/{!Ops.End},
    expander raw memory access, and
    transitional scalar {!Ops.Bind} nodes. It has no catch-all rule. *)

(** {1:verify Verification} *)

exception Verification_failed of Uop.t
(** Raised by {!type_verify} on the first node rejected by the spec. The
    registered printer reports the op name and dtype. *)

val type_verify : t -> Uop.t -> unit
(** [type_verify spec root] validates every node in [root]'s DAG against
    [spec], walking {!Uop.toposort}[ root] in dependency order.

    Raises [Verification_failed] on the first node that no rule of [spec]
    accepts. *)

val verify_list : t -> Uop.t list -> unit
(** [verify_list spec program] validates [program] exactly in the supplied
    order. This is intended for already-linearized programs where callers
    should not synthesize a root just to run program-stage validation.

    Raises [Verification_failed] on the first node that no rule of [spec]
    accepts. *)

val accepts : t -> Uop.t -> bool
(** [accepts spec u] is [true] iff the first rule of [spec] whose
    pattern matches [u] has a predicate returning [true] on some binding
    of that match. First-match semantics: a specific rule that rejects
    [u] is not overridden by a later catch-all. *)
