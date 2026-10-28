(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [[@@deriving ptree]]: the walk of a structure, derived from its type.

    A structure ([Nx.Ptree.S]) is a type [('a, 'd) t] with one function, [walk],
    that walks the parts of a value with a [Nx.Ptree.Walk.cursor]: the positions
    of its payload ['a] and its fixed arrays, on the set ['d]. The deriver
    writes that function from the type's definition:

    {[
    type ('a, 'd) t = {
      l1 : ('a, 'd) Kaun.Linear.t;
      l2 : ('a, 'd) Kaun.Linear.t;
      steps : 'd Nx.int32_t;
      window : int option; [@ptree.int]
    }
    [@@deriving ptree]
    ]}

    derives a walk equivalent to

    {[
    let walk c x =
      let open Nx.Ptree.Walk in
      let l1 = field c "l1" Kaun.Linear.walk x.l1 in
      let l2 = field c "l2" Kaun.Linear.walk x.l2 in
      let steps = field c "steps" tensor x.steps in
      let window = field c "window" (option int) x.window in
      { l1; l2; steps; window }
    ]}

    so the enclosing module is a structure, and
    [Nx.Ptree.instantiate (module M)] describes its values. The derived walk
    keeps [Nx.Ptree.S]'s contract: it walks every part once, in declaration
    order, and reports every case, length, presence and marked integer. A
    hand-written walk and a derived one are interchangeable.

    {1:declarations Declarations}

    A derived type has at most two parameters, read by position:
    - [('a, 'd) t]: the payload ['a], then the set ['d]. Either may be
      anonymous: [('a, _) t] holds no fixed array, as Kaun's layers, and
      [(_, 'd) t] no position of a payload.
    - ['d t]: the set alone, for a type of fixed arrays only, as [Nx_quant.t],
      [Kaun.Cache_index.t] and [Nx.Rng.key]. At one set it is [Gpu.d t].
    - [t]: no array of either kind, as a record of integers under
      [[@ptree.int]].

    A fixed array lies on the type's set parameter: a tensor type whose set is
    anything else, the host's [Nx.host] or the payload ['a], is an error at the
    part.

    {1:generated Generated values}

    For each type of the declaration group, [walk] for a type named [t] and
    [walk_name] for a type [name], of type
    - [('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t] for a
      type of two parameters;
    - [('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t] for a type of one;
    - [('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> t -> t] for a type of none.

    In an interface, [[@@deriving ptree]] declares the same values.

    No description is generated: a description is at one dtype and one set,
    which its first use fixes. Make one where it is used,
    [Nx.Ptree.instantiate (module M)] for a type of two parameters and
    [Nx.Ptree.v M.walk] for the others. [Nx.Ptree.place] and [Nx.Ptree.cast]
    take the module [M] of a type of two parameters.

    A type with no payload is declared either way, and the choice decides how it
    is described and moved. [(_, 'd) t] is a module that [Nx.Ptree.place] and
    [Nx.Ptree.cast] take, and [Nx.Ptree.instantiate (module M)] describes it,
    leaving the dtype of the absent payload free: a top-level description in an
    implementation without an interface needs an annotation, as
    [((Nx.host Nx.float32_t, Nx.host) M.t, Nx.host) Nx.Ptree.t]. ['d t] is
    described by [Nx.Ptree.v M.walk] without one, but placing or casting it
    takes a module [struct type ('a, 'd) t = 'd M.t let walk = M.walk end].

    {1:rules How a part is walked}

    A record walks its fields in declaration order, each with
    [Nx.Ptree.Walk.field] at the field's name. A variant reports the
    constructor's name with [Nx.Ptree.Walk.case], then walks a single argument
    at the constructor's path, several arguments with [Nx.Ptree.Walk.index] at
    their positions, and an inline record's fields by name. The type of each
    part decides its walk:
    - the payload parameter ['a]: [Nx.Ptree.Walk.leaf];
    - a tensor type on the set parameter, [('v, 's, 'd) Nx.t] and each of
      Nx's aliases ['d Nx.float32_t],
      ['d Nx.bit_t], ... (qualified or opened): [Nx.Ptree.Walk.tensor];
    - ['d Nx.Rng.key] (or ['d Rng.key] under [open Nx]): [Nx.Rng.walk];
    - ['d Nx.packed]: [fun c (Nx.P x) -> Nx.P (Nx.Ptree.Walk.tensor c x)];
    - [ty option] and [ty list]: [Nx.Ptree.Walk.option] and
      [Nx.Ptree.Walk.list]; [ty array]: its length reported with
      [Nx.Ptree.Walk.int], then each element at its index;
    - a tuple: each component with [Nx.Ptree.Walk.index] at its position;
    - any other application [(args) M.t], [(args) M.name] or [(args) name], of
      the declaration group or defined before it: [M.walk], [M.walk_name] and
      [walk_name]. [M.t] without argument is [M.walk] too;
    - such an application [(p, s) M.t] whose payload argument [p] mentions a
      parameter without being the payload parameter is a structure nested at
      [M]'s positions: [Nx.Ptree.Walk.under M.walk w], [w] the walk of [p].
      [(('a, 'd) Mlp.t, 'd) Vega.Adam_state.t] is
      [under Vega.Adam_state.walk Mlp.walk], and a frozen layer
      [('d Nx.float32_t, 'd) Kaun.Linear.t] is [under Kaun.Linear.walk tensor];
    - [int] and [bool] under [[@ptree.int]]: [Nx.Ptree.Walk.int], a bool as [0]
      or [1].

    The deriver names a walk by convention and checks no role: each walk it
    names, and each [[@ptree.walk f]], is constrained to walk the part's type,
    so OCaml reports a mismatch at the part. A module without the walk fails
    there with [Unbound value M.walk], as does [M.t] without argument where [M]
    has no [walk]. A structure applied in another order, or at a closed payload
    as [(int, 'd) Kaun.Linear.t], fails with a type error at the part.
    [Nx_wide.t]'s first argument is a float format, which this rule walks with
    [Nx_wide.walk].

    {1:attributes Attributes}

    An attribute goes after a record field, after a constructor's single
    argument, or on a type to annotate a nested part, as in
    [(int [@ptree.int]) list]. A part takes at most one. On a constructor with
    no argument or several, or on the type declaration, it is an error.
    - [[@ptree.int]] reports the part's [int]s and [bool]s. A compiled program
      is cached per reported value, so mark an integer that changes what a
      program computes (a window, a block size, a layer kind), and hold a value
      that changes on every call in a tensor.
    - [[@ptree.skip]] leaves the part out of the walk: the rebuilt value holds
      the same part, and neither paths, keys nor checkpoints see it. It is for
      data no program on the set reads, such as a name or an array on the host.
      The part's type mentions neither parameter.
    - [[@ptree.walk f]] walks the part with [f], an expression of type
      [('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ty -> ty'], where [ty'] is the
      part's type [ty] with ['b] for the payload and ['e] for the set. The walk
      moves values between sets, so an [f] at one set, as
      [Nx.Ptree.Walk.structure s], does not type: OCaml reports at [f] that
      [ptree_d] is not compatible with [ptree_e]. Walk such a part with its
      structure's own walk, or with [Nx.Ptree.Walk.under].

    An [int] or [bool] without an attribute is an error: reporting it would
    compile a program per value for data no program reads, and leaving it out
    would freeze data a program reads into its first trace. The attribute says
    which.

    {1:errors Errors}

    The deriver reports, at the offending part:
    - an [int] or [bool] without an attribute, and other scalar data ([float],
      [string], [char], [unit], [bytes], [int32], [int64], [nativeint], [exn])
      and [Nx.dtype] without [[@ptree.skip]];
    - a tensor type, a key or a packed array whose set is not the set parameter,
      the set parameter alone as a part, and a tensor type that mentions a
      parameter in its element types;
    - [Nx.Placement.t], [Nx.Mesh.t] and [Nx.devices], which hold no array to
      walk, other [Nx] types without a walk, and an [Nx] name ending in [_t]
      that is not a tensor alias;
    - [ref], [lazy_t] and [result], functions, objects, classes, polymorphic
      variants, first-class modules, polymorphic types, [as] aliases, local
      opens, extension nodes and [_], without [[@ptree.walk]] or
      [[@ptree.skip]];
    - GADT and existential constructors, which a walk cannot rebuild at other
      parameters;
    - abstract, extensible, and (in an implementation) private types, types with
      constraints, and types with more than two parameters;
    - a parameter inside a skipped part, and a type variable that is not a
      parameter;
    - a misplaced or repeated attribute. *)
