(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [[@@deriving ptree]]: the walk of a structure, derived from its type.

    A structure ([Nx.Ptree.S]) is a type ['a t] with one function, [walk], that
    walks the parts of a value with a [Nx.Ptree.Walk.cursor]. The deriver writes
    that function from the type's definition:

    {[
    type 'a t = {
      l1 : 'a Kaun.Linear.t;
      l2 : 'a Kaun.Linear.t;
      window : int option; [@ptree.int]
    }
    [@@deriving ptree]
    ]}

    derives a walk equivalent to

    {[
    let walk c x =
      let l1 = Nx.Ptree.Walk.field c "l1" Kaun.Linear.walk x.l1 in
      let l2 = Nx.Ptree.Walk.field c "l2" Kaun.Linear.walk x.l2 in
      let window =
        Nx.Ptree.Walk.field c "window" Nx.Ptree.Walk.(option int) x.window
      in
      { l1; l2; window }
    ]}

    so the enclosing module is a structure, and
    [Nx.Ptree.instantiate (module M)] makes it a value. The derived walk keeps
    [Nx.Ptree.S]'s contract: it walks every part once, in declaration order, and
    reports every case, length, presence and marked integer. A hand-written walk
    and a derived one are interchangeable.

    {1:generated Generated values}

    For each type of the declaration group:
    - [walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t] for a type named
      [t], and [walk_name] for a type [name]. A type without a parameter gets
      [walk : ('a, 'b) Nx.Ptree.Walk.cursor -> t -> t].
    - For a type without a parameter, also [ptree : t Nx.Ptree.t] (or
      [ptree_name]), its structure at its one type. A type with a parameter gets
      no [ptree]: it has one structure per payload type, which
      [Nx.Ptree.instantiate (module M)] builds where the dtype is known.

    A type [type _ t] with an anonymous parameter is a structure whose parts are
    all fixed: its module is a [Nx.Ptree.S], and it gets no [ptree].

    In an interface, [[@@deriving ptree]] declares the same values.

    {1:rules How a part is walked}

    A record walks its fields in declaration order, each with
    [Nx.Ptree.Walk.field] at the field's name. A variant reports the
    constructor's name with [Nx.Ptree.Walk.case], then walks a single argument
    at the constructor's path, several arguments with [Nx.Ptree.Walk.index] at
    their positions, and an inline record's fields by name. The type of each
    part decides its walk:
    - the type's parameter ['a]: [Nx.Ptree.Walk.leaf];
    - a tensor type, [('x, 'y) Nx.t], [Nx.float32_t] and Nx's other tensor
      aliases (qualified or opened), or [Nx.Rng.key]: [Nx.Ptree.Walk.tensor];
    - [ty option] and [ty list]: [Nx.Ptree.Walk.option] and
      [Nx.Ptree.Walk.list]; [ty array]: its length reported with
      [Nx.Ptree.Walk.int], then each element at its index;
    - a tuple: each component with [Nx.Ptree.Walk.index] at its position;
    - ['a M.t] and ['a M.name]: [M.walk] and [M.walk_name]; ['a name] of the
      same declaration group, or defined before it: [walk_name];
    - [M.t] and [M.name] without argument: [Nx.Ptree.Walk.structure M.ptree] and
      [Nx.Ptree.Walk.structure M.ptree_name]; [name] of the same declaration
      group: [walk_name]; [name] defined before it:
      [Nx.Ptree.Walk.structure ptree_name];
    - [ty M.t], where [ty] does not mention the parameter:
      [Nx.Ptree.Walk.structure (Nx.Ptree.nest (module M) s)], where [s] is
      [ty]'s structure at one type, built from [Nx.Ptree.tensor],
      [Nx.Ptree.unit], [Nx.Ptree.option], [Nx.Ptree.list], [Nx.Ptree.pair],
      [N.ptree], [ptree_name] of a type defined before and [Nx.Ptree.nest];
    - [int] and [bool] under [[@ptree.int]]: [Nx.Ptree.Walk.int], a bool as [0]
      or [1].

    A qualified or earlier type without argument is taken to be a structure at
    one type named by the [ptree] convention, as [Kaun.Cache_index.ptree] and
    [Nx_quant.ptree] are. A type that follows neither convention is a compile
    error at the part's type, such as [Unbound value M.walk]. Each [ptree] the
    deriver names is constrained to the part's type, and each [[@ptree.walk f]]
    to a walk of it, so a structure of another type or a mistyped [f] is
    reported where it is written.

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
      data no compiled program depends on, such as a name. The part's type must
      not mention the parameter.
    - [[@ptree.walk f]] walks the part with [f], an expression of type
      [('a, 'b) Nx.Ptree.Walk.cursor -> ty -> ty']. A part with a structure at
      one type and no module, such as an optimiser state, is
      [[@ptree.walk Nx.Ptree.Walk.structure (Vega.adam_ptree mlp)]].

    An [int] or [bool] without an attribute is an error: reporting it would
    compile a program per value for data no program reads, and leaving it out
    would freeze data a program reads into its first trace. The attribute says
    which.

    {1:errors Errors}

    The deriver reports, at the offending part:
    - an [int] or [bool] without an attribute, and other scalar data ([float],
      [string], [char], [unit], [bytes], [int32], [int64], [nativeint], [exn])
      and [Nx.dtype] without [[@ptree.skip]];
    - [ref], [lazy_t] and [result], functions, objects, classes, polymorphic
      variants, first-class modules, polymorphic types, [as] aliases, local
      opens, extension nodes and [_], without [[@ptree.walk]] or
      [[@ptree.skip]];
    - GADT and existential constructors, which a walk cannot rebuild at another
      parameter;
    - abstract, extensible, and (in an implementation) private types, types with
      constraints, and types with more than one parameter;
    - the parameter inside a tensor type or a skipped part, a structure applied
      to anything but the parameter alone or a type without it, and a type
      variable that is not the parameter;
    - a misplaced or repeated attribute. *)
