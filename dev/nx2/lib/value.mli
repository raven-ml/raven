(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values and operations: nx's central types, defined together because each
    mentions the other. This module has no implementation.

    A value is concrete, arrays on the devices of its placement; a constant not
    yet computed, a value of every set; or an interpretation's stand-in. The
    brand ['d] is phantom: every function that makes a value checks that its
    arrays lie on its placement's devices, and nothing at run time reads ['d]. A
    constant has no placement. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

type ('v, 's, +'d) payload = ..
(** What an interpretation keeps in its traced values. A payload holds values,
    never a function of ['d]. *)

(** {1:values Values} *)

type ('v, 's, 'd) t =
  | Array of {
      at : 'd Devices.placement;
      a : ('v, 's) Nx_array.t;
      mutable dead : string; [@atomic]
    }
      (** On [at]'s one device. [dead] is [""] while the value lives, and the
          function it was donated to once it died ({!Exec.donate}). *)
  | Shards of {
      at : 'd Devices.placement;
      arrays : ('v, 's) Nx_array.t array;
      mutable dead : string; [@atomic]
    }
      (** On [at]'s devices, two or more: per device, in
          [Grid.devices (Devices.grid at)]'s order, its window of the value. *)
  | Donated of {
      at : 'd Devices.placement;
      arrays : ('v, 's) Nx_array.t array;
      chain : chain;
      mutable spent : string; [@atomic]
    }
      (** A handle one operation reads: [arrays] over its donor's memory, one
          per device of [at]. [spent] is [""] until an operation reads it: the
          one that consumed it, or a movement that passed a new handle on. *)
  | Deferred of { form : ('v, 's, 'd) form; node : node; k : int }
      (** Result [k] of the constant operation [node]. *)
  | Traced of {
      form : ('v, 's, 'd) form;
      owner : interpretation;
      payload : ('v, 's, 'd) payload;
    }  (** An interpretation's stand-in: its form, and what [owner] keeps. *)

and chain = {
  origin : int;  (** Unique among a process's donations. *)
  claim : string -> string;
      (** [claim by] kills the donor, naming [by], and is [""]; or, where it
          died before, is the name it died to. Two claims on two domains: one
          wins. *)
  mutable consumer : string; [@atomic]
      (** [""] until an operation consumed the chain, then its name: the handles
          passed on die with it. *)
}
(** The handles one {!Exec.donate} made, the first and those movements passed
    on. *)

and ('v, 's, 'd) form = {
  dtype : ('v, 's) dtype;
  layout : Nx_array.Layout.t;
  placement : 'd Devices.placement option;  (** [None] for every set. *)
}
(** A value without its bytes. A sharded value's layout is the C-contiguous
    layout of the whole. *)

and node =
  | Node : {
      id : int;  (** Unique among a process's nodes. *)
      by : string;
      op : 'r prim;
      memo : (unit Devices.placement * Nx_array.any array array) list Atomic.t;
    }
      -> node
      (** An operation whose every operand is a constant, applied by [by].
          [memo] holds its results computed so far at the placements an
          operation read it at: per result, one array per device, of the
          device's window. *)

and 'd any = Any : ('v, 's, 'd) t -> 'd any

(** {1:interpretations Interpretations} *)

and reach =
  | Values  (** Reaches the operations on its traced values. *)
  | Extent
      (** Also reaches every operation its starting fiber applies inside its
          extent, on its domain. *)

and interpretation = {
  name : string;  (** In messages, as ["Rune.grad"]. *)
  reach : reach;
  rule : rule;
  start : int;  (** Its order of start on [domain]. *)
  domain : Domain.id;  (** Where it started. *)
  extents : int Atomic.t;  (** [domain]'s count of live [Extent]s. *)
  live : bool Atomic.t;  (** Cleared when its extent ends. *)
  mutable running : bool;  (** Whether its rule runs, on [domain]. *)
}
(** An interpretation. Only {!Interp} writes its fields. *)

and rule = { rule : 'r. interpretation -> by:string -> 'r prim -> 'r }
(** What an interpretation makes of the operations it receives. *)

(** {1:operations Operations}

    An operation's arrays are its own: whoever builds one from a caller's array
    copies it, so that nothing a caller changes later reaches it.

    A loop's loads have exactly its iteration shape: no operation broadcasts or
    promotes. *)

and 'd load =
  | Plain : ('v, 's, 'd) t -> 'd load
      (** How a loop reads an operand: through its layout. *)

and ('d, _) outs =
  | [] : ('d, unit) outs
  | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) t * 'r) outs

and _ prim =
  | Map : {
      layout : Nx_array.Layout.t;
      prog : Nx_kernel.Prog.t;
      outs : ('d, 'r) outs;
      loads : 'd load array;
    }
      -> 'r prim
      (** [prog] at every index of [layout]'s shape, reading load [i] as its
          operand [i]; result [k] is its output [k], laid out as [layout], which
          is C-contiguous. A one-result map is ['v * unit]. With no loads, a
          creation. *)
  | Copy : ('v, 's, 'd) t -> ('v, 's, 'd) t prim
      (** The value stored afresh, C-contiguous. *)
  | Move : Nx_array.Move.t * ('v, 's, 'd) t -> ('v, 's, 'd) t prim
  | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) t -> ('w, 'r, 'd) t prim
      (** The bits read in another dtype: one width keeps the shape, a narrower
          one appends an axis of the ratio, a wider one consumes a trailing axis
          of it. *)
  | Place : 'e Devices.placement * ('v, 's, 'd) t -> ('v, 's, 'e) t prim
  | Check : {
      ok : (bool, Nx_array.Dtype.bool_elt, 'd) t;
      data : 'd any list;
      fail : int array -> 'd any list -> exn;
    }
      -> unit prim
      (** Raises [fail i data_i] at the first index [i], in C order, where [ok]
          is [false], [data_i] each of [data] at [i]. *)
