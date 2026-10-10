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
          device's window. A movement or a bitcast keeps none: an operation
          reads it as a view of its operand's results. *)

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
  rule : 'r. interpretation -> by:string -> 'r prim -> 'r;
      (** What it makes of the operations it receives. *)
  start : int;  (** Its order of start on [domain]. *)
  domain : Domain.id;  (** Where it started. *)
  extents : int Atomic.t;  (** [domain]'s count of live [Extent]s. *)
  live : bool Atomic.t;  (** Cleared when its extent ends. *)
  mutable running : bool;  (** Whether its rule runs, on [domain]. *)
}
(** An interpretation. Only {!Interp} writes its fields. *)

(** {1:operations Operations}

    An operation's arrays are its own: whoever builds one from a caller's array
    copies it, so that nothing a caller changes later reaches it.

    A loop's loads have exactly its iteration shape: no operation broadcasts or
    promotes. *)

and 'd load =
  | Plain : ('v, 's, 'd) t -> 'd load
      (** How a loop reads an operand: through its layout. *)

and ('d, _) reduction =
  | Monoid :
      Nx_kernel.Spec.monoid * int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t) reduction
      (** Output [k] of a loop's program folded by the monoid in that output's
          dtype, rounded once to the dtype. *)
  | Moments :
      int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) reduction
      (** The population mean of output [k], then its variance. *)
  | Arg :
      Nx_kernel.Spec.extreme * int * ('v, 's) dtype
      -> ( 'd,
           ('v, 's, 'd) t * (int64, Nx_array.Dtype.int64_elt, 'd) t )
         reduction
      (** The extreme of output [k], then its first position, in C order of the
          reduced indices. *)

and ('d, _) reductions =
  | [] : ('d, unit) reductions
  | ( :: ) :
      ('d, 'a) reduction * ('d, 'r) reductions
      -> ('d, 'a * 'r) reductions

and ('d, _) outs =
  | [] : ('d, unit) outs
  | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) t * 'r) outs

and 'd index = (int64, Nx_array.Dtype.int64_elt, 'd) t
(** Positions held in data. *)

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
  | Reduce : {
      layout : Nx_array.Layout.t;
      axes : int array;
      prog : Nx_kernel.Prog.t;
      reductions : ('d, 'r) reductions;
      loads : 'd load array;
    }
      -> 'r prim
      (** [prog] at every index of [layout]'s shape, as a map's, each reduction
          folding its output along [axes], strictly increasing. Results drop
          [axes] and are C-contiguous. *)
  | Scan : {
      layout : Nx_array.Layout.t;
      axis : int;
      prog : Nx_kernel.Prog.t;
      reduction : ('d, 'r) reduction;
      loads : 'd load array;
    }
      -> 'r prim
      (** The reduction of [prog]'s output over the indices along [axis] up to
          each index's, inclusive: [layout]'s shape, C-contiguous. *)
  | Gather : {
      axis : int;
      idx : 'd index;
      x : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
      (** [x] read at the positions [idx] holds along [axis]: [idx] has [x]'s
          rank and its extents off [axis], and the result [idx]'s shape. A
          position outside [x]'s axis reads the element of zero bits. *)
  | Scatter : {
      combine : Nx_kernel.Spec.combine;
      unique : bool;
      axis : int;
      idx : 'd index;
      updates : ('v, 's, 'd) t;
      into : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
      (** [into] with each update combined at its own index with axis [axis]
          replaced by [idx]'s element there, in C order of [updates]
          ({!Nx_kernel.Spec.scatter}); [idx] and [updates] have one shape,
          [into]'s off [axis]. [unique] promises distinct targets. *)
  | Assemble : {
      dtype : ('v, 's) dtype;
      shape : int array;
      fill : 'v;
      pieces : (Nx_array.Move.range array * ('v, 's, 'd) t) list;
    }
      -> ('v, 's, 'd) t prim
      (** The value of [shape] whose element at an index is the last piece's
          whose region, a [Slice] of [shape] by its ranges, holds it, and [fill]
          where none does. Each piece has its region's shape. *)
  | Contract : {
      spec : Nx_kernel.Spec.contract Nx_kernel.Spec.t;
      out : ('v, 's) dtype;  (** [spec]'s [out]. *)
      a : ('a, 'b, 'd) t;
      b : ('c, 'e, 'd) t;
      init : ('v, 's, 'd) t option;  (** Present iff [spec] has one. *)
    }
      -> ('v, 's, 'd) t prim
      (** [spec] of [a] and [b], from [init]: its result C-contiguous, of the
          shape {!Nx_kernel.Spec.shapes} gives. *)
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
