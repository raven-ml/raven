(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* The representation of nodes, which the modules of the library that keep
   properties with each node read: [Ops], which exports it, and [Shape]. *)

module Axis_type = struct
  type t =
    | Device
    | Global
    | Warp
    | Local
    | Weak
    | Reduce
    | Upcast
    | Unroll
    | Placeholder
    | Loop
end

type device = Single of string | Multi of string list

module Tag = struct
  type t =
    | Bool of bool
    | Int of int
    | String of string
    | Bytes of string
    | Dtype of Dtype.t
    | Tuple of t list
end

type param_arg = {
  slot : int;
  dtype : Dtype.t;
  size : int option;
  vmin_vmax : (Dtype.value * Dtype.value) option;
  multiple_of : int option;
  name : string option;
  addrspace : Dtype.addr_space option;
  device : device option;
  volatile : bool;
  bind_on_realize : bool;
  bound : Dtype.value option;
  phase : int;
  align : int;
}

type keep = Removable | Broadcast | Whole

type bufferize_opts = {
  device : device option;
  addrspace : Dtype.addr_space;
  keep : keep;
}

type wmma = {
  dims : int * int * int;
  dtype_in : Dtype.t;
  threads : int;
  upcast_axes :
    ((int list * int) list * (int list * int) list * (int list * int) list)
    option;
}

(* The payloads of calls name nodes, and nodes carry them: the two groups of
   types are recursive modules, since one group of types cannot repeat a field
   name. *)
module rec Calls : sig
  type hcq_kernel = {
    devices : string list;
    name : string;
    estimates : Node.estimates;
    stamps : int list;
    profile_key : string option;
    input_slots : int list;
    outs : int list;
    ins : int list;
  }

  type hcq_info = {
    device : string list;
    kernels : hcq_kernel list;
    estimates : Node.estimates;
    nargs : int;
    table : int;
    inputs : (Node.t * int * string) list;
    slots : (string * int) list;
    written_bufs : Node.t list;
    writes : Node.t list;
    copies : (string * string * int) list;
  }

  type call_info = {
    name : string option;
    precompile : bool;
    aux : hcq_info option;
    dtype : Dtype.t;
  }
end =
  Calls

and Node : sig
  type t = {
    op : Op.t;
    src : t list;
    arg : arg;
    tag : Tag.t option;
    dtype : Dtype.t;
    id : int;
    memos : memos;
  }

  (* Properties computed on first use. Each is a function of the node, so
     domains racing to fill one write the same value; the fields are atomic so
     that a reader that sees a value sees it whole. They are a record of their
     own so that the probe a lookup builds ({!v}) shares one empty record. *)
  and memos = {
    mutable shape_memo : sint list option option; [@atomic]
    mutable ranges_memo : nodes option; [@atomic]
    mutable ended_ranges_memo : t list option; [@atomic]
    mutable min_max_memo : (Dtype.value * Dtype.value) option; [@atomic]
    mutable device_memo : device option option; [@atomic]
    mutable addrspace_memo : Dtype.addr_space option option; [@atomic]
    mutable backward_slice_memo : nodes option; [@atomic]
    mutable ops_reached_memo : Op.Set.t option; [@atomic]
    mutable axis_memo : int option option; [@atomic]
    mutable marg_memo : movement option; [@atomic]
    mutable key_memo : string option; [@atomic]
    mutable arg_repr_memo : string option; [@atomic]
  }

  (* A set in the order its nodes joined it. Small sets are searched in order;
     larger ones carry a table of their nodes' ids. *)
  and nodes = {
    order : t list;
    cardinal : int;
    ids : (int, unit) Hashtbl.t option;
  }

  and sint = Int of int | Sym of t
  and estimates = { ops : sint; lds : sint; mem : sint }
  and split = { iterations : sint; lo : int; hi : int }

  and kernel_info = {
    name : string;
    applied_opts : Opt.t list;
    opts_to_apply : Opt.t list option;
    estimates : estimates option;
    beam : int;
    split : split option;
  }

  and program_info = {
    global_size : sint list;
    local_size : sint list;
    vars : t list;
    globals : int list;
    outs : int list;
    ins : int list;
    target : Helpers.Target.t;
  }

  and arg =
    | No_arg
    | Const of Dtype.const
    | Dtype of Dtype.t
    | Param of param_arg
    | Range of { axis_id : int list; axis_type : Axis_type.t }
    | Reduce of { op : Op.t; num_axes : int }
    | Allreduce of { op : Op.t; device : device }
    | Device of device
    | Shard of int
    | Axes of int list
    | Flips of bool list
    | String of string
    | Bytes of string
    | Queue of { devices : string list; queue : string }
    | Region of { name : string; align : int }
    | Code of { code : string; dtype : Dtype.t }
    | Bufferize of bufferize_opts
    | Kernel of kernel_info
    | Program of program_info
    | Call of Calls.call_info
    | Wmma of wmma

  and movement =
    | Reshape of sint list
    | Expand of sint list
    | Pad of (sint * sint) list
    | Shrink of (sint * sint) list
    | Permute of int list
    | Flip of bool list
end =
  Node

include Calls
include Node
