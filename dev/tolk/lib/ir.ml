(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Three IRs at different abstraction levels: - Tensor: scheduling-level (shape
   ops, multi-device, lazy evaluation) - Kernel: codegen-oriented DAG (ranges,
   reductions, memory indexing) - Program: flat SSA for code emission (loops,
   if/endif, no graph structure)

   All three share a flat instruction-array representation where [type ref =
   int] indexes into the array. Instructions may only reference earlier indices
   (SSA dominance). Each module provides [dtype_of], [refs_of], [map_refs],
   [validate], and pretty-printing. *)

type special_dim = Group_id of int | Local_id of int | Global_idx of int

let special_axis = function Group_id a | Local_id a | Global_idx a -> a

type axis_kind =
  | Global
  | Thread
  | Local
  | Warp
  | Loop
  | Group_reduce
  | Reduce
  | Upcast
  | Unroll
  | Outer
  | Placeholder

(* ───── Shared Validation Helpers ───── *)

(* Factored out of Kernel/Tensor/Program.validate to avoid duplication. Each is
   parameterized by [fail] and [get_dtype] so modules can supply their own error
   reporting and dtype lookup. *)

let check_dtype_eq fail i ~ctx ~expected ~got =
  match (expected, got) with
  | Some e, Some g when Dtype.equal e g -> ()
  | Some e, Some g ->
      fail i
        (Printf.sprintf "%s: expected %s, got %s" ctx (Dtype.to_string e)
           (Dtype.to_string g))
  | None, _ -> fail i (Printf.sprintf "%s: expected dtype not available" ctx)
  | _, None -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)

let check_dtype_match fail i ~ctx dt1 dt2 =
  match (dt1, dt2) with
  | Some d1, Some d2 when Dtype.equal d1 d2 -> ()
  | Some _, Some _ ->
      fail i (Printf.sprintf "%s: operand dtypes don't match" ctx)
  | _ -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)

let check_bool_scalar fail (get_dtype : int -> Dtype.t option) i ~ctx r =
  match get_dtype r with
  | Some dt when dt.scalar = Dtype.Bool && dt.count = 1 -> ()
  | Some _ -> fail i (Printf.sprintf "%s must be bool scalar" ctx)
  | None -> fail i (Printf.sprintf "%s dtype not available" ctx)

let check_shift_rhs fail (get_dtype : int -> Dtype.t option) i rhs dtype =
  match get_dtype rhs with
  | Some dt when Dtype.equal dt dtype -> ()
  | Some dt when Dtype.equal dt Dtype.uint32 -> ()
  | Some _ -> fail i "shift rhs must match lhs dtype or be uint32"
  | None -> fail i "shift rhs dtype not available"

let check_index_like fail (get_dtype : int -> Dtype.t option) i ~ctx r =
  match get_dtype r with
  | Some dt
    when dt.count = 1 && (dt.scalar = Dtype.Index || dt.scalar = Dtype.Int32) ->
      ()
  | Some _ -> fail i (Printf.sprintf "%s must be index or int32 scalar" ctx)
  | None -> fail i (Printf.sprintf "%s dtype not available" ctx)

(* ───── Shared Pretty Printing Helpers ───── *)

let pp_comma fmt () = Format.fprintf fmt ", "
let pp_ref fmt r = Format.fprintf fmt "%%%d" r
let pp_refs fmt refs = Format.pp_print_list ~pp_sep:pp_comma pp_ref fmt refs

let pp_special_dim fmt = function
  | Group_id i -> Format.fprintf fmt "gid%d" i
  | Local_id i -> Format.fprintf fmt "lid%d" i
  | Global_idx i -> Format.fprintf fmt "idx%d" i

let pp_axis_kind fmt k =
  Format.pp_print_string fmt
    (match k with
    | Global -> "global"
    | Thread -> "thread"
    | Local -> "local"
    | Warp -> "warp"
    | Loop -> "loop"
    | Group_reduce -> "group_reduce"
    | Reduce -> "reduce"
    | Upcast -> "upcast"
    | Unroll -> "unroll"
    | Outer -> "outer"
    | Placeholder -> "placeholder")

let pp_ptr fmt (dtype : Dtype.ptr) =
  Format.fprintf fmt "%a*%s [%a]" Dtype.pp dtype.base
    (if dtype.v = 1 then "" else Printf.sprintf ".vec(%d)" dtype.v)
    Dtype.pp_addr_space dtype.addrspace

(* ───── Shared Deduplication Helpers ───── *)

let pp_program pp_instr fmt t =
  Array.iteri
    (fun i instr -> Format.fprintf fmt "%3d: %a@\n" i pp_instr instr)
    t

let intern_generic (type a) ~(map_refs : (int -> int) -> a -> a)
    (program : a array) : a array =
  let len = Array.length program in
  let module Tbl = Hashtbl.Make (struct
    type t = a

    let equal = ( = )
    let hash x = Hashtbl.hash_param 100 100 x
  end) in
  let tbl = Tbl.create (max 16 (len * 2)) in
  let map = Array.make len (-1) in
  let next = ref 0 in
  let acc = ref [] in
  for i = 0 to len - 1 do
    let mapped = map_refs (fun r -> map.(r)) program.(i) in
    match Tbl.find_opt tbl mapped with
    | Some idx -> map.(i) <- idx
    | None ->
        let idx = !next in
        next := idx + 1;
        map.(i) <- idx;
        Tbl.add tbl mapped idx;
        acc := mapped :: !acc
  done;
  Array.of_list (List.rev !acc)

module Kernel = struct
  (* ───── Types ───── *)

  type const = Bool of bool | Int of int | Float of float | Invalid
  type reduce_op = Add | Mul | Max

  type bufferize_device =
    | Device_single of string
    | Device_multi of string list
    | Device_index of int

  type estimate = Int of int | Symbolic of string
  type estimates = { ops : estimate; lds : estimate; mem : estimate }

  type bufferize_opts = {
    device : bufferize_device option;
    addrspace : Dtype.addr_space;
    removable : bool;
  }

  type kernel_info = {
    name : string;
    axis_kinds : axis_kind list;
    dont_use_locals : bool;
    applied_opts : string list;
    opts_to_apply : string list option;
    estimates : estimates option;
    metadata_tags : string list;
  }

  type ref = int

  type instr =
    (* Graph management *)
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
    | Group of { srcs : ref list }
    | After of { src : ref; deps : ref list }
    (* Memory definitions *)
    | Param of { idx : int; dtype : Dtype.ptr }
    | Define_local of { size : int; dtype : Dtype.ptr }
    | Define_reg of { size : int; dtype : Dtype.ptr }
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
    | Bufferize of {
        src : ref;
        idx : ref;
        ranges : ref list;
        dtype : Dtype.ptr;
        opts : bufferize_opts;
      }
    (* Constants *)
    | Const of { value : const; dtype : Dtype.t }
    (* Memory operations *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }
    | Ptrcat of { srcs : ref list; dtype : Dtype.ptr }
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
    | Store of { dst : ref; value : ref; ranges : ref list }
    (* Unary ALU *)
    | Neg of { src : ref; dtype : Dtype.t }
    | Exp2 of { src : ref; dtype : Dtype.t }
    | Log2 of { src : ref; dtype : Dtype.t }
    | Sin of { src : ref; dtype : Dtype.t }
    | Sqrt of { src : ref; dtype : Dtype.t }
    | Recip of { src : ref; dtype : Dtype.t }
    | Trunc of { src : ref; dtype : Dtype.t }
    (* Binary ALU *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
    (* Ternary ALU *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
    (* Type conversion *)
    | Cast of { src : ref; dtype : Dtype.t }
    | Bitcast of { src : ref; dtype : Dtype.t }
    (* Vector ops *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
    | Cat of { srcs : ref list; dtype : Dtype.t }
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
    (* Control flow / ranges *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
    | End of { value : ref; ranges : ref list }
    | Barrier
    (* GPU special *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
    (* Reduction and tensor core *)
    | Reduce of {
        op : reduce_op;
        src : ref;
        ranges : ref list;
        dtype : Dtype.t;
      }
    | Unroll of { src : ref; axes : (int * int) list; dtype : Dtype.t }
    | Contract of { src : ref; axes : (int * int) list; dtype : Dtype.t }
    | Wmma of {
        name : string;
        a : ref;
        b : ref;
        c : ref;
        ranges : ref list;
        dtype : Dtype.t;
        dims : int * int * int; (* N, M, K matrix dimensions *)
        dtype_in : Dtype.scalar;
        dtype_out : Dtype.scalar;
        device : string;
        threads : int;
        upcast_axes : (int * int) list * (int * int) list * (int * int) list;
        reduce_axes : int list;
      }
    (* Custom code injection *)
    | Custom of { fmt : string; args : ref list }
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }

  type t = instr array

  (* ───── Inspection ───── *)

  let dtype_of = function
    (* Pointer-producing: return base type *)
    | Param { dtype; _ }
    | Define_local { dtype; _ }
    | Define_reg { dtype; _ }
    | Bufferize { dtype; _ }
    | Index { dtype; _ }
    | Ptrcat { dtype; _ } ->
        Some dtype.base
    (* Value-producing with dtype field *)
    | Define_var { dtype; _ }
    | Const { dtype; _ }
    | Load { dtype; _ }
    | Neg { dtype; _ }
    | Exp2 { dtype; _ }
    | Log2 { dtype; _ }
    | Sin { dtype; _ }
    | Sqrt { dtype; _ }
    | Recip { dtype; _ }
    | Trunc { dtype; _ }
    | Add { dtype; _ }
    | Sub { dtype; _ }
    | Mul { dtype; _ }
    | Fdiv { dtype; _ }
    | Idiv { dtype; _ }
    | Mod { dtype; _ }
    | Max { dtype; _ }
    | Pow { dtype; _ }
    | Shl { dtype; _ }
    | Shr { dtype; _ }
    | And { dtype; _ }
    | Or { dtype; _ }
    | Xor { dtype; _ }
    | Threefry { dtype; _ }
    | Cmplt { dtype; _ }
    | Cmpeq { dtype; _ }
    | Cmpne { dtype; _ }
    | Where { dtype; _ }
    | Mulacc { dtype; _ }
    | Cast { dtype; _ }
    | Bitcast { dtype; _ }
    | Vectorize { dtype; _ }
    | Cat { dtype; _ }
    | Gep { dtype; _ }
    | Range { dtype; _ }
    | Special { dtype; _ }
    | Reduce { dtype; _ }
    | Unroll { dtype; _ }
    | Contract { dtype; _ }
    | Wmma { dtype; _ }
    | Custom_inline { dtype; _ } ->
        Some dtype
    (* No result value *)
    | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ -> None

  let refs_of = function
    | Sink { srcs; _ } | Group { srcs } -> srcs
    | After { src; deps } -> src :: deps
    | Param _ | Define_local _ | Define_reg _ | Define_var _ | Const _ | Barrier
      ->
        []
    | Bufferize { src; idx; ranges; _ } -> src :: idx :: ranges
    | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
    | Ptrcat { srcs; _ } -> srcs
    | Load { src; alt; _ } -> src :: Option.to_list alt
    | Store { dst; value; ranges } -> dst :: value :: ranges
    | Neg { src; _ }
    | Exp2 { src; _ }
    | Log2 { src; _ }
    | Sin { src; _ }
    | Sqrt { src; _ }
    | Recip { src; _ }
    | Trunc { src; _ }
    | Cast { src; _ }
    | Bitcast { src; _ }
    | Gep { src; _ }
    | Unroll { src; _ }
    | Contract { src; _ } ->
        [ src ]
    | Range { size; _ } | Special { size; _ } -> [ size ]
    | End { value; ranges } -> value :: ranges
    | Add { lhs; rhs; _ }
    | Sub { lhs; rhs; _ }
    | Mul { lhs; rhs; _ }
    | Fdiv { lhs; rhs; _ }
    | Idiv { lhs; rhs; _ }
    | Mod { lhs; rhs; _ }
    | Max { lhs; rhs; _ }
    | Pow { lhs; rhs; _ }
    | Shl { lhs; rhs; _ }
    | Shr { lhs; rhs; _ }
    | And { lhs; rhs; _ }
    | Or { lhs; rhs; _ }
    | Xor { lhs; rhs; _ }
    | Threefry { lhs; rhs; _ }
    | Cmplt { lhs; rhs; _ }
    | Cmpeq { lhs; rhs; _ }
    | Cmpne { lhs; rhs; _ } ->
        [ lhs; rhs ]
    | Where { cond; a; b; _ } -> [ cond; a; b ]
    | Mulacc { a; b; c; _ } -> [ a; b; c ]
    | Vectorize { srcs; _ } | Cat { srcs; _ } -> srcs
    | Reduce { src; ranges; _ } -> src :: ranges
    | Wmma { a; b; c; ranges; _ } -> a :: b :: c :: ranges
    | Custom { args; _ } | Custom_inline { args; _ } -> args

  let map_refs map_ref (instr : instr) : instr =
    let map_ref_list = List.map map_ref in
    let map_ref_opt = Option.map map_ref in
    match instr with
    | Sink { srcs; kernel_info } ->
        Sink { srcs = map_ref_list srcs; kernel_info }
    | Group { srcs } -> Group { srcs = map_ref_list srcs }
    | After { src; deps } ->
        After { src = map_ref src; deps = map_ref_list deps }
    | Param _ | Define_local _ | Define_reg _ | Define_var _ | Const _ -> instr
    | Bufferize { src; idx; ranges; dtype; opts } ->
        Bufferize
          {
            src = map_ref src;
            idx = map_ref idx;
            ranges = map_ref_list ranges;
            dtype;
            opts;
          }
    | Index { ptr; idxs; gate; dtype } ->
        Index
          {
            ptr = map_ref ptr;
            idxs = map_ref_list idxs;
            gate = map_ref_opt gate;
            dtype;
          }
    | Ptrcat { srcs; dtype } -> Ptrcat { srcs = map_ref_list srcs; dtype }
    | Load { src; alt; dtype } ->
        Load { src = map_ref src; alt = map_ref_opt alt; dtype }
    | Store { dst; value; ranges } ->
        Store
          {
            dst = map_ref dst;
            value = map_ref value;
            ranges = map_ref_list ranges;
          }
    | Neg { src; dtype } -> Neg { src = map_ref src; dtype }
    | Exp2 { src; dtype } -> Exp2 { src = map_ref src; dtype }
    | Log2 { src; dtype } -> Log2 { src = map_ref src; dtype }
    | Sin { src; dtype } -> Sin { src = map_ref src; dtype }
    | Sqrt { src; dtype } -> Sqrt { src = map_ref src; dtype }
    | Recip { src; dtype } -> Recip { src = map_ref src; dtype }
    | Trunc { src; dtype } -> Trunc { src = map_ref src; dtype }
    | Add { lhs; rhs; dtype } ->
        Add { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Sub { lhs; rhs; dtype } ->
        Sub { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mul { lhs; rhs; dtype } ->
        Mul { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Fdiv { lhs; rhs; dtype } ->
        Fdiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Idiv { lhs; rhs; dtype } ->
        Idiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mod { lhs; rhs; dtype } ->
        Mod { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Max { lhs; rhs; dtype } ->
        Max { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Pow { lhs; rhs; dtype } ->
        Pow { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shl { lhs; rhs; dtype } ->
        Shl { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shr { lhs; rhs; dtype } ->
        Shr { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | And { lhs; rhs; dtype } ->
        And { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Or { lhs; rhs; dtype } ->
        Or { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Xor { lhs; rhs; dtype } ->
        Xor { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Threefry { lhs; rhs; dtype } ->
        Threefry { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmplt { lhs; rhs; dtype } ->
        Cmplt { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpeq { lhs; rhs; dtype } ->
        Cmpeq { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpne { lhs; rhs; dtype } ->
        Cmpne { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Where { cond; a; b; dtype } ->
        Where { cond = map_ref cond; a = map_ref a; b = map_ref b; dtype }
    | Mulacc { a; b; c; dtype } ->
        Mulacc { a = map_ref a; b = map_ref b; c = map_ref c; dtype }
    | Cast { src; dtype } -> Cast { src = map_ref src; dtype }
    | Bitcast { src; dtype } -> Bitcast { src = map_ref src; dtype }
    | Vectorize { srcs; dtype } -> Vectorize { srcs = map_ref_list srcs; dtype }
    | Cat { srcs; dtype } -> Cat { srcs = map_ref_list srcs; dtype }
    | Gep { src; idx; dtype } -> Gep { src = map_ref src; idx; dtype }
    | Range { size; dtype; axis; kind } ->
        Range { size = map_ref size; dtype; axis; kind }
    | End { value; ranges } ->
        End { value = map_ref value; ranges = map_ref_list ranges }
    | Barrier -> Barrier
    | Special { dim; size; dtype } ->
        Special { dim; size = map_ref size; dtype }
    | Reduce { op; src; ranges; dtype } ->
        Reduce { op; src = map_ref src; ranges = map_ref_list ranges; dtype }
    | Unroll { src; axes; dtype } -> Unroll { src = map_ref src; axes; dtype }
    | Contract { src; axes; dtype } ->
        Contract { src = map_ref src; axes; dtype }
    | Wmma
        {
          name;
          a;
          b;
          c;
          ranges;
          dtype;
          dims;
          dtype_in;
          dtype_out;
          device;
          threads;
          upcast_axes;
          reduce_axes;
        } ->
        Wmma
          {
            name;
            a = map_ref a;
            b = map_ref b;
            c = map_ref c;
            ranges = map_ref_list ranges;
            dtype;
            dims;
            dtype_in;
            dtype_out;
            device;
            threads;
            upcast_axes;
            reduce_axes;
          }
    | Custom { fmt; args } -> Custom { fmt; args = map_ref_list args }
    | Custom_inline { fmt; args; dtype } ->
        Custom_inline { fmt; args = map_ref_list args; dtype }

  let is_unary = function
    | Neg _ | Exp2 _ | Log2 _ | Sin _ | Sqrt _ | Recip _ | Trunc _ -> true
    | _ -> false

  let is_binary = function
    | Add _ | Sub _ | Mul _ | Fdiv _ | Idiv _ | Mod _ | Max _ | Pow _ | Shl _
    | Shr _ | And _ | Or _ | Xor _ | Threefry _ | Cmplt _ | Cmpeq _ | Cmpne _ ->
        true
    | _ -> false

  let is_ternary = function Where _ | Mulacc _ -> true | _ -> false
  let is_alu instr = is_unary instr || is_binary instr || is_ternary instr
  let intern program = intern_generic ~map_refs program

  (* ───── Validation ───── *)

  let validate (program : t) : unit =
    let fail i msg =
      failwith (Printf.sprintf "Kernel.validate: instruction %d: %s" i msg)
    in

    let rec get_dtype r =
      if r < 0 || r >= Array.length program then None
      else
        match program.(r) with
        | After { src; _ } -> get_dtype src
        | End { value; _ } -> get_dtype value
        | _ -> dtype_of program.(r)
    in

    let check_dtype_eq = check_dtype_eq fail in
    let check_dtype_match = check_dtype_match fail in
    let check_bool_scalar = check_bool_scalar fail get_dtype in
    let check_shift_rhs = check_shift_rhs fail get_dtype in
    let check_index_like = check_index_like fail get_dtype in

    let check_index_range i ~ctx r =
      match get_dtype r with
      | Some dt when dt.scalar = Dtype.Index && dt.count = 1 -> ()
      | Some _ -> fail i (Printf.sprintf "%s must be index scalar" ctx)
      | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
    in

    let rec index_base r =
      match program.(r) with
      | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } ->
          index_base src
      | Param _ | Define_local _ | Define_reg _ | Bufferize _ | Ptrcat _ ->
          Some r
      | _ -> None
    in

    let rec get_ptr_dtype r =
      match program.(r) with
      | Param { dtype; _ }
      | Define_local { dtype; _ }
      | Define_reg { dtype; _ }
      | Bufferize { dtype; _ }
      | Index { dtype; _ }
      | Ptrcat { dtype; _ } ->
          Some dtype
      | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } ->
          get_ptr_dtype src
      | _ -> None
    in

    let rec ptr_ref r =
      match program.(r) with
      | Index { dtype; gate; _ } -> Some (r, dtype, gate)
      | Ptrcat { dtype; _ } -> Some (r, dtype, None)
      | Param { dtype; _ }
      | Define_local { dtype; _ }
      | Define_reg { dtype; _ }
      | Bufferize { dtype; _ } ->
          Some (r, dtype, None)
      | Gep { src; dtype; _ } -> (
          match ptr_ref src with
          | Some (_, pty, gate) ->
              let pty = { pty with base = dtype } in
              Some (r, pty, gate)
          | None -> None)
      | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> ptr_ref src
      | _ -> None
    in

    let prod lst = List.fold_left ( * ) 1 lst in

    Array.iteri
      (fun i instr ->
        (* 1. SSA dominance: all refs must point to earlier instructions *)
        List.iter
          (fun r ->
            if r < 0 || r >= i then
              fail i
                (Printf.sprintf "references %%%d (out of bounds or forward)" r))
          (refs_of instr);

        (* 2. Instruction-specific checks *)
        match instr with
        | Sink _ | Group _ | After _ -> ()
        | Param { dtype; _ } ->
            if dtype.addrspace <> Dtype.Global then
              fail i "Param must have Global addrspace"
        | Define_local { dtype; _ } ->
            if dtype.addrspace <> Dtype.Local then
              fail i "Define_local must have Local addrspace"
        | Define_reg { dtype; _ } ->
            if dtype.addrspace <> Dtype.Reg then
              fail i "Define_reg must have Reg addrspace"
        | Define_var { lo; hi; dtype; _ } ->
            if dtype.count <> 1 then fail i "Define_var must be scalar";
            if not (Dtype.is_int dtype || dtype.scalar = Dtype.Index) then
              fail i "Define_var must be int/index";
            if lo > hi then fail i "Define_var bounds invalid (lo > hi)"
        | Bufferize { src; idx; ranges; dtype; opts } ->
            if dtype.addrspace <> opts.addrspace then
              fail i "Bufferize dtype addrspace mismatch";
            ignore src;
            check_index_like i ~ctx:"Bufferize idx" idx;
            List.iter (check_index_like i ~ctx:"Bufferize range") ranges
        | Const { value; dtype } -> (
            match value with
            | Bool _ ->
                if dtype.scalar <> Dtype.Bool then
                  fail i "Bool const must have bool dtype"
            | Int _ ->
                if not (Dtype.is_int dtype) then
                  fail i "Int const must have int/index dtype"
            | Float _ ->
                if not (Dtype.is_float dtype) then
                  fail i "Float const must have float dtype"
            | Invalid ->
                if dtype.scalar <> Dtype.Index then
                  fail i "Invalid const must have Index dtype")
        | Range { size; dtype; _ } ->
            if not (Dtype.is_int dtype) then fail i "Range must have int/index";
            if dtype.count <> 1 then fail i "Range must be scalar";
            check_dtype_eq i ~ctx:"Range size" ~expected:(Some dtype)
              ~got:(get_dtype size)
        | End { ranges; _ } ->
            List.iter (check_index_like i ~ctx:"End range") ranges
        | Barrier -> ()
        | Special { size; dtype; _ } ->
            if dtype.count <> 1 then fail i "Special must be scalar";
            if not (dtype.scalar = Dtype.Index || dtype.scalar = Dtype.Int32)
            then fail i "Special must be index or int32";
            check_dtype_eq i ~ctx:"Special size" ~expected:(Some dtype)
              ~got:(get_dtype size)
        | Index { ptr; idxs; gate; dtype } -> (
            if idxs = [] then fail i "Index must have at least one index";
            (match index_base ptr with
            | Some _ -> ()
            | None -> fail i "Index base must be a buffer define/bufferize");
            List.iter (check_index_range i ~ctx:"Index operand") idxs;
            Option.iter (check_bool_scalar i ~ctx:"Index gate") gate;
            match get_ptr_dtype ptr with
            | Some base
              when Dtype.equal base.base dtype.base
                   && base.addrspace = dtype.addrspace
                   && base.v = dtype.v && base.image = dtype.image ->
                ()
            | Some _ -> fail i "Index dtype must match base pointer type"
            | None -> fail i "Index base dtype not available")
        | Ptrcat { srcs; dtype } ->
            if srcs = [] then fail i "Ptrcat must have at least one source";
            let total_vcount = ref 0 in
            List.iter
              (fun r ->
                match ptr_ref r with
                | Some (_, pty, _) ->
                    if pty.addrspace <> dtype.addrspace then
                      fail i "Ptrcat addrspace mismatch";
                    if not (Dtype.equal pty.base dtype.base) then
                      fail i "Ptrcat base dtype mismatch";
                    if pty.image <> dtype.image then
                      fail i "Ptrcat image metadata mismatch";
                    total_vcount := !total_vcount + pty.base.count
                | None -> fail i "Ptrcat sources must be pointers")
              srcs;
            if !total_vcount <> dtype.v then fail i "Ptrcat vcount mismatch"
        | Load { src; alt; dtype } -> (
            match ptr_ref src with
            | Some (_, pty, gate) -> (
                check_dtype_eq i ~ctx:"Load dtype" ~expected:(Some pty.base)
                  ~got:(Some dtype);
                match alt with
                | None -> ()
                | Some alt_ref -> (
                    check_dtype_eq i ~ctx:"Load alt" ~expected:(Some dtype)
                      ~got:(get_dtype alt_ref);
                    match gate with
                    | None -> fail i "Load alt requires gated Index"
                    | Some _ -> ()))
            | None -> fail i "Load src must reference a pointer")
        | Store { dst; value; ranges } -> (
            List.iter (check_index_like i ~ctx:"Store range") ranges;
            match ptr_ref dst with
            | Some (_, pty, _) ->
                check_dtype_eq i ~ctx:"Store value" ~expected:(Some pty.base)
                  ~got:(get_dtype value)
            | None -> fail i "Store dst must reference a pointer")
        | Neg { src; dtype }
        | Exp2 { src; dtype }
        | Log2 { src; dtype }
        | Sin { src; dtype }
        | Sqrt { src; dtype }
        | Recip { src; dtype }
        | Trunc { src; dtype } ->
            check_dtype_eq i ~ctx:"Unary operand" ~expected:(Some dtype)
              ~got:(get_dtype src)
        | Add { lhs; rhs; dtype }
        | Sub { lhs; rhs; dtype }
        | Mul { lhs; rhs; dtype }
        | Fdiv { lhs; rhs; dtype }
        | Max { lhs; rhs; dtype }
        | Pow { lhs; rhs; dtype }
        | And { lhs; rhs; dtype }
        | Or { lhs; rhs; dtype }
        | Xor { lhs; rhs; dtype }
        | Threefry { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
              (get_dtype rhs);
            check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
              ~got:(get_dtype lhs)
        | Idiv { lhs; rhs; dtype } | Mod { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
              (get_dtype rhs);
            check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
              ~got:(get_dtype lhs);
            if not (Dtype.is_int dtype) then
              fail i "Idiv/Mod must have int/index dtype"
        | Shl { lhs; rhs; dtype } | Shr { lhs; rhs; dtype } ->
            check_dtype_eq i ~ctx:"Shift lhs" ~expected:(Some dtype)
              ~got:(get_dtype lhs);
            check_shift_rhs i rhs dtype;
            if not (Dtype.is_int dtype) then
              fail i "Shift must have int/index dtype"
        | Cmplt { lhs; rhs; dtype }
        | Cmpeq { lhs; rhs; dtype }
        | Cmpne { lhs; rhs; dtype } ->
            if dtype.scalar <> Dtype.Bool then
              fail i "Comparison must produce bool";
            check_dtype_match i ~ctx:"Comparison operands" (get_dtype lhs)
              (get_dtype rhs)
        | Where { cond; a; b; dtype } ->
            check_bool_scalar i ~ctx:"Where condition" cond;
            check_dtype_match i ~ctx:"Where arms" (get_dtype a) (get_dtype b);
            check_dtype_eq i ~ctx:"Where result" ~expected:(Some dtype)
              ~got:(get_dtype a)
        | Mulacc { a; b; c; dtype } ->
            check_dtype_match i ~ctx:"Mulacc a/b" (get_dtype a) (get_dtype b);
            check_dtype_match i ~ctx:"Mulacc a/c" (get_dtype a) (get_dtype c);
            check_dtype_eq i ~ctx:"Mulacc result" ~expected:(Some dtype)
              ~got:(get_dtype a)
        | Vectorize { srcs; dtype } ->
            if srcs = [] then fail i "Vectorize must have at least one operand";
            if dtype.count <> List.length srcs then
              fail i "Vectorize dtype count must match operand count";
            let scalar = dtype.scalar in
            List.iter
              (fun r ->
                match get_dtype r with
                | Some dt when dt.count = 1 && dt.scalar = scalar -> ()
                | Some _ -> fail i "Vectorize operands must be scalar and match"
                | None -> fail i "Vectorize operand dtype not available")
              srcs
        | Cat { srcs; dtype } ->
            if srcs = [] then fail i "Cat must have at least one operand";
            let total = ref 0 in
            List.iter
              (fun r ->
                match get_dtype r with
                | Some dt ->
                    if dt.scalar <> dtype.scalar then
                      fail i "Cat operand scalar mismatch";
                    total := !total + dt.count
                | None -> fail i "Cat operand dtype not available")
              srcs;
            if !total <> dtype.count then fail i "Cat count mismatch"
        | Gep { src; idx; dtype } -> (
            match get_dtype src with
            | Some dt when dt.count > 1 ->
                if idx < 0 || idx >= dt.count then
                  fail i "Gep index out of bounds";
                if dtype.count <> 1 || dtype.scalar <> dt.scalar then
                  fail i "Gep dtype must be scalar of source";
                ()
            | Some _ -> fail i "Gep source must be a vector"
            | None -> fail i "Gep source dtype not available")
        | Reduce { src; ranges; dtype; _ } ->
            check_dtype_eq i ~ctx:"Reduce src" ~expected:(Some dtype)
              ~got:(get_dtype src);
            List.iter (check_index_like i ~ctx:"Reduce range") ranges
        | Unroll { src; axes; dtype } -> (
            let expected = prod (List.map snd axes) * dtype.count in
            match get_dtype src with
            | Some dt when dt.count = expected -> ()
            | Some _ -> fail i "Unroll source count mismatch"
            | None -> fail i "Unroll source dtype not available")
        | Contract { src; axes; dtype } ->
            let expected = prod (List.map snd axes) in
            if dtype.count <> expected then
              fail i "Contract dtype count mismatch";
            ignore src
        | Wmma { a; b; c; ranges; _ } ->
            List.iter (check_index_like i ~ctx:"Wmma range") ranges;
            ignore a;
            ignore b;
            ignore c
        | Cast _ | Bitcast _ | Custom _ | Custom_inline _ -> ())
      program

  (* ───── Pretty Printing ───── *)

  let pp_const fmt = function
    | Bool b -> Format.fprintf fmt "%b" b
    | Int i -> Format.fprintf fmt "%d" i
    | Float f -> Format.fprintf fmt "%g" f
    | Invalid -> Format.fprintf fmt "invalid"

  let pp_reduce_op fmt (op : reduce_op) =
    Format.pp_print_string fmt
      (match op with Add -> "add" | Mul -> "mul" | Max -> "max")

  let pp_axes fmt axes =
    Format.pp_print_list ~pp_sep:pp_comma
      (fun fmt (a, s) -> Format.fprintf fmt "(%d, %d)" a s)
      fmt axes

  let pp_instr fmt = function
    | Sink { srcs; kernel_info = _ } ->
        Format.fprintf fmt "sink %a" pp_refs srcs
    | Group { srcs } -> Format.fprintf fmt "group %a" pp_refs srcs
    | After { src; deps } ->
        Format.fprintf fmt "after %a, deps=[%a]" pp_ref src pp_refs deps
    | Param { idx; dtype } ->
        Format.fprintf fmt "param %d : %a" idx pp_ptr dtype
    | Define_local { size; dtype } ->
        Format.fprintf fmt "define_local %a, size=%d" pp_ptr dtype size
    | Define_reg { size; dtype } ->
        Format.fprintf fmt "define_reg %a, size=%d" pp_ptr dtype size
    | Define_var { name; lo; hi; dtype } ->
        Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo
          hi
    | Bufferize { src; idx; ranges; dtype; _ } ->
        Format.fprintf fmt "bufferize %a, idx=%a, ranges=[%a] : %a" pp_ref src
          pp_ref idx pp_refs ranges pp_ptr dtype
    | Const { value; dtype } ->
        Format.fprintf fmt "const %a : %a" pp_const value Dtype.pp dtype
    | Index { ptr; idxs; gate; dtype } ->
        Format.fprintf fmt "index %a, %a%a : %a" pp_ref ptr pp_refs idxs
          (fun fmt -> function
            | None -> () | Some g -> Format.fprintf fmt " gate=%%%d" g)
          gate pp_ptr dtype
    | Ptrcat { srcs; dtype } ->
        Format.fprintf fmt "ptrcat %a : %a" pp_refs srcs pp_ptr dtype
    | Load { src; alt; dtype } ->
        Format.fprintf fmt "load %a%a : %a" pp_ref src
          (fun fmt -> function
            | None -> () | Some a -> Format.fprintf fmt " alt=%%%d" a)
          alt Dtype.pp dtype
    | Store { dst; value; ranges } ->
        Format.fprintf fmt "store %a, %a, ranges=[%a]" pp_ref dst pp_ref value
          pp_refs ranges
    | ( Neg { src; dtype }
      | Exp2 { src; dtype }
      | Log2 { src; dtype }
      | Sin { src; dtype }
      | Sqrt { src; dtype }
      | Recip { src; dtype }
      | Trunc { src; dtype }
      | Cast { src; dtype }
      | Bitcast { src; dtype } ) as instr ->
        let name =
          match instr with
          | Neg _ -> "neg"
          | Exp2 _ -> "exp2"
          | Log2 _ -> "log2"
          | Sin _ -> "sin"
          | Sqrt _ -> "sqrt"
          | Recip _ -> "recip"
          | Trunc _ -> "trunc"
          | Cast _ -> "cast"
          | Bitcast _ -> "bitcast"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a : %a" name pp_ref src Dtype.pp dtype
    | ( Add { lhs; rhs; dtype }
      | Sub { lhs; rhs; dtype }
      | Mul { lhs; rhs; dtype }
      | Fdiv { lhs; rhs; dtype }
      | Idiv { lhs; rhs; dtype }
      | Mod { lhs; rhs; dtype }
      | Max { lhs; rhs; dtype }
      | Pow { lhs; rhs; dtype }
      | Shl { lhs; rhs; dtype }
      | Shr { lhs; rhs; dtype }
      | And { lhs; rhs; dtype }
      | Or { lhs; rhs; dtype }
      | Xor { lhs; rhs; dtype }
      | Threefry { lhs; rhs; dtype }
      | Cmplt { lhs; rhs; dtype }
      | Cmpeq { lhs; rhs; dtype }
      | Cmpne { lhs; rhs; dtype } ) as instr ->
        let name =
          match instr with
          | Add _ -> "add"
          | Sub _ -> "sub"
          | Mul _ -> "mul"
          | Fdiv _ -> "fdiv"
          | Idiv _ -> "idiv"
          | Mod _ -> "mod"
          | Max _ -> "max"
          | Pow _ -> "pow"
          | Shl _ -> "shl"
          | Shr _ -> "shr"
          | And _ -> "and"
          | Or _ -> "or"
          | Xor _ -> "xor"
          | Threefry _ -> "threefry"
          | Cmplt _ -> "cmplt"
          | Cmpeq _ -> "cmpeq"
          | Cmpne _ -> "cmpne"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a, %a : %a" name pp_ref lhs pp_ref rhs Dtype.pp
          dtype
    | Where { cond; a; b; dtype } ->
        Format.fprintf fmt "where %a, %a, %a : %a" pp_ref cond pp_ref a pp_ref b
          Dtype.pp dtype
    | Mulacc { a; b; c; dtype } ->
        Format.fprintf fmt "mulacc %a, %a, %a : %a" pp_ref a pp_ref b pp_ref c
          Dtype.pp dtype
    | Vectorize { srcs; dtype } ->
        Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
    | Cat { srcs; dtype } ->
        Format.fprintf fmt "cat %a : %a" pp_refs srcs Dtype.pp dtype
    | Gep { src; idx; dtype } ->
        Format.fprintf fmt "gep %a, %d : %a" pp_ref src idx Dtype.pp dtype
    | Range { size; dtype; axis; kind } ->
        Format.fprintf fmt "range %a : %a [axis=%d, %a]" pp_ref size Dtype.pp
          dtype axis pp_axis_kind kind
    | End { value; ranges } ->
        Format.fprintf fmt "end %a, ranges=[%a]" pp_ref value pp_refs ranges
    | Barrier -> Format.fprintf fmt "barrier"
    | Special { dim; size; dtype } ->
        Format.fprintf fmt "special %a, %a : %a" pp_special_dim dim pp_ref size
          Dtype.pp dtype
    | Reduce { op; src; ranges; dtype } ->
        Format.fprintf fmt "reduce.%a %a, ranges=[%a] : %a" pp_reduce_op op
          pp_ref src pp_refs ranges Dtype.pp dtype
    | Unroll { src; axes; dtype } ->
        Format.fprintf fmt "unroll %a, axes=[%a] : %a" pp_ref src pp_axes axes
          Dtype.pp dtype
    | Contract { src; axes; dtype } ->
        Format.fprintf fmt "contract %a, axes=[%a] : %a" pp_ref src pp_axes axes
          Dtype.pp dtype
    | Wmma
        {
          name;
          a;
          b;
          c;
          ranges;
          dtype;
          dims = n, m, k;
          dtype_in;
          dtype_out;
          device;
          threads;
          _;
        } ->
        Format.fprintf fmt
          "wmma.%s %a, %a, %a, ranges=[%a] : %a [%dx%dx%d, %a -> %a, %s, \
           threads=%d]"
          name pp_ref a pp_ref b pp_ref c pp_refs ranges Dtype.pp dtype n m k
          Dtype.pp_scalar dtype_in Dtype.pp_scalar dtype_out device threads
    | Custom { fmt = f; args } ->
        Format.fprintf fmt "custom \"%s\" %a" f pp_refs args
    | Custom_inline { fmt = f; args; dtype } ->
        Format.fprintf fmt "custom_inline \"%s\" %a : %a" f pp_refs args
          Dtype.pp dtype

  let pp fmt t = pp_program pp_instr fmt t
end

module Tensor = struct
  (* ───── Types ───── *)

  (* Registry: type-safe global ID-to-value maps for metadata, kernel info, and
     gradient functions. Avoids embedding large values directly in the
     instruction array, keeping instructions small and comparable. *)

  module Registry = struct
    type 'a id = int
    type 'a t = { next : int ref; table : (int, 'a) Hashtbl.t }

    let create () = { next = ref 0; table = Hashtbl.create 16 }

    let add reg value =
      let id = !(reg.next) in
      reg.next := id + 1;
      Hashtbl.add reg.table id value;
      id

    let get reg id = Hashtbl.find reg.table id
  end

  module Metadata = struct
    type t = { name : string; caller : string; backward : bool }
    type id = t Registry.id

    let registry : t Registry.t = Registry.create ()
    let register value = Registry.add registry value
    let get id = Registry.get registry id
  end

  module Kernel_info = struct
    type t = Kernel.kernel_info
    type id = t Registry.id

    let registry : t Registry.t = Registry.create ()
    let register value = Registry.add registry value
    let get id = Registry.get registry id
  end

  module Grad_fxn = struct
    type t = { name : string }
    type id = t Registry.id

    let registry : t Registry.t = Registry.create ()
    let register value = Registry.add registry value
    let get id = Registry.get registry id
  end

  module Custom_kernel = struct
    type t = {
      name : string;
      grad : Grad_fxn.id option;
      ast : Kernel.t option;
      metadata : Metadata.id list;
    }

    type id = t Registry.id

    let registry : t Registry.t = Registry.create ()
    let register value = Registry.add registry value
    let get id = Registry.get registry id
  end

  type const = Bool of bool | Int of int | Float of float | Invalid
  type reduce_op = Add | Mul | Max
  type device = Single of string | Multi of string list
  type metadata = Metadata.id
  type kernel_info = Kernel_info.id
  type grad_fxn = Grad_fxn.id
  type custom_kernel = Custom_kernel.id

  type kernel = {
    ast : Kernel.t;
    metadata : metadata list;
    grad : grad_fxn option;
  }

  type ref = int

  type instr =
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
    | Group of { srcs : ref list }
    | After of { src : ref; deps : ref list; dtype : Dtype.t }
    | Unique of { id : int }
    | Lunique of { id : int }
    | Device of { device : device }
    | Buffer of { unique : ref; device : ref; size : int; dtype : Dtype.t }
    | Buffer_view of { src : ref; size : int; offset : int; dtype : Dtype.t }
    | Const of { value : const; dtype : Dtype.t; srcs : ref list }
    | Vconst of { values : const list; dtype : Dtype.t; srcs : ref list }
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
    | Bind of { var : ref; value : ref option; dtype : Dtype.t }
    | Param of {
        slot : int;
        dtype : Dtype.t;
        shape : ref option;
        device : ref option;
      }
    | Call of {
        fn : ref;
        args : ref list;
        grad : grad_fxn option;
        dtype : Dtype.t;
      }
    | Custom_kernel of { srcs : ref list; kernel : custom_kernel }
    | Kernel of { srcs : ref list; kernel : kernel }
    | Assign of {
        target : ref;
        value : ref;
        extras : ref list;
        dtype : Dtype.t;
      }
    | Detach of { src : ref; dtype : Dtype.t }
    | Contiguous of { src : ref; ranges : ref list; dtype : Dtype.t }
    | Contiguous_backward of { src : ref; dtype : Dtype.t }
    | Copy of { src : ref; device : ref; dtype : Dtype.t }
    | Allreduce of { src : ref; device : ref; op : reduce_op; dtype : Dtype.t }
    | Multi of { src : ref; axis : int; dtype : Dtype.t }
    | Mstack of { srcs : ref list; dtype : Dtype.t }
    | Mselect of { src : ref; index : int; dtype : Dtype.t }
    | Encdec of { srcs : ref list; shape : int list; dtype : Dtype.t }
    | Reduce_axis of {
        src : ref;
        op : reduce_op;
        axes : int list;
        dtype : Dtype.t;
      }
    | Reduce of {
        src : ref;
        ranges : ref list;
        op : reduce_op;
        dtype : Dtype.t;
      }
    | Reshape of { src : ref; shape : ref; dtype : Dtype.t }
    | Expand of { src : ref; shape : ref; dtype : Dtype.t }
    | Pad of { src : ref; before : ref; after : ref; dtype : Dtype.t }
    | Shrink of { src : ref; before : ref; after : ref; dtype : Dtype.t }
    | Permute of { src : ref; order : int list; dtype : Dtype.t }
    | Flip of { src : ref; dims : bool list; dtype : Dtype.t }
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
    | End of { value : ref; ranges : ref list }
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.t;
      }
    | Store of { dst : ref; value : ref }
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
    | Cast of { src : ref; dtype : Dtype.t }
    | Bitcast of { src : ref; dtype : Dtype.t }
    | Neg of { src : ref; dtype : Dtype.t }
    | Exp2 of { src : ref; dtype : Dtype.t }
    | Log2 of { src : ref; dtype : Dtype.t }
    | Sin of { src : ref; dtype : Dtype.t }
    | Sqrt of { src : ref; dtype : Dtype.t }
    | Recip of { src : ref; dtype : Dtype.t }
    | Trunc of { src : ref; dtype : Dtype.t }
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }

  type t = instr array

  (* ───── Inspection ───── *)

  let dtype_of = function
    | Buffer { dtype; _ }
    | Buffer_view { dtype; _ }
    | Const { dtype; _ }
    | Vconst { dtype; _ }
    | Define_var { dtype; _ }
    | Bind { dtype; _ }
    | Param { dtype; _ }
    | Call { dtype; _ }
    | After { dtype; _ }
    | Assign { dtype; _ }
    | Detach { dtype; _ }
    | Contiguous { dtype; _ }
    | Contiguous_backward { dtype; _ }
    | Copy { dtype; _ }
    | Allreduce { dtype; _ }
    | Multi { dtype; _ }
    | Mstack { dtype; _ }
    | Mselect { dtype; _ }
    | Encdec { dtype; _ }
    | Reduce_axis { dtype; _ }
    | Reduce { dtype; _ }
    | Reshape { dtype; _ }
    | Expand { dtype; _ }
    | Pad { dtype; _ }
    | Shrink { dtype; _ }
    | Permute { dtype; _ }
    | Flip { dtype; _ }
    | Range { dtype; _ }
    | Index { dtype; _ }
    | Vectorize { dtype; _ }
    | Cast { dtype; _ }
    | Bitcast { dtype; _ }
    | Neg { dtype; _ }
    | Exp2 { dtype; _ }
    | Log2 { dtype; _ }
    | Sin { dtype; _ }
    | Sqrt { dtype; _ }
    | Recip { dtype; _ }
    | Trunc { dtype; _ }
    | Add { dtype; _ }
    | Sub { dtype; _ }
    | Mul { dtype; _ }
    | Fdiv { dtype; _ }
    | Idiv { dtype; _ }
    | Mod { dtype; _ }
    | Max { dtype; _ }
    | Pow { dtype; _ }
    | Shl { dtype; _ }
    | Shr { dtype; _ }
    | And { dtype; _ }
    | Or { dtype; _ }
    | Xor { dtype; _ }
    | Threefry { dtype; _ }
    | Cmplt { dtype; _ }
    | Cmpeq { dtype; _ }
    | Cmpne { dtype; _ }
    | Where { dtype; _ }
    | Mulacc { dtype; _ } ->
        Some dtype
    | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | Custom_kernel _
    | Kernel _ | End _ | Store _ ->
        None

  let refs_of = function
    | Sink { srcs; _ } -> srcs
    | Group { srcs } -> srcs
    | After { src; deps; _ } -> src :: deps
    | Unique _ | Lunique _ | Device _ | Define_var _ -> []
    | Buffer { unique; device; _ } -> [ unique; device ]
    | Buffer_view { src; _ } -> [ src ]
    | Const { srcs; _ } | Vconst { srcs; _ } -> srcs
    | Bind { var; value; _ } -> var :: Option.to_list value
    | Param { shape; device; _ } -> Option.to_list shape @ Option.to_list device
    | Call { fn; args; _ } -> fn :: args
    | Custom_kernel { srcs; _ } -> srcs
    | Kernel { srcs; _ } -> srcs
    | Assign { target; value; extras; _ } -> target :: value :: extras
    | Detach { src; _ } -> [ src ]
    | Contiguous { src; ranges; _ } -> src :: ranges
    | Contiguous_backward { src; _ } -> [ src ]
    | Copy { src; device; _ } -> [ src; device ]
    | Allreduce { src; device; _ } -> [ src; device ]
    | Multi { src; _ } -> [ src ]
    | Mstack { srcs; _ } -> srcs
    | Mselect { src; _ } -> [ src ]
    | Encdec { srcs; _ } -> srcs
    | Reduce_axis { src; _ } -> [ src ]
    | Reduce { src; ranges; _ } -> src :: ranges
    | Reshape { src; shape; _ } -> [ src; shape ]
    | Expand { src; shape; _ } -> [ src; shape ]
    | Pad { src; before; after; _ } -> [ src; before; after ]
    | Shrink { src; before; after; _ } -> [ src; before; after ]
    | Permute { src; _ } -> [ src ]
    | Flip { src; _ } -> [ src ]
    | Range { size; _ } -> [ size ]
    | End { value; ranges } -> value :: ranges
    | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
    | Store { dst; value } -> [ dst; value ]
    | Vectorize { srcs; _ } -> srcs
    | Cast { src; _ } -> [ src ]
    | Bitcast { src; _ } -> [ src ]
    | Neg { src; _ }
    | Exp2 { src; _ }
    | Log2 { src; _ }
    | Sin { src; _ }
    | Sqrt { src; _ }
    | Recip { src; _ }
    | Trunc { src; _ } ->
        [ src ]
    | Add { lhs; rhs; _ }
    | Sub { lhs; rhs; _ }
    | Mul { lhs; rhs; _ }
    | Fdiv { lhs; rhs; _ }
    | Idiv { lhs; rhs; _ }
    | Mod { lhs; rhs; _ }
    | Max { lhs; rhs; _ }
    | Pow { lhs; rhs; _ }
    | Shl { lhs; rhs; _ }
    | Shr { lhs; rhs; _ }
    | And { lhs; rhs; _ }
    | Or { lhs; rhs; _ }
    | Xor { lhs; rhs; _ }
    | Threefry { lhs; rhs; _ }
    | Cmplt { lhs; rhs; _ }
    | Cmpeq { lhs; rhs; _ }
    | Cmpne { lhs; rhs; _ } ->
        [ lhs; rhs ]
    | Where { cond; a; b; _ } -> [ cond; a; b ]
    | Mulacc { a; b; c; _ } -> [ a; b; c ]

  let map_refs map_ref (instr : instr) : instr =
    let map_ref_list = List.map map_ref in
    let map_ref_opt = Option.map map_ref in
    match instr with
    | Sink { srcs; kernel_info } ->
        Sink { srcs = map_ref_list srcs; kernel_info }
    | Group { srcs } -> Group { srcs = map_ref_list srcs }
    | After { src; deps; dtype } ->
        After { src = map_ref src; deps = map_ref_list deps; dtype }
    | Unique _ | Lunique _ | Device _ | Define_var _ -> instr
    | Buffer { unique; device; size; dtype } ->
        Buffer { unique = map_ref unique; device = map_ref device; size; dtype }
    | Buffer_view { src; size; offset; dtype } ->
        Buffer_view { src = map_ref src; size; offset; dtype }
    | Const { value; dtype; srcs } ->
        Const { value; dtype; srcs = map_ref_list srcs }
    | Vconst { values; dtype; srcs } ->
        Vconst { values; dtype; srcs = map_ref_list srcs }
    | Bind { var; value; dtype } ->
        Bind { var = map_ref var; value = map_ref_opt value; dtype }
    | Param { slot; dtype; shape; device } ->
        Param
          {
            slot;
            dtype;
            shape = map_ref_opt shape;
            device = map_ref_opt device;
          }
    | Call { fn; args; grad; dtype } ->
        Call { fn = map_ref fn; args = map_ref_list args; grad; dtype }
    | Custom_kernel { srcs; kernel } ->
        Custom_kernel { srcs = map_ref_list srcs; kernel }
    | Kernel { srcs; kernel } -> Kernel { srcs = map_ref_list srcs; kernel }
    | Assign { target; value; extras; dtype } ->
        Assign
          {
            target = map_ref target;
            value = map_ref value;
            extras = map_ref_list extras;
            dtype;
          }
    | Detach { src; dtype } -> Detach { src = map_ref src; dtype }
    | Contiguous { src; ranges; dtype } ->
        Contiguous { src = map_ref src; ranges = map_ref_list ranges; dtype }
    | Contiguous_backward { src; dtype } ->
        Contiguous_backward { src = map_ref src; dtype }
    | Copy { src; device; dtype } ->
        Copy { src = map_ref src; device = map_ref device; dtype }
    | Allreduce { src; device; op; dtype } ->
        Allreduce { src = map_ref src; device = map_ref device; op; dtype }
    | Multi { src; axis; dtype } -> Multi { src = map_ref src; axis; dtype }
    | Mstack { srcs; dtype } -> Mstack { srcs = map_ref_list srcs; dtype }
    | Mselect { src; index; dtype } ->
        Mselect { src = map_ref src; index; dtype }
    | Encdec { srcs; shape; dtype } ->
        Encdec { srcs = map_ref_list srcs; shape; dtype }
    | Reduce_axis { src; op; axes; dtype } ->
        Reduce_axis { src = map_ref src; op; axes; dtype }
    | Reduce { src; ranges; op; dtype } ->
        Reduce { src = map_ref src; ranges = map_ref_list ranges; op; dtype }
    | Reshape { src; shape; dtype } ->
        Reshape { src = map_ref src; shape = map_ref shape; dtype }
    | Expand { src; shape; dtype } ->
        Expand { src = map_ref src; shape = map_ref shape; dtype }
    | Pad { src; before; after; dtype } ->
        Pad
          {
            src = map_ref src;
            before = map_ref before;
            after = map_ref after;
            dtype;
          }
    | Shrink { src; before; after; dtype } ->
        Shrink
          {
            src = map_ref src;
            before = map_ref before;
            after = map_ref after;
            dtype;
          }
    | Permute { src; order; dtype } ->
        Permute { src = map_ref src; order; dtype }
    | Flip { src; dims; dtype } -> Flip { src = map_ref src; dims; dtype }
    | Range { size; dtype; axis; kind } ->
        Range { size = map_ref size; dtype; axis; kind }
    | End { value; ranges } ->
        End { value = map_ref value; ranges = map_ref_list ranges }
    | Index { ptr; idxs; gate; dtype } ->
        Index
          {
            ptr = map_ref ptr;
            idxs = map_ref_list idxs;
            gate = map_ref_opt gate;
            dtype;
          }
    | Store { dst; value } -> Store { dst = map_ref dst; value = map_ref value }
    | Vectorize { srcs; dtype } -> Vectorize { srcs = map_ref_list srcs; dtype }
    | Cast { src; dtype } -> Cast { src = map_ref src; dtype }
    | Bitcast { src; dtype } -> Bitcast { src = map_ref src; dtype }
    | Neg { src; dtype } -> Neg { src = map_ref src; dtype }
    | Exp2 { src; dtype } -> Exp2 { src = map_ref src; dtype }
    | Log2 { src; dtype } -> Log2 { src = map_ref src; dtype }
    | Sin { src; dtype } -> Sin { src = map_ref src; dtype }
    | Sqrt { src; dtype } -> Sqrt { src = map_ref src; dtype }
    | Recip { src; dtype } -> Recip { src = map_ref src; dtype }
    | Trunc { src; dtype } -> Trunc { src = map_ref src; dtype }
    | Add { lhs; rhs; dtype } ->
        Add { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Sub { lhs; rhs; dtype } ->
        Sub { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mul { lhs; rhs; dtype } ->
        Mul { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Fdiv { lhs; rhs; dtype } ->
        Fdiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Idiv { lhs; rhs; dtype } ->
        Idiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mod { lhs; rhs; dtype } ->
        Mod { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Max { lhs; rhs; dtype } ->
        Max { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Pow { lhs; rhs; dtype } ->
        Pow { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shl { lhs; rhs; dtype } ->
        Shl { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shr { lhs; rhs; dtype } ->
        Shr { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | And { lhs; rhs; dtype } ->
        And { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Or { lhs; rhs; dtype } ->
        Or { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Xor { lhs; rhs; dtype } ->
        Xor { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Threefry { lhs; rhs; dtype } ->
        Threefry { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmplt { lhs; rhs; dtype } ->
        Cmplt { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpeq { lhs; rhs; dtype } ->
        Cmpeq { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpne { lhs; rhs; dtype } ->
        Cmpne { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Where { cond; a; b; dtype } ->
        Where { cond = map_ref cond; a = map_ref a; b = map_ref b; dtype }
    | Mulacc { a; b; c; dtype } ->
        Mulacc { a = map_ref a; b = map_ref b; c = map_ref c; dtype }

  let is_unary = function
    | Neg _ | Exp2 _ | Log2 _ | Sin _ | Sqrt _ | Recip _ | Trunc _ -> true
    | _ -> false

  let is_binary = function
    | Add _ | Sub _ | Mul _ | Fdiv _ | Idiv _ | Mod _ | Max _ | Pow _ | Shl _
    | Shr _ | And _ | Or _ | Xor _ | Threefry _ | Cmplt _ | Cmpeq _ | Cmpne _ ->
        true
    | _ -> false

  let is_ternary = function Where _ | Mulacc _ -> true | _ -> false
  let is_alu instr = is_unary instr || is_binary instr || is_ternary instr
  let intern program = intern_generic ~map_refs program

  (* ───── Validation ───── *)

  let validate (program : t) : unit =
    let fail i msg =
      failwith (Printf.sprintf "Tensor.validate: instruction %d: %s" i msg)
    in
    let get_dtype r =
      if r < 0 || r >= Array.length program then None else dtype_of program.(r)
    in
    let check_dtype_eq = check_dtype_eq fail in
    let check_dtype_match = check_dtype_match fail in
    let check_bool_scalar = check_bool_scalar fail get_dtype in
    let check_index_like = check_index_like fail get_dtype in
    let check_shift_rhs = check_shift_rhs fail get_dtype in
    let check_index_scalar i ~ctx r =
      match get_dtype r with
      | Some dt when dt.scalar = Dtype.Index && dt.count = 1 -> ()
      | Some _ -> fail i (Printf.sprintf "%s must be index scalar" ctx)
      | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
    in
    let check_index_vector i ~ctx r =
      match get_dtype r with
      | Some dt when dt.scalar = Dtype.Index && dt.count >= 0 -> ()
      | Some _ -> fail i (Printf.sprintf "%s must be index vector" ctx)
      | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
    in
    let check_src_dtype i ~ctx src dtype =
      check_dtype_eq i ~ctx ~expected:(Some dtype) ~got:(get_dtype src)
    in
    (* Extract concrete int list from a shape ref (Vectorize of Const), or None
       when the shape contains symbolic expressions. *)
    let try_extract_shape r =
      if r < 0 || r >= Array.length program then None
      else
        match program.(r) with
        | Vectorize { srcs; _ } ->
            let dims =
              List.filter_map
                (fun s ->
                  if s < 0 || s >= Array.length program then None
                  else
                    match program.(s) with
                    | Const { value = Int n; _ } -> Some n
                    | _ -> None)
                srcs
            in
            if List.length dims = List.length srcs then Some dims else None
        | Const { value = Int n; _ } -> Some [ n ]
        | _ -> None
    in
    Array.iteri
      (fun i instr ->
        List.iter
          (fun r ->
            if r < 0 || r >= i then
              fail i
                (Printf.sprintf "references %%%d (out of bounds or forward)" r))
          (refs_of instr);
        match instr with
        | Sink _ | Group _ | Unique _ | Lunique _ | Device _ -> ()
        | Buffer { unique; device; size; _ } -> (
            if size < 0 then fail i "Buffer size must be non-negative";
            (match program.(unique) with
            | Unique _ | Lunique _ -> ()
            | _ -> fail i "Buffer unique must reference Unique/Lunique");
            match program.(device) with
            | Device _ -> ()
            | _ -> fail i "Buffer device must reference Device")
        | Buffer_view { src; size; offset; _ } -> (
            if size < 0 then fail i "Buffer_view size must be non-negative";
            if offset < 0 then fail i "Buffer_view offset must be non-negative";
            match program.(src) with
            | Index _ -> ()
            | _ -> fail i "Buffer_view src must be Index")
        | Const { value; dtype; _ } -> (
            match value with
            | Bool _ ->
                if dtype.scalar <> Dtype.Bool then
                  fail i "Bool const must have bool dtype"
            | Int _ ->
                if not (Dtype.is_int dtype) then
                  fail i "Int const must have int/index dtype"
            | Float _ ->
                if not (Dtype.is_float dtype) then
                  fail i "Float const must have float dtype"
            | Invalid ->
                if dtype.scalar <> Dtype.Index then
                  fail i "Invalid const must have Index dtype")
        | Vconst { values; dtype; _ } ->
            if dtype.count <> List.length values then
              fail i "Vconst values must match vector width";
            List.iter
              (function
                | Bool _ ->
                    if dtype.scalar <> Dtype.Bool then
                      fail i "Vconst bool elements must have bool dtype"
                | Int _ ->
                    if not (Dtype.is_int dtype) then
                      fail i "Vconst int elements must have int/index dtype"
                | Float _ ->
                    if not (Dtype.is_float dtype) then
                      fail i "Vconst float elements must have float dtype"
                | Invalid ->
                    if dtype.scalar <> Dtype.Index then
                      fail i "Vconst invalid requires Index dtype")
              values
        | Define_var { lo; hi; dtype; _ } ->
            if dtype.count <> 1 then fail i "Define_var must be scalar";
            if not (Dtype.is_int dtype || dtype.scalar = Dtype.Index) then
              fail i "Define_var must be int/index";
            if lo > hi then fail i "Define_var bounds invalid (lo > hi)"
        | Bind { var; value; dtype } -> (
            (match program.(var) with
            | Define_var { dtype = vdt; _ } ->
                check_dtype_eq i ~ctx:"Bind dtype" ~expected:(Some vdt)
                  ~got:(Some dtype)
            | _ -> fail i "Bind var must reference Define_var");
            match value with
            | None -> ()
            | Some v -> check_src_dtype i ~ctx:"Bind value" v dtype)
        | Param { shape; device; _ } ->
            Option.iter (check_index_vector i ~ctx:"Param shape") shape;
            Option.iter
              (fun d ->
                match program.(d) with
                | Device _ -> ()
                | _ -> fail i "Param device must reference Device")
              device
        | Call { fn; dtype; _ } -> check_src_dtype i ~ctx:"Call dtype" fn dtype
        | Custom_kernel _ -> ()
        | Kernel _ -> ()
        | After { src; dtype; _ } ->
            check_src_dtype i ~ctx:"After dtype" src dtype
        | Assign { target; value; dtype; _ } ->
            check_src_dtype i ~ctx:"Assign target" target dtype;
            check_src_dtype i ~ctx:"Assign value" value dtype
        | Detach { src; dtype }
        | Contiguous_backward { src; dtype }
        | Multi { src; dtype; _ }
        | Mselect { src; dtype; _ }
        | Encdec { srcs = src :: _; dtype; _ } ->
            check_src_dtype i ~ctx:"dtype" src dtype
        | Reduce_axis { src; axes; dtype; _ } ->
            check_src_dtype i ~ctx:"Reduce_axis dtype" src dtype;
            if axes = [] then fail i "Reduce_axis must have at least one axis";
            if List.length axes <> List.length (List.sort_uniq compare axes)
            then fail i "Reduce_axis axes must be unique"
        | Permute { src; order; dtype } ->
            check_src_dtype i ~ctx:"Permute dtype" src dtype;
            let n = List.length order in
            if List.sort compare order <> List.init n Fun.id then
              fail i
                (Printf.sprintf
                   "Permute order must be a valid permutation of 0..%d" (n - 1))
        | Flip { src; dtype; _ } ->
            check_src_dtype i ~ctx:"Flip dtype" src dtype
        | Copy { src; device; dtype } -> (
            check_src_dtype i ~ctx:"Copy dtype" src dtype;
            match program.(device) with
            | Device _ -> ()
            | _ -> fail i "Copy device must reference Device")
        | Allreduce { src; device; dtype; _ } -> (
            check_src_dtype i ~ctx:"Allreduce dtype" src dtype;
            match program.(device) with
            | Device _ -> ()
            | _ -> fail i "Allreduce device must reference Device")
        | Encdec { srcs = []; _ } -> ()
        | Contiguous { src; ranges; dtype } ->
            check_src_dtype i ~ctx:"Contiguous dtype" src dtype;
            List.iter (check_index_scalar i ~ctx:"Contiguous range") ranges
        | Mstack { srcs; dtype } ->
            if srcs = [] then fail i "Mstack must have srcs";
            List.iter
              (fun r -> check_src_dtype i ~ctx:"Mstack src" r dtype)
              srcs
        | Reduce { src; ranges; dtype; _ } ->
            check_src_dtype i ~ctx:"Reduce dtype" src dtype;
            List.iter (check_index_scalar i ~ctx:"Reduce range") ranges
        | Reshape { src; shape; dtype } -> (
            check_src_dtype i ~ctx:"Reshape dtype" src dtype;
            check_index_vector i ~ctx:"Shape" shape;
            match try_extract_shape shape with
            | Some dims ->
                if List.exists (fun d -> d < 0) dims then
                  fail i "Reshape shape must not contain negative numbers"
            | None -> ())
        | Expand { src; shape; dtype } ->
            check_src_dtype i ~ctx:"Expand dtype" src dtype;
            check_index_vector i ~ctx:"Shape" shape
        | Pad { src; before; after; dtype }
        | Shrink { src; before; after; dtype } -> (
            check_src_dtype i ~ctx:"Pad/Shrink dtype" src dtype;
            check_index_vector i ~ctx:"Pad/Shrink before" before;
            check_index_vector i ~ctx:"Pad/Shrink after" after;
            match (get_dtype before, get_dtype after) with
            | Some bdt, Some adt when bdt.count = adt.count -> ()
            | Some _, Some _ ->
                fail i "Pad/Shrink before/after vector width mismatch"
            | _ -> fail i "Pad/Shrink dtype not available")
        | Range { size; dtype; _ } ->
            if not (Dtype.is_int dtype) then fail i "Range must have int/index";
            if dtype.count <> 1 then fail i "Range must be scalar";
            check_src_dtype i ~ctx:"Range size" size dtype
        | End { ranges; _ } ->
            List.iter (check_index_like i ~ctx:"End range") ranges
        | Index { idxs; gate; _ } ->
            if idxs = [] then fail i "Index must have at least one index";
            List.iter (check_index_scalar i ~ctx:"Index operand") idxs;
            Option.iter (check_bool_scalar i ~ctx:"Index gate") gate
        | Store { dst = _; value = _ } -> ()
        | Vectorize { srcs; dtype } ->
            if srcs = [] then fail i "Vectorize must have at least one operand";
            if dtype.count <> List.length srcs then
              fail i "Vectorize dtype count must match operand count";
            let scalar = dtype.scalar in
            List.iter
              (fun r ->
                match get_dtype r with
                | Some dt when dt.count = 1 && dt.scalar = scalar -> ()
                | Some _ -> fail i "Vectorize operands must be scalar and match"
                | None -> fail i "Vectorize operand dtype not available")
              srcs
        | Cast { src; dtype } -> (
            match get_dtype src with
            | Some sdt ->
                if sdt.count <> dtype.count then
                  fail i "Cast must preserve vector width"
            | None -> fail i "Cast src dtype not available")
        | Bitcast { src; _ } -> (
            match get_dtype src with
            | Some _ -> ()
            | None -> fail i "Bitcast src dtype not available")
        | Neg { src; dtype }
        | Exp2 { src; dtype }
        | Log2 { src; dtype }
        | Sin { src; dtype }
        | Sqrt { src; dtype }
        | Recip { src; dtype }
        | Trunc { src; dtype } ->
            check_src_dtype i ~ctx:"Unary operand" src dtype
        | Add { lhs; rhs; dtype }
        | Sub { lhs; rhs; dtype }
        | Mul { lhs; rhs; dtype }
        | Fdiv { lhs; rhs; dtype }
        | Max { lhs; rhs; dtype }
        | Pow { lhs; rhs; dtype }
        | And { lhs; rhs; dtype }
        | Or { lhs; rhs; dtype }
        | Xor { lhs; rhs; dtype }
        | Threefry { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
              (get_dtype rhs);
            check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
              ~got:(get_dtype lhs)
        | Idiv { lhs; rhs; dtype } | Mod { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
              (get_dtype rhs);
            check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
              ~got:(get_dtype lhs);
            if not (Dtype.is_int dtype) then
              fail i "Idiv/Mod must have int/index dtype"
        | Shl { lhs; rhs; dtype } | Shr { lhs; rhs; dtype } ->
            check_src_dtype i ~ctx:"Shift lhs" lhs dtype;
            check_shift_rhs i rhs dtype;
            if not (Dtype.is_int dtype) then
              fail i "Shift must have int/index dtype"
        | Cmplt { lhs; rhs; dtype }
        | Cmpeq { lhs; rhs; dtype }
        | Cmpne { lhs; rhs; dtype } ->
            if dtype.scalar <> Dtype.Bool then
              fail i "Comparison must produce bool";
            check_dtype_match i ~ctx:"Comparison operands" (get_dtype lhs)
              (get_dtype rhs)
        | Where { cond; a; b; dtype } ->
            check_bool_scalar i ~ctx:"Where condition" cond;
            check_dtype_match i ~ctx:"Where arms" (get_dtype a) (get_dtype b);
            check_dtype_eq i ~ctx:"Where result" ~expected:(Some dtype)
              ~got:(get_dtype a)
        | Mulacc { a; b; c; dtype } ->
            check_dtype_match i ~ctx:"Mulacc a/b" (get_dtype a) (get_dtype b);
            check_dtype_match i ~ctx:"Mulacc a/c" (get_dtype a) (get_dtype c);
            check_dtype_eq i ~ctx:"Mulacc result" ~expected:(Some dtype)
              ~got:(get_dtype a))
      program

  (* ───── Pretty Printing ───── *)

  let pp_const fmt = function
    | Bool b -> Format.fprintf fmt "%b" b
    | Int i -> Format.fprintf fmt "%d" i
    | Float f -> Format.fprintf fmt "%g" f
    | Invalid -> Format.fprintf fmt "invalid"

  let pp_reduce_op fmt (op : reduce_op) =
    Format.pp_print_string fmt
      (match op with Add -> "add" | Mul -> "mul" | Max -> "max")

  let pp_device fmt = function
    | Single s -> Format.pp_print_string fmt s
    | Multi devs ->
        Format.fprintf fmt "[%a]"
          (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_string)
          devs

  let pp_instr fmt = function
    | Sink { srcs; kernel_info = _ } ->
        Format.fprintf fmt "sink %a" pp_refs srcs
    | Group { srcs } -> Format.fprintf fmt "group %a" pp_refs srcs
    | After { src; deps; dtype } ->
        Format.fprintf fmt "after %a, deps=[%a] : %a" pp_ref src pp_refs deps
          Dtype.pp dtype
    | Unique { id } -> Format.fprintf fmt "unique #%d" id
    | Lunique { id } -> Format.fprintf fmt "lunique #%d" id
    | Device { device } -> Format.fprintf fmt "device %a" pp_device device
    | Buffer { unique; device; size; dtype } ->
        Format.fprintf fmt "buffer %a, %a, size=%d : %a" pp_ref unique pp_ref
          device size Dtype.pp dtype
    | Buffer_view { src; size; offset; dtype } ->
        Format.fprintf fmt "buffer_view %a, size=%d, offset=%d : %a" pp_ref src
          size offset Dtype.pp dtype
    | Const { value; dtype; srcs } ->
        Format.fprintf fmt "const %a : %a" pp_const value Dtype.pp dtype;
        if srcs <> [] then Format.fprintf fmt " [%a]" pp_refs srcs
    | Vconst { values; dtype; srcs } ->
        Format.fprintf fmt "vconst [%a] : %a"
          (Format.pp_print_list ~pp_sep:pp_comma pp_const)
          values Dtype.pp dtype;
        if srcs <> [] then Format.fprintf fmt " [%a]" pp_refs srcs
    | Define_var { name; lo; hi; dtype } ->
        Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo
          hi
    | Bind { var; value; dtype } ->
        Format.fprintf fmt "bind %a" pp_ref var;
        (match value with
        | Some v -> Format.fprintf fmt " = %a" pp_ref v
        | None -> ());
        Format.fprintf fmt " : %a" Dtype.pp dtype
    | Param { slot; dtype; shape; device } -> (
        Format.fprintf fmt "param %d : %a" slot Dtype.pp dtype;
        (match shape with
        | Some s -> Format.fprintf fmt " shape=%a" pp_ref s
        | None -> ());
        match device with
        | Some d -> Format.fprintf fmt " device=%a" pp_ref d
        | None -> ())
    | Call { fn; args; dtype; _ } ->
        Format.fprintf fmt "call %a(%a) : %a" pp_ref fn pp_refs args Dtype.pp
          dtype
    | Custom_kernel { srcs; _ } ->
        Format.fprintf fmt "custom_kernel %a" pp_refs srcs
    | Kernel { srcs; kernel } ->
        Format.fprintf fmt "kernel [%d instrs] %a" (Array.length kernel.ast)
          pp_refs srcs
    | Assign { target; value; extras; dtype } ->
        Format.fprintf fmt "assign %a, %a" pp_ref target pp_ref value;
        if extras <> [] then Format.fprintf fmt ", extras=[%a]" pp_refs extras;
        Format.fprintf fmt " : %a" Dtype.pp dtype
    | Detach { src; dtype } ->
        Format.fprintf fmt "detach %a : %a" pp_ref src Dtype.pp dtype
    | Contiguous { src; ranges; dtype } ->
        Format.fprintf fmt "contiguous %a" pp_ref src;
        if ranges <> [] then Format.fprintf fmt ", ranges=[%a]" pp_refs ranges;
        Format.fprintf fmt " : %a" Dtype.pp dtype
    | Contiguous_backward { src; dtype } ->
        Format.fprintf fmt "contiguous_backward %a : %a" pp_ref src Dtype.pp
          dtype
    | Copy { src; device; dtype } ->
        Format.fprintf fmt "copy %a, %a : %a" pp_ref src pp_ref device Dtype.pp
          dtype
    | Allreduce { src; device; op; dtype } ->
        Format.fprintf fmt "allreduce.%a %a, %a : %a" pp_reduce_op op pp_ref src
          pp_ref device Dtype.pp dtype
    | Multi { src; axis; dtype } ->
        Format.fprintf fmt "multi %a, axis=%d : %a" pp_ref src axis Dtype.pp
          dtype
    | Mstack { srcs; dtype } ->
        Format.fprintf fmt "mstack %a : %a" pp_refs srcs Dtype.pp dtype
    | Mselect { src; index; dtype } ->
        Format.fprintf fmt "mselect %a, index=%d : %a" pp_ref src index Dtype.pp
          dtype
    | Encdec { srcs; shape; dtype } ->
        Format.fprintf fmt "encdec %a, shape=[%a] : %a" pp_refs srcs
          (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_int)
          shape Dtype.pp dtype
    | Reduce_axis { src; op; axes; dtype } ->
        Format.fprintf fmt "reduce_axis.%a %a, axes=[%a] : %a" pp_reduce_op op
          pp_ref src
          (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_int)
          axes Dtype.pp dtype
    | Reduce { src; ranges; op; dtype } ->
        Format.fprintf fmt "reduce.%a %a, ranges=[%a] : %a" pp_reduce_op op
          pp_ref src pp_refs ranges Dtype.pp dtype
    | Reshape { src; shape; dtype } ->
        Format.fprintf fmt "reshape %a, %a : %a" pp_ref src pp_ref shape
          Dtype.pp dtype
    | Expand { src; shape; dtype } ->
        Format.fprintf fmt "expand %a, %a : %a" pp_ref src pp_ref shape Dtype.pp
          dtype
    | Pad { src; before; after; dtype } ->
        Format.fprintf fmt "pad %a, %a, %a : %a" pp_ref src pp_ref before pp_ref
          after Dtype.pp dtype
    | Shrink { src; before; after; dtype } ->
        Format.fprintf fmt "shrink %a, %a, %a : %a" pp_ref src pp_ref before
          pp_ref after Dtype.pp dtype
    | Permute { src; order; dtype } ->
        Format.fprintf fmt "permute %a, [%a] : %a" pp_ref src
          (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_int)
          order Dtype.pp dtype
    | Flip { src; dims; dtype } ->
        Format.fprintf fmt "flip %a, [%a] : %a" pp_ref src
          (Format.pp_print_list ~pp_sep:pp_comma (fun fmt b ->
               Format.fprintf fmt "%b" b))
          dims Dtype.pp dtype
    | Range { size; dtype; axis; kind } ->
        Format.fprintf fmt "range %a : %a [axis=%d, %a]" pp_ref size Dtype.pp
          dtype axis pp_axis_kind kind
    | End { value; ranges } ->
        Format.fprintf fmt "end %a, ranges=[%a]" pp_ref value pp_refs ranges
    | Index { ptr; idxs; gate; dtype } ->
        Format.fprintf fmt "index %a, %a%a : %a" pp_ref ptr pp_refs idxs
          (fun fmt -> function
            | None -> () | Some g -> Format.fprintf fmt " gate=%%%d" g)
          gate Dtype.pp dtype
    | Store { dst; value } ->
        Format.fprintf fmt "store %a, %a" pp_ref dst pp_ref value
    | Vectorize { srcs; dtype } ->
        Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
    | ( Cast { src; dtype }
      | Bitcast { src; dtype }
      | Neg { src; dtype }
      | Exp2 { src; dtype }
      | Log2 { src; dtype }
      | Sin { src; dtype }
      | Sqrt { src; dtype }
      | Recip { src; dtype }
      | Trunc { src; dtype } ) as instr ->
        let name =
          match instr with
          | Cast _ -> "cast"
          | Bitcast _ -> "bitcast"
          | Neg _ -> "neg"
          | Exp2 _ -> "exp2"
          | Log2 _ -> "log2"
          | Sin _ -> "sin"
          | Sqrt _ -> "sqrt"
          | Recip _ -> "recip"
          | Trunc _ -> "trunc"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a : %a" name pp_ref src Dtype.pp dtype
    | ( Add { lhs; rhs; dtype }
      | Sub { lhs; rhs; dtype }
      | Mul { lhs; rhs; dtype }
      | Fdiv { lhs; rhs; dtype }
      | Idiv { lhs; rhs; dtype }
      | Mod { lhs; rhs; dtype }
      | Max { lhs; rhs; dtype }
      | Pow { lhs; rhs; dtype }
      | Shl { lhs; rhs; dtype }
      | Shr { lhs; rhs; dtype }
      | And { lhs; rhs; dtype }
      | Or { lhs; rhs; dtype }
      | Xor { lhs; rhs; dtype }
      | Threefry { lhs; rhs; dtype }
      | Cmplt { lhs; rhs; dtype }
      | Cmpeq { lhs; rhs; dtype }
      | Cmpne { lhs; rhs; dtype } ) as instr ->
        let name =
          match instr with
          | Add _ -> "add"
          | Sub _ -> "sub"
          | Mul _ -> "mul"
          | Fdiv _ -> "fdiv"
          | Idiv _ -> "idiv"
          | Mod _ -> "mod"
          | Max _ -> "max"
          | Pow _ -> "pow"
          | Shl _ -> "shl"
          | Shr _ -> "shr"
          | And _ -> "and"
          | Or _ -> "or"
          | Xor _ -> "xor"
          | Threefry _ -> "threefry"
          | Cmplt _ -> "cmplt"
          | Cmpeq _ -> "cmpeq"
          | Cmpne _ -> "cmpne"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a, %a : %a" name pp_ref lhs pp_ref rhs Dtype.pp
          dtype
    | Where { cond; a; b; dtype } ->
        Format.fprintf fmt "where %a, %a, %a : %a" pp_ref cond pp_ref a pp_ref b
          Dtype.pp dtype
    | Mulacc { a; b; c; dtype } ->
        Format.fprintf fmt "mulacc %a, %a, %a : %a" pp_ref a pp_ref b pp_ref c
          Dtype.pp dtype

  let pp fmt t = pp_program pp_instr fmt t
end

module Program = struct
  (* ───── Types ───── *)

  type const = Bool of bool | Int of int | Float of float
  type ref = int

  type instr =
    (* Memory definitions *)
    | Param of { idx : int; dtype : Dtype.ptr }
    | Define_local of { size : int; dtype : Dtype.ptr }
    | Define_reg of { size : int; dtype : Dtype.ptr }
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
    (* Constants *)
    | Const of { value : const; dtype : Dtype.t }
    (* Memory operations *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
    | Store of { dst : ref; value : ref }
    (* Unary ALU *)
    | Neg of { src : ref; dtype : Dtype.t }
    | Exp2 of { src : ref; dtype : Dtype.t }
    | Log2 of { src : ref; dtype : Dtype.t }
    | Sin of { src : ref; dtype : Dtype.t }
    | Sqrt of { src : ref; dtype : Dtype.t }
    | Recip of { src : ref; dtype : Dtype.t }
    | Trunc of { src : ref; dtype : Dtype.t }
    (* Binary ALU *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
    (* Ternary ALU *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
    (* Type conversion *)
    | Cast of { src : ref; dtype : Dtype.t }
    | Bitcast of { src : ref; dtype : Dtype.t }
    (* Vector operations *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
    (* Control flow *)
    (* axis: identifier used by renderer for variable naming (0=x, 1=y, ...).
       kind: controls loop semantics (global/local/reduce/etc). *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
    | End_range of { range : ref }
    | If of { cond : ref; idx_for_dedup : ref }
    | Endif of { if_ : ref }
    (* Synchronization *)
    | Barrier
    (* GPU special *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
    (* Tensor core *)
    | Wmma of {
        name : string;
        a : ref;
        b : ref;
        c : ref;
        dtype : Dtype.t;
        dims : int * int * int; (* N, M, K matrix dimensions *)
        dtype_in : Dtype.scalar;
        dtype_out : Dtype.scalar;
        device : string;
        threads : int;
        upcast_axes : (int * int) list * (int * int) list * (int * int) list;
        reduce_axes : int list;
      }
    (* Custom code injection *)
    | Custom of { fmt : string; args : ref list }
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }

  type t = instr array

  (* ───── Inspection ───── *)

  let dtype_of = function
    (* Pointer-producing: return base type *)
    | Param { dtype; _ }
    | Define_local { dtype; _ }
    | Define_reg { dtype; _ }
    | Index { dtype; _ } ->
        Some dtype.base
    (* Value-producing with dtype field *)
    | Define_var { dtype; _ }
    | Const { dtype; _ }
    | Load { dtype; _ }
    | Neg { dtype; _ }
    | Exp2 { dtype; _ }
    | Log2 { dtype; _ }
    | Sin { dtype; _ }
    | Sqrt { dtype; _ }
    | Recip { dtype; _ }
    | Trunc { dtype; _ }
    | Add { dtype; _ }
    | Sub { dtype; _ }
    | Mul { dtype; _ }
    | Fdiv { dtype; _ }
    | Idiv { dtype; _ }
    | Mod { dtype; _ }
    | Max { dtype; _ }
    | Pow { dtype; _ }
    | Shl { dtype; _ }
    | Shr { dtype; _ }
    | And { dtype; _ }
    | Or { dtype; _ }
    | Xor { dtype; _ }
    | Threefry { dtype; _ }
    | Cmplt { dtype; _ }
    | Cmpeq { dtype; _ }
    | Cmpne { dtype; _ }
    | Where { dtype; _ }
    | Mulacc { dtype; _ }
    | Cast { dtype; _ }
    | Bitcast { dtype; _ }
    | Vectorize { dtype; _ }
    | Gep { dtype; _ }
    | Range { dtype; _ }
    | Special { dtype; _ }
    | Wmma { dtype; _ }
    | Custom_inline { dtype; _ } ->
        Some dtype
    (* No result value *)
    | Store _ | End_range _ | If _ | Endif _ | Barrier | Custom _ -> None

  let refs_of = function
    (* No operands *)
    | Param _ | Define_local _ | Define_reg _ | Define_var _ | Const _ | Barrier
      ->
        []
    (* Single operand *)
    | Neg { src; _ }
    | Exp2 { src; _ }
    | Log2 { src; _ }
    | Sin { src; _ }
    | Sqrt { src; _ }
    | Recip { src; _ }
    | Trunc { src; _ }
    | Cast { src; _ }
    | Bitcast { src; _ }
    | Gep { src; _ } ->
        [ src ]
    | Range { size; _ } | Special { size; _ } -> [ size ]
    | End_range { range } -> [ range ]
    | Endif { if_ } -> [ if_ ]
    (* Two operands *)
    | Store { dst; value } -> [ dst; value ]
    | If { cond; idx_for_dedup } -> [ cond; idx_for_dedup ]
    | Add { lhs; rhs; _ }
    | Sub { lhs; rhs; _ }
    | Mul { lhs; rhs; _ }
    | Fdiv { lhs; rhs; _ }
    | Idiv { lhs; rhs; _ }
    | Mod { lhs; rhs; _ }
    | Max { lhs; rhs; _ }
    | Pow { lhs; rhs; _ }
    | Shl { lhs; rhs; _ }
    | Shr { lhs; rhs; _ }
    | And { lhs; rhs; _ }
    | Or { lhs; rhs; _ }
    | Xor { lhs; rhs; _ }
    | Threefry { lhs; rhs; _ }
    | Cmplt { lhs; rhs; _ }
    | Cmpeq { lhs; rhs; _ }
    | Cmpne { lhs; rhs; _ } ->
        [ lhs; rhs ]
    (* Three operands *)
    | Where { cond; a; b; _ } -> [ cond; a; b ]
    | Mulacc { a; b; c; _ } | Wmma { a; b; c; _ } -> [ a; b; c ]
    (* Variable operands *)
    | Vectorize { srcs; _ } -> srcs
    | Custom { args; _ } | Custom_inline { args; _ } -> args
    (* With optional operands *)
    | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
    | Load { src; alt; _ } -> src :: Option.to_list alt

  let map_refs map_ref (instr : instr) : instr =
    let map_ref_opt = Option.map map_ref in
    match instr with
    | Param _ | Define_local _ | Define_reg _ | Define_var _ | Const _ | Barrier
      ->
        instr
    | Index { ptr; idxs; gate; dtype } ->
        Index
          {
            ptr = map_ref ptr;
            idxs = List.map map_ref idxs;
            gate = map_ref_opt gate;
            dtype;
          }
    | Load { src; alt; dtype } ->
        Load { src = map_ref src; alt = map_ref_opt alt; dtype }
    | Store { dst; value } -> Store { dst = map_ref dst; value = map_ref value }
    | Neg { src; dtype } -> Neg { src = map_ref src; dtype }
    | Exp2 { src; dtype } -> Exp2 { src = map_ref src; dtype }
    | Log2 { src; dtype } -> Log2 { src = map_ref src; dtype }
    | Sin { src; dtype } -> Sin { src = map_ref src; dtype }
    | Sqrt { src; dtype } -> Sqrt { src = map_ref src; dtype }
    | Recip { src; dtype } -> Recip { src = map_ref src; dtype }
    | Trunc { src; dtype } -> Trunc { src = map_ref src; dtype }
    | Add { lhs; rhs; dtype } ->
        Add { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Sub { lhs; rhs; dtype } ->
        Sub { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mul { lhs; rhs; dtype } ->
        Mul { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Fdiv { lhs; rhs; dtype } ->
        Fdiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Idiv { lhs; rhs; dtype } ->
        Idiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mod { lhs; rhs; dtype } ->
        Mod { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Max { lhs; rhs; dtype } ->
        Max { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Pow { lhs; rhs; dtype } ->
        Pow { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shl { lhs; rhs; dtype } ->
        Shl { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shr { lhs; rhs; dtype } ->
        Shr { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | And { lhs; rhs; dtype } ->
        And { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Or { lhs; rhs; dtype } ->
        Or { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Xor { lhs; rhs; dtype } ->
        Xor { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Threefry { lhs; rhs; dtype } ->
        Threefry { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmplt { lhs; rhs; dtype } ->
        Cmplt { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpeq { lhs; rhs; dtype } ->
        Cmpeq { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpne { lhs; rhs; dtype } ->
        Cmpne { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Where { cond; a; b; dtype } ->
        Where { cond = map_ref cond; a = map_ref a; b = map_ref b; dtype }
    | Mulacc { a; b; c; dtype } ->
        Mulacc { a = map_ref a; b = map_ref b; c = map_ref c; dtype }
    | Cast { src; dtype } -> Cast { src = map_ref src; dtype }
    | Bitcast { src; dtype } -> Bitcast { src = map_ref src; dtype }
    | Vectorize { srcs; dtype } ->
        Vectorize { srcs = List.map map_ref srcs; dtype }
    | Gep { src; idx; dtype } -> Gep { src = map_ref src; idx; dtype }
    | Range { size; dtype; axis; kind } ->
        Range { size = map_ref size; dtype; axis; kind }
    | End_range { range } -> End_range { range = map_ref range }
    | If { cond; idx_for_dedup } ->
        If { cond = map_ref cond; idx_for_dedup = map_ref idx_for_dedup }
    | Endif { if_ } -> Endif { if_ = map_ref if_ }
    | Special { dim; size; dtype } ->
        Special { dim; size = map_ref size; dtype }
    | Wmma
        {
          name;
          a;
          b;
          c;
          dtype;
          dims;
          dtype_in;
          dtype_out;
          device;
          threads;
          upcast_axes;
          reduce_axes;
        } ->
        Wmma
          {
            name;
            a = map_ref a;
            b = map_ref b;
            c = map_ref c;
            dtype;
            dims;
            dtype_in;
            dtype_out;
            device;
            threads;
            upcast_axes;
            reduce_axes;
          }
    | Custom { fmt; args } -> Custom { fmt; args = List.map map_ref args }
    | Custom_inline { fmt; args; dtype } ->
        Custom_inline { fmt; args = List.map map_ref args; dtype }

  let map_alu ~(map_ref : ref -> ref) ~(dtype : Dtype.t) (instr : instr) : instr
      =
    match instr with
    | Neg { src; _ } -> Neg { src = map_ref src; dtype }
    | Exp2 { src; _ } -> Exp2 { src = map_ref src; dtype }
    | Log2 { src; _ } -> Log2 { src = map_ref src; dtype }
    | Sin { src; _ } -> Sin { src = map_ref src; dtype }
    | Sqrt { src; _ } -> Sqrt { src = map_ref src; dtype }
    | Recip { src; _ } -> Recip { src = map_ref src; dtype }
    | Trunc { src; _ } -> Trunc { src = map_ref src; dtype }
    | Add { lhs; rhs; _ } -> Add { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Sub { lhs; rhs; _ } -> Sub { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mul { lhs; rhs; _ } -> Mul { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Fdiv { lhs; rhs; _ } ->
        Fdiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Idiv { lhs; rhs; _ } ->
        Idiv { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Mod { lhs; rhs; _ } -> Mod { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Max { lhs; rhs; _ } -> Max { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Pow { lhs; rhs; _ } -> Pow { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shl { lhs; rhs; _ } -> Shl { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Shr { lhs; rhs; _ } -> Shr { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | And { lhs; rhs; _ } -> And { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Or { lhs; rhs; _ } -> Or { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Xor { lhs; rhs; _ } -> Xor { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Threefry { lhs; rhs; _ } ->
        Threefry { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmplt { lhs; rhs; _ } ->
        Cmplt { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpeq { lhs; rhs; _ } ->
        Cmpeq { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Cmpne { lhs; rhs; _ } ->
        Cmpne { lhs = map_ref lhs; rhs = map_ref rhs; dtype }
    | Where { cond; a; b; _ } ->
        Where { cond = map_ref cond; a = map_ref a; b = map_ref b; dtype }
    | Mulacc { a; b; c; _ } ->
        Mulacc { a = map_ref a; b = map_ref b; c = map_ref c; dtype }
    | _ -> failwith "map_alu: non-ALU instruction"

  let is_unary = function
    | Neg _ | Exp2 _ | Log2 _ | Sin _ | Sqrt _ | Recip _ | Trunc _ -> true
    | _ -> false

  let is_binary = function
    | Add _ | Sub _ | Mul _ | Fdiv _ | Idiv _ | Mod _ | Max _ | Pow _ | Shl _
    | Shr _ | And _ | Or _ | Xor _ | Threefry _ | Cmplt _ | Cmpeq _ | Cmpne _ ->
        true
    | _ -> false

  let is_ternary = function Where _ | Mulacc _ -> true | _ -> false
  let is_alu instr = is_unary instr || is_binary instr || is_ternary instr

  (* ───── Validation ───── *)

  let validate (program : t) : unit =
    let range_stack = ref [] in
    let if_stack = ref [] in
    let seen_specials = Hashtbl.create 8 in

    let fail i msg =
      failwith (Printf.sprintf "Program.validate: instruction %d: %s" i msg)
    in

    let pp_dim = function
      | Group_id i -> Printf.sprintf "Group %d" i
      | Local_id i -> Printf.sprintf "Local %d" i
      | Global_idx i -> Printf.sprintf "GlobalIdx %d" i
    in

    let get_dtype r =
      if r < 0 || r >= Array.length program then None else dtype_of program.(r)
    in

    let check_dtype_eq = check_dtype_eq fail in
    let check_dtype_match = check_dtype_match fail in
    let check_shift_rhs = check_shift_rhs fail get_dtype in

    let check_scalar_bool i ~ctx r =
      match get_dtype r with
      | Some dt when dt.scalar = Dtype.Bool && dt.count = 1 -> ()
      | Some dt when dt.count > 1 ->
          fail i (Printf.sprintf "%s must be scalar (not vector)" ctx)
      | Some _ -> fail i (Printf.sprintf "%s must be bool" ctx)
      | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
    in

    let rec index_ref r =
      match program.(r) with
      | Index _ -> Some r
      | Cast { src; _ } | Bitcast { src; _ } -> index_ref src
      | _ -> None
    in
    let is_index_ref r = Option.is_some (index_ref r) in

    let check_int_scalar i ~ctx r =
      match get_dtype r with
      | Some dt when Dtype.is_int dt && dt.count = 1 -> ()
      | Some dt when dt.count <> 1 ->
          fail i (Printf.sprintf "%s must be scalar (not vector)" ctx)
      | Some _ -> fail i (Printf.sprintf "%s must be int" ctx)
      | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
    in

    let check_index_base i ptr =
      match program.(ptr) with
      | Param _ | Define_local _ | Define_reg _ -> ()
      | _ -> fail i "Index base must be a Param/Define_local/Define_reg"
    in

    Array.iteri
      (fun i instr ->
        (* 1. SSA dominance: all refs must point to earlier instructions *)
        List.iter
          (fun r ->
            if r < 0 || r >= i then
              fail i
                (Printf.sprintf "references %%%d (out of bounds or forward)" r))
          (refs_of instr);

        (* 2. No Index dtype in linearized program (should be lowered) *)
        (match dtype_of instr with
        | Some dt when dt.scalar = Dtype.Index ->
            fail i
              "Index dtype not allowed in linearized program (should be \
               lowered)"
        | _ -> ());

        (* 3. Instruction-specific checks *)
        match instr with
        | Param { dtype; _ } ->
            if dtype.addrspace <> Dtype.Global then
              fail i "Param must have Global addrspace"
        | Define_local { dtype; _ } ->
            if dtype.addrspace <> Dtype.Local then
              fail i "Define_local must have Local addrspace"
        | Define_reg { dtype; _ } ->
            if dtype.addrspace <> Dtype.Reg then
              fail i "Define_reg must have Reg addrspace"
        | Define_var { lo; hi; dtype; _ } ->
            if dtype.count <> 1 then fail i "Define_var must be scalar";
            if lo > hi then fail i "Define_var bounds invalid (lo > hi)"
        | Range { size; dtype; _ } ->
            if not (Dtype.is_int dtype) then fail i "Range must have int dtype";
            if dtype.count <> 1 then fail i "Range must be scalar";
            check_dtype_eq i ~ctx:"Range size" ~expected:(Some dtype)
              ~got:(get_dtype size);
            range_stack := i :: !range_stack
        | End_range { range } -> (
            (match program.(range) with
            | Range _ -> ()
            | _ -> fail i "End_range must reference a Range");
            match !range_stack with
            | top :: rest when top = range -> range_stack := rest
            | _ -> fail i "unbalanced End_range")
        | If { cond; idx_for_dedup } ->
            if_stack := i :: !if_stack;
            check_scalar_bool i ~ctx:"If condition" cond;
            if not (is_index_ref idx_for_dedup) then
              fail i "If idx_for_dedup must reference Index (or casted Index)"
        | Endif { if_ } -> (
            (match program.(if_) with
            | If _ -> ()
            | _ -> fail i "Endif must reference an If");
            match !if_stack with
            | top :: rest when top = if_ -> if_stack := rest
            | _ -> fail i "unbalanced Endif")
        | Special { dim; size; dtype } ->
            (match Hashtbl.find_opt seen_specials dim with
            | Some first_idx ->
                fail i
                  (Printf.sprintf "duplicate Special %s (first at %d)"
                     (pp_dim dim) first_idx)
            | None -> Hashtbl.add seen_specials dim i);
            if dtype.scalar <> Dtype.Int32 || dtype.count <> 1 then
              fail i "Special must be int32 scalar";
            check_dtype_eq i ~ctx:"Special size" ~expected:(Some dtype)
              ~got:(get_dtype size)
        | Index { ptr; idxs; gate; _ } ->
            check_index_base i ptr;
            if idxs = [] then fail i "Index must have at least one index";
            List.iter (check_int_scalar i ~ctx:"Index operand") idxs;
            Option.iter (check_scalar_bool i ~ctx:"Index gate") gate
        (* alt (fallback value) is only valid when the Index has a gate; without
           a gate, loads are unconditional so alt would be dead code. *)
        | Load { src; alt; dtype } -> (
            if not (is_index_ref src) then
              fail i "Load src must reference Index (or casted Index)";
            match alt with
            | None -> ()
            | Some alt_ref -> (
                match index_ref src with
                | Some idx_ref -> (
                    match program.(idx_ref) with
                    | Index { gate = Some _; _ } ->
                        check_dtype_eq i ~ctx:"Load alt" ~expected:(Some dtype)
                          ~got:(get_dtype alt_ref)
                    | _ -> fail i "Load alt requires gated Index")
                | None -> fail i "Load alt requires Index (or casted Index)"))
        | Where { cond; a; b; dtype } ->
            check_scalar_bool i ~ctx:"Where condition" cond;
            check_dtype_eq i ~ctx:"Where branch a" ~expected:(Some dtype)
              ~got:(get_dtype a);
            check_dtype_eq i ~ctx:"Where branch b" ~expected:(Some dtype)
              ~got:(get_dtype b)
        | Cmplt { lhs; rhs; dtype }
        | Cmpeq { lhs; rhs; dtype }
        | Cmpne { lhs; rhs; dtype } ->
            if dtype.scalar <> Dtype.Bool then
              fail i "comparison result must be bool";
            check_dtype_match i ~ctx:"comparison operands" (get_dtype lhs)
              (get_dtype rhs)
        | Idiv { dtype; _ } | Mod { dtype; _ } ->
            if not (Dtype.is_int dtype) then
              fail i "Idiv/Mod must have int dtype"
        | Add { lhs; rhs; dtype }
        | Sub { lhs; rhs; dtype }
        | Mul { lhs; rhs; dtype }
        | Fdiv { lhs; rhs; dtype }
        | Max { lhs; rhs; dtype }
        | Pow { lhs; rhs; dtype }
        | And { lhs; rhs; dtype }
        | Or { lhs; rhs; dtype }
        | Xor { lhs; rhs; dtype }
        | Threefry { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"binary ALU lhs" (Some dtype)
              (get_dtype lhs);
            check_dtype_match i ~ctx:"binary ALU rhs" (Some dtype)
              (get_dtype rhs)
        | Shl { lhs; rhs; dtype } | Shr { lhs; rhs; dtype } ->
            check_dtype_match i ~ctx:"shift operand" (Some dtype)
              (get_dtype lhs);
            check_shift_rhs i rhs dtype
        | Neg { src; dtype }
        | Exp2 { src; dtype }
        | Log2 { src; dtype }
        | Sin { src; dtype }
        | Sqrt { src; dtype }
        | Recip { src; dtype }
        | Trunc { src; dtype } ->
            check_dtype_match i ~ctx:"unary ALU" (Some dtype) (get_dtype src)
        | Mulacc { a; b; c; dtype } ->
            check_dtype_match i ~ctx:"Mulacc a" (Some dtype) (get_dtype a);
            check_dtype_match i ~ctx:"Mulacc b" (Some dtype) (get_dtype b);
            check_dtype_match i ~ctx:"Mulacc c" (Some dtype) (get_dtype c)
        | Vectorize { srcs; dtype } ->
            if List.length srcs <= 1 then
              fail i "Vectorize must have more than one source";
            if List.length srcs <> dtype.count then
              fail i
                (Printf.sprintf "Vectorize has %d sources but dtype.count=%d"
                   (List.length srcs) dtype.count);
            List.iteri
              (fun j src_ref ->
                match get_dtype src_ref with
                | Some src_dt ->
                    if src_dt.count <> 1 then
                      fail i
                        (Printf.sprintf "Vectorize source %d must be scalar" j);
                    if src_dt.scalar <> dtype.scalar then
                      fail i
                        (Printf.sprintf
                           "Vectorize source %d has wrong scalar type" j)
                | None ->
                    fail i
                      (Printf.sprintf "Vectorize source %d dtype not available"
                         j))
              srcs
        | Gep { src; idx; dtype } -> (
            match get_dtype src with
            | Some src_dt ->
                if src_dt.count <= 1 then fail i "Gep source must be a vector";
                if idx < 0 || idx >= src_dt.count then
                  fail i
                    (Printf.sprintf
                       "Gep index %d out of bounds (vector has %d elements)" idx
                       src_dt.count);
                if dtype.scalar <> src_dt.scalar || dtype.count <> 1 then
                  fail i "Gep result must be scalar of source vector type"
            | None -> fail i "Gep source dtype not available")
        | Wmma { dims = n, m, k; dtype; dtype_out; _ } ->
            if n <= 0 || m <= 0 || k <= 0 then
              fail i "Wmma dims must be positive";
            if dtype.scalar <> dtype_out then
              fail i "Wmma result dtype must match dtype_out"
        | Store { dst; _ } ->
            if not (is_index_ref dst) then
              fail i "Store dst must reference Index (or casted Index)"
        | Const _ | Cast _ | Bitcast _ | Barrier | Custom _ | Custom_inline _ ->
            ())
      program;

    (* Check final nesting depth *)
    if !range_stack <> [] then
      failwith
        (Printf.sprintf
           "Program.validate: %d unclosed Range(s) at end of program"
           (List.length !range_stack));
    if !if_stack <> [] then
      failwith
        (Printf.sprintf "Program.validate: %d unclosed If(s) at end of program"
           (List.length !if_stack))

  (* ───── Pretty Printing ───── *)

  let pp_const fmt = function
    | Bool b -> Format.fprintf fmt "%b" b
    | Int i -> Format.fprintf fmt "%d" i
    | Float f -> Format.fprintf fmt "%g" f

  let pp_instr fmt = function
    | Param { idx; dtype } ->
        Format.fprintf fmt "param %d : %a" idx pp_ptr dtype
    | Define_local { size; dtype } ->
        Format.fprintf fmt "define_local %a, size=%d" pp_ptr dtype size
    | Define_reg { size; dtype } ->
        Format.fprintf fmt "define_reg %a, size=%d" pp_ptr dtype size
    | Define_var { name; lo; hi; dtype } ->
        Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo
          hi
    | Const { value; dtype } ->
        Format.fprintf fmt "const %a : %a" pp_const value Dtype.pp dtype
    | Index { ptr; idxs; gate; dtype } ->
        Format.fprintf fmt "index %a, %a%a : %a" pp_ref ptr pp_refs idxs
          (fun fmt -> function
            | None -> () | Some g -> Format.fprintf fmt " gate=%%%d" g)
          gate pp_ptr dtype
    | Load { src; alt; dtype } ->
        Format.fprintf fmt "load %a%a : %a" pp_ref src
          (fun fmt -> function
            | None -> () | Some a -> Format.fprintf fmt " alt=%%%d" a)
          alt Dtype.pp dtype
    | Store { dst; value } ->
        Format.fprintf fmt "store %a, %a" pp_ref dst pp_ref value
    | ( Neg { src; dtype }
      | Exp2 { src; dtype }
      | Log2 { src; dtype }
      | Sin { src; dtype }
      | Sqrt { src; dtype }
      | Recip { src; dtype }
      | Trunc { src; dtype }
      | Cast { src; dtype }
      | Bitcast { src; dtype } ) as instr ->
        let name =
          match instr with
          | Neg _ -> "neg"
          | Exp2 _ -> "exp2"
          | Log2 _ -> "log2"
          | Sin _ -> "sin"
          | Sqrt _ -> "sqrt"
          | Recip _ -> "recip"
          | Trunc _ -> "trunc"
          | Cast _ -> "cast"
          | Bitcast _ -> "bitcast"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a : %a" name pp_ref src Dtype.pp dtype
    | ( Add { lhs; rhs; dtype }
      | Sub { lhs; rhs; dtype }
      | Mul { lhs; rhs; dtype }
      | Fdiv { lhs; rhs; dtype }
      | Idiv { lhs; rhs; dtype }
      | Mod { lhs; rhs; dtype }
      | Max { lhs; rhs; dtype }
      | Pow { lhs; rhs; dtype }
      | Shl { lhs; rhs; dtype }
      | Shr { lhs; rhs; dtype }
      | And { lhs; rhs; dtype }
      | Or { lhs; rhs; dtype }
      | Xor { lhs; rhs; dtype }
      | Threefry { lhs; rhs; dtype }
      | Cmplt { lhs; rhs; dtype }
      | Cmpeq { lhs; rhs; dtype }
      | Cmpne { lhs; rhs; dtype } ) as instr ->
        let name =
          match instr with
          | Add _ -> "add"
          | Sub _ -> "sub"
          | Mul _ -> "mul"
          | Fdiv _ -> "fdiv"
          | Idiv _ -> "idiv"
          | Mod _ -> "mod"
          | Max _ -> "max"
          | Pow _ -> "pow"
          | Shl _ -> "shl"
          | Shr _ -> "shr"
          | And _ -> "and"
          | Or _ -> "or"
          | Xor _ -> "xor"
          | Threefry _ -> "threefry"
          | Cmplt _ -> "cmplt"
          | Cmpeq _ -> "cmpeq"
          | Cmpne _ -> "cmpne"
          | _ -> assert false
        in
        Format.fprintf fmt "%s %a, %a : %a" name pp_ref lhs pp_ref rhs Dtype.pp
          dtype
    | Where { cond; a; b; dtype } ->
        Format.fprintf fmt "where %a, %a, %a : %a" pp_ref cond pp_ref a pp_ref b
          Dtype.pp dtype
    | Mulacc { a; b; c; dtype } ->
        Format.fprintf fmt "mulacc %a, %a, %a : %a" pp_ref a pp_ref b pp_ref c
          Dtype.pp dtype
    | Vectorize { srcs; dtype } ->
        Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
    | Gep { src; idx; dtype } ->
        Format.fprintf fmt "gep %a, %d : %a" pp_ref src idx Dtype.pp dtype
    | Range { size; dtype; axis; kind } ->
        Format.fprintf fmt "range %a : %a [axis=%d, %a]" pp_ref size Dtype.pp
          dtype axis pp_axis_kind kind
    | End_range { range } -> Format.fprintf fmt "end_range %a" pp_ref range
    | If { cond; idx_for_dedup } ->
        Format.fprintf fmt "if %a, %a" pp_ref cond pp_ref idx_for_dedup
    | Endif { if_ } -> Format.fprintf fmt "endif %a" pp_ref if_
    | Barrier -> Format.fprintf fmt "barrier"
    | Special { dim; size; dtype } ->
        Format.fprintf fmt "special %a, %a : %a" pp_special_dim dim pp_ref size
          Dtype.pp dtype
    | Wmma
        {
          name;
          a;
          b;
          c;
          dtype;
          dims = n, m, k;
          dtype_in;
          dtype_out;
          device;
          threads;
          _;
        } ->
        Format.fprintf fmt
          "wmma.%s %a, %a, %a : %a [%dx%dx%d, %a -> %a, %s, threads=%d]" name
          pp_ref a pp_ref b pp_ref c Dtype.pp dtype n m k Dtype.pp_scalar
          dtype_in Dtype.pp_scalar dtype_out device threads
    | Custom { fmt = f; args } ->
        Format.fprintf fmt "custom \"%s\" %a" f pp_refs args
    | Custom_inline { fmt = f; args; dtype } ->
        Format.fprintf fmt "custom_inline \"%s\" %a : %a" f pp_refs args
          Dtype.pp dtype

  let pp fmt t = pp_program pp_instr fmt t

  (* ───── Transformation ───── *)

  (* Rebuild the program by visiting each instruction in order. The callback [f]
     can emit replacement instructions via [~emit] and return [Some idx] to
     override the default, or [None] to keep the original (with refs
     remapped). *)
  let rebuild f program =
    let n = Array.length program in
    let remap = Array.make n (-1) in
    let acc = ref [] in
    let next = ref 0 in
    let emit instr =
      let idx = !next in
      acc := instr :: !acc;
      incr next;
      idx
    in
    let map_ref r = remap.(r) in
    Array.iteri
      (fun i instr ->
        match f ~emit ~map_ref instr with
        | Some idx -> remap.(i) <- idx
        | None -> remap.(i) <- emit (map_refs map_ref instr))
      program;
    Array.of_list (List.rev !acc)
end
