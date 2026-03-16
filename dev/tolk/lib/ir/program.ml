(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type id = int
type sort = Value | Pointer | Index | Effect

type view =
  | Param of { idx : int; dtype : Dtype.ptr }
  | Param_image of { idx : int; dtype : Dtype.ptr; width : int; height : int }
  | Define_local of { size : int; dtype : Dtype.ptr }
  | Define_reg of { size : int; dtype : Dtype.ptr }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Const of { value : Const.t; dtype : Dtype.t }
  | Index of { ptr : id; idxs : id list; gate : id option; dtype : Dtype.ptr }
  | Load of { src : id; alt : id option; dtype : Dtype.t }
  | After of { src : id; deps : id list; dtype : Dtype.t }
  | Store of { dst : id; value : id }
  | Unary of { op : Op.unary; src : id; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : id; rhs : id; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : id; b : id; c : id; dtype : Dtype.t }
  | Cast of { src : id; dtype : Dtype.t }
  | Bitcast of { src : id; dtype : Dtype.t }
  | Vectorize of { srcs : id list; dtype : Dtype.t }
  | Gep of { src : id; idx : int; dtype : Dtype.t }
  | Range of { size : id; dtype : Dtype.t; axis : int; kind : Axis_kind.t }
  | End_range of { range : id }
  | If of { cond : id; idx_for_dedup : id }
  | Endif of { if_ : id }
  | Barrier
  | Special of { dim : Special_dim.t; size : id; dtype : Dtype.t }
  | Wmma of {
      name : string;
      a : id;
      b : id;
      c : id;
      dtype : Dtype.t;
      dims : int * int * int;
      dtype_in : Dtype.scalar;
      dtype_out : Dtype.scalar;
      device : string;
      threads : int;
      upcast_axes : (int * int) list * (int * int) list * (int * int) list;
      reduce_axes : int list;
    }
  | Custom of { fmt : string; args : id list }
  | Custom_inline of { fmt : string; args : id list; dtype : Dtype.t }

type t = view array

let unary ~op ~src ~dtype = Unary { op; src; dtype }
let binary ~op ~lhs ~rhs ~dtype = Binary { op; lhs; rhs; dtype }
let ternary ~op ~a ~b ~c ~dtype = Ternary { op; a; b; c; dtype }

let dtype_of = function
  | Param { dtype; _ }
  | Param_image { dtype; _ }
  | Define_local { dtype; _ }
  | Define_reg { dtype; _ }
  | Index { dtype; _ } ->
      Some dtype.base
  | Define_var { dtype; _ }
  | Const { dtype; _ }
  | Load { dtype; _ }
  | After { dtype; _ }
  | Unary { dtype; _ }
  | Binary { dtype; _ }
  | Ternary { dtype; _ }
  | Cast { dtype; _ }
  | Bitcast { dtype; _ }
  | Vectorize { dtype; _ }
  | Gep { dtype; _ }
  | Range { dtype; _ }
  | Special { dtype; _ }
  | Wmma { dtype; _ }
  | Custom_inline { dtype; _ } ->
      Some dtype
  | Store _ | End_range _ | If _ | Endif _ | Barrier | Custom _ -> None

let refs_of = function
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Barrier ->
      []
  | Unary { src; _ } | Cast { src; _ } | Bitcast { src; _ } | Gep { src; _ } ->
      [ src ]
  | Range { size; _ } | Special { size; _ } -> [ size ]
  | End_range { range } -> [ range ]
  | Endif { if_ } -> [ if_ ]
  | Store { dst; value } -> [ dst; value ]
  | After { src; deps; _ } -> src :: deps
  | If { cond; idx_for_dedup } -> [ cond; idx_for_dedup ]
  | Binary { lhs; rhs; _ } -> [ lhs; rhs ]
  | Ternary { a; b; c; _ } | Wmma { a; b; c; _ } -> [ a; b; c ]
  | Vectorize { srcs; _ } -> srcs
  | Custom { args; _ } | Custom_inline { args; _ } -> args
  | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
  | Load { src; alt; _ } -> src :: Option.to_list alt

let map_refs map_ref (instr : view) : view =
  let map_ref_opt = Option.map map_ref in
  match instr with
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Barrier ->
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
  | After { src; deps; dtype } ->
      After { src = map_ref src; deps = List.map map_ref deps; dtype }
  | Store { dst; value } -> Store { dst = map_ref dst; value = map_ref value }
  | Unary { op; src; dtype } -> Unary { op; src = map_ref src; dtype }
  | Binary { op; lhs; rhs; dtype } ->
      Binary { op; lhs = map_ref lhs; rhs = map_ref rhs; dtype }
  | Ternary { op; a; b; c; dtype } ->
      Ternary { op; a = map_ref a; b = map_ref b; c = map_ref c; dtype }
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
  | Special { dim; size; dtype } -> Special { dim; size = map_ref size; dtype }
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

(* Validation *)

let validate (program : t) : unit =
  let range_stack = ref [] in
  let if_stack = ref [] in
  let seen_specials = Hashtbl.create 8 in
  let fail i msg =
    failwith (Printf.sprintf "Program.validate: instruction %d: %s" i msg)
  in
  let get_dtype r =
    if r < 0 || r >= Array.length program then None else dtype_of program.(r)
  in
  let check_dtype_eq i ~ctx ~expected ~got =
    match expected, got with
    | Some expected, Some got when Dtype.equal expected got -> ()
    | Some expected, Some got ->
        fail i
          (Printf.sprintf "%s: expected %s, got %s" ctx
             (Dtype.to_string expected) (Dtype.to_string got))
    | None, _ -> fail i (Printf.sprintf "%s: expected dtype not available" ctx)
    | _, None -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)
  in
  let check_dtype_match i ~ctx left right =
    match left, right with
    | Some left, Some right when Dtype.equal left right -> ()
    | Some _, Some _ ->
        fail i (Printf.sprintf "%s: operand dtypes don't match" ctx)
    | _ -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)
  in
  let check_shift_rhs i rhs dtype =
    match get_dtype rhs with
    | Some (rhs_dtype : Dtype.t) when Dtype.equal rhs_dtype dtype -> ()
    | Some (rhs_dtype : Dtype.t) when Dtype.equal rhs_dtype Dtype.uint32 -> ()
    | Some _ -> fail i "shift rhs must match lhs dtype or be uint32"
    | None -> fail i "shift rhs dtype not available"
  in
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
    | Cast { src; _ } | Bitcast { src; _ } | After { src; _ } -> index_ref src
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
    | Param _ | Param_image _ | Define_local _ | Define_reg _ -> ()
    | _ ->
        fail i "Index base must be a Param/Param_image/Define_local/Define_reg"
  in
  Array.iteri
    (fun i instr ->
      List.iter
        (fun r ->
          if r < 0 || r >= i then
            fail i
              (Printf.sprintf "references %%%d (out of bounds or forward)" r))
        (refs_of instr);
      (match dtype_of instr with
      | Some dt when dt.scalar = Dtype.Index ->
          fail i
            "Index dtype not allowed in linearized program (should be lowered)"
      | _ -> ());
      match instr with
      | Param { dtype; _ } | Param_image { dtype; _ } ->
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
                (Format.asprintf "duplicate Special %a (first at %d)"
                   Special_dim.pp dim first_idx)
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
      | Load { src; alt; dtype } -> (
          if not (is_index_ref src) then
            fail i "Load src must reference Index (or casted Index)";
          let idx_ref =
            match index_ref src with
            | Some idx_ref -> idx_ref
            | None -> fail i "Load src requires Index (or casted Index)"
          in
          let ptr, gate =
            match program.(idx_ref) with
            | Index { ptr; gate; _ } -> (ptr, gate)
            | _ -> fail i "Load src requires Index"
          in
          let ptr_dtype =
            match program.(ptr) with
            | Param { dtype; _ }
            | Param_image { dtype; _ }
            | Define_local { dtype; _ }
            | Define_reg { dtype; _ } ->
                dtype
            | _ -> fail i "Load src must index a pointer definition"
          in
          check_dtype_eq i ~ctx:"Load dtype" ~expected:(Some ptr_dtype.base)
            ~got:(Some dtype);
          match alt with
          | None -> ()
          | Some alt_ref -> (
              check_dtype_eq i ~ctx:"Load alt" ~expected:(Some dtype)
                ~got:(get_dtype alt_ref);
              match gate with
              | Some _ -> ()
              | None -> fail i "Load alt requires gated Index"))
      | After { src; dtype; _ } -> (
          match program.(src) with
          | Barrier | Store _ | End_range _ | Custom _ ->
              if not (Dtype.equal dtype Dtype.void) then
                fail i "After void-source must have void dtype"
          | _ ->
              check_dtype_eq i ~ctx:"After src" ~expected:(Some dtype)
                ~got:(get_dtype src))
      | Ternary { op = `Where; a = cond; b = then_; c = else_; dtype } ->
          check_scalar_bool i ~ctx:"Where condition" cond;
          check_dtype_eq i ~ctx:"Where branch then" ~expected:(Some dtype)
            ~got:(get_dtype then_);
          check_dtype_eq i ~ctx:"Where branch else" ~expected:(Some dtype)
            ~got:(get_dtype else_)
      | Binary { op = `Cmplt | `Cmpeq | `Cmpne; lhs; rhs; dtype } ->
          if dtype.scalar <> Dtype.Bool then
            fail i "comparison result must be bool";
          check_dtype_match i ~ctx:"comparison operands" (get_dtype lhs)
            (get_dtype rhs)
      | Binary { op = `Idiv | `Mod; dtype; _ } ->
          if not (Dtype.is_int dtype) then fail i "Idiv/Mod must have int dtype"
      | Binary
          {
            op =
              ( `Add | `Sub | `Mul | `Fdiv | `Max | `Pow | `And | `Or | `Xor
              | `Threefry );
            lhs;
            rhs;
            dtype;
          } ->
          check_dtype_match i ~ctx:"binary ALU lhs" (Some dtype) (get_dtype lhs);
          check_dtype_match i ~ctx:"binary ALU rhs" (Some dtype) (get_dtype rhs)
      | Binary { op = `Shl | `Shr; lhs; rhs; dtype } ->
          check_dtype_match i ~ctx:"shift operand" (Some dtype) (get_dtype lhs);
          check_shift_rhs i rhs dtype
      | Unary { src; dtype; _ } ->
          check_dtype_match i ~ctx:"unary ALU" (Some dtype) (get_dtype src)
      | Ternary { op = `Mulacc; a; b; c; dtype } ->
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
                    (Printf.sprintf "Vectorize source %d dtype not available" j))
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
          if n <= 0 || m <= 0 || k <= 0 then fail i "Wmma dims must be positive";
          if dtype.scalar <> dtype_out then
            fail i "Wmma result dtype must match dtype_out"
      | Store { dst; value } -> (
          if not (is_index_ref dst) then
            fail i "Store dst must reference Index (or casted Index)";
          match index_ref dst with
          | Some idx_ref -> (
              match program.(idx_ref) with
              | Index { ptr; _ } -> (
                  match program.(ptr) with
                  | Param { dtype = ptr_dtype; _ }
                  | Param_image { dtype = ptr_dtype; _ }
                  | Define_local { dtype = ptr_dtype; _ }
                  | Define_reg { dtype = ptr_dtype; _ } ->
                      check_dtype_eq i ~ctx:"Store value"
                        ~expected:(Some ptr_dtype.base) ~got:(get_dtype value)
                  | _ -> fail i "Store dst must index a pointer definition")
              | _ -> fail i "Store dst requires Index")
          | None -> fail i "Store dst requires Index (or casted Index)")
      | Const _ | Cast _ | Bitcast _ | Barrier | Custom _ | Custom_inline _ ->
          ())
    program;
  if !range_stack <> [] then
    failwith
      (Printf.sprintf "Program.validate: %d unclosed Range(s) at end of program"
         (List.length !range_stack));
  if !if_stack <> [] then
    failwith
      (Printf.sprintf "Program.validate: %d unclosed If(s) at end of program"
         (List.length !if_stack))

(* Formatting *)

let pp_comma fmt () = Format.fprintf fmt ", "
let pp_ref fmt r = Format.fprintf fmt "%%%d" r
let pp_refs fmt refs = Format.pp_print_list ~pp_sep:pp_comma pp_ref fmt refs

let pp_ptr fmt (dtype : Dtype.ptr) =
  Format.fprintf fmt "%a*%s [%a]" Dtype.pp dtype.base
    (if dtype.v = 1 then "" else Printf.sprintf ".vec(%d)" dtype.v)
    Dtype.pp_addr_space dtype.addrspace

let pp_program pp_view fmt t =
  Array.iteri
    (fun i instr -> Format.fprintf fmt "%3d: %a@\n" i pp_view instr)
    t

let pp_view fmt = function
  | Param { idx; dtype } -> Format.fprintf fmt "param %d : %a" idx pp_ptr dtype
  | Param_image { idx; dtype; width; height } ->
      Format.fprintf fmt "param_image %d : %a [%dx%d]" idx pp_ptr dtype width
        height
  | Define_local { size; dtype } ->
      Format.fprintf fmt "define_local %a, size=%d" pp_ptr dtype size
  | Define_reg { size; dtype } ->
      Format.fprintf fmt "define_reg %a, size=%d" pp_ptr dtype size
  | Define_var { name; lo; hi; dtype } ->
      Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo hi
  | Const { value; dtype } ->
      Format.fprintf fmt "const %a : %a" Const.pp value Dtype.pp dtype
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
  | After { src; deps; dtype } ->
      Format.fprintf fmt "after %a, deps=[%a] : %a" pp_ref src pp_refs deps
        Dtype.pp dtype
  | Store { dst; value } ->
      Format.fprintf fmt "store %a, %a" pp_ref dst pp_ref value
  | Unary { op; src; dtype } ->
      Format.fprintf fmt "%a %a : %a" Op.pp_unary op pp_ref src Dtype.pp dtype
  | Cast { src; dtype } ->
      Format.fprintf fmt "cast %a : %a" pp_ref src Dtype.pp dtype
  | Bitcast { src; dtype } ->
      Format.fprintf fmt "bitcast %a : %a" pp_ref src Dtype.pp dtype
  | Binary { op; lhs; rhs; dtype } ->
      Format.fprintf fmt "%a %a, %a : %a" Op.pp_binary op pp_ref lhs pp_ref rhs
        Dtype.pp dtype
  | Ternary { op; a; b; c; dtype } ->
      Format.fprintf fmt "%a %a, %a, %a : %a" Op.pp_ternary op pp_ref a pp_ref b
        pp_ref c Dtype.pp dtype
  | Vectorize { srcs; dtype } ->
      Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
  | Gep { src; idx; dtype } ->
      Format.fprintf fmt "gep %a, %d : %a" pp_ref src idx Dtype.pp dtype
  | Range { size; dtype; axis; kind } ->
      Format.fprintf fmt "range %a : %a [axis=%d, %a]" pp_ref size Dtype.pp
        dtype axis Axis_kind.pp kind
  | End_range { range } -> Format.fprintf fmt "end_range %a" pp_ref range
  | If { cond; idx_for_dedup } ->
      Format.fprintf fmt "if %a, %a" pp_ref cond pp_ref idx_for_dedup
  | Endif { if_ } -> Format.fprintf fmt "endif %a" pp_ref if_
  | Barrier -> Format.fprintf fmt "barrier"
  | Special { dim; size; dtype } ->
      Format.fprintf fmt "special %a, %a : %a" Special_dim.pp dim pp_ref size
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
        pp_ref a pp_ref b pp_ref c Dtype.pp dtype n m k Dtype.pp_scalar dtype_in
        Dtype.pp_scalar dtype_out device threads
  | Custom { fmt = f; args } ->
      Format.fprintf fmt "custom \"%s\" %a" f pp_refs args
  | Custom_inline { fmt = f; args; dtype } ->
      Format.fprintf fmt "custom_inline \"%s\" %a : %a" f pp_refs args Dtype.pp
        dtype

let pp fmt t = pp_program pp_view fmt t

(* Rewriting *)

let rebuild_internal f program =
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

(* Building *)

type builder = { mutable data : view array; mutable len : int }

let create () = { data = Array.make 32 Barrier; len = 0 }

let ensure builder =
  if builder.len = Array.length builder.data then begin
    let next = Array.make (max 1 (builder.len * 2)) Barrier in
    Array.blit builder.data 0 next 0 builder.len;
    builder.data <- next
  end

let emit builder (instr : view) =
  ensure builder;
  let id = builder.len in
  builder.data.(id) <- instr;
  builder.len <- id + 1;
  id

let finish builder = Array.init builder.len (fun i -> builder.data.(i))
let view (program : t) (id : id) : view = program.(id)

(* Inspecting *)

let length program = Array.length program

let dtype program id =
  if id < 0 || id >= Array.length program then None else dtype_of program.(id)

let children program id = refs_of program.(id)
let iteri f program = Array.iteri f program
let is_alu = function Unary _ | Binary _ | Ternary _ -> true | _ -> false
let dtype_of_view = dtype_of

let index_gate program id =
  let rec walk id =
    match program.(id) with
    | Index { gate; _ } -> gate
    | Cast { src; _ } | Bitcast { src; _ } | After { src; _ } -> walk src
    | _ -> None
  in
  walk id

let map_children = map_refs

let map_alu ~map_ref ~dtype = function
  | Unary { op; src; _ } -> Unary { op; src = map_ref src; dtype }
  | Binary { op; lhs; rhs; _ } ->
      Binary { op; lhs = map_ref lhs; rhs = map_ref rhs; dtype }
  | Ternary { op; a; b; c; _ } ->
      Ternary { op; a = map_ref a; b = map_ref b; c = map_ref c; dtype }
  | _ -> invalid_arg "Program.map_alu expects an ALU view"

let sort program id =
  match view program id with
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Index _ -> Pointer
  | Define_var _ | Range _ | Special _ -> Index
  | Store _ | End_range _ | If _ | Endif _ | Barrier | Custom _ -> Effect
  | After { dtype; _ } when Dtype.equal dtype Dtype.void -> Effect
  | Const _ | Load _ | After _ | Unary _ | Binary _ | Ternary _ | Cast _
  | Bitcast _ | Vectorize _ | Gep _ | Wmma _ | Custom_inline _ ->
      Value

let rebuild = rebuild_internal
