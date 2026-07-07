(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Uop
open Opt

let axis_type_repr kind =
  "AxisType." ^ String.uppercase_ascii (Axis_type.to_string kind)

let python_bool = function true -> "True" | false -> "False"

let python_quote s =
  let buf = Buffer.create (String.length s + 2) in
  Buffer.add_char buf '\'';
  String.iter
    (function
      | '\'' -> Buffer.add_string buf "\\'"
      | '\\' -> Buffer.add_string buf "\\\\"
      | '\n' -> Buffer.add_string buf "\\n"
      | '\r' -> Buffer.add_string buf "\\r"
      | '\t' -> Buffer.add_string buf "\\t"
      | c -> Buffer.add_char buf c)
    s;
  Buffer.add_char buf '\'';
  Buffer.contents buf

let tuple_string items =
  match items with
  | [] -> "()"
  | [ x ] -> "(" ^ x ^ ",)"
  | _ -> "(" ^ String.concat ", " items ^ ")"

let list_string items = "[" ^ String.concat ", " items ^ "]"

let option_string f = function
  | None -> "None"
  | Some x -> f x

let dataclass_string name fields =
  name ^ "("
  ^ String.concat ", " (List.map (fun (k, v) -> k ^ "=" ^ v) fields)
  ^ ")"

let addrspace_debug_string a =
  "AddrSpace." ^ String.uppercase_ascii (Dtype.addr_space_to_string a)

let dtype_debug_string = Dtype.repr

(* Reproduce CPython's [repr(float)]: the shortest decimal string that
   round-trips to the same double, formatted in fixed notation when the
   decimal point falls in [-4, 16] and scientific notation otherwise, with a
   signed two-or-more-digit exponent. *)
let python_float_string f =
  if Float.is_nan f then "nan"
  else if f = Float.infinity then "inf"
  else if f = Float.neg_infinity then "-inf"
  else if f = 0.0 then if 1.0 /. f = Float.neg_infinity then "-0.0" else "0.0"
  else
    let neg = f < 0.0 in
    let a = Float.abs f in
    (* shortest [%e] rendering that parses back to [a] *)
    let rec shortest p =
      if p >= 17 then Printf.sprintf "%.17e" a
      else
        let s = Printf.sprintf "%.*e" p a in
        if float_of_string s = a then s else shortest (p + 1)
    in
    let mant, exp =
      match String.split_on_char 'e' (shortest 0) with
      | [ m; e ] -> m, int_of_string e
      | _ -> shortest 0, 0
    in
    let digits = String.concat "" (String.split_on_char '.' mant) in
    let ndigits = String.length digits in
    let decpt = exp + 1 in
    let body =
      if decpt <= -4 || decpt > 16 then
        let mantissa =
          if ndigits > 1 then
            String.sub digits 0 1 ^ "." ^ String.sub digits 1 (ndigits - 1)
          else digits
        in
        Printf.sprintf "%se%+03d" mantissa exp
      else if decpt <= 0 then "0." ^ String.make (-decpt) '0' ^ digits
      else if decpt >= ndigits then
        digits ^ String.make (decpt - ndigits) '0' ^ ".0"
      else
        String.sub digits 0 decpt ^ "."
        ^ String.sub digits decpt (ndigits - decpt)
    in
    if neg then "-" ^ body else body

let const_debug_string c =
  match Const.view c with
  | Const.Bool b -> python_bool b
  | Const.Int n -> Int64.to_string n
  | Const.Float f -> python_float_string f
  | Const.Invalid -> "Invalid"

let device_repr_string = function
  | Single d -> python_quote d
  | Multi ds -> tuple_string (List.map python_quote ds)
  | Index i -> string_of_int i

let device_arg_string = function
  | Single d -> d
  | Multi ds -> tuple_string (List.map python_quote ds)
  | Index i -> string_of_int i

let int_pair_string (a, b) =
  tuple_string [ string_of_int a; string_of_int b ]

let metadata_debug_string (m : metadata) =
  dataclass_string "Metadata"
    [
      "name", python_quote m.name;
      "caller", python_quote m.caller;
      "backward", python_bool m.backward;
    ]

let param_arg_debug_string (p : param_arg) =
  let fields = ref [ string_of_int p.slot ] in
  let add name render = function
    | None -> ()
    | Some value -> fields := !fields @ [ name ^ "=" ^ render value ]
  in
  add "vmin_vmax" int_pair_string p.vmin_vmax;
  add "name" python_quote p.name;
  (match p.addrspace with
   | Dtype.Global -> ()
   | addrspace ->
       fields := !fields @ [ "addrspace=" ^ addrspace_debug_string addrspace ]);
  add "axis" string_of_int p.axis;
  add "device" device_repr_string p.device;
  "ParamArg(" ^ String.concat ", " !fields ^ ")"

let reduce_arg_debug_string (r : reduce_arg) =
  tuple_string
    [ "Ops." ^ Ops.name r.op; tuple_string (List.map string_of_int r.axes) ]

let estimate_debug_string = function
  | Int n -> string_of_int n
  | Sym u -> "%" ^ string_of_int (tag u)

let estimates_debug_string (e : estimates) =
  Printf.sprintf "Estimates(ops=%s, lds=%s, mem=%s)"
    (estimate_debug_string e.ops)
    (estimate_debug_string e.lds)
    (estimate_debug_string e.mem)

let opt_op_string name = "OptOps." ^ name

let opt_debug_string = function
  | Tc { axis; tc_select; tc_opt; use_tc } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "TC";
          "axis", string_of_int axis;
          ( "arg",
            tuple_string (List.map string_of_int [ tc_select; tc_opt; use_tc ])
          );
        ]
  | Upcast { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "UPCAST";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Unroll { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "UNROLL";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Local { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "LOCAL";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Thread { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "THREAD";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Group { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "GROUP";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Grouptop { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "GROUPTOP";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Nolocals ->
      dataclass_string "Opt"
        [ "op", opt_op_string "NOLOCALS"; "axis", "None"; "arg", "None" ]
  | Padto { axis; amount } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "PADTO";
          "axis", string_of_int axis;
          "arg", string_of_int amount;
        ]
  | Swap { axis; with_axis } ->
      dataclass_string "Opt"
        [
          "op", opt_op_string "SWAP";
          "axis", string_of_int axis;
          "arg", string_of_int with_axis;
        ]

let stage_opts_debug_string (o : stage_opts) =
  dataclass_string "BufferizeOpts"
    [
      "device", option_string device_repr_string o.device;
      "addrspace", addrspace_debug_string o.addrspace;
      "removable", python_bool o.removable;
    ]

let kernel_info_debug_string (k : kernel_info) =
  dataclass_string "KernelInfo"
    [
      "name", python_quote k.name;
      "axis_types", tuple_string (List.map axis_type_repr k.axis_types);
      "dont_use_locals", python_bool k.dont_use_locals;
      "applied_opts", tuple_string (List.map opt_debug_string k.applied_opts);
      "opts_to_apply",
      option_string (fun opts -> tuple_string (List.map opt_debug_string opts))
        k.opts_to_apply;
      "estimates", option_string estimates_debug_string k.estimates;
      "beam", string_of_int k.beam;
    ]

let call_info_debug_string (c : call_info) =
  Printf.sprintf "CallInfo(%s, %s, %s, %s, %s)"
    (if Option.is_some c.grad_fxn then "<function>" else "None")
    (tuple_string (List.map metadata_debug_string c.metadata))
    (option_string python_quote c.name)
    (python_bool c.precompile)
    (python_bool c.precompile_backward)

let launch_dim_debug_string = function
  | Launch_int n -> string_of_int n
  | Launch_float f -> python_float_string f
  | Launch_sym u -> "%" ^ string_of_int (tag u)

let rec uop_repr_debug_string ?(indent = 0) u =
  let arg =
    match arg u with
    | Arg.Empty -> "None"
    | Arg.Int n -> string_of_int n
    | Arg.Ints xs -> tuple_string (List.map string_of_int xs)
    | Arg.Bools xs -> tuple_string (List.map python_bool xs)
    | Arg.String s -> python_quote s
    | Arg.Value c -> const_debug_string c
    | Arg.Op op -> "Ops." ^ Ops.name op
    | Arg.Range_info { axis; sub; kind } ->
        tuple_string
          (string_of_int axis
          :: (List.map string_of_int sub @ [ axis_type_repr kind ]))
    | Arg.Param_arg p -> param_arg_debug_string p
    | Arg.Reduce_arg r -> reduce_arg_debug_string r
    | Arg.Device d -> device_arg_string d
    | Arg.Op_device (op, device) ->
        tuple_string [ "Ops." ^ Ops.name op; device_repr_string device ]
    | Arg.Stage_info opts -> stage_opts_debug_string opts
    | Arg.Opts opts -> tuple_string (List.map opt_debug_string opts)
    | Arg.Kernel_info info -> kernel_info_debug_string info
    | Arg.Call_info info -> call_info_debug_string info
    | Arg.Program_info _ -> "<program>"
    | Arg.Wmma_info _ -> "<wmma>"
    | Arg.Shaped_wmma_info _ -> "<shaped_wmma>"
  in
  let srcs =
    Array.to_list (src u)
    |> List.map (fun s ->
      "\n" ^ String.make (indent + 2) ' '
      ^ uop_repr_debug_string ~indent:(indent + 2) s ^ ",")
    |> String.concat ""
  in
  Printf.sprintf "UOp(Ops.%s, %s, arg=%s, src=(%s))"
    (Ops.name (op u))
    (dtype_debug_string (dtype u))
    arg srcs

let program_info_debug_string (p : program_info) =
  dataclass_string "ProgramInfo"
    [
      "name", python_quote p.name;
      "global_size", tuple_string (List.map launch_dim_debug_string p.global_size);
      "local_size",
      option_string (fun xs -> tuple_string (List.map string_of_int xs))
        p.local_size;
      ( "vars",
        tuple_string (List.map uop_repr_debug_string p.vars)
      );
      "globals", tuple_string (List.map string_of_int p.globals);
      "outs", tuple_string (List.map string_of_int p.outs);
      "ins", tuple_string (List.map string_of_int p.ins);
      "aux", tuple_string (List.map python_quote p.aux);
    ]

let wmma_info_debug_string (w : wmma_info) =
  let pair (a, b) = tuple_string [ string_of_int a; string_of_int b ] in
  let pairs xs = tuple_string (List.map pair xs) in
  let a, b, c = w.upcast_axes in
  tuple_string
    [
      python_quote w.name;
      tuple_string
        (List.map string_of_int (let x, y, z = w.dims in [ x; y; z ]));
      Dtype.Val.repr (Dtype.Val.of_scalar w.dtype_in);
      Dtype.Val.repr (Dtype.Val.of_scalar w.dtype_out);
      python_quote w.device;
      string_of_int w.threads;
      tuple_string [ pairs a; pairs b; pairs c ];
      tuple_string (List.map string_of_int w.reduce_axes);
    ]

let shaped_wmma_info_debug_string (w : shaped_wmma_info) =
  Printf.sprintf "ShapeWMMAInfo(%s, %s, threads=%d)"
    (tuple_string
       (List.map string_of_int
          (let x, y, z = w.dims in [ x; y; z ])))
    (python_quote w.device)
    w.threads

let arg_debug_string = function
  | Arg.Empty -> "None"
  | Arg.Int n -> string_of_int n
  | Arg.Ints xs -> tuple_string (List.map string_of_int xs)
  | Arg.Bools xs -> tuple_string (List.map python_bool xs)
  | Arg.String s -> python_quote s
  | Arg.Value c -> const_debug_string c
  | Arg.Op op -> "Ops." ^ Ops.name op
  | Arg.Range_info { axis; sub; kind } ->
      tuple_string
        (string_of_int axis
        :: (List.map string_of_int sub @ [ axis_type_repr kind ]))
  | Arg.Param_arg p -> param_arg_debug_string p
  | Arg.Reduce_arg r -> reduce_arg_debug_string r
  | Arg.Device d -> device_arg_string d
  | Arg.Op_device (op, device) ->
      tuple_string [ "Ops." ^ Ops.name op; device_repr_string device ]
  | Arg.Stage_info opts -> stage_opts_debug_string opts
  | Arg.Opts opts -> tuple_string (List.map opt_debug_string opts)
  | Arg.Kernel_info info -> kernel_info_debug_string info
  | Arg.Call_info info -> call_info_debug_string info
  | Arg.Program_info info -> program_info_debug_string info
  | Arg.Wmma_info info -> wmma_info_debug_string info
  | Arg.Shaped_wmma_info info -> shaped_wmma_info_debug_string info

let range_debug_key r =
  match as_range r with
  | Some { axis; sub; kind; _ } -> axis :: sub, kind
  | None -> [ max_int ], Axis_type.Placeholder

let range_debug_string r =
  match as_range r with
  | Some { axis; sub; _ } ->
      String.concat "_"
        (List.map
           (fun n -> if n >= 0 then string_of_int n else "m" ^ string_of_int (-n))
           (axis :: sub))
  | None -> ""

let ranges_debug_string u =
  let rs = ranges u in
  let rs =
    List.sort
      (fun a b -> Stdlib.compare (range_debug_key a) (range_debug_key b))
      rs
  in
  let s = String.concat "," (List.map range_debug_string rs) in
  s ^ String.make (max 0 (10 - String.length s)) ' '

let source_debug_string index s =
  try
    let i = Ref_tbl.find index s in
    if op s = Ops.Const then python_quote (arg_debug_string (arg s))
    else string_of_int i
  with Not_found -> python_quote "--"

let uop_arg_debug_string u =
  match op u, arg u with
  | _, Arg.String s -> s
  | _ -> arg_debug_string (arg u)

let uops_list_to_string uops =
  let buf = Buffer.create (List.length uops * 96) in
  let index = Ref_tbl.create (List.length uops) in
  List.iteri (fun i u -> Ref_tbl.replace index u i) uops;
  List.iteri (fun i u ->
    let srcs =
      Array.to_list (src u)
      |> List.map (source_debug_string index)
      |> list_string
    in
    Buffer.add_string buf
      (Printf.sprintf "%4d %-20s: %s %-40s %-32s %s\n"
         i ("Ops." ^ Ops.name (op u))
         (ranges_debug_string u)
         (dtype_debug_string (dtype u))
         srcs
         (uop_arg_debug_string u)))
    uops;
  Buffer.contents buf

let pp_uops fmt uops = Format.pp_print_string fmt (uops_list_to_string uops)

(* Compact scalar-expression rendering (kernel names, debug shapes). *)

let strip_parens s =
  let n = String.length s in
  if n < 2 || s.[0] <> '(' || s.[n - 1] <> ')' then s
  else
    let d = ref 0 in
    try
      for i = 1 to n - 2 do
        if s.[i] = '(' then incr d
        else if s.[i] = ')' then (
          decr d;
          if !d < 0 then raise_notrace Exit)
      done;
      if !d = 0 then String.sub s 1 (n - 2) else s
    with Exit -> s

let binary_sym = function
  | Ops.Add -> Some "+"
  | Ops.Sub -> Some "-"
  | Ops.Floordiv -> Some "//"
  | Ops.Floormod -> Some "%"
  | Ops.Shl -> Some "<<"
  | Ops.Shr -> Some ">>"
  | Ops.Mul -> Some "*"
  | Ops.Cmplt -> Some "<"
  | Ops.Cmpne -> Some "!="
  | Ops.And -> Some "&"
  | Ops.Or -> Some "|"
  | Ops.Xor -> Some "^"
  | _ -> None

(* Comparisons have no precedence entry: they always keep their parens. *)
let precedence = function
  | Ops.Mul | Ops.Floordiv | Ops.Floormod -> Some 1
  | Ops.Add | Ops.Sub -> Some 2
  | Ops.Shl | Ops.Shr -> Some 3
  | Ops.And -> Some 4
  | Ops.Xor -> Some 5
  | Ops.Or -> Some 6
  | _ -> None

let expr_to_string ?(simplify = true) u =
  let u = if simplify then Uop.simplify u else u in
  let memo = Ref_tbl.create 16 in
  let rec go u =
    match Ref_tbl.find_opt memo u with
    | Some s -> s
    | None ->
        let s = render u in
        Ref_tbl.replace memo u s;
        s
  and s0 u = go (src u).(0)
  and s1 u = go (src u).(1)
  and s2 u = go (src u).(2)
  and render u =
    match op u with
    | Ops.Param -> (
        match as_param u with
        | Some { param = { name = Some name; _ }; _ } -> name
        | Some { param = { slot; _ }; _ } -> "p" ^ string_of_int slot
        | None -> uop_repr_debug_string u)
    | Ops.Special -> (
        match as_special u with
        | Some { name; _ } -> name
        | None -> uop_repr_debug_string u)
    | Ops.Range -> "r" ^ range_debug_string u
    | Ops.Const -> (
        match arg u with
        | Arg.Value c -> const_debug_string c
        | _ -> uop_repr_debug_string u)
    | Ops.Cast ->
        let dt = dtype_debug_string (dtype u) in
        let dt =
          if String.length dt > 7 then String.sub dt 7 (String.length dt - 7)
          else dt
        in
        Printf.sprintf "(%s)(%s)" dt (s0 u)
    | Ops.Bind -> s0 u
    | Ops.Neg -> Printf.sprintf "(-%s)" (s0 u)
    | Ops.Reciprocal -> Printf.sprintf "(1/%s)" (s0 u)
    | Ops.Max -> Printf.sprintf "max(%s, %s)" (s0 u) (s1 u)
    | Ops.Mulacc -> Printf.sprintf "(%s*%s+%s)" (s0 u) (s1 u) (s2 u)
    | Ops.Where -> Printf.sprintf "(%s if %s else %s)" (s1 u) (s0 u) (s2 u)
    | Ops.Cdiv -> Printf.sprintf "cdiv(%s, %s)" (s0 u) (s1 u)
    | Ops.Cmod -> Printf.sprintf "cmod(%s, %s)" (s0 u) (s1 u)
    | o when Option.is_some (binary_sym o) ->
        let sym = Option.get (binary_sym o) in
        let left = s0 u and right = s1 u in
        let prec op' = match precedence op' with Some p -> p | None -> 99 in
        let left, right =
          match precedence o with
          | None -> left, right
          | Some p ->
              ( (if prec (op (src u).(0)) <= p then strip_parens left else left),
                if prec (op (src u).(1)) < p then strip_parens right else right
              )
        in
        Printf.sprintf "(%s%s%s)" left sym right
    | Ops.Index | Ops.Stage ->
        let srcs = src u in
        let parts = ref [] in
        for i = Array.length srcs - 1 downto 1 do
          parts := Printf.sprintf "[%s]" (strip_parens (go srcs.(i))) :: !parts
        done;
        String.concat "" !parts
    | Ops.Stack ->
        let srcs = src u |> Array.to_list |> List.map go in
        "{" ^ String.concat "," srcs ^ "}"
    | _ -> uop_repr_debug_string u
  in
  go u

let uops_to_string ?label root =
  let buf = Buffer.create 256 in
  let nodes = toposort root in
  (match label with
   | Option.Some l -> Buffer.add_string buf (Printf.sprintf "=== %s ===\n" l)
   | Option.None -> ());
  Buffer.add_string buf (uops_list_to_string nodes);
  Buffer.contents buf
