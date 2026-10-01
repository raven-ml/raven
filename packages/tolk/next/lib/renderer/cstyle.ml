(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let strf = Printf.sprintf
let is o u = Op.equal (op u) o

(* the integer [s] holds from its [i]th character on *)
let number_from i s =
  if (String.length s < i) [@mutate off "an empty suffix is no integer either"]
  then None
  else int_of_string_opt (String.sub s i (String.length s - i))

let dedup l =
  List.rev (List.fold_left (fun a x -> if List.mem x a then a else x :: a) [] l)

let const_str u = Format.asprintf "%a" Dtype.pp_const (value u)

let cval u =
  match value u with
  | #Dtype.value as v -> v
  | `Invalid -> invalid_arg "the invalid constant has no value"

let is_ptr = function Some (Dtype.Global | Dtype.Local) -> true | _ -> false

(* Python's str.format on positional arguments: {} and {i}, {{ and }}. *)
let format fmt args =
  let b = Buffer.create 64 and next = ref 0 and n = String.length fmt in
  let fail why = invalid_arg (strf "%S %s" fmt why) in
  let rec go i =
    if i < n then
      match fmt.[i] with
      | ('{' | '}') as c when i + 1 < n && fmt.[i + 1] = c ->
          Buffer.add_char b c;
          go (i + 2)
      | '}' -> fail "has a single }"
      | '{' -> (
          match String.index_from_opt fmt i '}' with
          | None -> fail "has an unmatched {"
          | Some j -> (
              let k =
                match String.sub fmt (i + 1) (j - i - 1) with
                | "" ->
                    incr next;
                    Some (!next - 1)
                | k -> int_of_string_opt k
              in
              match Option.bind k (List.nth_opt args) with
              | Some arg ->
                  Buffer.add_string b arg;
                  go (j + 1)
              | None -> fail "names no argument"))
      | c ->
          Buffer.add_char b c;
          go (i + 1)
  in
  go 0;
  Buffer.contents b

(* Languages *)

type lang = {
  abi : string;
  kernel_typedef : int -> string; (* of the launch bounds *)
  buffer_prefix : string;
  buffer_suffix : string;
  smem_align : string;
  smem_prefix : string;
  smem_prefix_for_cast : bool;
  var_prefix : string;
  var_suffix : string;
  barrier : string;
  code_for_workitem : (char * (char -> string)) list;
  extra_args : string list;
  float4 : string -> string; (* the constructor of a vector type *)
  float4_style : string * string;
  gep_arr_threshold : int;
  type_map : (Dtype.t * string) list;
  infinity : string;
  nan : string;
  promoted : Dtype.t list; (* scalars whose operations compute wider *)
  vector_names : (Dtype.t * string) list;
      (* the element names of vector types, where they are not type_map's *)
  code_for_op : (Op.t * (string list -> Dtype.t -> string)) list;
  string_rewrite : (ctx, string) Pattern_matcher.t;
}

and ctx = {
  lang : lang;
  r : string Tbl.t;
  narrowed : unit Tbl.t; (* the inlined operations cast to their type *)
}

let ( .%{} ) ctx u =
  match Tbl.find_opt ctx.r u with
  | Some s -> s
  | None ->
      invalid_arg (strf "%s is used before it is rendered" (Op.name (op u)))

let render_dtype ?(sz = 1) ?(addrspace = Some Dtype.Alu) ?(override_ptr = false)
    l dt =
  let prefix =
    match addrspace with
    | Some Dtype.Local when l.smem_prefix_for_cast -> l.smem_prefix
    | Some Dtype.Global -> l.buffer_prefix
    | _ -> ""
  in
  let suffix = if is_ptr addrspace || override_ptr then "*" else "" in
  let name =
    Option.value (List.assoc_opt dt l.type_map) ~default:(Dtype.name dt)
  in
  if sz > 1 then
    let element =
      match List.assoc_opt dt l.vector_names with
      | Some element -> element
      | None -> String.map (function ' ' -> '_' | c -> c) name
    in
    prefix ^ element ^ string_of_int sz ^ suffix
  else prefix ^ name ^ suffix

let render_scalar l dt = render_dtype l ~addrspace:(Some Dtype.Reg) dt

let render_type l u =
  let addrspace = addrspace u in
  render_dtype l (dtype u) ~sz:(max_numel u) ~addrspace
    ~override_ptr:(is Op.Index u && addrspace = Some Dtype.Reg)

(* the address of an access, vector-cast if it moves more lanes than the
   pointer's scalar type *)
let render_ptr ctx u =
  if max_numel u > 1 || not (Dtype.equal (dtype u) (dtype (nth u 0))) then
    let t =
      render_dtype ctx.lang (dtype u) ~sz:(max_numel u) ~addrspace:(addrspace u)
        ~override_ptr:true
    in
    strf "((%s)(%s))" t ctx.%{u}
  else ctx.%{u}

let render_access ctx u = "*" ^ render_ptr ctx u
let render_cast ctx u v = strf "(%s)(%s)" (render_type ctx.lang u) v

let render_index ctx buf idx =
  if addrspace buf = Some Dtype.Alu then
    (* lane access in C *)
    match (op idx, src idx) with
    | Op.Cast, [ c ] when is Op.Const c ->
        let lane = Dtype.Value.to_int (cval c) in
        if max_numel buf > ctx.lang.gep_arr_threshold then
          strf "%s[%d]" ctx.%{buf} lane
        else strf "%s.%c" ctx.%{buf} "xyzwabcd".[lane]
    | _ -> strf "(%s)[%s]" ctx.%{buf} ctx.%{idx}
  else strf "(%s+%s)" ctx.%{buf} ctx.%{idx}

let render_buffer ctx x =
  let prefix =
    if addrspace x = Some Dtype.Local then
      ctx.lang.smem_align ^ ctx.lang.smem_prefix
    else ""
  in
  strf "%s%s %s[%d];" prefix
    (render_dtype ctx.lang (dtype x))
    ctx.%{x} (max_numel x)

let wmma_name u =
  match arg u with
  | Wmma { dims = n, m, k; dtype_in; _ } ->
      String.map
        (function ' ' -> '_' | c -> c)
        (strf "WMMA_%d_%d_%d_%s_%s" n m k (Dtype.name dtype_in)
           (Dtype.name (dtype u)))
  | _ -> invalid_arg "a WMMA without its tensor core argument"

(* Base rewrite *)

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let cast_of ?dtype ?name p = Upat.op ?dtype ?name ~src:[ p ] Op.Cast
let c = Upat.cvar "c"

let base_rewrite =
  let r = rule_ctx in
  let str p s = r p (fun _ _ -> Some s) in
  Pattern_matcher.fold
    (fun () -> [
      (* local/reg buffers *)
      r (Upat.op ~name:"x" Op.Buffer) (fun ctx m ->
          Some (render_buffer ctx (m "x")));
      r (Upat.op ~name:"x" Op.Binary) (fun ctx m ->
          match arg (m "x") with
          | Bytes b ->
              let hex =
                String.concat ""
                  (List.map
                     (fun c -> strf "\\x%02x" (Char.code c))
                     (List.of_seq (String.to_seq b)))
              in
              Some (strf "const unsigned char %s[] = \"%s\";" ctx.%{m "x"} hex)
          | _ -> None);
      (* range/loop/if/endif *)
      str (Upat.op ~dtype:[ Dtype.Void ] Op.Range) "for (;;) {";
      r (Upat.op ~name:"x" Op.Range) (fun ctx m ->
          let x = m "x" in
          Some
            (strf "for (%s %s = 0; %s < %s; %s++) {"
               (render_scalar ctx.lang (dtype x))
               ctx.%{x} ctx.%{x}
               ctx.%{nth x 0}
               ctx.%{x}));
      r
        (Upat.op
           ~src:
             [ Upat.wild; Upat.op Op.Range; Upat.var ~dtype:[ Dtype.Bool ] "c" ]
           Op.Backedge)
        (fun ctx m -> Some (strf "  if (!(%s)) { break; }\n}" ctx.%{m "c"}));
      r (Upat.op ~name:"x" Op.If) (fun ctx m ->
          Some (strf "if (%s) {" ctx.%{nth (m "x") 0}));
      str (Upat.v ~op:(Op.Set.of_list [ Op.Endif; Op.End ]) ()) "}";
      (* const *)
      r (cast_of ~dtype:Dtype.floats ~name:"x" c) (fun ctx m ->
          match value (m "c") with
          | `Float v when not (Float.is_finite v) ->
              let l = ctx.lang in
              let s =
                if Float.is_nan v then l.nan
                else if Float.sign_bit v then "-" ^ l.infinity
                else l.infinity
              in
              Some (strf "(%s)" (render_cast ctx (m "x") s))
          | _ -> None);
      r (cast_of ~dtype:[ Dtype.Float32 ] c) (fun _ m ->
          Some (const_str (m "c") ^ "f"));
      r (cast_of ~dtype:[ Dtype.Int64 ] c) (fun _ m ->
          Some (const_str (m "c") ^ "l"));
      r
        (cast_of ~dtype:[ Dtype.Uint64; Dtype.Uint32 ] ~name:"x" c)
        (fun _ m ->
          let dt = dtype (m "x") in
          let t = Dtype.truncate dt (cval (m "c")) in
          Some
            (Format.asprintf "%a%s" Dtype.pp_const t
               (if Dtype.equal dt Dtype.Uint64 then "ul" else "u")));
      r (cast_of ~dtype:[ Dtype.Bool ] c) (fun _ m ->
          Some (if Dtype.Value.to_bool (cval (m "c")) then "1" else "0"));
      (* consts are rendered to larger type and casted *)
      r
        (cast_of
           ~dtype:(Dtype.fp8s @ [ Dtype.Bfloat16; Dtype.Float16 ])
           ~name:"x" c)
        (fun ctx m ->
          Some (strf "(%s)" (render_cast ctx (m "x") (const_str (m "c") ^ "f"))));
      r
        (cast_of ~dtype:[ Dtype.Uint8; Dtype.Uint16 ] ~name:"x" c)
        (fun ctx m ->
          Some (strf "(%s)" (render_cast ctx (m "x") (const_str (m "c") ^ "u"))));
      r
        (cast_of ~dtype:[ Dtype.Int8; Dtype.Int16 ] ~name:"x" c)
        (fun ctx m ->
          Some (strf "(%s)" (render_cast ctx (m "x") (const_str (m "c")))));
      (* default const render *)
      r (cast_of c) (fun _ m -> Some (const_str (m "c")));
      (* casting *)
      r (Upat.op ~name:"x" Op.Cast) (fun ctx m ->
          let x = m "x" in
          if max_numel x > 1 && addrspace x = Some Dtype.Reg then
            Some
              (strf "__builtin_convertvector(%s, %s)"
                 ctx.%{nth x 0}
                 (render_type ctx.lang x))
          else None);
      r (Upat.op ~name:"x" Op.Cast) (fun ctx m ->
          let x = m "x" in
          Some (strf "(%s)" (render_cast ctx x ctx.%{nth x 0})));
      r (Upat.op ~name:"x" Op.Bitcast) (fun ctx m ->
          let x = m "x" in
          if is_ptr (addrspace x) then
            let t = render_dtype ctx.lang (dtype x) ~addrspace:(addrspace x) in
            Some (strf "((%s)(%s))" t ctx.%{nth x 0})
          else None);
      r (Upat.op ~name:"x" Op.Bitcast) (fun ctx m ->
          let x = m "x" in
          let s = nth x 0 in
          Some
            (strf "__builtin_bit_cast(%s, (%s)(%s))" (render_type ctx.lang x)
               (render_type ctx.lang s) ctx.%{s}));
      (* GPU stuff *)
      r (Upat.op Op.Barrier) (fun ctx _ -> Some ctx.lang.barrier);
      r (Upat.op ~name:"x" Op.Special) (fun ctx m ->
          let x = m "x" in
          match arg x with
          | String s -> (
              match List.assoc_opt s.[0] ctx.lang.code_for_workitem with
              | Some f ->
                  Some
                    (strf "%s; /* %s */"
                       (f s.[String.length s - 1])
                       (Render.render (nth x 0)))
              | None -> None)
          | _ -> None);
      (* SHRINK/INDEX *)
      r
        (Upat.op ~src:[ Upat.var "buf"; Upat.var "idx" ] Op.Index)
        (fun ctx m -> Some (render_index ctx (m "buf") (m "idx")));
      r
        (Upat.op
           ~src:[ Upat.var "buf"; Upat.var "idx"; cast_of (Upat.op Op.Const) ]
           Op.Shrink)
        (fun ctx m -> Some (render_index ctx (m "buf") (m "idx")));
      r (Upat.op ~name:"x" Op.Stack) (fun ctx m ->
          let x = m "x" and l = ctx.lang in
          let first, last = l.float4_style in
          let elems =
            String.concat "," (List.map (fun y -> ctx.%{y}) (src x))
          in
          Some (l.float4 (render_type l x) ^ first ^ elems ^ last));
      (* load/store *)
      r
        (Upat.op ~src:[ Upat.var "bidx" ] Op.Load)
        (fun ctx m -> Some (strf "(%s)" (render_access ctx (m "bidx"))));
      r
        (Upat.op
           ~src:[ Upat.var "bidx"; Upat.var "var"; Upat.var "gate" ]
           Op.Load)
        (fun ctx m ->
          Some
            (strf "(%s?%s:%s)"
               ctx.%{m "gate"}
               (render_access ctx (m "bidx"))
               ctx.%{m "var"}));
      r
        (Upat.op ~src:[ Upat.var "bidx"; Upat.var "var" ] Op.Store)
        (fun ctx m ->
          Some (strf "%s = %s;" (render_access ctx (m "bidx")) ctx.%{m "var"}));
      (* alu/gep *)
      r (Upat.op ~name:"x" Op.Wmma) (fun ctx m ->
          let x = m "x" in
          Some
            (strf "__%s(%s, %s, %s)" (wmma_name x)
               ctx.%{nth x 0}
               ctx.%{nth x 1}
               ctx.%{nth x 2}));
      r (Upat.v ~op:Op.Set.alu ~name:"x" ()) (fun ctx m ->
          let x = m "x" in
          let assoc = Op.Set.of_list Op.[ Add; Mul; Xor; Or; And ] in
          let operand v =
            if
              is (op x) v
              && Op.Set.mem (op x) assoc
              && not (Tbl.mem ctx.narrowed v)
            then Helpers.strip_parens ctx.%{v}
            else ctx.%{v}
          in
          Option.map
            (fun f -> f (List.map operand (src x)) (dtype x))
            (List.assoc_opt (op x) ctx.lang.code_for_op));
      (* a division is written whether or not the target lists it, which decides
         only whether code generation makes reciprocals divisions *)
      r
        (Upat.op ~src:[ Upat.var "a"; Upat.var "b" ] Op.Fdiv)
        (fun ctx m -> Some (strf "(%s/%s)" ctx.%{m "a"} ctx.%{m "b"}));
      (* call an external function: the CUSTOM_FUNCTION body holds the callee (a
         function pointer), the other sources are the arguments *)
      r
        (Upat.op ~name:"x" ~allow_any_len:true
           ~src:[ Upat.op ~src:[ Upat.var "fptr" ] Op.Custom_function ]
           Op.Call)
        (fun ctx m ->
          let x = m "x" and l = ctx.lang in
          let args = List.tl (src x) in
          let types = List.map (render_type l) args in
          let values =
            List.map2 (fun t y -> strf "(%s)(%s)" t ctx.%{y}) types args
          in
          Some
            (strf "(((%s%s(*)(%s))(%s))(%s))%s" l.abi
               (render_scalar l (dtype x))
               (String.concat ", " types)
               ctx.%{m "fptr"}
               (String.concat ", " values)
               (if Dtype.equal (dtype x) Dtype.Void then ";" else "")));
      (* custom passes through with format *)
      r
        (Upat.v ~op:(Op.Set.of_list [ Op.Custom; Op.Customi ]) ~name:"x" ())
        (fun ctx m ->
          let x = m "x" in
          match arg x with
          | Code { code; _ } ->
              Some (format code (List.map (fun y -> ctx.%{y}) (src x)))
          | _ -> None);
    ])

(* Non-native floats *)

let alu_but_where = Op.Set.diff Op.Set.alu (Op.Set.of_list [ Op.Where ])

(* [x] computed on its sources cast to float32 *)
let on_floats x =
  Ops.v
    ~src:(List.map (fun v -> cast v Dtype.Float32) (src x))
    ~arg:(arg x) (op x)

let create_non_native_float_pats ?(casting = true) dts =
  let in_dts u = List.exists (Dtype.equal (dtype u)) dts in
  let x = Upat.var ~dtype:dts "x" and y = Upat.var ~dtype:dts "y" in
  Pattern_matcher.v
    (fun () -> [
       (* a weak CONST states no width and cannot be restated: commit it at the
          emulated dtype a sibling src states *)
       rule (Upat.v ~op:Op.Set.alu ~name:"x" ()) (fun m ->
           let x = m "x" in
           let dt = Option.map dtype (List.find_opt in_dts (src x)) in
           Some (Uop_weak.commit_weak_consts x dt));
       rule (Upat.v ~op:alu_but_where ~dtype:dts ~name:"x" ()) (fun m ->
           Some (cast (on_floats (m "x")) (dtype (m "x"))));
       rule
         (Upat.v ~op:Op.Set.alu ~dtype:[ Dtype.Bool ] ~src:[ x; y ] ~name:"alu"
            ())
         (fun m -> Some (on_floats (m "alu")));
     ]
    @
    if not casting then []
    else
      (* add float intermediate casting *)
      [
        rule
          (Upat.op ~dtype:dts ~src:[ Upat.var "x" ] ~name:"y" Op.Cast)
          (fun m ->
            let x = m "x" in
            if Dtype.equal (dtype x) Dtype.Float32 || is Op.Const x then None
            else Some (cast (cast x Dtype.Float32) (dtype (m "y"))));
        rule (Upat.op ~src:[ y ] ~name:"x" Op.Cast) (fun m ->
            let x = m "x" in
            if Dtype.equal (dtype x) Dtype.Float32 then None
            else Some (cast (cast (m "y") Dtype.Float32) (dtype x)));
      ])

let cast_float_to_bf16 x =
  let x = bitcast x Dtype.Uint32 in
  let x =
    O.(
      where
        (-x land int 0x7f800000 <> int 0)
        (x + ((x lsr int 16) land int 1) + int 0x7fff)
        (where (x land int 0xffff <> int 0) (x lor int 0x10000) x))
  in
  bitcast (cast O.(x lsr int 16) Dtype.Uint16) Dtype.Bfloat16

(* manual bfloat16 casting patterns, which need no compiler intrinsics *)
let pm_manual_bf16_cast =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.op ~dtype:[ Dtype.Float32 ]
           ~src:[ Upat.var ~dtype:[ Dtype.Bfloat16 ] "x" ]
           Op.Cast)
        (fun m ->
          let bits = cast (bitcast (m "x") Dtype.Uint16) Dtype.Uint32 in
          Some (bitcast O.(bits lsl int 16) Dtype.Float32));
      rule
        (Upat.op ~dtype:[ Dtype.Bfloat16 ]
           ~src:[ Upat.var ~dtype:[ Dtype.Float32 ] "x" ]
           Op.Cast)
        (fun m -> Some (cast_float_to_bf16 (m "x")));
    ])

(* a bfloat16 stored as ushort renders its const as the bit pattern *)
let pm_bf16_ushort_const =
  Pattern_matcher.fold
    (fun () -> [
      rule (cast_of ~dtype:[ Dtype.Bfloat16 ] c) (fun m ->
          let bits = Dtype.to_storage_scalar Dtype.Bfloat16 (cval (m "c")) in
          Some (Format.asprintf "%au" Dtype.pp_const bits));
    ])

let uops_to_dtypes uops =
  let dtypes u =
    match addrspace u with
    | (Some Dtype.Alu | None)
      when ((not (Dtype.equal (dtype u) Dtype.Void))
           && Option.is_some (shape_opt u))
           [@mutate
             off "the value nodes of a kernel are exactly its shaped ones"] ->
        Some (dtype u, max_numel u)
    | _ -> None
  in
  dedup (List.filter_map dtypes uops)

(* (name, dims, dtype_in, dtype_out, upcast_sizes) *)
let wmma_args uops =
  let args u =
    match arg u with
    | Wmma { dims; dtype_in; _ } when is Op.Wmma u ->
        let last x = List.hd (List.rev (max_shape x)) in
        Some (wmma_name u, dims, dtype_in, dtype u, List.map last (src u))
    | _ -> None
  in
  dedup (List.filter_map args uops)

(* C-style languages *)

let unary f args dt =
  match args with
  | [ x ] -> f x dt
  | _ -> invalid_arg "a unary operation takes one operand"

let call name = unary (fun x _ -> strf "%s(%s)" name x)

(* [o ^ x], with a space where a minus sign would meet the minus that starts
   [x], a negation or a negative constant: [--] is the decrement operator. *)
let joined o x =
  if String.ends_with ~suffix:"-" o && String.starts_with ~prefix:"-" x then
    o ^ " " ^ x
  else o ^ x

let infix o args _ =
  match args with
  | [ a; b ] -> strf "(%s%s)" a (joined o b)
  | _ -> invalid_arg "a binary operation takes two operands"

(* [override table l] is [table] with the operations of [l] replaced, and those
   it lacks appended, as a Python dict is updated *)
let override table l =
  List.map
    (fun (o, f) -> (o, Option.value (List.assoc_opt o l) ~default:f))
    table
  @ List.filter (fun (o, _) -> not (List.mem_assoc o table)) l

let code_for_op =
  Op.
    [
      (Sqrt, call "sqrt");
      (Reciprocal, unary (fun x _ -> strf "(1/%s)" x));
      (Neg, unary (fun x _ -> joined "-" x));
      (Exp2, call "exp2");
      (Log2, call "log2");
      (Sin, call "sin");
      (Trunc, call "trunc");
      (And, infix "&");
      (Xor, infix "^");
      (Or, infix "|");
      (Add, infix "+");
      (Sub, infix "-");
      (Mul, infix "*");
      (Cmod, infix "%");
      (Cdiv, infix "/");
      (Cmpne, infix "!=");
      (Shr, infix ">>");
      (Shl, infix "<<");
      (Cmplt, infix "<");
      ( Where,
        fun args _ ->
          match args with
          | [ a; b; c ] -> strf "(%s?%s:%s)" a b c
          | _ -> invalid_arg "a selection takes three operands" );
      (Cmpeq, infix "==");
    ]

let cstyle =
  {
    abi = "";
    kernel_typedef = (fun _ -> "void");
    buffer_prefix = "";
    buffer_suffix = "";
    smem_align = "";
    smem_prefix = "";
    smem_prefix_for_cast = true;
    var_prefix = "const ";
    var_suffix = "";
    barrier = "";
    code_for_workitem = [];
    extra_args = [];
    float4 = (fun _ -> invalid_arg "the language has no vector constructor");
    float4_style = ("(", ")");
    gep_arr_threshold = 4;
    type_map = [];
    infinity = "INFINITY";
    nan = "NAN";
    promoted = Dtype.[ Int8; Uint8; Int16; Uint16 ];
    vector_names = [];
    code_for_op;
    string_rewrite = base_rewrite;
  }

let render_kernel ?prefix l ~name kernel bufs uops =
  let buftype u =
    let alu = addrspace u = Some Dtype.Alu in
    let volatile = match arg u with Param p -> p.volatile | _ -> false in
    (if volatile then "volatile " else "")
    ^ (if alu then l.var_prefix else "")
    ^ render_dtype l (dtype u) ~addrspace:(addrspace u)
    ^ if alu then l.var_suffix else l.buffer_suffix
  in
  let local_bound u =
    match arg u with
    | String s when is Op.Special u && s.[0] = 'l' ->
        [ Dtype.Value.to_int (vmax (nth u 0)) ]
    | _ -> []
  in
  let launch_bounds = Helpers.prod (List.concat_map local_bound uops) in
  let params =
    List.map (fun (n, u) -> buftype u ^ " " ^ n) bufs @ l.extra_args
  in
  let prg =
    strf "%s %s(%s) {\n%s\n}"
      (l.kernel_typedef launch_bounds)
      name
      (String.concat ", " params)
      (String.concat "\n" kernel)
  in
  match prefix with None -> prg | Some p -> String.concat "\n" p ^ "\n" ^ prg

let closes = Op.Set.of_list [ Op.Endif; Op.End; Op.Backedge ]
let opens = Op.Set.of_list [ Op.If; Op.Range ]
let undeclared = Op.Set.of_list [ Op.Range; Op.Buffer; Op.Binary ]
let always_inlined = Op.Set.of_list [ Op.Index; Op.Shrink; Op.Customi ]
let casts = Op.Set.of_list [ Op.Cast; Op.Bitcast ]

let special_name u =
  match arg u with
  | String s -> s
  | _ -> invalid_arg "a SPECIAL without its name"

let render_uops l uops =
  let r = Tbl.create 256 and child_count = Tbl.create 256 in
  let user = Tbl.create 256 in
  let ctx = { lang = l; r; narrowed = Tbl.create 16 } in
  let children u = Option.value (Tbl.find_opt child_count u) ~default:0 in
  List.iter
    (fun u ->
      List.iter
        (fun v ->
          Tbl.replace child_count v (children v + 1);
          Tbl.replace user v u)
        (src u))
    uops;
  (* C computes an operation on narrow scalars in a wider type, which only an
     assignment narrows back: an inlined operation that is not stored is cast to
     its type, so that each operation rounds or wraps as the kernel says *)
  let narrowed u =
    Op.Set.mem (op u) Op.Set.alu
    && List.mem (dtype u) l.promoted
    && max_numel u = 1
    &&
    match Tbl.find_opt user u with
    | Some s -> not (is Op.Store s && nth s 1 == u)
    | None -> true
  in
  let expand_ssa = Helpers.getenv "EXPAND_SSA" 0 <> 0 in
  let bufs = ref []
  and kernel = ref []
  and depth = ref 1
  and name = ref "test" in
  let counts = Hashtbl.create 8 in
  let count p = Option.value (Hashtbl.find_opt counts p) ~default:0 in
  let failed u =
    let srcs =
      List.map
        (fun x -> Format.asprintf "(%a, %a)" Op.pp (op x) Dtype.pp (dtype x))
        (src u)
    in
    invalid_arg
      (Format.asprintf "failed to render %a %a [%s] %a" Op.pp (op u) Dtype.pp
         (dtype u) (String.concat ", " srcs) pp_arg (arg u))
  in
  let inlined u =
    let o = op u in
    let cast = Op.equal o Op.Cast and casts = Op.Set.mem o casts in
    ((not cast) || max_numel u = 1)
    && ((cast && is Op.Const (nth u 0))
       || Op.Set.mem o always_inlined
       || Op.equal o Op.Load
          && addrspace (nth u 0) = Some Dtype.Reg
          && children u = 1
       || (casts && is_ptr (addrspace u))
       || (casts || Op.equal o Op.Stack || Op.Set.mem o alu_but_where)
          && children u = 1
          && not expand_ssa)
  in
  let visit u =
    match op u with
    | Op.Noop | Op.Group | Op.Const | Op.Custom_function -> ()
    | Op.Stack when src u = [] -> ()
    | Op.After -> Tbl.replace r u ctx.%{nth u 0}
    | Op.Sink -> (
        match arg u with Kernel k -> name := function_name k | _ -> ())
    | Op.Param -> (
        match arg u with
        | Param p ->
            let base =
              match p.name with
              | Some n -> String.map (function ':' -> '_' | c -> c) n
              | None -> strf "data%d" p.slot
            in
            let shape = Option.fold ~none:"" ~some:string_of_int p.size in
            Tbl.replace r u (base ^ "_" ^ shape);
            bufs := (base ^ "_" ^ shape, u) :: !bufs
        | _ -> failed u)
    | o ->
        (* naming *)
        let prefix =
          match o with
          | Op.Special ->
              Tbl.replace r u (special_name u);
              None
          | Op.Range ->
              Tbl.replace r u
                (Axis_type.letter (axis_type u) ^ "idx" ^ range_str u);
              None
          | _ ->
              let p =
                match o with
                | Op.Wmma -> "wmma"
                | Op.Buffer -> "buf"
                | Op.Index -> "bidx"
                | Op.Load -> "val"
                | Op.Cast | Op.Bitcast | Op.Stack -> "cast"
                | _ -> "alu"
              in
              Tbl.replace r u (p ^ string_of_int (count p));
              Some p
        in
        let line =
          match Pattern_matcher.rewrite l.string_rewrite ctx u with
          | Some line -> line
          | None -> failed u
        in
        if Op.Set.mem o closes then decr depth;
        if inlined u then
          if narrowed u then begin
            Tbl.replace ctx.narrowed u ();
            Tbl.replace r u (strf "(%s)" (render_cast ctx u line))
          end
          else Tbl.replace r u line
        else begin
          let declared =
            (not (Op.Set.mem o undeclared))
            && not (Dtype.equal (dtype u) Dtype.Void)
          in
          let line =
            if declared then
              strf "%s %s = %s%s" (render_type l u) ctx.%{u} line
                (if Op.equal o Op.Special then "" else ";")
            else if Op.equal o Op.Cast then
              (* a discarded value renders as a bare statement *)
              line ^ ";"
            else line
          in
          let indent s = String.make (2 * !depth) ' ' ^ s in
          kernel :=
            String.concat "\n"
              (List.map indent (String.split_on_char '\n' line))
            :: !kernel;
          (* if it was used, increment *)
          Option.iter (fun p -> Hashtbl.replace counts p (count p + 1)) prefix
        end;
        if Op.Set.mem o opens then incr depth
  in
  List.iter visit uops;
  (!name, List.rev !kernel, List.rev !bufs)

let render render_kernel l uops =
  let name, kernel, bufs = render_uops l uops in
  render_kernel l ~name kernel bufs uops

(* Clang *)

let clang_lang =
  let abi = if Sys.win32 then "__attribute__((ms_abi)) " else "" in
  let builtin name x dt =
    if Dtype.equal dt Dtype.Float64 then strf "__builtin_%s(%s)" name x
    else strf "__builtin_%sf(%s)" name x
  in
  let dropped = Op.[ Exp2; Sin; Log2; Trunc; Reciprocal ] in
  {
    cstyle with
    float4 = (fun t -> "(" ^ t ^ ")");
    float4_style = ("{", "}");
    gep_arr_threshold = 0;
    infinity = "__builtin_inff()";
    nan = "__builtin_nanf(\"\")";
    barrier = "__atomic_thread_fence(__ATOMIC_SEQ_CST);";
    buffer_suffix = " restrict";
    type_map = [ (Dtype.Bool, "_Bool"); (Dtype.Float16, "__fp16") ];
    (* __fp16 is a storage format: its operations compute in float *)
    promoted = Dtype.Float16 :: cstyle.promoted;
    code_for_op =
      override
        (List.filter (fun (o, _) -> not (List.mem o dropped)) code_for_op)
        Op.
          [
            (Sqrt, unary (builtin "sqrt"));
            (Trunc, unary (builtin "trunc"));
            (Fdiv, infix "/");
          ];
    abi;
    kernel_typedef = (fun _ -> abi ^ "void");
  }

(* LLVM legalizes a double to half cast on CPUs without native support (such as
   x86 without AVX512-FP16) into a compiler-rt libcall *)
let clang_extra_matcher =
  Pattern_matcher.concat
    [
      Pattern_matcher.v
        (fun () -> [
          rule
            (Upat.cast (Upat.var ~dtype:[ Dtype.Float64 ] "x") Dtype.Float16)
            (fun m -> Some (cast (cast (m "x") Dtype.Float32) Dtype.Float16));
        ]);
      create_non_native_float_pats [ Dtype.Bfloat16 ];
      pm_manual_bf16_cast;
    ]

let clang_vector_prefix l (dt, count) =
  let rec pow2_floor n p = if 2 * p > n then p else pow2_floor n (2 * p) in
  (* round (down) to a power of two, as clang does by default *)
  let alignment =
    if Helpers.getenv "ALIGNED" 1 <> 0 && not (Dtype.is_bool dt) then
      pow2_floor (Dtype.itemsize dt * count) 1
    else 1
  in
  let vec = render_dtype l dt ~sz:count ~addrspace:(Some Dtype.Reg) in
  strf "typedef %s %s __attribute__((aligned(%d),ext_vector_type(%d)));"
    (render_scalar l dt) vec alignment count

let clang_kernel l ~name kernel bufs uops =
  let vectors =
    List.filter (fun (_, count) -> count > 1) (uops_to_dtypes uops)
  in
  let defines = String.concat "\n" (List.map (clang_vector_prefix l) vectors) in
  defines ^ "\n" ^ render_kernel l ~name kernel bufs uops ^ "\n"

let clang (target : Helpers.Target.t) =
  let arch = target.arch in
  let native dt =
    ((not (Dtype.equal dt Dtype.Bfloat16))
    || String.starts_with ~prefix:"x86" arch
    || String.starts_with ~prefix:"arm" arch)
    && not (List.mem dt Dtype.fp8s)
  in
  Renderer.v ~name:"ClangRenderer" ~has_local:false ~global_max:[ 1; 0; 0 ]
    ~extra_matcher:clang_extra_matcher ~code_for_op:clang_lang.code_for_op
    ~native
    ~render:(render clang_kernel clang_lang)
    ~compiler:(Compiler_cpu.clang arch) target

(* Metal *)

let metal_lang =
  let axis base c =
    strf "%s.%c" base (Char.chr (Char.code 'x' + Char.code c - Char.code '0'))
  in
  {
    cstyle with
    kernel_typedef = (fun _ -> "kernel void");
    buffer_prefix = "device ";
    smem_prefix = "threadgroup __attribute__((aligned(16))) ";
    var_prefix = "constant ";
    var_suffix = "&";
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    float4 = Fun.id;
    code_for_workitem = [ ('g', axis "gid"); ('l', axis "lid") ];
    extra_args =
      [
        "constant args_t& args [[buffer(0)]]";
        "uint3 gid [[threadgroup_position_in_grid]]";
        "uint3 lid [[thread_position_in_threadgroup]]";
      ];
    type_map = [ (Dtype.Uint32, "uint"); (Dtype.Bfloat16, "bfloat") ];
    (* Metal's vector types are named after one-word elements *)
    vector_names =
      Dtype.
        [
          (Int8, "char"); (Uint8, "uchar"); (Uint16, "ushort"); (Uint64, "ulong");
        ];
    code_for_op = override code_for_op [ (Op.Sin, call "precise::sin") ];
    string_rewrite =
      Pattern_matcher.append
        (Pattern_matcher.fold
           (fun () -> [
             rule_ctx (Upat.op ~name:"x" Op.Bitcast) (fun ctx m ->
                 let x = m "x" and l = ctx.lang in
                 if is_ptr (addrspace x) then None
                 else
                   let s = nth x 0 in
                   Some
                     (strf "as_type<%s>((%s)(%s))"
                        (render_scalar l (dtype x))
                        (render_scalar l (dtype s))
                        ctx.%{s}));
           ]))
        base_rewrite;
  }

(* upcast to float32 the operations that have no bfloat16: Metal computes them
   in float, and a float does not convert to a bfloat implicitly *)
let metal_extra_matcher =
  Pattern_matcher.append
    (Pattern_matcher.v
       (fun () -> [
         rule
           (Upat.v
              ~op:(Op.Set.of_list Op.[ Sqrt; Exp2; Log2; Sin; Trunc ])
              ~dtype:[ Dtype.Bfloat16 ] ~name:"x" ())
           (fun m -> Some (cast (on_floats (m "x")) Dtype.Bfloat16));
       ]))
    pm_manual_bf16_cast

let metal_wmma l (name, _, dtype_in, dtype_out, _) =
  let vec dt = render_dtype l dt ~sz:2 ~addrspace:(Some Dtype.Reg) in
  let out = vec dtype_out and in_ = vec dtype_in in
  let elements i =
    strf
      "  mat_a.thread_elements()[%d] = a[%d]; mat_b.thread_elements()[%d] = \
       b[%d]; mat_c.thread_elements()[%d] = c[%d];"
      i i i i i i
  in
  String.concat "\n"
    [
      strf "%s __%s(%s a, %s b, %s c){" out name in_ in_ out;
      strf "  simdgroup_%s8x8 mat_a, mat_b; simdgroup_%s8x8 mat_c;"
        (render_scalar l dtype_in)
        (render_scalar l dtype_out);
      elements 0;
      elements 1;
      "  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);";
      strf
        "  return %s(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);"
        out;
      "}";
    ]

(* one argument buffer: a struct of the buffer pointers and the scalars, so a
   binding is a GPU address *)
let metal_kernel l ~name kernel bufs uops =
  let args =
    List.map
      (fun (n, u) -> (n, render_dtype l (dtype u) ~addrspace:(addrspace u)))
      bufs
  in
  let fields = List.map (fun (n, t) -> strf "%s %s;" t n) args in
  let loads = List.map (fun (n, t) -> strf "%s %s = args.%s;" t n n) args in
  let prefix =
    [ "#include <metal_stdlib>"; "using namespace metal;" ]
    @ List.map (metal_wmma l) (wmma_args uops)
    @ [ "struct args_t { " ^ String.concat " " fields ^ " };" ]
  in
  render_kernel l ~prefix ~name
    (("  " ^ String.concat " " loads) :: kernel)
    [] uops

let metal (target : Helpers.Target.t) =
  let arch = target.arch in
  let family =
    if not (String.starts_with ~prefix:"Apple" arch) then None
    else
      match number_from 5 arch with
      | Some f -> Some f
      | None -> invalid_arg (strf "%S is not an Apple GPU family" arch)
  in
  let from f = match family with Some n -> n >= f | None -> false in
  let native dt =
    ((not (Dtype.equal dt Dtype.Bfloat16)) || from 6)
    && not (List.mem dt (Dtype.Float64 :: Dtype.fp8s))
  in
  Renderer.v ~name:"MetalRenderer"
    ~tensor_cores:(if from 7 then Tc.metal else [])
    ~extra_matcher:metal_extra_matcher ~code_for_op:metal_lang.code_for_op
    ~native
    ~render:(render metal_kernel metal_lang)
    ~compiler:(Compiler_metal.compiler ()) target

(* CUDA *)

let nms =
  List.map (String.make 1) (List.of_seq (String.to_seq "xyzwabcdefghijkl"))
  @ List.init 16 (fun i -> strf "v%d" (i + 16))

let rec take n = function x :: l when n > 0 -> x :: take (n - 1) l | _ -> []

(* an infinity keeps its meaning, which the saturating conversions to an 8-bit
   float lose: e5m2 has infinities, and e4m3 gives NaN *)
let fp8_infinity dt = if Dtype.equal dt Dtype.Fp8e5m2 then 0x7c else 0x7f

let cuda_fp8_guard =
  "template <class T, class F> __device__ __forceinline__ T tg_fp8(F v, \
   unsigned char inf) { T r = T(v); if (isinf((double)v)) r.__x = \
   (signbit((double)v) ? 0x80 : 0) | inf; return r; }"

let is_fp8_guarded u =
  is Op.Cast u
  && List.mem (dtype u) Dtype.fp8_ocp
  && List.mem (dtype (nth u 0)) Dtype.floats
  && max_numel u = 1

let cuda_lang =
  let half dt = Dtype.equal dt Dtype.Float16 || Dtype.equal dt Dtype.Bfloat16 in
  let h name =
    unary (fun x dt -> strf "%s%s(%s)" (if half dt then "h" else "") name x)
  in
  let axis base c =
    strf "%s.%c" base (Char.chr (Char.code 'x' + Char.code c - Char.code '0'))
  in
  {
    cstyle with
    (* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html *)
    kernel_typedef = strf "extern \"C\" __global__ void __launch_bounds__(%d)";
    smem_prefix = "__shared__ __align__(16) ";
    smem_prefix_for_cast = false;
    barrier = "__syncthreads();";
    float4 = (fun t -> "make_" ^ t);
    gep_arr_threshold = 8;
    code_for_workitem = [ ('g', axis "blockIdx"); ('l', axis "threadIdx") ];
    code_for_op =
      override code_for_op
        Op.
          [
            (Trunc, h "trunc");
            (Sin, h "sin");
            (Log2, h "log2");
            (Exp2, h "exp2");
            (Sqrt, h "sqrt");
            ( Reciprocal,
              unary (fun x dt ->
                  if half dt then strf "hrcp(%s)" x else strf "(1/%s)" x) );
          ];
    type_map =
      [
        (Dtype.Uint32, "uint");
        (Dtype.Bfloat16, "nv_bfloat16");
        (Dtype.Fp8e4m3, "__nv_fp8_e4m3");
        (Dtype.Fp8e5m2, "__nv_fp8_e5m2");
      ];
    string_rewrite =
      Pattern_matcher.append
        (Pattern_matcher.fold
           (fun () -> [
             rule_ctx (cast_of ~dtype:Dtype.fp8_ocp ~name:"x" c) (fun ctx m ->
                 match cval (m "c") with
                 | `Float v when Float.abs v = Float.infinity ->
                     let x = m "x" in
                     let sign = if Float.sign_bit v then 0x80 else 0 in
                     Some
                       (strf "(tg_bitcast<%s>((unsigned char)0x%x))"
                          (render_type ctx.lang x)
                          (sign lor fp8_infinity (dtype x)))
                 | _ -> None);
             rule_ctx (Upat.op ~dtype:Dtype.fp8_ocp ~name:"x" Op.Cast)
               (fun ctx m ->
                 let x = m "x" in
                 if is_fp8_guarded x then
                   Some
                     (strf "tg_fp8<%s>(%s, 0x%x)" (render_type ctx.lang x)
                        ctx.%{nth x 0}
                        (fp8_infinity (dtype x)))
                 else None);
             rule_ctx (Upat.op ~name:"x" Op.Bitcast) (fun ctx m ->
                 let x = m "x" and l = ctx.lang in
                 if is_ptr (addrspace x) then None
                 else
                   let s = nth x 0 in
                   Some
                     (strf "tg_bitcast<%s>((%s)(%s))"
                        (render_scalar l (dtype x))
                        (render_scalar l (dtype s))
                        ctx.%{s}));
           ]))
        base_rewrite;
  }

let cuda_extra_matcher =
  Pattern_matcher.append
    (create_non_native_float_pats ~casting:false Dtype.fp8s)
    (Pattern_matcher.v
       (fun () -> [
         rule
           (Upat.op ~dtype:Dtype.fp8s
              ~src:[ Upat.var ~dtype:Dtype.fp8s "x" ]
              ~name:"y" Op.Cast)
           (fun m ->
             let x = m "x" and y = m "y" in
             if Dtype.equal (dtype x) (dtype y) then None
             else Some (cast (cast x Dtype.Float32) (dtype y)));
       ]))

let cuda_vector_prefix l (dt, count) =
  let vec = render_dtype l dt ~sz:count ~addrspace:(Some Dtype.Reg)
  and scal = render_scalar l dt in
  let names = take count nms in
  let elems = String.concat ", " names in
  let header = String.concat ", " (List.map (fun x -> scal ^ " " ^ x) names) in
  strf
    "struct __align__(%d) %s { %s %s; }; __device__ %s make_%s(%s) { %s \
     r={%s}; return r; }"
    (Dtype.itemsize dt * count)
    vec scal elems vec vec header vec elems

let cuda_wmma l (name, (n, m, k), dtype_in, dtype_out, upcast_sizes) =
  let dt_map_in = function
    | Dtype.Float32 -> "tf32"
    | Dtype.Float16 -> "f16"
    | Dtype.Bfloat16 -> "bf16"
    | Dtype.Fp8e4m3 -> "e4m3"
    | Dtype.Fp8e5m2 -> "e5m2"
    | dt -> invalid_arg (Format.asprintf "no tensor core takes %a" Dtype.pp dt)
  in
  let dt_map_out = function
    | Dtype.Float32 -> "f32"
    | Dtype.Float16 -> "f16"
    | dt -> invalid_arg (Format.asprintf "no tensor core gives %a" Dtype.pp dt)
  in
  let sa, sb, sc =
    match upcast_sizes with
    | [ a; b; c ] -> (a, b, c)
    | _ -> invalid_arg "a WMMA takes three operands"
  in
  let vec dt size = render_dtype l dt ~sz:size ~addrspace:(Some Dtype.Reg) in
  let ta = vec dtype_in sa and tb = vec dtype_in sb and tc = vec dtype_out sc in
  (* 4 bytes is the size of a CUDA register *)
  let regs dt size = size * Dtype.itemsize dt / 4 in
  let na = regs dtype_in sa
  and nb = regs dtype_in sb
  and nc = regs dtype_out sc in
  let operands from len =
    String.concat ", " (List.init len (fun i -> strf "%%%d" (from + i)))
  in
  let constraints kind name len =
    String.concat ", "
      (List.init len (fun i -> strf "\"%s\"(%s_pk[%d])" kind name i))
  in
  let dt_in = dt_map_in dtype_in and dt_out = dt_map_out dtype_out in
  (* mma operands => {c}, {a}, {b}, {c} *)
  String.concat "\n"
    [
      strf "__device__ %s __%s(%s a, %s b, %s c){" tc name ta tb tc;
      "  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);";
      strf "  asm(\"mma.sync.aligned.m%dn%dk%d.row.col.%s.%s.%s.%s\"" m n k
        dt_out dt_in dt_in dt_out;
      strf "      \"{%s}, {%s},\"" (operands 0 nc) (operands nc na);
      strf "      \"{%s}, {%s};\"" (operands (nc + na) nb) (operands 0 nc);
      strf "    : %s" (constraints "+r" "c" nc);
      strf "    : %s, %s);" (constraints "r" "a" na) (constraints "r" "b" nb);
      "  return c;";
      "}";
    ]

let cuda_kernel l ~name kernel bufs uops =
  let used = uops_to_dtypes uops in
  let uses p = List.exists (fun (dt, _) -> p dt) used in
  let is_fp8 dt = List.mem dt Dtype.fp8s in
  let vector (dt, count) =
    (List.mem count [ 4; 8 ] && List.mem dt [ Dtype.Float16; Dtype.Bfloat16 ])
    || (List.mem count [ 2; 4; 8; 16 ] && is_fp8 dt)
  in
  let prefix =
    [
      "typedef unsigned int uint;";
      "#define INFINITY (__int_as_float(0x7f800000))";
      "#define NAN (__int_as_float(0x7fffffff))";
      "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F \
       v) { union U { F f; T t; }; U u; u.f = v; return u.t; }";
    ]
    @ (if uses is_fp8 then [ "#include <cuda_fp8.h>" ] else [])
    @ (if List.exists is_fp8_guarded uops then [ cuda_fp8_guard ] else [])
    @ (if uses (Dtype.equal Dtype.Float16) then [ "#include <cuda_fp16.h>" ]
       else [])
    @ (if uses (Dtype.equal Dtype.Bfloat16) then [ "#include <cuda_bf16.h>" ]
       else [])
    @ List.map (cuda_vector_prefix l) (List.filter vector used)
    @ List.map (cuda_wmma l) (wmma_args uops)
  in
  render_kernel l ~prefix ~name kernel bufs uops

let cuda (target : Helpers.Target.t) =
  let arch = target.arch in
  let tensor_cores = Tc.cuda arch in
  let ver =
    match number_from 3 arch with
    | Some v -> v
    | None -> invalid_arg (strf "%S has no compute capability" arch)
  in
  let native dt =
    ((not (Dtype.equal dt Dtype.Float16)) || ver >= 53)
    && ((not (Dtype.equal dt Dtype.Bfloat16)) || ver >= 80)
    && ((not (List.mem dt Dtype.fp8_ocp)) || ver >= 89)
    && not (List.mem dt Dtype.fp8_fnuz)
  in
  let compiler =
    Compiler_cuda.nvrtc ~ptx:(target.device = "CUDA")
      ~cache_key:(String.lowercase_ascii target.device)
      arch
  in
  Renderer.v ~name:"CUDARenderer"
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~local_max:[ 1024; 1024; 64 ] ~shared_max:49152 ~tensor_cores
    ~extra_matcher:cuda_extra_matcher ~code_for_op:cuda_lang.code_for_op ~native
    ~render:(render cuda_kernel cuda_lang)
    ~compiler target

(* HIP *)

let fp8_index dt =
  match dt with
  | Dtype.Fp8e4m3 | Dtype.Fp8e4m3fnuz -> 0
  | Dtype.Fp8e5m2 | Dtype.Fp8e5m2fnuz -> 1
  | dt -> invalid_arg (Format.asprintf "%a is not an 8-bit float" Dtype.pp dt)

let amd_fp8s = function
  | "gfx942" -> Dtype.fp8_fnuz
  | "gfx950" -> Dtype.fp8_ocp
  | _ -> []

let ocml op =
  unary (fun x dt ->
      let bits =
        match dt with Dtype.Float16 -> 16 | Dtype.Float64 -> 64 | _ -> 32
      in
      strf "__ocml_%s_f%d(%s)" op bits x)

let gpu arch = List.hd (String.split_on_char ':' arch)
let is_cdna arch = List.mem (gpu arch) [ "gfx942"; "gfx950" ]
let is_cdna4 arch = gpu arch = "gfx950"

let cdna_rewrite =
  Pattern_matcher.fold
    (fun () -> [
      rule_ctx (Upat.op ~name:"x" Op.Wmma) (fun ctx m ->
          let x = m "x" in
          match arg x with
          | Wmma { dims = _, _, 128; _ } ->
              let i = fp8_index (dtype (nth x 0)) in
              Some
                (strf "__%s(%s, %s, %s, %d, %d, 0, 0, 0, 0)" (wmma_name x)
                   ctx.%{nth x 0}
                   ctx.%{nth x 1}
                   ctx.%{nth x 2}
                   i i)
          | _ -> None);
      rule_ctx (Upat.op ~name:"x" Op.Wmma) (fun ctx m ->
          let x = m "x" in
          Some
            (strf "__%s(%s, %s, %s, 0, 0, 0)" (wmma_name x)
               ctx.%{nth x 0}
               ctx.%{nth x 1}
               ctx.%{nth x 2}));
      rule_ctx (cast_of ~dtype:Dtype.fp8s ~name:"x" c) (fun ctx m ->
          let l = ctx.lang in
          let v =
            match cval (m "c") with
            | `Float v when Float.is_nan v -> l.nan
            | `Float v when v = Float.infinity -> l.infinity
            | `Float v when v = Float.neg_infinity -> "-" ^ l.infinity
            | _ -> const_str (m "c") ^ "f"
          in
          Some (strf "f32_to_fp8(%s, %d)" v (fp8_index (dtype (m "x")))));
      rule_ctx
        (Upat.op ~dtype:Dtype.fp8s
           ~src:[ Upat.v ~dtype:[ Dtype.Float32 ] () ]
           ~name:"x" Op.Cast)
        (fun ctx m ->
          let x = m "x" in
          Some (strf "f32_to_fp8(%s, %d)" ctx.%{nth x 0} (fp8_index (dtype x))));
      rule_ctx
        (Upat.op ~dtype:[ Dtype.Float32 ]
           ~src:[ Upat.var ~dtype:Dtype.fp8s "y" ]
           ~name:"x" Op.Cast)
        (fun ctx m ->
          let y = m "y" in
          let kind = if fp8_index (dtype y) = 0 then "fp8" else "bf8" in
          Some
            (strf "__builtin_amdgcn_cvt_f32_%s((unsigned int)%s, 0)" kind
               ctx.%{nth (m "x") 0}));
    ])

(* a load flagged nontemporal bypasses the caches (only used on global loads) *)
let nontemporal_rewrite =
  Pattern_matcher.fold
    (fun () -> [
      rule_ctx
        (Upat.op ~arg:(String "nontemporal") ~src:[ Upat.var "bidx" ] Op.Load)
        (fun ctx m ->
          Some
            (strf "__builtin_nontemporal_load(%s)" (render_ptr ctx (m "bidx"))));
    ])

let hip_lang arch =
  let rewrite =
    if is_cdna arch then Pattern_matcher.append cdna_rewrite base_rewrite
    else base_rewrite
  in
  let rewrite = Pattern_matcher.append nontemporal_rewrite rewrite in
  {
    cstyle with
    (* https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size *)
    kernel_typedef =
      strf
        "extern \"C\" __attribute__((global)) void \
         __attribute__((amdgpu_flat_work_group_size(1, %d)))";
    code_for_workitem =
      [
        ('g', strf "__ockl_get_group_id(%c)");
        ('l', strf "__ockl_get_local_id(%c)");
      ];
    code_for_op =
      override code_for_op
        Op.
          [
            (Trunc, ocml "trunc");
            (Sin, ocml "sin");
            (Log2, ocml "log2");
            (Exp2, ocml "exp2");
            (Sqrt, ocml "sqrt");
          ];
    smem_prefix = "__attribute__((shared, aligned(16)))";
    smem_prefix_for_cast = false;
    barrier =
      "__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");"
      ^ "__builtin_amdgcn_s_barrier();"
      ^ "__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");";
    float4 = (fun t -> "make_" ^ t);
    type_map =
      (Dtype.Bfloat16, "hip_bfloat16")
      :: List.map
           (fun d -> (d, if fp8_index d = 0 then "hip_fp8" else "hip_bf8"))
           Dtype.fp8s;
    string_rewrite =
      (if is_cdna4 arch then rewrite
       else Pattern_matcher.append pm_bf16_ushort_const rewrite);
  }

let hip_extra_matcher arch =
  Pattern_matcher.concat
    ([
       create_non_native_float_pats (Dtype.Bfloat16 :: Dtype.fp8s);
       Pattern_matcher.v
         (fun () -> [
           rule (Upat.op ~dtype:[ Dtype.Float32 ] ~name:"x" Op.Wmma) (fun m ->
               match src (m "x") with
               | [ a; b; acc ]
                 when max_numel a = 8 && List.mem (dtype a) Dtype.fp8s ->
                   let u64 u = bitcast u Dtype.Uint64 in
                   Some (replace (m "x") ~src:[ u64 a; u64 b; acc ])
               | _ -> None);
         ]);
     ]
    @ if is_cdna4 arch then [] else [ pm_manual_bf16_cast ])

let hip_vector_prefix l (dt, count) =
  let vec = render_dtype l dt ~sz:count ~addrspace:(Some Dtype.Reg)
  and scal = render_scalar l dt in
  let names = take count nms in
  strf
    "typedef %s %s __attribute__((ext_vector_type(%d)));\n\
     static inline __attribute__((device)) %s make_%s(%s) { return { %s }; }"
    scal vec count vec vec
    (String.concat ", " (List.map (fun x -> scal ^ " " ^ x) names))
    (String.concat ", " names)

let hip_wmma arch tensor_cores type_map (name, (n, m, k), dtype_in, dtype_out, _)
    =
  let tm dt =
    match List.assoc_opt dt !type_map with
    | Some s -> s
    | None ->
        invalid_arg (Format.asprintf "no tensor core takes %a" Dtype.pp dt)
  in
  if is_cdna arch then begin
    (match (n, m, k) with
    | 16, 16, 16 -> type_map := (Dtype.Bfloat16, "bf16_1k") :: !type_map
    | 16, 16, 32 ->
        type_map :=
          (Dtype.Bfloat16, "_bf16") :: (Dtype.Float16, "_f16") :: !type_map
    | 16, 16, 128 ->
        type_map :=
          (Dtype.Fp8e4m3, "_f8f6f4") :: (Dtype.Fp8e5m2, "_f8f6f4") :: !type_map
    | _ -> ());
    strf "#define __%s __builtin_amdgcn_mfma_%sf32_%dx%dx%d%s" name
      (if k = 128 then "scale_" else "")
      n m k (tm dtype_in)
  end
  else if List.equal Tc.equal tensor_cores Tc.amd_rdna4 then
    (* #define __WMMA_16_16_16_half_half
       __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12 *)
    strf "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s_w32_gfx12" name
      (tm dtype_out) (tm dtype_in)
  else if Dtype.equal dtype_out Dtype.Int32 then
    String.concat "\n"
      [
        "typedef int wmma_int4 __attribute__((ext_vector_type(4)));";
        strf
          "static inline __attribute__((device)) int8 __%s(signed_char16 a, \
           signed_char16 b, int8 c) {"
          name;
        "  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, \
         __builtin_bit_cast(wmma_int4, a),";
        "    true, __builtin_bit_cast(wmma_int4, b), c, false);";
        "}";
      ]
  else if Dtype.equal dtype_out Dtype.Float32 then
    strf "#define __%s __builtin_amdgcn_wmma_f32_16x16x16_%s_w32" name
      (if Dtype.equal dtype_in Dtype.Float16 then "f16" else "bf16")
  else
    String.concat "\n"
      [
        strf
          "static inline __attribute__((device)) half8 __%s(half16 a, half16 \
           b, half8 c) {"
          name;
        "  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { \
         c_frag[n*2] = c[n]; }";
        "  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, \
         false);";
        "  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;";
        "}";
      ]

let hip_kernel arch tensor_cores l ~name kernel bufs uops =
  let used = uops_to_dtypes uops in
  let uses p = List.exists (fun (dt, _) -> p dt) used in
  let is_fp8 dt = List.mem dt Dtype.fp8s in
  let const_cast u = is Op.Cast u && is Op.Const (nth u 0) in
  let non_finite u =
    match cval (nth u 0) with `Float v -> not (Float.is_finite v) | _ -> false
  in
  let specials = List.exists (is Op.Special) uops in
  let ockl =
    if not specials then []
    else
      List.map
        (fun n -> (strf "__ockl_get_%s" n, "unsigned int", "size_t", "const"))
        [ "local_id"; "group_id"; "local_size" ]
  in
  let ocml_ops =
    Op.
      [
        (Exp2, ("exp2", "pure"));
        (Log2, ("log2", "pure"));
        (Sqrt, ("sqrt", "const"));
        (Sin, ("sin", ""));
        (Trunc, ("trunc", ""));
      ]
  in
  let ocml (o, dt) =
    match List.assoc_opt o ocml_ops with
    | Some (n, attr) when List.mem dt Dtype.[ Float16; Float32; Float64 ] ->
        Some
          ( strf "__ocml_%s_f%d" n (Dtype.bitsize dt),
            Dtype.name dt,
            Dtype.name dt,
            attr )
    | _ -> None
  in
  let ocml =
    List.filter_map ocml (dedup (List.map (fun u -> (op u, dtype u)) uops))
  in
  let to_fp8 u =
    is Op.Cast u
    && is_fp8 (dtype u)
    && (Dtype.equal (dtype (nth u 0)) Dtype.Float32 || is Op.Const (nth u 0))
  in
  let f32_to_fp8 () =
    let fp8_max =
      match amd_fp8s arch with
      | dt :: _ -> Format.asprintf "%a" Dtype.pp_const (Dtype.max dt)
      | [] -> invalid_arg (strf "%S has no 8-bit float" arch)
    in
    String.concat "\n"
      [
        "static inline __attribute__((device)) unsigned char f32_to_fp8(float \
         v, int is_bf8) {";
        strf
          "  v = \
           (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:%sf,is_bf8?-57344.0f:-%sf) \
           : v;"
          fp8_max fp8_max;
        "  return (unsigned \
         char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));";
        "}";
      ]
  in
  let declare (meth, dti, dto, attr) =
    strf "extern \"C\" __attribute__((device%s)) %s %s(%s);"
      (if attr = "" then "" else ", " ^ attr)
      dto meth dti
  in
  let type_map =
    ref
      ((Dtype.Bfloat16, "bf16") :: (Dtype.Float32, "f32")
     :: (Dtype.Float16, "f16")
      :: List.map
           (fun d -> (d, if fp8_index d = 0 then "_fp8_fp8" else "_bf8_bf8"))
           Dtype.fp8s)
  in
  let prefix =
    (if List.exists (fun u -> const_cast u && non_finite u) uops then
       [
         "#define INFINITY (__builtin_inff())";
         "#define NAN (__builtin_nanf(\"\"))";
       ]
     else [])
    @ (if specials then [ "typedef long unsigned int size_t;" ] else [])
    @ (if uses (Dtype.equal Dtype.Bfloat16) then
         [
           strf "typedef %s hip_bfloat16;"
             (if is_cdna4 arch then "__bf16" else "unsigned short");
         ]
       else [])
    @ (if uses (Dtype.equal Dtype.Float16) then [ "#define half _Float16" ]
       else [])
    @ (if uses is_fp8 then
         [ "typedef unsigned char hip_bf8;"; "typedef unsigned char hip_fp8;" ]
       else [])
    @ (if List.exists to_fp8 uops then [ f32_to_fp8 () ] else [])
    @ List.map declare (ockl @ ocml)
    @ List.map (hip_vector_prefix l)
        (List.filter (fun (_, count) -> count > 1) used)
    @ List.map (hip_wmma arch tensor_cores type_map) (wmma_args uops)
  in
  render_kernel l ~prefix ~name kernel bufs uops

let hip (target : Helpers.Target.t) =
  (* gfx942 => MI300, gfx1100 => RX 7900, gfx1201 => RX 9700 *)
  let arch = target.arch in
  let tensor_cores = Tc.amd arch and lang = hip_lang arch in
  let native dt =
    (not (List.mem dt Dtype.fp8s)) || List.mem dt (amd_fp8s arch)
  in
  (* the global limit is only really needed on gfx12, though gfx11 reports the
     same *)
  Renderer.v ~name:"HIPRenderer" ~shared_max:65536
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~global_prod_max:[ 0xFFFFFFFF; 0xFFFFFFFF; 0xFFFFFFFF ]
    ~tensor_cores ~extra_matcher:(hip_extra_matcher arch)
    ~code_for_op:lang.code_for_op ~native
    ~render:(render (hip_kernel arch tensor_cores) lang)
    ~compiler:(Compiler_amd.hip arch) target
