(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
open Program

let strf = Printf.sprintf

(* Helpers *)

let strip_parens s =
  let n = String.length s in
  if n >= 2 && s.[0] = '(' && s.[n - 1] = ')' then String.sub s 1 (n - 2)
  else s

let dedup lst =
  let seen = Hashtbl.create 16 in
  List.filter (fun x ->
    if Hashtbl.mem seen x then false
    else (Hashtbl.replace seen x (); true)) lst

let render_custom_fmt fmt args =
  let a = Array.of_list args in
  let n = Array.length a in
  let next = ref 0 in
  let buf = Buffer.create (String.length fmt) in
  let len = String.length fmt in
  let i = ref 0 in
  while !i < len do
    match fmt.[!i] with
    | '{' when !i + 1 < len && fmt.[!i + 1] = '{' ->
        Buffer.add_char buf '{'; i := !i + 2
    | '{' ->
        let j = ref (!i + 1) in
        while !j < len && fmt.[!j] <> '}' do incr j done;
        let field = String.sub fmt (!i + 1) (!j - !i - 1) |> String.trim in
        let idx =
          if field = "" then (let k = !next in incr next; k)
          else match int_of_string_opt field with
            | Some k -> k
            | None -> invalid_arg (strf "render_custom_fmt: non-numeric placeholder {%s}" field)
        in
        if idx >= 0 && idx < n then Buffer.add_string buf a.(idx);
        i := !j + 1
    | '}' when !i + 1 < len && fmt.[!i + 1] = '}' ->
        Buffer.add_char buf '}'; i := !i + 2
    | ch -> Buffer.add_char buf ch; incr i
  done;
  Buffer.contents buf

let vec_elem_name i =
  if i < 16 then
    let names = "xyzwabcdefghijkl" in
    String.make 1 names.[i]
  else strf "v%d" i

let prod = List.fold_left ( * ) 1

(* Type rendering *)

type scalar_name = Dtype.scalar -> string

let base_scalar_name : scalar_name = function
  | Dtype.Void -> "void" | Bool -> "bool"
  | Int8 -> "signed char" | Int16 -> "short" | Int32 -> "int"
  | Int64 -> "long long" | Uint8 -> "unsigned char" | Uint16 -> "unsigned short"
  | Uint32 -> "unsigned int" | Uint64 -> "unsigned long long"
  | Float16 -> "half" | Bfloat16 -> "__bf16" | Float32 -> "float"
  | Float64 -> "double" | Fp8e4m3 -> "unsigned char"
  | Fp8e5m2 -> "unsigned char" | Index -> "long long"

let render_dtype_str (sn : scalar_name) (dt : Dtype.t) =
  let base = sn (Dtype.scalar dt) in
  if Dtype.count dt > 1 then
    strf "%s%d" (String.map (fun c -> if c = ' ' then '_' else c) base) (Dtype.count dt)
  else base

(* Constant rendering *)

let render_float_lit f =
  let s = strf "%.17g" f in
  if String.contains s '.' || String.contains s 'e' || String.contains s 'E'
  then s else s ^ ".0"

let truncate_u32 i = Int64.to_int (Int64.logand (Int64.of_int i) 0xFFFFFFFFL)

let float_to_bf16_bits (f : float) =
  let bits = Int32.bits_of_float f in
  if not (Float.is_finite f) then
    Int32.to_int (Int32.shift_right_logical bits 16)
  else
    let rounded =
      Int32.add bits
        (Int32.add 0x7fffl
           (Int32.logand (Int32.shift_right_logical bits 16) 1l))
    in
    Int32.to_int (Int32.shift_right_logical rounded 16)

let render_const_base ~infinity ~nan_ ~render_cast (c : Const.t) (dt : Dtype.t) =
  match Const.view c with
  | Bool b -> if b then "1" else "0"
  | Float f ->
      if Float.is_nan f then strf "(%s)" (render_cast dt nan_)
      else if f = Float.infinity then strf "(%s)" (render_cast dt infinity)
      else if f = Float.neg_infinity then strf "(%s)" (render_cast dt ("-" ^ infinity))
      else (match Dtype.scalar dt with
        | Float64 -> render_float_lit f
        | Float16 | Bfloat16 | Fp8e4m3 | Fp8e5m2 ->
            strf "(%s)" (render_cast dt (render_float_lit f ^ "f"))
        | _ -> render_float_lit f ^ "f")
  | Int v -> (match Dtype.scalar dt with
      | Int64 -> strf "%Ldll" v
      | Uint64 -> strf "%Luull" v
      | Uint32 -> strf "%uu" (truncate_u32 (Int64.to_int v))
      | Uint8 | Uint16 -> strf "(%s)" (render_cast dt (strf "%Ldu" v))
      | Int8 | Int16 -> strf "(%s)" (render_cast dt (strf "%Ld" v))
      | _ -> strf "%Ld" v)

(* Code for op *)

type code_for_op = {
  unary : Op.unary -> string -> Dtype.t -> string;
  binary : Op.binary -> string -> string -> Dtype.t -> string;
  ternary : Op.ternary -> string -> string -> string -> Dtype.t -> string;
}

(* Ops handled by base_code_for_op — passed to Renderer.make so
   supported_ops_of_code_for_op derives accurate decomposition flags. *)
let base_code_for_op_list : Renderer.code_op list =
  [ Sqrt; Recip; Neg; Exp2; Log2; Sin; Trunc;
    And; Xor; Or; Add; Sub; Mul; Mod; Idiv; Cmpne;
    Shr; Shl; Cmplt; Where; Cmpeq; Fdiv ]

let base_code_for_op = {
  unary = (fun op x _dt -> match op with
    | `Neg -> strf "(-%s)" x | `Exp2 -> strf "exp2(%s)" x
    | `Log2 -> strf "log2(%s)" x | `Sin -> strf "sin(%s)" x
    | `Sqrt -> strf "sqrt(%s)" x | `Recip -> strf "(1/%s)" x
    | `Trunc -> strf "trunc(%s)" x);
  binary = (fun op a b _dt -> match op with
    | `Add -> strf "(%s+%s)" a b | `Sub -> strf "(%s-%s)" a b
    | `Mul -> strf "(%s*%s)" a b | `Fdiv -> strf "(%s/%s)" a b
    | `Idiv -> strf "(%s/%s)" a b | `Mod -> strf "(%s%%%s)" a b
    | `Shl -> strf "(%s<<%s)" a b | `Shr -> strf "(%s>>%s)" a b
    | `And -> strf "(%s&%s)" a b | `Or -> strf "(%s|%s)" a b
    | `Xor -> strf "(%s^%s)" a b
    | `Cmplt -> strf "(%s<%s)" a b | `Cmpeq -> strf "(%s==%s)" a b
    | `Cmpne -> strf "(%s!=%s)" a b
    | _ -> invalid_arg (strf "binary op not handled in renderer"));
  ternary = (fun op a b c _dt -> match op with
    | `Where -> strf "(%s?%s:%s)" a b c
    | _ -> invalid_arg (strf "ternary op not handled in renderer"));
}

(* Language configuration *)

type rule = Program.t -> Program.id -> Program.view -> lang -> string array -> string option

and lang = {
  kernel_typedef : int -> string;
  buffer_prefix : string;
  buffer_suffix : string;
  smem_align : string;
  smem_prefix : string;
  barrier : string;
  extra_args : string list;
  float4_ctor : Dtype.t -> string;
  float4_style : string * string;
  gep_arr_threshold : int;
  code_for_workitem : Special_dim.t -> string;
  type_map : Dtype.t -> string;
  render_const : Const.t -> Dtype.t -> string;
  render_cast : Dtype.t -> string -> string;
  render_bitcast : Program.t -> id -> Dtype.t -> string -> string;
  code_for_op : code_for_op;
  rules : rule list;
  render_kernel_hook : lang -> string -> string list -> rendered_buf list -> Program.t -> string;
  infinity : string;
  nan_ : string;
}

and buf_kind =
  | Buf_ptr of Dtype.ptr  (** Pointer parameter (buffer). *)
  | Buf_image of Dtype.ptr  (** OpenCL image parameter. *)
  | Buf_int  (** Scalar integer parameter (Define_var). *)

and rendered_buf = {
  buf_name : string;
  buf_kind : buf_kind;
  buf_mutable : bool;
}

(* Base rendering rule *)

let is_associative = function
  | `Add | `Mul | `Xor | `Or | `And -> true | _ -> false

let base_render : rule = fun program id v lang r ->
  match v with
  | Define_reg { size; dtype } ->
      Some (strf "%s %s[%d];" (lang.type_map (Dtype.base dtype)) r.(id) size)
  | If { cond; _ } -> Some (strf "if (%s) {" r.(cond))
  | End_range _ | Endif _ -> Some "}"
  | Wmma { name; a; b; c; _ } ->
      Some (strf "__%s(%s, %s, %s)" name r.(a) r.(b) r.(c))
  | Range { size; dtype; _ } ->
      let n = r.(id) in
      Some (strf "for (%s %s = 0; %s < %s; %s++) {"
        (lang.type_map dtype) n n r.(size) n)
  | Vectorize { srcs; dtype } ->
      let l, rr = lang.float4_style in
      Some (strf "%s%s%s%s" (lang.float4_ctor dtype) l
        (String.concat "," (List.map (fun s -> r.(s)) srcs)) rr)
  | Cast { src; dtype } when Dtype.count dtype > 1 ->
      Some (strf "__builtin_convertvector(%s, %s)" r.(src) (lang.type_map dtype))
  | Cast { src; dtype } ->
      Some (strf "(%s)" (lang.render_cast dtype r.(src)))
  | Bitcast { src; dtype } ->
      Some (lang.render_bitcast program src dtype r.(src))
  | Define_local { size; dtype } ->
      Some (strf "%s%s%s %s[%d];"
        lang.smem_align lang.smem_prefix (lang.type_map (Dtype.base dtype)) r.(id) size)
  | Barrier -> Some lang.barrier
  | Special { dim; size; _ } ->
      Some (strf "%s; /* %s */" (lang.code_for_workitem dim) r.(size))
  | Const { value; dtype } -> Some (lang.render_const value dtype)
  | Index { ptr; idxs; _ } ->
      let idx_str = match idxs with
        | [] -> "0"
        | [idx] -> r.(idx)
        | _ -> String.concat "+" (List.map (fun s -> r.(s)) idxs)
      in
      Some (strf "(%s+%s)" r.(ptr) idx_str)
  | Load { src; alt = Some alt; _ } ->
      (match Program.index_gate program src with
      | Some gate -> Some (strf "(%s?*%s:%s)" r.(gate) r.(src) r.(alt))
      | None -> Some (strf "(*%s)" r.(src)))
  | Load { src; _ } -> Some (strf "(*%s)" r.(src))
  | Store { dst; value } -> Some (strf "*%s = %s;" r.(dst) r.(value))
  | Unary { op; src; dtype } ->
      Some (lang.code_for_op.unary op r.(src) dtype)
  | Binary { op; lhs; rhs; dtype } ->
      (* Strip parens only when the child is the SAME associative op as the
         parent. This avoids incorrectly flattening e.g. (a+b)*c -> a+b*c. *)
      let strip_if_same_op child =
        match Program.view program child with
        | Binary { op = child_op; _ } when child_op = op && is_associative op ->
            strip_parens r.(child)
        | _ -> r.(child)
      in
      Some (lang.code_for_op.binary op (strip_if_same_op lhs) (strip_if_same_op rhs) dtype)
  | Ternary { op; a; b; c; dtype } ->
      Some (lang.code_for_op.ternary op r.(a) r.(b) r.(c) dtype)
  | Gep { src; idxs; dtype } ->
      let src_count = match Program.dtype program src with
        | Some dt -> Dtype.count dt | None -> 1
      in
      let elem idx =
        if src_count > lang.gep_arr_threshold then r.(src) ^ strf "[%d]" idx
        else r.(src) ^ strf ".%s" (vec_elem_name idx)
      in
      (match idxs with
      | [idx] -> Some (elem idx)
      | _ ->
          let l, rr = lang.float4_style in
          Some (strf "%s%s%s%s" (lang.float4_ctor dtype) l
            (String.concat "," (List.map elem idxs)) rr))
  | Custom { fmt; args } | Custom_inline { fmt; args; _ } ->
      Some (render_custom_fmt fmt (List.map (fun s -> r.(s)) args))
  | _ -> None

let apply_rules rules program id v lang r =
  let rec loop = function
    | [] -> None
    | rule :: rest ->
        (match rule program id v lang r with Some _ as s -> s | None -> loop rest)
  in
  loop rules

(* Inlining heuristic — decides which instructions to inline into their
   use site rather than assigning to a temporary variable. *)

let should_inline use_count program id (v : view) =
  match v with
  | Const _ | Gep _ | Index _ | Custom_inline _ -> true
  | Load { src; alt = None; _ } -> (
      match Program.view program src with
      | Index { dtype; _ } -> Dtype.addrspace dtype = Dtype.Reg
      | _ -> false)
  | Unary _ | Binary _ | Ternary _ -> (
      match v with Ternary { op = `Where; _ } -> false | _ -> use_count <= 1)
  | Cast { dtype; _ } -> Dtype.count dtype = 1 && use_count <= 1
  | Bitcast _ | Vectorize _ -> use_count <= 1
  | _ -> false

(* Naming *)

let prefix_of = function
  | Wmma _ -> "wmma" | Define_local _ -> "temp" | Const _ -> "const"
  | Cast _ | Bitcast _ | Vectorize _ -> "cast" | Gep _ -> "gep"
  | Index _ -> "bidx" | Define_reg _ -> "acc" | Load _ -> "val"
  | _ -> "alu"

let special_name = function
  | Special_dim.Group_id a -> strf "gidx%d" a
  | Local_id a -> strf "lidx%d" a
  | Global_idx a -> strf "idx%d" a

let axis_letter_of_kind : Axis_kind.t -> string = function
  | Loop -> "L" | Reduce -> "R" | Upcast -> "U" | Global -> "G"
  | _ -> "X"

let range_name kind axis sub =
  let base = strf "%sidx%d" (axis_letter_of_kind kind) axis in
  match sub with
  | [] -> base
  | _ -> strf "%s_%s" base (String.concat "_" (List.map string_of_int sub))

(* Metadata collection *)

type wmma_info = {
  wi_name : string;
  wi_dims : int * int * int;
  wi_dtype_in : Dtype.scalar;
  wi_dtype_out : Dtype.scalar;
  wi_upcast_axes : (int * int) list * (int * int) list * (int * int) list;
}

let collect_used_dtypes program =
  let dtypes = ref [] in
  Program.iteri (fun id _v ->
    match Program.dtype program id with
    | Some dt -> dtypes := dt :: !dtypes
    | None -> ()
  ) program;
  dedup (List.rev !dtypes)

let collect_wmma_args program =
  let wmmas = ref [] in
  Program.iteri (fun _id v -> match v with
    | Wmma { name; dims; dtype_in; dtype_out; upcast_axes; _ } ->
        wmmas := { wi_name = name; wi_dims = dims;
                   wi_dtype_in = dtype_in; wi_dtype_out = dtype_out;
                   wi_upcast_axes = upcast_axes } :: !wmmas
    | _ -> ()
  ) program;
  dedup (List.rev !wmmas)

(* Multi-pass rendering pipeline: first computes per-node use-counts for
   inlining decisions (single-use expressions are emitted inline rather than
   assigned to temporaries), then marks writable refs by following Store→Index
   chains so buffer parameters are declared mutable, and finally walks the
   program in topological order to emit C-style source. *)

let render_program (lang : lang) (program : Program.t) =
  let n = Program.length program in
  let r = Array.make n "" in
  let use_counts = Array.make n 0 in
  Program.iteri (fun _id _v ->
    List.iter (fun c ->
      if c >= 0 && c < n then use_counts.(c) <- use_counts.(c) + 1
    ) (Program.children program _id)
  ) program;
  let writable = Array.make n false in
  let rec mark_writable id =
    if id >= 0 && id < n then begin
      writable.(id) <- true;
      match Program.view program id with
      | Index { ptr; _ } -> mark_writable ptr
      | Cast { src; _ } | Bitcast { src; _ } | After { src; _ } -> mark_writable src
      | _ -> ()
    end
  in
  Program.iteri (fun _id v -> match v with
    | Store { dst; _ } -> mark_writable dst
    | Custom { args; _ } ->
        (* Side-effecting custom ops that reference image params (e.g. image
           store rewrites) mark the image as mutable. *)
        List.iter (fun arg ->
          if arg >= 0 && arg < n then
            match Program.view program arg with
            | Param_image _ -> mark_writable arg
            | _ -> ()
        ) args
    | _ -> ()) program;
  let bufs = ref [] in
  let kernel = ref [] in
  let depth = ref 1 in
  let counters : (string, int) Hashtbl.t = Hashtbl.create 16 in
  let peek_name prefix =
    let c = Hashtbl.find_opt counters prefix |> Option.value ~default:0 in
    strf "%s%d" prefix c
  in
  let bump_counter prefix =
    let c = Hashtbl.find_opt counters prefix |> Option.value ~default:0 in
    Hashtbl.replace counters prefix (c + 1)
  in
  Program.iteri (fun id v ->
    match v with
    | After { src; _ } -> r.(id) <- r.(src)
    | Param { idx; dtype } ->
        let name =
          if Dtype.ptr_size dtype > 0 then strf "data%d_%d" idx (Dtype.ptr_size dtype)
          else strf "data%d" idx
        in
        r.(id) <- name;
        bufs := { buf_name = name; buf_kind = Buf_ptr dtype;
                  buf_mutable = writable.(id) } :: !bufs
    | Param_image { idx; dtype; width; height } ->
        let name = strf "data%d_%dx%d" idx width height in
        r.(id) <- name;
        bufs := { buf_name = name; buf_kind = Buf_image dtype;
                  buf_mutable = writable.(id) } :: !bufs
    | Define_var { name; _ } ->
        r.(id) <- name;
        bufs := { buf_name = name; buf_kind = Buf_int;
                  buf_mutable = false } :: !bufs
    | _ ->
        let prefix = match v with
          | Special { dim; _ } -> r.(id) <- special_name dim; None
          | Range { axis; kind; sub; _ } -> r.(id) <- range_name kind axis sub; None
          | _ ->
              let p = prefix_of v in
              r.(id) <- peek_name p;
              Some p
        in
        let l = match apply_rules lang.rules program id v lang r with
          | Some s -> s
          | None -> Format.asprintf "/* unhandled %a */" Program.pp_view v
        in
        (match v with End_range _ | Endif _ -> decr depth | _ -> ());
        if should_inline use_counts.(id) program id v then
          r.(id) <- l
        else begin
          let line = match v with
            | Range _ | Define_local _ | Store _ | Define_reg _ -> l
            | Special _ ->
                strf "%s %s = %s" (lang.type_map Dtype.int32) r.(id) l
            | _ -> (match Program.dtype program id with
                | Some dt when not (Dtype.equal dt Dtype.void) ->
                    strf "%s %s = %s;" (lang.type_map dt) r.(id) l
                | _ -> l)
          in
          kernel := (String.make (!depth * 2) ' ' ^ line) :: !kernel;
          (match prefix with Some p -> bump_counter p | None -> ())
        end;
        (match v with If _ | Range _ -> incr depth | _ -> ())
  ) program;
  (List.rev !kernel, List.rev !bufs)

(* Default kernel assembly *)

let default_render_kernel lang name kernel bufs program =
  let has_images = List.exists (fun b -> match b.buf_kind with Buf_image _ -> true | _ -> false) bufs in
  let sampler_preamble =
    if has_images then
      "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    else ""
  in
  let param_strs = List.map (fun b -> match b.buf_kind with
    | Buf_image _ ->
      let qualifier = if b.buf_mutable then "write_only" else "read_only" in
      strf "%s image2d_t %s" qualifier b.buf_name
    | Buf_ptr dtype ->
      let type_str =
        lang.buffer_prefix ^
        (render_dtype_str (fun s -> lang.type_map (Dtype.of_scalar s))
           (Dtype.base dtype)) ^
        "*" ^ lang.buffer_suffix
      in
      strf "%s %s" type_str b.buf_name
    | Buf_int ->
      strf "const int %s" b.buf_name
  ) bufs in
  let all_params = param_strs @ lang.extra_args in
  let launch_bounds = ref 1 in
  Program.iteri (fun _id v -> match v with
    | Special { dim = Local_id _; size; _ } -> (
        match Program.view program size with
        | Const { value; _ } -> (match Const.view value with
            | Int n -> launch_bounds := !launch_bounds * Int64.to_int n | _ -> ())
        | _ -> ())
    | _ -> ()) program;
  let typedef = lang.kernel_typedef !launch_bounds in
  strf "%s %s(%s) {\n%s%s\n}" typedef name (String.concat ", " all_params)
    sampler_preamble (String.concat "\n" kernel)

let render_kernel (lang : lang) ?(name = "kernel") (program : Program.t) =
  let kernel, bufs = render_program lang program in
  lang.render_kernel_hook lang name kernel bufs program

(* Base language constructor *)

let base_render_cast type_map dt v = strf "(%s)(%s)" (type_map dt) v
let base_render_bitcast type_map program src_id dst v =
  let src_dt = match Program.dtype program src_id with Some dt -> dt | None -> dst in
  strf "__builtin_bit_cast(%s, (%s)(%s))" (type_map dst) (type_map src_dt) v

let make_lang ~scalar_name
    ?(kernel_typedef = fun _ -> "void")
    ?(buffer_prefix = "") ?(buffer_suffix = "")
    ?(smem_align = "") ?(smem_prefix = "")
    ?(barrier = "")
    ?(extra_args = [])
    ?(float4_ctor = fun _type_map _dt -> "(float4)")
    ?(float4_style = ("(", ")"))
    ?(gep_arr_threshold = 4)
    ?(code_for_workitem = fun _ -> failwith "no workitem support")
    ?(code_for_op = base_code_for_op)
    ?render_bitcast:render_bitcast_opt
    ?(rules = [base_render])
    ?(render_kernel_hook = fun lang name kernel bufs program ->
        default_render_kernel lang name kernel bufs program)
    ?(infinity = "INFINITY") ?(nan_ = "NAN")
    () =
  let type_map = render_dtype_str scalar_name in
  let render_cast = base_render_cast type_map in
  let render_bitcast = match render_bitcast_opt with
    | Some f -> f type_map
    | None -> base_render_bitcast type_map
  in
  let render_const = render_const_base ~infinity ~nan_ ~render_cast in
  let float4_ctor = float4_ctor type_map in
  { kernel_typedef; buffer_prefix; buffer_suffix; smem_align; smem_prefix;
    barrier; extra_args; float4_ctor; float4_style; gep_arr_threshold;
    code_for_workitem; type_map; render_const; render_cast; render_bitcast;
    code_for_op; rules; render_kernel_hook; infinity; nan_ }

(* Clang *)

let clang_scalar_name : scalar_name = function
  | Dtype.Bool -> "_Bool" | Float16 -> "__fp16" | s -> base_scalar_name s

let clang_render_vector_prefix scalar_name (dt : Dtype.t) =
  let type_map = render_dtype_str scalar_name in
  let scalar = type_map (Dtype.scalar_of dt) in
  let vec = type_map dt in
  let alignment =
    let sz = Dtype.itemsize dt in
    1 lsl (int_of_float (log (float_of_int sz) /. log 2.0))
  in
  strf "typedef %s %s __attribute__((aligned(%d),ext_vector_type(%d)));"
    scalar vec alignment (Dtype.count dt)

let clang_render_kernel lang name kernel bufs program =
  let used = collect_used_dtypes program in
  let vec_defs = List.filter_map (fun (dt : Dtype.t) ->
    if Dtype.count dt > 1 then Some (clang_render_vector_prefix clang_scalar_name dt)
    else None) used
  in
  let wmma_defs = List.concat_map (fun wi ->
    let type_map = render_dtype_str clang_scalar_name in
    let n, m, _ = wi.wi_dims in
    let out = type_map (Dtype.vec (Dtype.of_scalar wi.wi_dtype_in) (n * n)) in
    let dt1 = type_map (Dtype.vec (Dtype.of_scalar wi.wi_dtype_in) n) in
    let dt2 = type_map (Dtype.vec (Dtype.of_scalar wi.wi_dtype_in) m) in
    [ {|#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")|};
      {|#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")|};
      strf {|static %s __%s(%s data1, %s data2, %s data0){
  AMX_SET(0);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }
  AMX_SET(1);
  return data0;
}|} out wi.wi_name dt1 dt2 out ]
  ) (collect_wmma_args program)
  in
  let prefix = String.concat "\n" (vec_defs @ wmma_defs) in
  let body = default_render_kernel lang name kernel bufs program in
  if prefix = "" then body else prefix ^ "\n" ^ body

let clang_lang = make_lang ~scalar_name:clang_scalar_name
  ~buffer_suffix:" restrict" ~gep_arr_threshold:0
  ~float4_ctor:(fun tm dt -> strf "(%s)" (tm dt)) ~float4_style:("{", "}")
  ~infinity:{|__builtin_inff()|} ~nan_:{|__builtin_nanf("")|}
  ~code_for_op:{ base_code_for_op with
    unary = (fun op x dt -> match op with
      | `Sqrt -> strf "%s(%s)" (if Dtype.scalar dt = Float64 then "__builtin_sqrt" else "__builtin_sqrtf") x
      | `Trunc -> strf "%s(%s)" (if Dtype.scalar dt = Float64 then "__builtin_trunc" else "__builtin_truncf") x
      | _ -> base_code_for_op.unary op x dt);
    binary = (fun op a b dt -> match op with
      | `Fdiv -> strf "(%s/%s)" a b
      | _ -> base_code_for_op.binary op a b dt);
  }
  ~render_kernel_hook:clang_render_kernel
  ()

(* OpenCL *)

let opencl_scalar_name : scalar_name = function
  | Dtype.Void -> "void" | Bool -> "bool"
  | Int8 -> "char" | Int16 -> "short" | Int32 -> "int" | Int64 -> "long"
  | Uint8 -> "uchar" | Uint16 -> "ushort" | Uint32 -> "uint" | Uint64 -> "ulong"
  | Float16 -> "half" | Bfloat16 -> "ushort" | Float32 -> "float" | Float64 -> "double"
  | Fp8e4m3 -> "uchar" | Fp8e5m2 -> "uchar" | Index -> "long"

let opencl_render_bitcast tm program src_id dst v =
  let src_dt = match Program.dtype program src_id with Some dt -> dt | None -> dst in
  strf "as_%s((%s)(%s))" (tm dst) (tm src_dt) v

let opencl_bf16_const_rule : rule = fun _program _id v _lang _r -> match v with
  | Const { value; dtype } when Dtype.scalar dtype = Dtype.Bfloat16 -> (
      match Const.view value with
      | Float f -> Some (strf "%uu" (float_to_bf16_bits f))
      | _ -> None)
  | _ -> None

let opencl_render_kernel lang name kernel bufs program =
  let has_half = List.exists (fun (dt : Dtype.t) ->
    Dtype.scalar dt = Dtype.Float16) (collect_used_dtypes program) in
  let prefix = if has_half then
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" else "" in
  prefix ^ default_render_kernel lang name kernel bufs program

let opencl_lang = make_lang ~scalar_name:opencl_scalar_name
  ~kernel_typedef:(fun _ -> "__kernel void")
  ~buffer_prefix:"__global "
  ~smem_align:{|__attribute__ ((aligned (16))) |}
  ~smem_prefix:"__local "
  ~barrier:"barrier(CLK_LOCAL_MEM_FENCE);"
  ~float4_ctor:(fun tm dt -> strf "(%s)" (tm dt))
  ~render_bitcast:(fun tm -> opencl_render_bitcast tm)
  ~code_for_workitem:(fun dim ->
    let a = Special_dim.axis dim in
    match dim with
    | Group_id _ -> strf "get_group_id(%d)" a
    | Local_id _ -> strf "get_local_id(%d)" a
    | Global_idx _ -> strf "get_global_id(%d)" a)
  ~rules:[opencl_bf16_const_rule; base_render]
  ~render_kernel_hook:opencl_render_kernel
  ()

(* Intel *)

let intel_bf16_cast_rule : rule = fun program _id v lang r -> match v with
  | Cast { src; dtype } when Dtype.scalar dtype = Dtype.Bfloat16 -> (
      match Program.dtype program src with
      | Some src_dt when Dtype.scalar src_dt = Dtype.Float32 ->
          Some (strf "intel_convert_bfloat16_as_ushort(%s)" r.(src))
      | _ -> None)
  | Cast { src; dtype } when Dtype.scalar dtype = Dtype.Float32 -> (
      match Program.dtype program src with
      | Some src_dt when Dtype.scalar src_dt = Dtype.Bfloat16 ->
          Some (strf "intel_convert_as_bfloat16_float(%s)" r.(src))
      | _ -> None)
  | _ -> None

let intel_render_kernel lang name kernel bufs program =
  let prefix = List.map (fun wi ->
    let dt_in_name, dt_in_sfx =
      if wi.wi_dtype_in = Dtype.Bfloat16 then ("ushort", "bf16")
      else ("half", "f16")
    in
    let dt_out = base_scalar_name wi.wi_dtype_out in
    strf {|%s8 __%s(%s16 a, %s16 b, %s8 c) {
    return intel_sub_group_%s_%s_matrix_mad_k16(as_int8(a), as_int8(b), c);
}|} dt_out wi.wi_name dt_in_name dt_in_name dt_out dt_in_sfx dt_in_sfx
  ) (collect_wmma_args program) in
  let preamble = opencl_render_kernel lang name kernel bufs program in
  if prefix = [] then preamble
  else String.concat "\n" prefix ^ "\n" ^ preamble

let intel_lang =
  { opencl_lang with
    kernel_typedef = (fun _ ->
      "__attribute__((intel_reqd_sub_group_size(8)))\n__kernel void");
    rules = [intel_bf16_cast_rule; opencl_bf16_const_rule; base_render];
    render_kernel_hook = intel_render_kernel;
  }

(* Metal *)

let metal_scalar_name : scalar_name = function
  | Dtype.Int8 -> "char" | Uint8 -> "uchar" | Uint16 -> "ushort"
  | Uint32 -> "uint" | Uint64 -> "ulong" | Int64 -> "long"
  | Bfloat16 -> "bfloat" | Index -> "long" | s -> base_scalar_name s

let metal_render_bitcast tm program src_id dst v =
  let src_dt = match Program.dtype program src_id with Some dt -> dt | None -> dst in
  strf "as_type<%s>((%s)(%s))" (tm dst) (tm src_dt) v

let metal_render_kernel lang name kernel bufs program =
  let type_map = render_dtype_str metal_scalar_name in
  let prefix = ["#include <metal_stdlib>"; "using namespace metal;"] in
  let wmma_prefix = List.map (fun wi ->
    let dstr_out = type_map (Dtype.vec (Dtype.of_scalar wi.wi_dtype_out) 2) in
    let dstr_in = type_map (Dtype.vec (Dtype.of_scalar wi.wi_dtype_in) 2) in
    let simd_in = type_map (Dtype.of_scalar wi.wi_dtype_in) in
    let simd_out = type_map (Dtype.of_scalar wi.wi_dtype_out) in
    strf {|%s __%s(%s a, %s b, %s c){
  simdgroup_%s8x8 mat_a, mat_b; simdgroup_%s8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
  return %s(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);
}|} dstr_out wi.wi_name dstr_in dstr_in dstr_out simd_in simd_out dstr_out
  ) (collect_wmma_args program) in
  let all_prefix = String.concat "\n" (prefix @ wmma_prefix) ^ "\n" in
  all_prefix ^ default_render_kernel lang name kernel bufs program

let metal_lang = make_lang ~scalar_name:metal_scalar_name
  ~kernel_typedef:(fun _ -> "kernel void")
  ~buffer_prefix:"device "
  ~smem_prefix:{|threadgroup __attribute__((aligned(16))) |}
  ~barrier:"threadgroup_barrier(mem_flags::mem_threadgroup);"
  ~float4_ctor:(fun tm dt -> tm dt)
  ~render_bitcast:(fun tm -> metal_render_bitcast tm)
  ~extra_args:[
    "uint3 gid [[threadgroup_position_in_grid]]";
    "uint3 lid [[thread_position_in_threadgroup]]"]
  ~code_for_workitem:(fun dim ->
    let a = Special_dim.axis dim in
    match dim with
    | Group_id _ -> strf "gid.%c" (Char.chr (120 + a))
    | Local_id _ -> strf "lid.%c" (Char.chr (120 + a))
    | Global_idx _ -> failwith "Metal does not support Global_idx specials")
  ~code_for_op:{ base_code_for_op with
    unary = (fun op x _dt -> match op with
      | `Sin -> strf "precise::sin(%s)" x
      | _ -> base_code_for_op.unary op x _dt);
  }
  ~rules:[base_render]
  ~render_kernel_hook:metal_render_kernel
  ()

(* CUDA *)

let cuda_scalar_name : scalar_name = function
  | Dtype.Bfloat16 -> "nv_bfloat16"
  | Fp8e4m3 -> "__nv_fp8_e4m3" | Fp8e5m2 -> "__nv_fp8_e5m2"
  | s -> base_scalar_name s

let is_half_or_bf16 (dt : Dtype.t) =
  match Dtype.scalar dt with Float16 | Bfloat16 -> true | _ -> false

let cuda_render_bitcast tm program src_id dst v =
  let src_dt = match Program.dtype program src_id with Some dt -> dt | None -> dst in
  strf "tg_bitcast<%s>((%s)(%s))" (tm dst) (tm src_dt) v

let cuda_render_vector_prefix scalar_name (dt : Dtype.t) =
  let type_map = render_dtype_str scalar_name in
  let vec = type_map dt in
  let scal = type_map (Dtype.scalar_of dt) in
  let nms = List.init (Dtype.count dt) vec_elem_name in
  let elems = String.concat ", " nms in
  let header = String.concat ", "
    (List.map (fun x -> strf "%s %s" scal x) nms) in
  strf "struct __align__(%d) %s { %s %s; }; __device__ %s make_%s(%s) { %s r={%s}; return r; }"
    (Dtype.itemsize dt) vec scal elems vec vec header vec elems

let cuda_dt_map_in = function
  | Dtype.Float32 -> "tf32" | Float16 -> "f16" | Bfloat16 -> "bf16"
  | Fp8e4m3 -> "e4m3" | Fp8e5m2 -> "e5m2" | _ -> "f32"
let cuda_dt_map_out = function
  | Dtype.Float32 -> "f32" | Float16 -> "f16" | _ -> "f32"

let cuda_render_kernel lang name kernel bufs program =
  let type_map = render_dtype_str cuda_scalar_name in
  let prefix = ref [
    {|#define INFINITY (__int_as_float(0x7f800000))|};
    {|#define NAN (__int_as_float(0x7fffffff))|};
    {|template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }|};
  ] in
  let used = collect_used_dtypes program in
  if List.exists (fun (dt : Dtype.t) ->
    Dtype.scalar dt = Fp8e4m3 || Dtype.scalar dt = Fp8e5m2) used then
    prefix := !prefix @ [("#include <cuda_fp8.h>")];
  if List.exists (fun (dt : Dtype.t) -> Dtype.scalar dt = Float16) used then
    prefix := !prefix @ ["#include <cuda_fp16.h>"];
  if List.exists (fun (dt : Dtype.t) -> Dtype.scalar dt = Bfloat16) used then
    prefix := !prefix @ ["#include <cuda_bf16.h>"];
  let vec_defs = List.filter_map (fun (dt : Dtype.t) ->
    let need = match Dtype.scalar dt with
      | Float16 | Bfloat16 -> List.mem (Dtype.count dt) [4; 8]
      | Fp8e4m3 | Fp8e5m2 -> List.mem (Dtype.count dt) [2; 4; 8; 16]
      | _ -> false
    in
    if need then Some (cuda_render_vector_prefix cuda_scalar_name dt)
    else None) used
  in
  prefix := !prefix @ vec_defs;
  (* WMMA preambles *)
  List.iter (fun wi ->
    let n, m, k = wi.wi_dims in
    let ua, ub, uc = wi.wi_upcast_axes in
    let upcast_sizes = [prod (List.map snd ua); prod (List.map snd ub); prod (List.map snd uc)] in
    let wmma_dtypes = List.map2 (fun dt size -> type_map (Dtype.vec (Dtype.of_scalar dt) size))
      [wi.wi_dtype_in; wi.wi_dtype_in; wi.wi_dtype_out] upcast_sizes in
    let n_operands = List.map2 (fun dt size ->
      size * (Dtype.itemsize (Dtype.of_scalar dt)) / 4)
      [wi.wi_dtype_in; wi.wi_dtype_in; wi.wi_dtype_out] upcast_sizes in
    let total_ops = List.fold_left (+) 0 n_operands in
    let operands = List.init total_ops (fun i -> strf "%%%d" i) in
    let nc = List.nth n_operands 2 in
    let na = List.nth n_operands 0 in
    let nb = List.nth n_operands 1 in
    let slice from len = List.filteri (fun i _ -> i >= from && i < from + len) operands in
    let join = String.concat ", " in
    prefix := !prefix @ [
      strf {|__device__ %s __%s(%s a, %s b, %s c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m%dn%dk%d.row.col.%s.%s.%s.%s"
      "{%s}, {%s},"
      "{%s}, {%s};"
    : %s
    : %s, %s);
  return c;
}|}
        (List.nth wmma_dtypes 2) wi.wi_name
        (List.nth wmma_dtypes 0) (List.nth wmma_dtypes 1) (List.nth wmma_dtypes 2)
        m n k
        (cuda_dt_map_out wi.wi_dtype_out)
        (cuda_dt_map_in wi.wi_dtype_in) (cuda_dt_map_in wi.wi_dtype_in)
        (cuda_dt_map_out wi.wi_dtype_out)
        (join (slice 0 nc)) (join (slice nc na))
        (join (slice (nc + na) nb)) (join (slice 0 nc))
        (join (List.init nc (fun i -> strf {|"+r"(c_pk[%d])|} i)))
        (join (List.init na (fun i -> strf {|"r"(a_pk[%d])|} i)))
        (join (List.init nb (fun i -> strf {|"r"(b_pk[%d])|} i)))
    ]
  ) (collect_wmma_args program);
  let preamble = String.concat "\n" !prefix ^ "\n" in
  preamble ^ default_render_kernel lang name kernel bufs program

let cuda_lang = make_lang ~scalar_name:cuda_scalar_name
  ~kernel_typedef:(fun lb ->
    strf {|extern "C" __global__ void __launch_bounds__(%d)|} lb)
  ~smem_prefix:"__shared__ __align__(16) "
  ~barrier:"__syncthreads();"
  ~float4_ctor:(fun tm dt -> strf "make_%s" (tm dt))
  ~render_bitcast:(fun tm -> cuda_render_bitcast tm)
  ~gep_arr_threshold:8
  ~infinity:"INFINITY"
  ~nan_:"NAN"
  ~code_for_workitem:(fun dim ->
    let a = Special_dim.axis dim in
    let c = Char.chr (120 + a) in
    match dim with
    | Group_id _ -> strf "blockIdx.%c" c
    | Local_id _ -> strf "threadIdx.%c" c
    | Global_idx _ -> strf "(blockIdx.%c*blockDim.%c+threadIdx.%c)" c c c)
  ~code_for_op:{ base_code_for_op with
    unary = (fun op x dt -> match op with
      | `Exp2 -> strf "%s(%s)" (if is_half_or_bf16 dt then "hexp2" else "exp2") x
      | `Log2 -> strf "%s(%s)" (if is_half_or_bf16 dt then "hlog2" else "log2") x
      | `Sin -> strf "%s(%s)" (if is_half_or_bf16 dt then "hsin" else "sin") x
      | `Sqrt -> strf "%s(%s)" (if is_half_or_bf16 dt then "hsqrt" else "sqrt") x
      | `Recip -> if is_half_or_bf16 dt then strf "hrcp(%s)" x else strf "(1/%s)" x
      | `Trunc -> strf "%s(%s)" (if is_half_or_bf16 dt then "htrunc" else "trunc") x
      | _ -> base_code_for_op.unary op x dt);
  }
  ~rules:[base_render]
  ~render_kernel_hook:cuda_render_kernel
  ()

(* AMD HIP *)

let amd_scalar_name ~cdna4 : scalar_name = function
  | Dtype.Bfloat16 -> if cdna4 then "__bf16" else "hip_bfloat16"
  | Fp8e4m3 -> "hip_fp8" | Fp8e5m2 -> "hip_bf8" | Float16 -> "_Float16"
  | s -> base_scalar_name s

let ocml op (dt : Dtype.t) =
  let bits = match Dtype.scalar dt with Float16 -> 16 | Float64 -> 64 | _ -> 32 in
  strf "__ocml_%s_f%d" op bits

let fp8_index = function
  | Dtype.Fp8e5m2 -> 1 | _ -> 0

let amd_render_vector_prefix scalar_name (dt : Dtype.t) =
  let type_map = render_dtype_str scalar_name in
  let vec = type_map dt in
  let scal = type_map (Dtype.scalar_of dt) in
  let nms = List.init (Dtype.count dt) vec_elem_name in
  strf "typedef %s %s __attribute__((ext_vector_type(%d)));\n\
        static inline __attribute__((device)) %s make_%s(%s) { return { %s }; }"
    scal vec (Dtype.count dt) vec vec
    (String.concat ", " (List.map (fun x -> strf "%s %s" scal x) nms))
    (String.concat ", " nms)

let amd_wmma_type_map = function
  | Dtype.Bfloat16 -> "bf16" | Float32 -> "f32" | Float16 -> "f16"
  | Fp8e4m3 -> "_fp8_fp8" | Fp8e5m2 -> "_bf8_bf8" | _ -> "f32"

let amd_wmma_out_map = function
  | Dtype.Float32 -> "f32" | Float16 -> "f16" | _ -> "f32"

let amd_cdna_wmma_rule : rule = fun _program _id v _lang r -> match v with
  | Wmma { name; a; b; c; dims = (_, _, k); _ } when k = 128 ->
      Some (strf "__%s(%s, %s, %s, 0, 0, 0, 0, 0, 0)" name r.(a) r.(b) r.(c))
  | Wmma { name; a; b; c; _ } ->
      Some (strf "__%s(%s, %s, %s, 0, 0, 0)" name r.(a) r.(b) r.(c))
  | _ -> None

let amd_cdna_fp8_cast_rule : rule = fun program _id v lang r -> match v with
  | Cast { src; dtype } when (Dtype.scalar dtype = Fp8e4m3 || Dtype.scalar dtype = Fp8e5m2) -> (
      match Program.dtype program src with
      | Some src_dt when Dtype.scalar src_dt = Float32 ->
          Some (strf "f32_to_fp8(%s, %d)" r.(src) (fp8_index (Dtype.scalar dtype)))
      | _ -> None)
  | Cast { src; dtype } when Dtype.scalar dtype = Float32 -> (
      match Program.dtype program src with
      | Some src_dt when Dtype.scalar src_dt = Fp8e4m3 || Dtype.scalar src_dt = Fp8e5m2 ->
          let cvt = if Dtype.scalar src_dt = Fp8e5m2 then "bf8" else "fp8" in
          Some (strf "__builtin_amdgcn_cvt_f32_%s((unsigned int)%s, 0)" cvt r.(src))
      | _ -> None)
  | _ -> None

let amd_render_kernel ~cdna ~cdna4 ~rdna4 ~tensor_cores scalar_name
    lang name kernel bufs program =
  let prefix = ref [] in
  let ockl = ref [] in
  let used = collect_used_dtypes program in
  let has_non_finite = ref false in
  Program.iteri (fun _id v -> match v with
    | Const { value; _ } -> (match Const.view value with
        | Float f when not (Float.is_finite f) -> has_non_finite := true
        | _ -> ())
    | _ -> ()) program;
  if !has_non_finite then
    prefix := [{|#define INFINITY (__builtin_inff())|}; {|#define NAN (__builtin_nanf(""))|}];
  let has_specials = ref false in
  Program.iteri (fun _id v -> match v with
    | Special _ -> has_specials := true | _ -> ()) program;
  if !has_specials then begin
    prefix := !prefix @ ["typedef long unsigned int size_t;"];
    ockl := List.map (fun n ->
      strf {|extern "C" __attribute__((device, const)) unsigned int __ockl_get_%s(size_t);|} n
    ) ["local_id"; "group_id"; "local_size"]
  end;
  (* OCML declarations *)
  let ocml_ops = [(`Exp2, "exp2", "pure"); (`Log2, "log2", "pure");
    (`Sqrt, "sqrt", "const"); (`Sin, "sin", ""); (`Trunc, "trunc", "")] in
  let ocml_decls = ref [] in
  Program.iteri (fun _id v -> match v with
    | Unary { op; dtype; _ } ->
        List.iter (fun (tag, name, attr) ->
          if op = tag && (Dtype.scalar dtype = Float16 || Dtype.scalar dtype = Float32 || Dtype.scalar dtype = Float64) then
            let bits = match Dtype.scalar dtype with Float16 -> 16 | Float64 -> 64 | _ -> 32 in
            let dt_name = base_scalar_name (Dtype.scalar dtype) in
            let decl = strf {|extern "C" __attribute__((device%s)) %s __ocml_%s_f%d(%s);|}
              (if attr = "" then "" else ", " ^ attr) dt_name name bits dt_name in
            ocml_decls := decl :: !ocml_decls
        ) ocml_ops
    | _ -> ()) program;
  prefix := !prefix @ !ockl @ dedup (List.rev !ocml_decls);
  if List.exists (fun (dt : Dtype.t) -> Dtype.scalar dt = Bfloat16) used then
    prefix := !prefix @
      [strf "typedef %s hip_bfloat16;" (if cdna4 then "__bf16" else "unsigned short")];
  if List.exists (fun (dt : Dtype.t) -> Dtype.scalar dt = Float16) used then
    prefix := !prefix @ ["#define half _Float16"];
  if List.exists (fun (dt : Dtype.t) ->
    Dtype.scalar dt = Fp8e4m3 || Dtype.scalar dt = Fp8e5m2) used then begin
    prefix := !prefix @ ["typedef unsigned char hip_bf8;"; "typedef unsigned char hip_fp8;"];
    prefix := !prefix @ [{|static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {
  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;
  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));
}|}]
  end;
  prefix := !prefix @ List.filter_map (fun (dt : Dtype.t) ->
    if Dtype.count dt > 1 then Some (amd_render_vector_prefix scalar_name dt)
    else None) used;
  (* WMMA defines *)
  let wmma_type_map = ref [(Dtype.Bfloat16, "bf16"); (Dtype.Float32, "f32"); (Dtype.Float16, "f16");
    (Dtype.Fp8e4m3, "_fp8_fp8"); (Dtype.Fp8e5m2, "_bf8_bf8")] in
  List.iter (fun wi ->
    let n, m, k = wi.wi_dims in
    let type_in = List.assoc wi.wi_dtype_in !wmma_type_map in
    let type_out = amd_wmma_out_map wi.wi_dtype_out in
    if cdna then begin
      (if (n, m, k) = (16, 16, 16) && wi.wi_dtype_in = Dtype.Bfloat16 then
        wmma_type_map := (Dtype.Bfloat16, "bf16_1k") :: !wmma_type_map);
      (if (n, m, k) = (16, 16, 32) then begin
        wmma_type_map := (Dtype.Bfloat16, "_bf16") :: (Dtype.Float16, "_f16") :: !wmma_type_map
      end);
      (if (n, m, k) = (16, 16, 128) then
        wmma_type_map := (Dtype.Fp8e4m3, "_f8f6f4") :: (Dtype.Fp8e5m2, "_f8f6f4") :: !wmma_type_map);
      let type_in' = List.assoc wi.wi_dtype_in !wmma_type_map in
      let scale = if k = 128 then "scale_" else "" in
      prefix := !prefix @
        [strf "#define __%s __builtin_amdgcn_mfma_%s%s_%dx%dx%d%s"
           wi.wi_name scale type_out n m k type_in']
    end else if rdna4 then
      prefix := !prefix @
        [strf "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s_w32_gfx12"
           wi.wi_name type_out type_in]
    else if wi.wi_dtype_out = Float32 then
      prefix := !prefix @
        [strf "#define __%s __builtin_amdgcn_wmma_f32_16x16x16_%s_w32"
           wi.wi_name (if wi.wi_dtype_in = Float16 then "f16" else "bf16")]
    else
      prefix := !prefix @
        [strf {|static inline __attribute__((device)) half8 __%s(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;
}|} wi.wi_name]
  ) (collect_wmma_args program);
  let preamble = String.concat "\n" !prefix ^ "\n" in
  preamble ^ default_render_kernel lang name kernel bufs program

let amd_lang ~cdna ~cdna4 ~rdna4 ~tensor_cores =
  let scalar_name = amd_scalar_name ~cdna4 in
  let base_rules =
    if cdna then [amd_cdna_fp8_cast_rule; amd_cdna_wmma_rule; base_render]
    else [base_render]
  in
  make_lang ~scalar_name
    ~kernel_typedef:(fun lb ->
      strf {|extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, %d)))|} lb)
    ~smem_prefix:{|__attribute__((shared, aligned(16)))|}
    ~float4_ctor:(fun tm dt -> strf "make_%s" (tm dt))
    ~barrier:(
      {|__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");|} ^
      "__builtin_amdgcn_s_barrier();" ^
      {|__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");|})
    ~code_for_workitem:(fun dim ->
      let a = Special_dim.axis dim in
      match dim with
      | Group_id _ -> strf "__ockl_get_group_id(%d)" a
      | Local_id _ -> strf "__ockl_get_local_id(%d)" a
      | Global_idx _ ->
          strf "(__ockl_get_group_id(%d)*__ockl_get_local_size(%d)+__ockl_get_local_id(%d))" a a a)
    ~code_for_op:{ base_code_for_op with
      unary = (fun op x dt -> match op with
        | `Exp2 -> strf "%s(%s)" (ocml "exp2" dt) x
        | `Log2 -> strf "%s(%s)" (ocml "log2" dt) x
        | `Sin -> strf "%s(%s)" (ocml "sin" dt) x
        | `Sqrt -> strf "%s(%s)" (ocml "sqrt" dt) x
        | `Trunc -> strf "%s(%s)" (ocml "trunc" dt) x
        | _ -> base_code_for_op.unary op x dt);
    }
    ~rules:base_rules
    ~render_kernel_hook:(amd_render_kernel ~cdna ~cdna4 ~rdna4 ~tensor_cores scalar_name)
    ()

(* Non-native float promotion — promotes ALU ops on non-native float types
   (e.g. bfloat16 on clang) through float32 intermediates. *)

let is_non_native_float (non_native : Dtype.scalar list) (dt : Dtype.t) =
  Dtype.count dt = 1 && List.mem (Dtype.scalar dt) non_native

let promote_non_native_floats (non_native : Dtype.scalar list) program =
  let f32 = Dtype.float32 in
  Program.rebuild (fun ~emit ~map_ref instr ->
    match instr with
    | Binary { op; lhs; rhs; dtype } when is_non_native_float non_native dtype ->
        let lhs' = emit (Cast { src = map_ref lhs; dtype = f32 }) in
        let rhs' = emit (Cast { src = map_ref rhs; dtype = f32 }) in
        let alu = emit (Binary { op; lhs = lhs'; rhs = rhs'; dtype = f32 }) in
        Some (emit (Cast { src = alu; dtype }))
    | Unary { op; src; dtype } when is_non_native_float non_native dtype ->
        let src' = emit (Cast { src = map_ref src; dtype = f32 }) in
        let alu = emit (Unary { op; src = src'; dtype = f32 }) in
        Some (emit (Cast { src = alu; dtype }))
    | Ternary { op = `Where; a; b; c; dtype } when is_non_native_float non_native dtype ->
        let b' = emit (Cast { src = map_ref b; dtype = f32 }) in
        let c' = emit (Cast { src = map_ref c; dtype = f32 }) in
        let w = emit (Ternary { op = `Where; a = map_ref a; b = b'; c = c'; dtype = f32 }) in
        Some (emit (Cast { src = w; dtype }))
    | _ -> None
  ) program

(* Clang fixed-ABI wrapper — generates a public entry point that unpacks
   bufs/vals arrays into the kernel's typed parameters. *)

let clang_abi_wrapper name bufs =
  let inner = name ^ "_" in
  let buf_idx = ref 0 in
  let val_idx = ref 0 in
  let call_args = List.map (fun b -> match b.buf_kind with
    | Buf_ptr dtype | Buf_image dtype ->
      let c_type = base_scalar_name (Dtype.scalar (Dtype.base dtype)) in
      let arg = strf "(%s*)bufs[%d]" c_type !buf_idx in
      incr buf_idx; arg
    | Buf_int ->
      let arg = strf "vals[%d]" !val_idx in
      incr val_idx; arg
  ) bufs in
  strf "void %s(const unsigned long long *bufs, const long long *vals) {\n  %s(%s);\n}"
    name inner (String.concat ", " call_args)

let clang_abi_render_kernel lang name kernel bufs program =
  let inner_name = name ^ "_" in
  let body = clang_render_kernel lang inner_name kernel bufs program in
  let wrapper = clang_abi_wrapper name bufs in
  body ^ "\n" ^ wrapper

(* Exported renderers *)

let render_fn lang ?(name = "kernel") program = render_kernel lang ~name program

let clang_bf16_promote program =
  promote_non_native_floats [Dtype.Bfloat16] program

let clang_has_threads =
  match Sys.getenv_opt "THREADS" with
  | Some "0" -> false
  | _ -> true

let clang_abi_lang = make_lang ~scalar_name:clang_scalar_name
  ~kernel_typedef:(fun _ -> "static void")
  ~buffer_suffix:" restrict" ~gep_arr_threshold:0
  ~float4_ctor:(fun tm dt -> strf "(%s)" (tm dt)) ~float4_style:("{", "}")
  ~infinity:{|__builtin_inff()|} ~nan_:{|__builtin_nanf("")|}
  ~code_for_op:{ base_code_for_op with
    unary = (fun op x dt -> match op with
      | `Sqrt -> strf "%s(%s)" (if Dtype.scalar dt = Float64 then "__builtin_sqrt" else "__builtin_sqrtf") x
      | `Trunc -> strf "%s(%s)" (if Dtype.scalar dt = Float64 then "__builtin_trunc" else "__builtin_truncf") x
      | _ -> base_code_for_op.unary op x dt);
    binary = (fun op a b dt -> match op with
      | `Fdiv -> strf "(%s/%s)" a b
      | _ -> base_code_for_op.binary op a b dt);
  }
  ~render_kernel_hook:clang_abi_render_kernel
  ()

let clang_amx =
  match Sys.getenv_opt "AMX" with
  | Some "1" -> true
  | _ -> false

let clang =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"clang" ~device:"CPU"
    ~has_local:false ~has_shared:false ~shared_max:0
    ~has_threads:clang_has_threads
    ~tensor_cores:(if clang_amx then Tc.amx else [])
    ~render:(fun ?name program ->
      render_fn clang_abi_lang ?name (clang_bf16_promote program)) ()

let clang_no_abi =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"clang" ~device:"CPU"
    ~has_local:false ~has_shared:false ~shared_max:0
    ~has_threads:clang_has_threads
    ~tensor_cores:(if clang_amx then Tc.amx else [])
    ~render:(fun ?name program ->
      render_fn clang_lang ?name (clang_bf16_promote program)) ()

let opencl =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"opencl" ~device:"CL"
    ~has_local:true ~has_shared:true ~shared_max:(32 * 1024)
    ~render:(render_fn opencl_lang) ()

let intel =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"intel" ~device:"CL"
    ~tensor_cores:Tc.intel
    ~has_local:true ~has_shared:true ~shared_max:(32 * 1024)
    ~render:(render_fn intel_lang) ()

let qcom =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"qcom" ~device:"QCOM"
    ~has_local:true ~has_shared:true ~shared_max:(32 * 1024)
    ~render:(render_fn opencl_lang) ()

let metal_is_arm64 =
  try
    let ic = Unix.open_process_in "uname -m" in
    let machine = String.trim (input_line ic) in
    ignore (Unix.close_process_in ic);
    machine = "arm64"
  with _ -> false

let metal =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"metal" ~device:"METAL"
    ~tensor_cores:(if metal_is_arm64 then Tc.metal else [])
    ~has_local:true ~has_shared:true ~shared_max:(32 * 1024)
    ~render:(render_fn metal_lang) ()

let cuda_tensor_cores = function
  | Gpu_target.SM75 -> Tc.cuda_sm75
  | Gpu_target.SM80 -> Tc.cuda_sm80
  | Gpu_target.SM89 -> Tc.cuda_sm89

let cuda (arch : Gpu_target.cuda) =
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"cuda" ~device:"CUDA"
    ~tensor_cores:(cuda_tensor_cores arch)
    ~has_local:true ~has_shared:true ~shared_max:(48 * 1024)
    ~global_max:(Some [0x7FFFFFFF; 65535; 65535])
    ~local_max:(Some [1024; 1024; 64])
    ~render:(render_fn cuda_lang) ()

let amd_tensor_cores = function
  | Gpu_target.RDNA3 -> Tc.amd_rdna3
  | Gpu_target.RDNA4 -> Tc.amd_rdna4
  | Gpu_target.CDNA3 -> Tc.amd_cdna3
  | Gpu_target.CDNA4 -> Tc.amd_cdna4

let amd (arch : Gpu_target.amd) =
  let tensor_cores = amd_tensor_cores arch in
  let cdna, cdna4, rdna4 = match arch with
    | Gpu_target.CDNA3 -> (true, false, false)
    | Gpu_target.CDNA4 -> (true, true, false)
    | Gpu_target.RDNA4 -> (false, false, true)
    | Gpu_target.RDNA3 -> (false, false, false)
  in
  let lang = amd_lang ~cdna ~cdna4 ~rdna4 ~tensor_cores in
  Renderer.make ~code_for_op:base_code_for_op_list
    ~name:"amd" ~device:"AMD"
    ~tensor_cores
    ~has_local:true ~has_shared:true ~shared_max:(64 * 1024)
    ~global_max:(Some [0x7FFFFFFF; 65535; 65535])
    ~render:(render_fn lang) ()
