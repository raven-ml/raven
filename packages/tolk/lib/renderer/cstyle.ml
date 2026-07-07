(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/renderer/cstyle.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

let strf = Printf.sprintf

(* Helpers *)

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

let has_top_level_char ch s =
  let depth = ref 0 in
  let found = ref false in
  String.iter
    (function
      | '(' -> incr depth
      | ')' -> decr depth
      | c when c = ch && !depth = 0 -> found := true
      | _ -> ())
    s;
  !found

let is_parenthesized_top_level_add s =
  let stripped = strip_parens s in
  not (String.equal stripped s) && has_top_level_char '+' stripped

let prod = List.fold_left ( * ) 1

(* Replace first occurrence of [needle] with [replacement] in [s]. *)
let replace_first ~needle ~replacement s =
  let nlen = String.length needle in
  let slen = String.length s in
  if nlen = 0 || slen < nlen then s
  else
    let rec find i =
      if i + nlen > slen then None
      else if String.sub s i nlen = needle then Some i
      else find (i + 1)
    in
    match find 0 with
    | None -> s
    | Some i ->
        String.sub s 0 i ^ replacement
        ^ String.sub s (i + nlen) (slen - i - nlen)

let contains_substring s sub =
  let slen = String.length s in
  let nlen = String.length sub in
  if nlen = 0 then true
  else if slen < nlen then false
  else
    let rec loop i =
      if i + nlen > slen then false
      else if String.sub s i nlen = sub then true
      else loop (i + 1)
    in
    loop 0

let arch_int_value ~prefix arch =
  String.split_on_char ',' arch
  |> List.find_map (fun part ->
         let part = String.trim part in
         let plen = String.length prefix in
         if String.length part > plen && String.starts_with ~prefix part then
           int_of_string_opt
             (String.sub part plen (String.length part - plen))
         else None)

let getenv name default =
  match Sys.getenv_opt name with
  | Some s -> ( try int_of_string s with Failure _ -> default)
  | None -> default

let host_cpu_count () =
  match Sys.getenv_opt "CPU_COUNT" with
  | Some s -> (try max 1 (int_of_string s) with Failure _ -> 1)
  | None -> (
      try
        let ic = Unix.open_process_in "getconf _NPROCESSORS_ONLN" in
        let line = input_line ic in
        ignore (Unix.close_process_in ic);
        max 1 (int_of_string (String.trim line))
      with _ -> 1)

(* Subset of python str.format(): positional {0}/{1}, auto-numbered {}, {{ }} escapes. *)
let render_custom_fmt fmt args =
  let a = Array.of_list args in
  let n = Array.length a in
  let buf = Buffer.create (String.length fmt) in
  let len = String.length fmt in
  let rec scan i auto =
    if i >= len then ()
    else
      match fmt.[i] with
      | '{' when i + 1 < len && fmt.[i + 1] = '{' ->
          Buffer.add_char buf '{';
          scan (i + 2) auto
      | '}' when i + 1 < len && fmt.[i + 1] = '}' ->
          Buffer.add_char buf '}';
          scan (i + 2) auto
      | '{' ->
          let j =
            match String.index_from_opt fmt (i + 1) '}' with
            | Some j -> j
            | None -> invalid_arg "render_custom_fmt: unclosed '{'"
          in
          let field = String.trim (String.sub fmt (i + 1) (j - i - 1)) in
          let idx, auto =
            if field = "" then (auto, auto + 1)
            else
              match int_of_string_opt field with
              | Some k -> (k, auto)
              | None ->
                  invalid_arg
                    (strf "render_custom_fmt: non-numeric field {%s}" field)
          in
          if idx >= 0 && idx < n then Buffer.add_string buf a.(idx);
          scan (j + 1) auto
      | c ->
          Buffer.add_char buf c;
          scan (i + 1) auto
  in
  scan 0 0;
  Buffer.contents buf

let vec_elem_letter i =
  if i < 16 then String.make 1 "xyzwabcdefghijkl".[i]
  else strf "v%d" i

let workitem_name name =
  match Gpu_dim.of_special_name name with
  | Some dim -> dim
  | None -> invalid_arg (strf "unknown SPECIAL name %S" name)

(* Const rendering helpers *)

let const_view_of_uop u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Some c
  | _ -> None

let truncate_uint32 (n : int64) = Int64.logand n 0xFFFFFFFFL

let uint64_decimal n =
  if Int64.equal n 0L then "0"
  else
    let rec loop acc n =
      if Int64.equal n 0L then acc
      else
        let q = Int64.unsigned_div n 10L in
        let r = Int64.unsigned_rem n 10L |> Int64.to_int in
        loop (Char.chr (Char.code '0' + r) :: acc) q
    in
    String.of_seq (List.to_seq (loop [] n))

(* Upper 16 bits of float32 encoding after round-to-nearest-even. *)
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

(* C-style language config and per-render context. *)

(* code_for_op dispatches are keyed by Ops.t. Some ops are unary, some binary,
   some ternary. Callbacks take the list of rendered operands plus the result
   dtype. *)
type code_for_op = Ops.t -> string list -> Dtype.t -> string

type 'ctx rule =
  Upat.t * ('ctx -> Upat.bindings -> U.t -> string option)

type ctx = {
  lang : language;
  r : string U.Tbl.t;
  lane_demand : int U.Tbl.t;
}

and language = {
  (* language options *)
  kernel_typedef : string;  (* may embed {launch_bounds} *)
  buffer_prefix : string;
  buffer_suffix : string;
  smem_align : string;
  smem_prefix : string;
  smem_prefix_for_cast : bool;
  arg_int_prefix : string;
  barrier : string;
  code_for_workitem : string -> string;
  extra_args : string list;
  supports_images : bool;
  float4 : string option;
  float4_style : string * string;
  gep_arr_threshold : int;
  type_map : Dtype.scalar -> string option;
  infinity : string;
  nan : string;
  code_for_op : code_for_op;
  (* rule sets *)
  string_rewrite : ctx rule list;
  extra_matcher : U.t -> U.t option;
  render_kernel :
    ctx -> function_name:string -> kernel:string list ->
    bufs:(U.t * string * (Dtype.t * bool)) list -> uops:U.t list ->
    prefix:string list option -> string;
  preamble : language -> U.t list -> string list;
}

(* Dtype rendering *)

let c_scalar_to_string = function
  | Dtype.Void -> "void"
  | Dtype.Weakint | Dtype.Int32 -> "int"
  | Dtype.Bool -> "bool"
  | Dtype.Int8 -> "signed char"
  | Dtype.Int16 -> "short"
  | Dtype.Int64 -> "long"
  | Dtype.Uint8 -> "unsigned char"
  | Dtype.Uint16 -> "unsigned short"
  | Dtype.Uint32 -> "unsigned int"
  | Dtype.Uint64 -> "unsigned long"
  | Dtype.Uint128 -> "ulong2"
  | Dtype.Uint256 -> "ulong4"
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "__bf16"
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Fp8e4m3 -> "float8_e4m3"
  | Dtype.Fp8e5m2 -> "float8_e5m2"
  | Dtype.Fp8e4m3fnuz -> "float8_e4m3fnuz"
  | Dtype.Fp8e5m2fnuz -> "float8_e5m2fnuz"

let clean_vector_base s = String.map (fun c -> if c = ' ' then '_' else c) s

let render_canonical_val (v : Dtype.Val.t) =
  let base = c_scalar_to_string (Dtype.Val.scalar v) in
  if Dtype.Val.count v > 1 then
    clean_vector_base base ^ string_of_int (Dtype.Val.count v)
  else base

let render_dtype_val (r : ctx) (v : Dtype.Val.t) =
  let scalar = Dtype.Val.scalar v in
  let base =
    match r.lang.type_map scalar with
    | Some s -> s
    | None -> c_scalar_to_string scalar
  in
  if Dtype.Val.count v > 1 then
    clean_vector_base base ^ string_of_int (Dtype.Val.count v)
  else base

let render_dtype (r : ctx) (dt : Dtype.t) =
  match dt with
  | Dtype.Ptr p ->
      let prefix =
        let a = Dtype.Ptr.addrspace p in
        if a = Dtype.Local && r.lang.smem_prefix_for_cast then r.lang.smem_prefix
        else if a = Dtype.Global then r.lang.buffer_prefix
        else ""
      in
      prefix ^ render_dtype_val r (Dtype.Ptr.base p) ^ "*"
  | Dtype.Val v -> render_dtype_val r v

let stack_dtype u =
  match U.op u, U.dtype u with
  | Ops.Stack, Dtype.Val v when Dtype.Val.count v = 1 ->
      let n = Array.length (U.src u) in
      if n <= 1 then U.dtype u
      else Dtype.Val (Dtype.Val.vec n (Dtype.Val.scalarize v))
  | _ -> U.dtype u

let max_numel u =
  try List.fold_left (fun acc d -> acc * U.vmax d) 1 (U.shape u)
  with Invalid_argument _ -> Dtype.count (stack_dtype u)

(* Memoized: an unmemoized walk revisits shared subgraphs and goes
   exponential on wide unrolled ALU chains. *)
let expr_numel_cache : int U.Ref_tbl.t = U.Ref_tbl.create 256

let rec expr_numel u =
  match U.Ref_tbl.find_opt expr_numel_cache u with
  | Some n -> n
  | None ->
      let n = compute_expr_numel u in
      U.Ref_tbl.add expr_numel_cache u n;
      n

and compute_expr_numel u =
  let dtype_count = Dtype.count (stack_dtype u) in
  match U.op u with
  | Ops.Stack ->
      let srcs = U.src u in
      if Array.length srcs = 0 then dtype_count else Array.length srcs
  | Ops.Index | Ops.Shrink -> max dtype_count (max_numel u)
  | Ops.Load ->
      let srcs = U.src u in
      if Array.length srcs = 0 then dtype_count
      else
        let access_count = expr_numel srcs.(0) in
        if access_count > 1 then access_count else dtype_count
  | Ops.Cast | Ops.Bitcast | Ops.Noop | Ops.After ->
      let srcs = U.src u in
      if Array.length srcs = 0 then dtype_count
      else max dtype_count (expr_numel srcs.(0))
  | op when Ops.Group.is_alu op ->
      U.src u |> Array.to_list |> List.map expr_numel
      |> List.fold_left max dtype_count
  | _ -> dtype_count

let base_render_numel u =
  match U.op u, U.src u with
  | (Ops.Buffer | Ops.Param), _ -> max_numel u
  | Ops.Load, srcs when Array.length srcs > 0 ->
      let access_count = expr_numel srcs.(0) in
      if access_count > 1 then access_count else max_numel u
  | _ -> expr_numel u

let lane_demand ctx u =
  try U.Tbl.find ctx.lane_demand u with Not_found -> 1

let render_numel ctx u = max (base_render_numel u) (lane_demand ctx u)

let scalar_view_source_is_value u =
  match U.addrspace u with
  | None | Some Dtype.Alu -> true
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> false

let render_value_dtype ctx u =
  match U.dtype u with
  | Dtype.Val dtype ->
      let count = render_numel ctx u in
      if count <= 1 then U.dtype u
      else Dtype.Val (Dtype.Val.vec count (Dtype.Val.scalarize dtype))
  | Dtype.Ptr _ -> U.dtype u

let render_type ctx u =
  match U.dtype u with
  | Dtype.Val dtype ->
      let count = render_numel ctx u in
      let dtype =
        if count <= 1 then dtype
        else Dtype.Val.vec count (Dtype.Val.scalarize dtype)
      in
      render_dtype_val ctx dtype
  | Dtype.Ptr ptr -> render_dtype ctx (Dtype.Ptr ptr)

let render_param_ptr_dtype ?(canonical = false) (ctx : ctx) p =
  let prefix =
    let a = Dtype.Ptr.addrspace p in
    if a = Dtype.Local && ctx.lang.smem_prefix_for_cast then ctx.lang.smem_prefix
    else if a = Dtype.Global then ctx.lang.buffer_prefix
    else ""
  in
  let base =
    if canonical then render_canonical_val (Dtype.Ptr.base p)
    else render_dtype_val ctx (Dtype.Ptr.base p)
  in
  prefix ^ base ^ "*"

let render_cast (r : ctx) (dt : Dtype.t) (v : string) =
  strf "(%s)(%s)" (render_dtype r dt) v

(* String rewrite engine: parallel of Upat.Pattern_matcher over string outputs. *)

let try_rewrite (rules : ctx rule list) (ctx : ctx) (u : U.t) :
    string option =
  List.find_map
    (fun (pat, fn) -> List.find_map (fun bs -> fn ctx bs u) (Upat.match_ pat u))
    rules

(* Rendered string of [u]. Returns "" if not yet rendered; the render loop
   always inserts a name before children are visited, so this is defensive. *)
let lookup (ctx : ctx) (u : U.t) : string =
  try U.Tbl.find ctx.r u with Not_found -> ""

let render_buffer (ctx : ctx) (u : U.t) =
  let render ~addrspace ~base ~size =
    let prefix =
      match addrspace with
      | Dtype.Local -> ctx.lang.smem_align ^ ctx.lang.smem_prefix
      | Dtype.Reg | Dtype.Global | Dtype.Alu -> ""
    in
    Some
      (strf "%s%s %s[%d];" prefix
         (render_dtype_val ctx base)
         (lookup ctx u) size)
  in
  match U.dtype u, U.as_buffer u with
  | Dtype.Ptr p, _ ->
      render ~addrspace:(Dtype.Ptr.addrspace p) ~base:(Dtype.Ptr.base p)
        ~size:(Dtype.Ptr.size p)
  | Dtype.Val base, Some { buffer; shape } -> (
      match U.const_int_value shape with
      | Some size -> render ~addrspace:buffer.addrspace ~base ~size
      | None -> None)
  | Dtype.Val _, None -> None

let render_index (ctx : ctx) ~ptr ~idxs =
  let flat_index_string () =
    let ptr_shape =
      try U.shape ptr with
      | Invalid_argument msg ->
          let node_summary u =
            let shape =
              try
                U.shape u
                |> List.map (fun s ->
                       match U.const_int_value s with
                       | Some n -> string_of_int n
                       | None -> Ops.name (U.op s))
                |> String.concat "x"
              with Invalid_argument e -> "!" ^ e
            in
            Printf.sprintf "%s/%s/tag=%d/shape=%s" (Ops.name (U.op u))
              (Dtype.to_string (U.dtype u)) (U.tag u) shape
          in
          invalid_arg
            (Printf.sprintf
               "render_index: pointer shape failed: %s ptr=%s srcs=[%s] idx_bounds=%s"
               msg (node_summary ptr)
               (U.children ptr
                |> List.map (fun s ->
                       Printf.sprintf "%s srcs=[%s]" (node_summary s)
                         (U.children s
                          |> List.map node_summary |> String.concat ";"))
                |> String.concat ",")
               (idxs
                |> List.map (fun idx -> string_of_int (U.vmax idx + 1))
                |> String.concat ","))
    in
    let used_shape =
      if List.length ptr_shape >= List.length idxs then
        List.filteri (fun i _ -> i < List.length idxs) ptr_shape |> List.map U.vmax
      else
        match ptr_shape with
        | [ flat ] ->
            let flat = U.vmax flat in
            let first_dim =
              match idxs with idx :: _ -> U.vmax idx + 1 | [] -> 1
            in
            if first_dim = flat then [ flat ]
            else
            let dims = List.map (fun idx -> U.vmax idx + 1) idxs in
            let prod = List.fold_left ( * ) 1 dims in
            if prod = flat then dims
            else
              invalid_arg
                (Printf.sprintf
                   "render_index: cannot infer %d logical dims from flat size %d bounds=%s"
                   (List.length idxs) flat
                   (String.concat ","
                      (List.map (fun idx -> string_of_int (U.vmax idx + 1)) idxs)))
        | _ ->
            invalid_arg
              (Printf.sprintf "render_index: rank mismatch, got %d idxs for rank %d"
                 (List.length idxs) (List.length ptr_shape))
    in
    let terms =
      let used_idxs =
        List.filteri (fun i _ -> i < List.length used_shape) idxs
        |> List.map (lookup ctx)
      in
      let rec go_strings stride acc rev_idxs rev_shape =
        match rev_idxs, rev_shape with
        | [], [] -> acc
        | idx :: idxs, dim :: shape ->
            let term =
              if stride = 1 then idx else strf "(%s*%d)" idx stride
            in
            go_strings (stride * dim) (term :: acc) idxs shape
        | _ -> acc
      in
      go_strings 1 [] (List.rev used_idxs) (List.rev used_shape)
    in
    match terms with
    | [] -> "0"
    | term :: terms -> List.fold_left (fun acc t -> strf "(%s+%s)" acc t) term terms
  in
  let idx_is_zero =
    match idxs with [ idx ] -> U.const_int_value idx = Some 0 | _ -> false
  in
  let render_add_index idx =
    let rendered = lookup ctx idx in
    let src = U.src idx in
    let has_named_child =
      Array.exists
        (fun child ->
          let s = lookup ctx child in
          String.length s > 0 && s.[0] <> '(')
        src
    in
    if has_named_child then rendered
    else
      let rec has_param u =
        U.op u = Ops.Param || Array.exists has_param (U.src u)
      in
      if not (has_param idx) then rendered
      else
        let rec terms acc u =
          if U.op u = Ops.Add then
            let src = U.src u in
            terms (terms acc src.(1)) src.(0)
          else u :: acc
        in
        let terms =
          terms [] idx
          |> List.stable_sort (fun a b -> Int.compare (U.vmax b) (U.vmax a))
          |> List.map (lookup ctx)
        in
        match terms with
        | [] -> rendered
        | [ term ] -> term
        | terms -> strf "(%s)" (String.concat "+" terms)
  in
  let idx =
    match idxs with
    | [ idx ] when U.op idx = Ops.Add -> render_add_index idx
    | [ idx ] -> lookup ctx idx
    | _ -> flat_index_string ()
  in
  if U.addrspace ptr = Some Dtype.Alu then begin
    match idxs with
    | [ idx ] -> (
    match const_view_of_uop idx with
    | Some c -> (
        match Const.view c with
        | Const.Int i ->
            let i = Int64.to_int i in
            let base = lookup ctx ptr in
            if Dtype.count (U.dtype ptr) > ctx.lang.gep_arr_threshold then
              strf "%s[%d]" base i
            else strf "%s.%s" base (vec_elem_letter i)
        | _ -> invalid_arg "render_index: ALU index must be integer CONST")
    | None -> invalid_arg "render_index: ALU index must be CONST")
    | _ -> invalid_arg "render_index: ALU index must be scalar"
  end else
    let base = lookup ctx ptr in
    if
      idx_is_zero
      && String.length base > 0
      && base.[0] = '('
    then base
    else strf "(%s+%s)" base idx

let access_ptr_dtype ~ptr access_dt =
  match access_dt with
  | Dtype.Val v -> Dtype.Ptr (Dtype.Ptr.with_base v ptr)
  | Dtype.Ptr _ -> access_dt

let access_cast_dtype u access_dt =
  match U.dtype u with
  | Dtype.Ptr p -> access_ptr_dtype ~ptr:p access_dt
  | Dtype.Val _ ->
      let addrspace = Option.value (U.addrspace u) ~default:Dtype.Reg in
      let base = Dtype.val_of access_dt in
      Dtype.Ptr (Dtype.Ptr.create base ~addrspace ~size:(-1))

let render_access (ctx : ctx) ~access_dt (u : U.t) =
  let expr_count = expr_numel u in
  let access_count =
    if expr_count > 1 then expr_count else max (Dtype.count access_dt) (max_numel u)
  in
  let render_vector_index_load ptrs vec =
    let n = Array.length ptrs in
    let scalar = Dtype.Val.scalarize (Dtype.val_of access_dt) in
    let dtype = if n <= 1 then scalar else Dtype.Val.vec n scalar in
    let ctor =
      match ctx.lang.float4 with
      | Some f ->
          replace_first ~needle:"float4"
            ~replacement:(render_dtype ctx (Dtype.Val dtype))
            f
      | None -> render_dtype ctx (Dtype.Val dtype)
    in
    let l, rr = ctx.lang.float4_style in
    let index_lane i =
      let base = lookup ctx vec in
      if expr_numel vec <= 1 then base
      else if access_count > ctx.lang.gep_arr_threshold then strf "%s[%d]" base i
      else strf "%s.%s" base (vec_elem_letter i)
    in
    let items =
      Array.to_list ptrs
      |> List.mapi (fun i ptr -> strf "*(%s+%s)" (lookup ctx ptr) (index_lane i))
    in
    strf "%s%s%s%s" ctor l (String.concat "," items) rr
  in
  match U.as_index u with
  | Some { ptr; idxs = [ vec ] }
    when U.op ptr = Ops.Stack && Array.length (U.src ptr) = access_count
         && access_count > 1 ->
      render_vector_index_load (U.src ptr) vec
  | _ ->
  let access_dt =
    if access_count = Dtype.count access_dt then access_dt
    else
      Dtype.Val
        (Dtype.Val.vec access_count
           (Dtype.Val.scalarize (Dtype.val_of access_dt)))
  in
  let cast =
    access_count > 1 ||
    match U.dtype u, U.src u with
    | Dtype.Ptr p, srcs when Array.length srcs >= 2 ->
        let ptr = srcs.(0) in
        not (Dtype.Val.equal (Dtype.Ptr.base p) (Dtype.val_of access_dt))
        || not (Dtype.equal (U.dtype u) (U.dtype ptr))
    | _ -> false
  in
  if cast then
    strf "*((%s)(%s))" (render_dtype ctx (access_cast_dtype u access_dt))
      (lookup ctx u)
  else strf "*%s" (lookup ctx u)

let image_index u =
  match U.as_index u with
  | Some { ptr; idxs } -> (
      match U.dtype ptr with
      | Dtype.Ptr p when Dtype.Ptr.is_image p ->
          let idx =
            match idxs with
            | [ idx ] -> idx
            | [ y; x ] -> U.stack [ y; x ]
            | _ -> invalid_arg "image_index: expected one int2 or two scalars"
          in
          Some (ptr, idx, p)
      | _ -> None)
  | None -> None

let image_coord (ctx : ctx) idx =
  match U.op idx, U.src idx, U.dtype idx with
  | Ops.Stack, lanes, Dtype.Val dtype
    when Array.length lanes = 2 && Dtype.Val.is_int dtype ->
      let y = lookup ctx lanes.(0) in
      let x = lookup ctx lanes.(1) in
      Some (strf "(int2)(%s,%s)" x y)
  | _ -> None

let check_image_support ctx =
  if not ctx.lang.supports_images then failwith "renderer does not support images"

let render_image_load ctx node src alt gate =
  match image_index src with
  | None -> None
  | Some (buf, idx, _p) ->
      check_image_support ctx;
      let coord =
        match image_coord ctx idx with
        | Some coord -> coord
        | None ->
            invalid_arg
              (strf "image load coordinate must be int2, got %s"
                 (Dtype.to_string (U.dtype idx)))
      in
      let read = strf "read_imagef(%s, smp, %s)" (lookup ctx buf) coord in
      let value =
        match alt, gate with
        | None, None -> read
        | Some alt, Some gate ->
            strf "(%s?%s:%s)" (lookup ctx gate) read (lookup ctx alt)
        | None, Some _ -> invalid_arg "gated image load requires alt value"
        | Some _, None -> invalid_arg "image load alt requires gated index"
      in
      Some
        (match U.dtype node with
         | Dtype.Val dtype when Dtype.Val.equal dtype Dtype.Val.float32 ->
             value
         | Dtype.Val dtype ->
             invalid_arg
               (strf "image load must produce float, got %s"
                  (Dtype.Val.to_string dtype))
         | Dtype.Ptr _ -> invalid_arg "image load must produce a value")

let render_image_store ctx dst value gate =
  match image_index dst with
  | None -> None
  | Some (buf, idx, p) ->
      check_image_support ctx;
      let coord =
        match image_coord ctx idx with
        | Some coord -> coord
        | None ->
            invalid_arg
              (strf "image store coordinate must be int2, got %s"
                 (Dtype.to_string (U.dtype idx)))
      in
      let value =
        match Dtype.Ptr.image_kind p, U.dtype value with
        | (Some Dtype.Imageh | Some Dtype.Imagef), Dtype.Val dtype
          when Dtype.Val.equal dtype Dtype.Val.float32 ->
            lookup ctx value
        | Some _, Dtype.Val dtype ->
            invalid_arg
              (strf "image store must write float, got %s"
                 (Dtype.Val.to_string dtype))
        | None, _ -> invalid_arg "image store requires an image dtype"
        | Some _, Dtype.Ptr _ -> invalid_arg "image store must write a value"
      in
      let write =
        strf "write_imagef(%s, %s, %s);" (lookup ctx buf) coord value
      in
      Some
        (match gate with
         | None -> write
         | Some gate -> strf "if (%s) %s" (lookup ctx gate) write)

let bitcast_passthrough_for_pointer_addrspace (ctx : ctx) (x : U.t) =
  match U.dtype x, U.src x with
  | Dtype.Ptr p, [| src |]
    when Dtype.Ptr.addrspace p = Dtype.Global
         || Dtype.Ptr.addrspace p = Dtype.Local ->
      Some (lookup ctx src)
  | _ -> None

(* Base rewrite rules *)

(* Render a float literal with a guaranteed decimal point or exponent so that
   C parses it as floating point, not integer. *)
let float_lit f =
  let bits = Int64.bits_of_float f in
  let rec shortest p =
    if p >= 17 then strf "%.17g" f
    else
      let s = strf "%.*g" p f in
      match float_of_string_opt s with
      | Some f' when Int64.equal (Int64.bits_of_float f') bits -> s
      | _ -> shortest (p + 1)
  in
  let s = shortest 1 in
  let s =
    let a = Float.abs f in
    if a >= 1e-4 && a < 1e16
       && (String.contains s 'e' || String.contains s 'E')
    then
      let fixed = strf "%.17f" f in
      let rec trim i =
        if i > 0 && fixed.[i] = '0' then trim (i - 1) else i
      in
      let i = trim (String.length fixed - 1) in
      if fixed.[i] = '.' then String.sub fixed 0 (i + 2)
      else String.sub fixed 0 (i + 1)
    else s
  in
  if String.contains s '.' || String.contains s 'e' || String.contains s 'E'
  then s
  else s ^ ".0"

let render_float (ctx : ctx) (dt : Dtype.t) f =
  if Float.is_nan f then strf "(%s)" (render_cast ctx dt ctx.lang.nan) else
  if f = Float.infinity then strf "(%s)" (render_cast ctx dt ctx.lang.infinity) else
  if f = Float.neg_infinity then
    strf "(%s)" (render_cast ctx dt ("-" ^ ctx.lang.infinity))
  else
    let lit = float_lit f in
    match Dtype.scalar dt with
    | Dtype.Float32 -> strf "%sf" lit
    | Dtype.Float64 -> lit
    | Dtype.Float16 | Dtype.Bfloat16 | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
    | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz ->
        strf "(%s)" (render_cast ctx dt (lit ^ "f"))
    | _ -> lit

let render_int (ctx : ctx) (dt : Dtype.t) n =
  match Dtype.scalar dt with
  | Dtype.Int64 -> strf "%Ldl" n
  | Dtype.Uint64 -> strf "%sul" (uint64_decimal n)
  | Dtype.Uint32 -> strf "%Ldu" (truncate_uint32 n)
  | Dtype.Uint8 | Dtype.Uint16 ->
      strf "(%s)" (render_cast ctx dt (strf "%Ldu" n))
  | Dtype.Int8 | Dtype.Int16 ->
      strf "(%s)" (render_cast ctx dt (strf "%Ld" n))
  | Dtype.Bool -> if Int64.compare n 0L = 0 then "0" else "1"
  | _ -> strf "%Ld" n

(* [Invalid] payloads are never rendered directly; rejecting here leaves a
   visible error rather than emitting empty source. *)
let render_const_any (ctx : ctx) (u : U.t) : string option =
  match const_view_of_uop u with
  | None -> None
  | Some c ->
      let dt = U.dtype u in
      match Const.view c with
      | Const.Bool b -> Some (if b then "1" else "0")
      | Const.Float f -> Some (render_float ctx dt f)
      | Const.Int n -> Some (render_int ctx dt n)
      | Const.Invalid -> None

(* base_rewrite rules. Each rule mirrors tinygrad's cstyle.py base_rewrite. *)
let base_rewrite : ctx rule list =
  let open Upat in
  [
    (* Local/register buffers. *)
    (op ~name:"x" Ops.Buffer, fun ctx bs _ -> render_buffer ctx (bs $ "x"));
    (* IF: "if (cond) {" *)
    ( op ~name:"x" Ops.If,
      fun ctx bs _ ->
        match U.as_if (bs $ "x") with
        | Some v -> Some (strf "if (%s) {" (lookup ctx v.cond))
        | None -> None );
    (* ENDIF / END: "}" *)
    (ops [ Ops.Endif; Ops.End ], fun _ _ _ -> Some "}");
    (* WMMA: "__name(a, b, c)" *)
    ( op ~name:"x" Ops.Wmma,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_wmma x with
        | Some v ->
            Some
              (strf "__%s(%s, %s, %s)" v.info.name (lookup ctx v.a)
                 (lookup ctx v.b) (lookup ctx v.c))
        | None -> None );
    (* RANGE: "for (dtype n = 0; n < size; n++) {" *)
    ( op ~name:"x" Ops.Range,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_range x with
        | Some v ->
            let n = lookup ctx x in
            Some
              (strf "for (%s %s = 0; %s < %s; %s++) {"
                 (render_dtype ctx (U.dtype x))
                 n n (lookup ctx v.size) n)
        | None -> None );
    (* STACK: "float4{a,b,c,d}" (or language-specific style) *)
    ( op ~name:"x" Ops.Stack,
      fun ctx bs _ ->
        let x = bs $ "x" in
        let srcs = U.src x in
        let dtype = stack_dtype x in
        let ctor =
          match ctx.lang.float4 with
          | Some f ->
              let dt = render_dtype ctx dtype in
              replace_first ~needle:"float4" ~replacement:dt f
          | None -> render_dtype ctx dtype
        in
        let l, rr = ctx.lang.float4_style in
        let items =
          Array.to_list srcs |> List.map (fun s -> lookup ctx s)
        in
        Some (strf "%s%s%s%s" ctor l (String.concat "," items) rr) );
    (* CAST vector (non-ptr): __builtin_convertvector *)
    ( op ~name:"x" Ops.Cast,
      fun ctx bs _ ->
        let x = bs $ "x" in
        if max_numel x > 1 && U.addrspace x = Some Dtype.Reg then
          Some
            (strf "__builtin_convertvector(%s, %s)"
               (lookup ctx (U.src x).(0))
               (render_type ctx x))
        else None );
    (* CAST scalar: (dtype)(x) *)
    ( op ~name:"x" Ops.Cast,
      fun ctx bs _ ->
        let x = bs $ "x" in
        Some (strf "(%s)" (render_cast ctx (U.dtype x) (lookup ctx (U.src x).(0))))
    );
    (* BITCAST: __builtin_bit_cast(dtype, (src_dtype)(src)) *)
    ( op ~name:"x" Ops.Bitcast,
      fun ctx bs _ -> bitcast_passthrough_for_pointer_addrspace ctx (bs $ "x") );
    ( op ~name:"x" Ops.Bitcast,
      fun ctx bs _ ->
        let x = bs $ "x" in
        let src = (U.src x).(0) in
        Some
          (strf "__builtin_bit_cast(%s, (%s)(%s))"
             (render_dtype ctx (U.dtype x))
             (render_dtype ctx (U.dtype src))
             (lookup ctx src)) );
    (* BARRIER *)
    (op Ops.Barrier, fun ctx _ _ -> Some ctx.lang.barrier);
    (* SPECIAL *)
    ( op ~name:"x" Ops.Special,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_special x with
        | Some v ->
            Some
              (strf "%s; /* %s */"
                 (ctx.lang.code_for_workitem v.name)
                 (Render.expr_to_string v.size))
        | None -> None );
    (* CONST rules *)
    ( op ~name:"x" Ops.Const,
      fun ctx _ u -> render_const_any ctx u );
    (* Scalar view metadata has no C expression form. It should normally be
       removed by movement/index rewrites; when a shape-1 view remains after
       linearization, rendering the source is the same scalar value. *)
    ( ops [ Ops.Reshape; Ops.Expand; Ops.Permute ] ~name:"x",
      fun ctx bs _ ->
        let x = bs $ "x" in
        let src = (U.src x).(0) in
        let rec uniform_const_base n =
          match U.op n with
          | Ops.Const when Dtype.count (U.dtype n) = 1 -> Some n
          | Ops.Reshape | Ops.Expand | Ops.Permute -> uniform_const_base (U.src n).(0)
          | _ -> None
        in
        if render_numel ctx x = 1 && scalar_view_source_is_value src
        then Some (lookup ctx src)
        else
          match uniform_const_base src with
          | Some c -> Some (lookup ctx c)
          | None -> None
    );
    (* INDEX/SHRINK: pointer arithmetic or value-lane extraction. *)
	    ( ops [ Ops.Index; Ops.Shrink ] ~name:"x",
	      fun ctx bs _ ->
	        let x = bs $ "x" in
	        let rec uniform_const_base n =
	          match U.op n with
	          | Ops.Const when Dtype.count (U.dtype n) = 1 -> Some n
	          | Ops.Reshape | Ops.Expand | Ops.Permute ->
	              uniform_const_base (U.src n).(0)
	          | _ -> None
	        in
	        match U.op x, U.as_index x, U.src x with
	        | Ops.Index, Some { ptr; idxs = [ idx ] }, _
	          when U.addrspace ptr = Some Dtype.Alu -> (
	            match U.const_int_value idx with
	            | None -> None
	            | Some i ->
	                let base = lookup ctx ptr in
	                let width = render_numel ctx ptr in
	                if Dtype.count (U.dtype ptr) = 1 && width <= 1 then
	                  Some base
	                else if Dtype.count (U.dtype ptr) > ctx.lang.gep_arr_threshold then
	                  Some (strf "%s[%d]" base i)
	                else Some (strf "%s.%s" base (vec_elem_letter i)))
	        | Ops.Index, Some v, _ -> (
	            match uniform_const_base v.ptr with
	            | Some c -> Some (lookup ctx c)
            | None ->
                Some (render_index ctx ~ptr:v.ptr ~idxs:v.idxs))
	        | Ops.Shrink, _, [| ptr; idx; _ |] ->
	            Some (render_index ctx ~ptr ~idxs:[ idx ])
	        | _ -> None );
    (* Image LOAD: read_imagef(buf, smp, (int2)(x,y)) *)
    ( op ~name:"x" Ops.Load,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_load x with
        | Some { src; alt; gate } -> render_image_load ctx x src alt gate
        | None -> None );
    (* LOAD with gate: (gate?*bidx:alt) *)
    ( op ~name:"x" Ops.Load,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_load x with
        | Some { src; alt = Some alt_u; gate = Some gate } ->
            Some
              (strf "(%s?%s:%s)" (lookup ctx gate)
                 (render_access ctx ~access_dt:(render_value_dtype ctx x) src)
                 (lookup ctx alt_u))
        | Some { src; alt = Some _; gate = None } ->
            Some
              (strf "(%s)"
                 (render_access ctx ~access_dt:(render_value_dtype ctx x) src))
        | Some { src; alt = None; gate = _ } ->
            Some
              (strf "(%s)"
                 (render_access ctx ~access_dt:(render_value_dtype ctx x) src))
        | None -> None );
    (* Image STORE: write_imagef(buf, (int2)(x,y), value); *)
    ( op ~name:"x" Ops.Store,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_store x with
        | Some { dst; value; gate } -> render_image_store ctx dst value gate
        | None -> None );
    (* STORE: *dst = value; *)
    ( op ~name:"x" Ops.Store,
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.as_store x with
        | Some v ->
            let store =
              strf "%s = %s;"
                (render_access ctx ~access_dt:(stack_dtype v.value) v.dst)
                (lookup ctx v.value)
            in
            Some
              (match v.gate with
               | None -> store
               | Some gate -> strf "if (%s) %s" (lookup ctx gate) store)
        | None -> None );
    (* ALU: dispatch to code_for_op *)
    ( ops Ops.Group.alu ~name:"x",
      fun ctx bs _ ->
        let x = bs $ "x" in
        let xop = U.op x in
        let assoc_strip =
          List.mem xop [ Ops.Add; Ops.Mul; Ops.Xor; Ops.Or; Ops.And ]
        in
        let args =
          Array.to_list (U.src x)
          |> List.map (fun s ->
                 let rendered = lookup ctx s in
                 if U.op (U.base s) = xop && assoc_strip then
                   strip_parens rendered
                 else if
                   xop = Ops.Add
                   && is_parenthesized_top_level_add rendered
                 then strip_parens rendered
                 else rendered)
        in
        Some (ctx.lang.code_for_op xop args (U.dtype x)) );
    (* CUSTOM / CUSTOMI: format the arg as a template with src strings. *)
    ( ops [ Ops.Custom; Ops.Customi ] ~name:"x",
      fun ctx bs _ ->
        let x = bs $ "x" in
        match U.arg x with
        | U.Arg.String fmt ->
            let args =
              Array.to_list (U.src x) |> List.map (fun s -> lookup ctx s)
            in
            Some (render_custom_fmt fmt args)
        | _ -> None );
  ]

(* Base code_for_op. Matches tinygrad's CStyleLanguage.code_for_op dict. *)
let base_code_for_op : code_for_op =
 fun op args dt ->
  let cdiv a b = strf "(%s/%s)" a b in
  let cmod a b = strf "(%s%%%s)" a b in
  match op, args with
  | Ops.Sqrt, [ x ] -> strf "sqrt(%s)" x
  | Ops.Reciprocal, [ x ] -> strf "(1/%s)" x
  | Ops.Neg, [ x ] -> strf "-%s" x
  | Ops.Exp2, [ x ] -> strf "exp2(%s)" x
  | Ops.Log2, [ x ] -> strf "log2(%s)" x
  | Ops.Sin, [ x ] -> strf "sin(%s)" x
  | Ops.Trunc, [ x ] -> strf "trunc(%s)" x
  | Ops.And, [ a; b ] -> strf "(%s&%s)" a b
  | Ops.Xor, [ a; b ] -> strf "(%s^%s)" a b
  | Ops.Or, [ a; b ] -> strf "(%s|%s)" a b
  | Ops.Add, [ a; b ] -> strf "(%s+%s)" a b
  | Ops.Sub, [ a; b ] -> strf "(%s-%s)" a b
  | Ops.Mul, [ a; b ] -> strf "(%s*%s)" a b
  | Ops.Cmod, [ a; b ] -> cmod a b
  | Ops.Cdiv, [ a; b ] -> cdiv a b
  | Ops.Cmpne, [ a; b ] -> strf "(%s!=%s)" a b
  | Ops.Shr, [ a; b ] -> strf "(%s>>%s)" a b
  | Ops.Shl, [ a; b ] -> strf "(%s<<%s)" a b
  | Ops.Cmplt, [ a; b ] -> strf "(%s<%s)" a b
  | Ops.Cmpeq, [ a; b ] -> strf "(%s==%s)" a b
  | Ops.Where, [ a; b; c ] -> strf "(%s?%s:%s)" a b c
  | _ ->
      invalid_arg
        (strf "base_code_for_op: unhandled op %s (arity %d)" (Ops.name op)
           (List.length args))

(* Extra matchers *)

(* no_vectorized_alu: split a vector ALU node into scalar indexes + STACK.
   Ported lazily; we only need the surface behavior here for bools and WHERE. *)
let no_vectorized_alu (u : U.t) : U.t option =
  let dt = U.dtype u in
  if Dtype.count dt <= 1 then None
  else
    let n = Dtype.count dt in
    let scalar_dt = Dtype.scalarize dt in
    let lanes =
      List.init n (fun i ->
        let scalar_srcs =
          Array.to_list (U.src u)
          |> List.map (fun s ->
                 if Dtype.count (U.dtype s) = n then
                   U.index ~ptr:s ~idxs:[ U.const_int i ] ()
                 else s)
        in
        U.replace u ~src:(Array.of_list scalar_srcs) ~dtype:scalar_dt ())
    in
    Some (U.stack ~dtype:(Dtype.Val.scalarize (Dtype.val_of dt)) lanes)

let extra_pm : U.t -> U.t option =
 fun node ->
  let dt = U.dtype node in
  let is_vec = Dtype.count dt > 1 in
  let is_bool_result = Dtype.is_bool dt && is_vec in
  match U.op node with
  | o when is_bool_result &&
           (Ops.Group.is_alu o || o = Ops.Cast || o = Ops.Bitcast
            || o = Ops.Index) ->
      no_vectorized_alu node
  | Ops.Cast when is_vec ->
      let srcs = U.src node in
      if Array.length srcs > 0 && Dtype.is_bool (U.dtype srcs.(0)) then
        no_vectorized_alu node
      else None
  | Ops.Where when is_vec -> no_vectorized_alu node
  | _ -> None

(* create_non_native_float_pats: promote ALU ops on non-native floats through
   float32.  Matches tinygrad's create_non_native_float_pats. *)
let create_non_native_float_pats ?(casting = true)
    (dts : Dtype.scalar list) : U.t -> U.t option =
 fun node ->
  let f32 = Dtype.float32 in
  let is_nn dt = List.mem (Dtype.scalar dt) dts in
  let cast_f32 src = U.cast ~src ~dtype:f32 in
  let dt = U.dtype node in
  match U.op node with
  | Ops.Where when is_nn dt ->
      let srcs = U.src node in
      if Array.length srcs = 3 then
        let a = srcs.(0) and b = srcs.(1) and c = srcs.(2) in
        let w =
          U.alu_ternary ~op:Ops.Where ~a ~b:(cast_f32 b) ~c:(cast_f32 c)
        in
        Some (U.cast ~src:w ~dtype:dt)
      else None
  | o when Ops.Group.is_alu o && is_nn dt ->
      let new_children =
        Array.to_list (U.src node)
        |> List.map (fun c -> if is_nn (U.dtype c) then cast_f32 c else c)
      in
      let promoted_dt =
        Dtype.Val (Dtype.Val.vec (Dtype.count dt) (Dtype.Val.float32))
      in
      let promoted =
        U.replace node ~src:(Array.of_list new_children) ~dtype:promoted_dt ()
      in
      Some (U.cast ~src:promoted ~dtype:dt)
  | o when Ops.Group.is_binary o && Dtype.is_bool dt ->
      let children = Array.to_list (U.src node) in
      if List.length children = 2 && List.for_all (fun c -> is_nn (U.dtype c)) children
      then
        let new_children = List.map cast_f32 children in
        Some (U.replace node ~src:(Array.of_list new_children) ())
      else None
  | Ops.Cast when casting ->
      let srcs = U.src node in
      if Array.length srcs = 0 then None
      else
        let src = srcs.(0) in
        let sdt = U.dtype src in
        if is_nn dt && Dtype.scalar sdt <> Dtype.Float32 then
          Some (U.cast ~src:(cast_f32 src) ~dtype:dt)
        else if is_nn sdt && Dtype.scalar dt <> Dtype.Float32 then
          Some (U.cast ~src:(cast_f32 src) ~dtype:dt)
        else None
  | _ -> None

(* Software bf16 ↔ f32 cast via bit manipulation. *)
let cast_float_to_bf16 (x : U.t) : U.t =
  let u32 = Dtype.Val.uint32 in
  let c_u32 n = U.const (Const.int u32 n) in
  let bits = U.bitcast ~src:x ~dtype:(Dtype.Val u32) in
  let neg_bits =
    U.alu_binary ~op:Ops.And
      ~lhs:(U.alu_unary ~op:Ops.Neg ~src:bits)
      ~rhs:(c_u32 0x7f800000)
  in
  let is_not_inf =
    U.alu_binary ~op:Ops.Cmpne ~lhs:neg_bits ~rhs:(c_u32 0)
  in
  let bit16 =
    U.alu_binary ~op:Ops.And
      ~lhs:(U.alu_binary ~op:Ops.Shr ~lhs:bits ~rhs:(c_u32 16))
      ~rhs:(c_u32 1)
  in
  let rounded =
    U.alu_binary ~op:Ops.Add ~lhs:bits
      ~rhs:
        (U.alu_binary ~op:Ops.Add ~lhs:bit16 ~rhs:(c_u32 0x7fff))
  in
  let mantissa_nz =
    U.alu_binary ~op:Ops.Cmpne
      ~lhs:(U.alu_binary ~op:Ops.And ~lhs:bits ~rhs:(c_u32 0xffff))
      ~rhs:(c_u32 0)
  in
  let inf_nan =
    U.alu_ternary ~op:Ops.Where ~a:mantissa_nz
      ~b:(U.alu_binary ~op:Ops.Or ~lhs:bits ~rhs:(c_u32 0x10000))
      ~c:bits
  in
  let result =
    U.alu_ternary ~op:Ops.Where ~a:is_not_inf ~b:rounded ~c:inf_nan
  in
  let shifted =
    U.alu_binary ~op:Ops.Shr ~lhs:result ~rhs:(c_u32 16)
  in
  U.bitcast
    ~src:(U.cast ~src:shifted ~dtype:(Dtype.of_scalar Dtype.Uint16))
    ~dtype:(Dtype.Val Dtype.Val.bfloat16)

let pm_manual_bf16_cast (node : U.t) : U.t option =
  match U.op node, U.src node with
  | Ops.Cast, [| src |]
    when Dtype.scalar (U.dtype node) = Dtype.Float32
         && Dtype.scalar (U.dtype src) = Dtype.Bfloat16 ->
      let bits =
        U.cast
          ~src:
            (U.bitcast ~src ~dtype:(Dtype.Val Dtype.Val.uint16))
          ~dtype:(Dtype.of_scalar Dtype.Uint32)
      in
      let shifted =
        U.alu_binary ~op:Ops.Shl ~lhs:bits
          ~rhs:(U.const (Const.int Dtype.Val.uint32 16))
      in
      Some (U.bitcast ~src:shifted ~dtype:(Dtype.Val Dtype.Val.float32))
  | Ops.Cast, [| src |]
    when Dtype.scalar (U.dtype node) = Dtype.Bfloat16
         && Dtype.scalar (U.dtype src) = Dtype.Float32 ->
      Some (cast_float_to_bf16 src)
  | _ -> None

(* Naming — range suffix, prefix per op. *)

let prefix_of (u : U.t) : string =
  match U.op u with
  | Ops.Wmma -> "wmma"
  | Ops.Const -> "const"
  | Ops.Buffer -> (
      match U.dtype u with
      | Dtype.Ptr p -> (
          match Dtype.Ptr.addrspace p with
          | Dtype.Local | Dtype.Global | Dtype.Alu | Dtype.Reg -> "buf")
      | Dtype.Val _ -> "buf")
  | Ops.Cast | Ops.Bitcast | Ops.Stack -> "cast"
  | Ops.Index -> "bidx"
  | Ops.Load -> "val"
  | _ -> "alu"

(* Core render loop. Mirrors CStyleLanguage._render. *)

let child_count_of (uops : U.t list) : int U.Tbl.t =
  let tbl = U.Tbl.create 64 in
  List.iter
    (fun u ->
      Array.iter
        (fun v ->
          let n = try U.Tbl.find tbl v with Not_found -> 0 in
          U.Tbl.replace tbl v (n + 1))
        (U.src u))
    uops;
  tbl

(* Params reachable from a STORE destination are rendered without a
   [const] qualifier. *)
let writable_params (uops : U.t list) : unit U.Ref_tbl.t =
  let tbl = U.Ref_tbl.create 16 in
  let store_dsts =
    List.filter_map
      (fun u -> Option.map (fun v -> v.U.dst) (U.as_store u))
      uops
  in
  let image_store_dsts =
    List.filter_map
      (fun u ->
        match U.op u, U.arg u, Array.to_list (U.src u) with
        | (Ops.Custom | Ops.Customi), U.Arg.String fmt, dst :: _
          when contains_substring fmt "write_imagef" ->
            Some dst
        | _ -> None)
      uops
  in
  let slice =
    U.toposort ~gate:(fun u -> U.op u <> Ops.End)
      (U.sink (store_dsts @ image_store_dsts))
  in
  List.iter
    (fun u ->
      if U.op u = Ops.Param then U.Ref_tbl.replace tbl u ())
    slice;
  tbl

(* Naming helpers derived from a Uop. *)

let sub_str i = if i >= 0 then string_of_int i else "m" ^ string_of_int (-i)

let effective_param_dtype u dt =
  match U.as_param u, dt with
  | Some { param; shape }, Dtype.Val base when param.addrspace <> Dtype.Alu -> (
      match U.const_int_value shape with
      | Some size ->
          Some
            (Dtype.Ptr
               (Dtype.Ptr.create base ~addrspace:param.addrspace ~size))
      | None -> None)
  | _ -> Some dt

let param_shape_dim_name dim =
  match U.const_int_value dim with
  | Some n -> string_of_int n
  | None -> (
      match U.as_param dim with
      | Some { param = { name = Some name; _ }; _ } -> name
      | _ -> strf "sym%d" (U.tag dim))

let param_shape_names u =
  match U.as_param u, U.dtype u with
  | Some { shape; _ }, _ when U.op shape <> Ops.Noop ->
      List.map param_shape_dim_name (U.as_shape shape)
  | _, Dtype.Ptr p -> [ string_of_int (Dtype.Ptr.size p) ]
  | _ -> []

let name_param (u : U.t) : string =
  match U.as_param u with
  | Some { param; _ } -> (
      match effective_param_dtype u (U.dtype u) with
      | Some (Dtype.Ptr _) ->
          strf "data%d_%s" param.slot (String.concat "_" (param_shape_names u))
      | Some (Dtype.Val _) when param.addrspace = Dtype.Alu ->
          if param.slot >= 0 then strf "data%d_" param.slot
          else Option.value param.name ~default:(strf "data%d_" param.slot)
      | _ -> strf "data%d" param.slot)
  | None -> "data"

let render_param_dtype u dt =
  match effective_param_dtype u dt with
  | Some dt -> dt
  | None -> dt

let name_range v =
  let base = strf "%sidx%d" (Axis_type.letter v.U.kind) v.U.axis in
  match v.U.sub with
  | [] -> base
  | sub -> base ^ "_" ^ String.concat "_" (List.map sub_str sub)

(* Decide whether a node should be inlined at its use site rather than
   assigned to a named temporary. Mirrors the predicate in
   CStyleLanguage._render. *)
let should_inline ~expand_ssa ~child_count (u : U.t) : bool =
  let cc =
    try U.Tbl.find child_count u with Not_found -> 0
  in
  let dt = U.dtype u in
  if U.op u = Ops.Cast && Dtype.vcount dt <> 1 then false
  else
    match U.op u with
    | Ops.Const | Ops.Index | Ops.Shrink | Ops.Customi -> true
    | Ops.Load ->
        U.addrspace (U.src u).(0) = Some Dtype.Reg
    | Ops.Cast
      when U.addrspace u = Some Dtype.Global || U.addrspace u = Some Dtype.Local
      ->
        true
    | Ops.Stack | Ops.Cast | Ops.Bitcast -> not expand_ssa && cc = 1
    | o when Ops.Group.is_alu o && o <> Ops.Where ->
        not expand_ssa && cc = 1
    | _ -> false

let address_view_parent op =
  op = Ops.Index || op = Ops.Shrink || op = Ops.Load

let scalar_view_source_can_be_forwarded parents u src =
  if scalar_view_source_is_value src then true
  else
    match U.Tbl.find_opt parents u with
    | Some (_ :: _ as ps) -> List.for_all (fun p -> address_view_parent (U.op p)) ps
    | Some [] | None -> false

let transparent_scalar_view parents (ctx : ctx) (u : U.t) : U.t option =
  match U.op u, U.src u with
  | (Ops.Reshape | Ops.Expand | Ops.Permute), srcs
    when Array.length srcs > 0
         && render_numel ctx u = 1
         && scalar_view_source_can_be_forwarded parents u srcs.(0) ->
      Some srcs.(0)
  | _ -> None

(* Counters per prefix for temporary naming. *)
module StrTbl = Hashtbl.Make (struct
  type t = string

  let equal = String.equal
  let hash = Hashtbl.hash
end)

(* _render: topologically walks the uops list and assigns names, rendering
   each uop as a C expression or statement. Mirrors CStyleLanguage._render. *)

type render_result = {
  name : string;
  kernel : string list;
  bufs : (U.t * string * (Dtype.t * bool)) list;
}

let render_uops (ctx : ctx) (uops : U.t list) : render_result =
  let r = ctx.r in
  let cc = child_count_of uops in
  let parents : U.t list U.Tbl.t = U.Tbl.create 128 in
  List.iter
    (fun parent ->
      Array.iter
        (fun child ->
          let prev =
            try U.Tbl.find parents child with Not_found -> []
          in
          U.Tbl.replace parents child (parent :: prev))
        (U.src parent))
    uops;
  let writable = writable_params uops in
  let bufs = ref [] in
  let seen_params = StrTbl.create 16 in
  let kernel = ref [] in
  let depth = ref 1 in
  let counters : int StrTbl.t = StrTbl.create 16 in
  let counter_get p = try StrTbl.find counters p with Not_found -> 0 in
  let expand_ssa = getenv "EXPAND_SSA" 0 <> 0 in
  let name = ref "test" in
  List.iter
    (fun u ->
      match U.op u with
      | Ops.Noop | Ops.Group -> ()
      | Ops.After ->
          let srcs = U.src u in
          if Array.length srcs > 0 then
            U.Tbl.replace r u (U.Tbl.find r srcs.(0))
      | Ops.Sink ->
          (match U.as_kernel_info u with
           | Some ki -> name := ki.name
           | None -> ())
      | Ops.Param ->
          let rendered = name_param u in
          U.Tbl.replace r u rendered;
          (match U.as_param u with
           | Some _ ->
               if not (StrTbl.mem seen_params rendered) then begin
                 StrTbl.add seen_params rendered ();
                 bufs :=
                   (u, rendered, (U.dtype u, U.Ref_tbl.mem writable u)) :: !bufs
               end
           | None -> ())
      | _ ->
          (match transparent_scalar_view parents ctx u with
           | Some src -> U.Tbl.replace r u (U.Tbl.find r src)
           | None ->
               (* Name assignment. Special and Range have semantic names;
                  other ops get a per-prefix counter. *)
               let prefix_opt =
                 match U.as_special u, U.as_range u with
                 | Some sv, _ ->
                     U.Tbl.replace r u sv.name;
                     None
                 | None, Some rv ->
                     U.Tbl.replace r u (name_range rv);
                     None
                 | None, None ->
                     let p = prefix_of u in
                     U.Tbl.replace r u (strf "%s%d" p (counter_get p));
                     Some p
               in
               let rendered =
                 match try_rewrite ctx.lang.string_rewrite ctx u with
                 | Some s -> s
                 | None ->
                     invalid_arg
                       (strf "failed to render %s tag=%d with %s"
                          (Ops.name (U.op u)) (U.tag u)
                          (Dtype.to_string (U.dtype u))
                       ^ " shape="
                       ^ (try
                            U.shape u
                            |> List.map (fun s ->
                                   match U.const_int_value s with
                                   | Some n -> string_of_int n
                                   | None -> Ops.name (U.op s))
                            |> String.concat "x"
                          with Invalid_argument _ -> "?")
                       ^ " marg="
                       ^ (match U.op u with
                          | Ops.Pad | Ops.Shrink -> (
                              try
                                match U.marg u with
                                | U.Marg_bounds bounds ->
                                    bounds
                                    |> List.map (fun (a, b) ->
                                           let show x =
                                             match U.const_int_value x with
                                             | Some n -> string_of_int n
                                             | None -> Format.asprintf "%a" U.pp x
                                           in
                                           Printf.sprintf "(%s,%s)"
                                             (show a) (show b))
                                    |> String.concat ","
                                | _ -> ""
                              with Invalid_argument _ -> "?")
                          | _ -> "")
                       ^ " ranges="
                       ^ (U.ranges u
                         |> List.map (fun r ->
                                match U.as_range r with
                                | Some v -> string_of_int v.axis
                                | None -> Ops.name (U.op r))
                         |> String.concat ",")
                       ^ " node="
                       ^ (match U.op u with
                          | Ops.Pad | Ops.Shrink | Ops.Expand | Ops.Reshape ->
                              Format.asprintf "%a" U.pp u
                          | _ -> "")
                       ^ " uses="
                       ^ string_of_int
                           (try U.Tbl.find cc u with Not_found -> 0)
                       ^ " parents="
                       ^ (try
                            U.Tbl.find parents u
                            |> List.map (fun p -> Ops.name (U.op p))
                            |> String.concat ","
                          with Not_found -> "")
                       ^ " chain="
                       ^ (let rec chain acc n =
                            let ps =
                              try U.Tbl.find parents n with Not_found -> []
                            in
                            match ps with
                            | [ p ] when Ops.Group.is_movement (U.op p) ->
                                chain (Ops.name (U.op p) :: acc) p
                            | _ ->
                                String.concat ">"
                                  (List.rev
                                     ((List.map
                                         (fun p -> Ops.name (U.op p))
                                         ps)
                                      @ acc))
                          in
                          chain [] u)
                       ^ " srcs="
                       ^ (U.children u
                         |> List.map (fun s ->
                                strf "%s/%s" (Ops.name (U.op s))
                                  (Dtype.to_string (U.dtype s)))
                         |> String.concat ","))
               in
               (match U.op u with Ops.Endif | Ops.End -> decr depth | _ -> ());
               if should_inline ~expand_ssa ~child_count:cc u then
                 U.Tbl.replace r u rendered
               else begin
                 let line =
                   match U.op u with
                   | Ops.Range | Ops.End | Ops.Buffer | Ops.Store -> rendered
                   | Ops.Special ->
                       strf "%s %s = %s" (render_type ctx u) (U.Tbl.find r u)
                         rendered
                   | _
                     when Dtype.equal (U.dtype u) (Dtype.Val Dtype.Val.void) ->
                       rendered
                   | _ ->
                       strf "%s %s = %s;" (render_type ctx u) (U.Tbl.find r u)
                         rendered
                 in
                 if !depth < 0 then
                   invalid_arg
                     (strf "negative render depth at %s rendered=%s node=%s"
                        (Ops.name (U.op u)) line (Format.asprintf "%a" U.pp u));
                 kernel := (String.make (!depth * 2) ' ' ^ line) :: !kernel;
                 (match prefix_opt with
                  | Some p -> StrTbl.replace counters p (counter_get p + 1)
                  | None -> ())
               end;
               (match U.op u with Ops.If | Ops.Range -> incr depth | _ -> ())))
    uops;
  { name = !name; kernel = List.rev !kernel; bufs = List.rev !bufs }

let buf_param ctx (u, nm, (dt, mut)) =
  if U.op u <> Ops.Param then invalid_arg "buf_param: expected Param";
  let dt = render_param_dtype u dt in
  match dt with
  | Dtype.Ptr p when Dtype.Ptr.is_image p ->
      strf "%s image2d_t %s" (if mut then "write_only" else "read_only") nm
  | Dtype.Ptr p ->
      let canonical =
        match U.dtype u with Dtype.Ptr _ -> true | Dtype.Val _ -> false
      in
      let ptr = render_param_ptr_dtype ~canonical ctx p in
      strf "%s%s %s" ptr ctx.lang.buffer_suffix nm
  | Dtype.Val v when Dtype.Val.equal v Dtype.Val.int32 ->
      strf "%s %s" ctx.lang.arg_int_prefix nm
  | _ -> strf "%s %s" (render_dtype ctx dt) nm

let collect_lane_demand uops =
  let tbl = U.Tbl.create 32 in
  let bump u width =
    if width > 1 then
      let old = try U.Tbl.find tbl u with Not_found -> 1 in
      if width > old then U.Tbl.replace tbl u width
  in
  List.iter
    (fun u ->
      match U.as_index u with
      | Some { ptr; idxs = [ idx ] } when U.addrspace ptr = Some Dtype.Alu -> (
          match U.const_int_value idx with
          | Some lane -> bump ptr (lane + 1)
          | None -> ())
      | _ -> ())
    uops;
  tbl

let local_size_of u =
  match U.as_special u with
  | Some { name; size; _ } ->
      (match Gpu_dim.of_special_name name with
       | Some (Gpu_dim.Local_id _) -> Some size
       | Some (Gpu_dim.Group_id _ | Gpu_dim.Global_idx _) | None -> None)
  | _ -> None

(* Default render_kernel: wraps the body in a function signature. *)
let default_render_kernel (ctx : ctx) ~function_name ~kernel ~bufs
    ~uops ~prefix =
  let launch_bounds =
    prod (List.map U.vmax (List.filter_map local_size_of uops))
  in
  let typedef =
    replace_first ~needle:"{launch_bounds}"
      ~replacement:(string_of_int launch_bounds)
      ctx.lang.kernel_typedef
  in
  let params =
    String.concat ", "
      (List.map (buf_param ctx) bufs @ ctx.lang.extra_args)
  in
  let body =
    let image_sampler =
      if
        List.exists
          (fun (_u, _nm, (dt, _mut)) ->
            match dt with
            | Dtype.Ptr p -> Dtype.Ptr.is_image p
            | Dtype.Val _ -> false)
          bufs
      then
        "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
      else ""
    in
    strf "%s %s(%s) {\n%s\n}" typedef function_name params
      (image_sampler ^ String.concat "\n" kernel)
  in
  match prefix with
  | None -> body
  | Some p -> String.concat "\n" p ^ "\n" ^ body

(* render: tie it all together. *)
let render (lang : language) ?name:name_override (uops : U.t list) : string =
  let ctx =
    { lang; r = U.Tbl.create (List.length uops);
      lane_demand = collect_lane_demand uops }
  in
  let { name; kernel; bufs } = render_uops ctx uops in
  let name = Option.value ~default:name name_override in
  let prefix = lang.preamble lang uops in
  let prefix = if prefix = [] then None else Some prefix in
  lang.render_kernel ctx
    ~function_name:(U.sanitize_function_name name)
    ~kernel ~bufs ~uops ~prefix

(* Default C-style renderer.  Mirrors CStyleLanguage defaults. *)
let default_type_map scalar = Some (c_scalar_to_string scalar)

let make_language
    ?(kernel_typedef = "void")
    ?(buffer_prefix = "") ?(buffer_suffix = "")
    ?(smem_align = "") ?(smem_prefix = "")
    ?(smem_prefix_for_cast = true)
    ?(arg_int_prefix = "const int")
    ?(barrier = "")
    ?(code_for_workitem = fun _ -> invalid_arg "no workitem support")
    ?(extra_args = [])
    ?(supports_images = false)
    ?(float4 = None)
    ?(float4_style = ("(", ")"))
    ?(gep_arr_threshold = 4)
    ?(type_map = default_type_map)
    ?(infinity = "INFINITY")
    ?(nan = "NAN")
    ?(code_for_op = base_code_for_op)
    ?(string_rewrite = base_rewrite)
    ?(extra_matcher = fun _ -> None)
    ?(render_kernel = default_render_kernel)
    ?(preamble = fun _ _ -> [])
    () : language =
  {
    kernel_typedef; buffer_prefix; buffer_suffix; smem_align; smem_prefix;
    smem_prefix_for_cast; arg_int_prefix; barrier; code_for_workitem;
    extra_args; supports_images; float4; float4_style; gep_arr_threshold; type_map;
    infinity; nan; code_for_op; string_rewrite; extra_matcher;
    render_kernel; preamble;
  }

(* ClangRenderer *)

let clang_type_map : Dtype.scalar -> string option = function
  | Dtype.Bool -> Some "_Bool"
  | Dtype.Float16 -> Some "__fp16"
  | _ -> None

let clang_code_for_op : code_for_op =
 fun op args dt ->
  match op, args with
  | Ops.Sqrt, [ x ] ->
      if Dtype.scalar dt = Dtype.Float64 then strf "__builtin_sqrt(%s)" x
      else strf "__builtin_sqrtf(%s)" x
  | Ops.Trunc, [ x ] ->
      if Dtype.scalar dt = Dtype.Float64 then strf "__builtin_trunc(%s)" x
      else strf "__builtin_truncf(%s)" x
  | Ops.Fdiv, [ a; b ] -> strf "(%s/%s)" a b
  | Ops.Exp2, _ | Ops.Log2, _ | Ops.Sin, _ | Ops.Reciprocal, _ ->
      invalid_arg (strf "clang does not provide %s" (Ops.name op))
  | _ -> base_code_for_op op args dt

let clang_extra_matcher (node : U.t) : U.t option =
  match U.op node, U.src node with
  | Ops.Cast, [| src |]
    when (Dtype.scalar (U.dtype src) = Dtype.Float64
          || Dtype.scalar (U.dtype src) = Dtype.Bfloat16)
         && (Dtype.scalar (U.dtype node) = Dtype.Float16
             || Dtype.scalar (U.dtype node) = Dtype.Bfloat16)
         && Dtype.scalar (U.dtype src) <> Dtype.scalar (U.dtype node) ->
      Some
        (U.cast
           ~src:
             (U.cast ~src ~dtype:(Dtype.of_scalar Dtype.Float32))
           ~dtype:(U.dtype node))
  | (Ops.Sqrt | Ops.Trunc), _ when Dtype.count (U.dtype node) > 1 ->
      no_vectorized_alu node
  | _ ->
      (match create_non_native_float_pats [ Dtype.Bfloat16 ] node with
       | Some _ as r -> r
       | None ->
           (match pm_manual_bf16_cast node with
	            | Some _ as r -> r
	            | None -> extra_pm node))

let floor_power_of_two n =
  let p = ref 1 in
  while !p * 2 <= n do
    p := !p * 2
  done;
  !p

(* Movement ops carry their shape/size metadata as STACK srcs (Reshape and
   Expand at index 1, Pad and Shrink at indices 1..2). In tinygrad this
   metadata lives in [.arg] as untyped tuples, so the renderer's dtype
   collection never sees it. Tolk's IR represents it as dtyped STACK nodes;
   mirror tinygrad's collection semantics by keying the exclusion on
   metadata position — a node reached only as a movement op's shape/size src
   is metadata, whereas the same node reached as a value elsewhere is kept. *)
let metadata_src_indices u =
  match U.op u with
  | Ops.Reshape | Ops.Expand -> [ 1 ]
  | Ops.Pad | Ops.Shrink -> [ 1; 2 ]
  | _ -> []

let used_alu_dtypes uops =
  (* Vector width must match the width used when the value is declared
     ({!render_type} -> {!render_numel}), not {!max_numel}: gated loads with
     out-of-bounds addressing have no valid shape, so [max_numel] falls back to
     a scalar while the value is still declared and indexed as a vector. *)
  let lane_demand = collect_lane_demand uops in
  let render_width u =
    max (base_render_numel u)
      (try U.Tbl.find lane_demand u with Not_found -> 1)
  in
  let metadata_only = U.Tbl.create 64 and value = U.Tbl.create 64 in
  List.iter
    (fun u ->
      let meta = metadata_src_indices u in
      Array.iteri
        (fun i s ->
          if List.mem i meta then
            (if not (U.Tbl.mem value s) then U.Tbl.replace metadata_only s ())
          else begin
            U.Tbl.replace value s ();
            U.Tbl.remove metadata_only s
          end)
        (U.src u))
    uops;
  let add dtype acc =
    match dtype with
    | Some dtype when not (Dtype.Val.equal dtype Dtype.Val.void) ->
        if List.exists (Dtype.Val.equal dtype) acc then acc else dtype :: acc
    | Some _ -> acc
    | None -> acc
  in
  let dtype_for u =
    if U.Tbl.mem metadata_only u then None
    else
      match U.addrspace u, stack_dtype u with
      | (Some Dtype.Alu | None), Dtype.Val dtype ->
          Some (Dtype.Val.vec (render_width u) (Dtype.Val.scalarize dtype))
      | Some (Dtype.Reg | Dtype.Global | Dtype.Local), _
      | (Some Dtype.Alu | None), Dtype.Ptr _ ->
          None
  in
  List.rev (List.fold_left (fun acc u -> add (dtype_for u) acc) [] uops)

let used_vector_dtypes uops =
  List.filter (fun dtype -> Dtype.Val.count dtype > 1) (used_alu_dtypes uops)

let clang_vector_prefix lang dtype =
  let ctx = { lang; r = U.Tbl.create 0; lane_demand = U.Tbl.create 0 } in
  let scalar = Dtype.Val.scalarize dtype in
  let alignment =
    if getenv "ALIGNED" 1 = 0 || Dtype.Val.scalar dtype = Dtype.Bool then 1
    else floor_power_of_two (Dtype.Val.itemsize dtype)
  in
  strf "typedef %s %s __attribute__((aligned(%d),ext_vector_type(%d)));"
    (render_dtype ctx (Dtype.Val scalar))
    (render_dtype ctx (Dtype.Val dtype))
    alignment (Dtype.Val.count dtype)

let clang_preamble lang uops =
  List.map (clang_vector_prefix lang) (used_vector_dtypes uops)

let clang_language : language =
  make_language
    ~buffer_suffix:" restrict"
    ~gep_arr_threshold:0
    ~type_map:clang_type_map
    ~float4:(Some "(float4)")
    ~float4_style:("{", "}")
    ~infinity:"__builtin_inff()"
    ~nan:"__builtin_nanf(\"\")"
    ~code_for_op:clang_code_for_op
    ~extra_matcher:clang_extra_matcher
    ~preamble:clang_preamble
    ()

let fixed_abi_arg ctx buf_idx val_idx (u, _nm, (dt, _mut)) =
  let dt = render_param_dtype u dt in
  match dt with
  | Dtype.Ptr _ ->
      let arg = strf "(%s)bufs[%d]" (render_dtype ctx dt) buf_idx in
      (arg, buf_idx + 1, val_idx)
  | Dtype.Val v when Dtype.Val.equal v Dtype.Val.int32 ->
      let arg = strf "(int)vals[%d]" val_idx in
      (arg, buf_idx, val_idx + 1)
  | _ ->
      invalid_arg
        (strf "fixed_abi_arg: unsupported parameter dtype %s"
           (Dtype.to_string dt))

let clang_fixed_abi_render_kernel ctx ~function_name ~kernel ~bufs ~uops
    ~prefix =
  let inner_name = function_name ^ "_" in
  let inner_ctx =
    { ctx with lang = { ctx.lang with kernel_typedef = "static void" } }
  in
  let inner =
    default_render_kernel inner_ctx ~function_name:inner_name ~kernel ~bufs
      ~uops ~prefix
  in
  let args, _, _ =
    List.fold_left
      (fun (args, buf_idx, val_idx) buf ->
        let arg, buf_idx, val_idx = fixed_abi_arg ctx buf_idx val_idx buf in
        (arg :: args, buf_idx, val_idx))
      ([], 0, 0) bufs
  in
  strf "%s\nvoid %s(const unsigned long long *bufs, const long long *vals) {\n  %s(%s);\n}"
    inner function_name inner_name (String.concat ", " (List.rev args))

let clang_fixed_abi_language : language =
  { clang_language with render_kernel = clang_fixed_abi_render_kernel }

(* OpenCLRenderer *)

let opencl_type_map : Dtype.scalar -> string option = function
  | Dtype.Int8 -> Some "char"
  | Dtype.Uint8 -> Some "uchar"
  | Dtype.Uint32 -> Some "uint"
  | Dtype.Uint16 -> Some "ushort"
  | Dtype.Uint64 -> Some "ulong"
  | Dtype.Bfloat16 -> Some "ushort"
  | _ -> None

let opencl_code_for_workitem name : string =
  let dim = workitem_name name in
  let a = Gpu_dim.axis dim in
  match dim with
  | Gpu_dim.Group_id _ -> strf "get_group_id(%d)" a
  | Gpu_dim.Local_id _ -> strf "get_local_id(%d)" a
  | Gpu_dim.Global_idx _ -> strf "get_global_id(%d)" a

let opencl_bf16_const_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Const,
    fun _ctx bs _ ->
      let x = bs $ "x" in
      match const_view_of_uop x, U.dtype x with
      | Some c, Dtype.Val v
        when Dtype.Val.scalar v = Dtype.Bfloat16 -> (
          match Const.view c with
          | Const.Float f -> Some (strf "%uu" (float_to_bf16_bits f))
          | _ -> None)
      | _ -> None )

(* OpenCL BITCAST: as_dtype((src_dtype)(src)) *)
let opencl_bitcast_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Bitcast,
    fun ctx bs _ ->
      let x = bs $ "x" in
      let srcs = U.src x in
      if Array.length srcs = 0 then None
      else
        Some
          (strf "as_%s((%s)(%s))"
             (render_dtype ctx (U.dtype x))
             (render_dtype ctx (U.dtype srcs.(0)))
             (lookup ctx srcs.(0))) )

let opencl_extra_matcher (node : U.t) : U.t option =
  match create_non_native_float_pats [ Dtype.Bfloat16 ] node with
  | Some _ as r -> r
  | None ->
      (match pm_manual_bf16_cast node with
       | Some _ as r -> r
       | None -> extra_pm node)

let has_dtype_scalar scalar uops =
  List.exists
    (fun u -> Dtype.scalar (U.dtype u) = scalar)
    uops

let opencl_preamble _ctx uops =
  let prefix =
    if has_dtype_scalar Dtype.Float16 uops then
      [ "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" ]
    else []
  in
  prefix

let opencl_aux uops =
  let params =
    List.filter_map
      (fun u ->
        match U.as_param u with
        | None -> None
        | Some { param = { slot; _ }; _ } -> Some (u, slot))
      uops
  in
  let add_param i (u, slot) =
    Some (slot, strf "(%d,%s)" i (Dtype.repr (U.dtype u)))
  in
  let by_slot = Hashtbl.create 8 in
  List.mapi add_param params
  |> List.filter_map Fun.id
  |> List.iter (fun (slot, item) ->
         let items = Option.value (Hashtbl.find_opt by_slot slot) ~default:[] in
         Hashtbl.replace by_slot slot (item :: items));
  let max_slot = Hashtbl.fold (fun slot _ acc -> max slot acc) by_slot (-1) in
  List.init (max_slot + 1) (fun slot ->
      match Hashtbl.find_opt by_slot slot with
      | None -> "()"
      | Some items -> "(" ^ String.concat "," (List.rev items) ^ ")")

let opencl_language : language =
  make_language
    ~kernel_typedef:"__kernel void"
    ~buffer_prefix:"__global "
    ~smem_align:"__attribute__ ((aligned (16))) "
    ~smem_prefix:"__local "
    ~barrier:"barrier(CLK_LOCAL_MEM_FENCE);"
    ~float4:(Some "(float4)")
    ~code_for_workitem:opencl_code_for_workitem
    ~type_map:opencl_type_map
    ~supports_images:true
    ~string_rewrite:
      (opencl_bitcast_rule :: opencl_bf16_const_rule :: base_rewrite)
    ~extra_matcher:opencl_extra_matcher
    ~preamble:opencl_preamble
    ()

(* MetalRenderer *)

let metal_type_map : Dtype.scalar -> string option = function
  | Dtype.Bfloat16 -> Some "bfloat"
  | _ -> None

let metal_code_for_workitem name : string =
  let dim = workitem_name name in
  let a = Gpu_dim.axis dim in
  let c = Char.chr (120 + a) in
  match dim with
  | Gpu_dim.Group_id _ -> strf "gid.%c" c
  | Gpu_dim.Local_id _ -> strf "lid.%c" c
  | Gpu_dim.Global_idx _ ->
      invalid_arg "Metal does not support Global_idx"

let metal_bitcast_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Bitcast,
    fun ctx bs _ ->
      let x = bs $ "x" in
      let srcs = U.src x in
      if Array.length srcs = 0 then None
      else
        Some
          (strf "as_type<%s>((%s)(%s))"
             (render_dtype ctx (U.dtype x))
             (render_dtype ctx (U.dtype srcs.(0)))
             (lookup ctx srcs.(0))) )

let metal_code_for_op : code_for_op =
 fun op args dt ->
  match op, args with
  | Ops.Sin, [ x ] -> strf "precise::sin(%s)" x
  | _ -> base_code_for_op op args dt

let metal_extra_matcher (node : U.t) : U.t option =
  match U.op node with
  | (Ops.Sqrt | Ops.Exp2 | Ops.Log2 | Ops.Sin)
    when Dtype.scalar (U.dtype node) = Dtype.Bfloat16 ->
      let f32 = Dtype.of_scalar Dtype.Float32 in
      let new_children =
        Array.to_list (U.src node)
        |> List.map (fun c -> U.cast ~src:c ~dtype:f32)
      in
      let promoted =
        U.replace node ~src:(Array.of_list new_children) ~dtype:f32 ()
      in
      Some (U.cast ~src:promoted ~dtype:(U.dtype node))
  | _ -> extra_pm node

let wmma_nodes uops : (U.wmma_info * Dtype.scalar) list =
  List.filter_map
    (fun u ->
      match U.as_wmma u, U.dtype u with
      | Some ({ info; _ } : U.wmma_view), Dtype.Val dtype ->
          Some (info, Dtype.Val.scalar dtype)
      | _ -> None)
    uops

let dedup_by_key key values =
  let rec loop seen acc = function
    | [] -> List.rev acc
    | x :: xs ->
        let k = key x in
        if List.exists (( = ) k) seen then loop seen acc xs
        else loop (k :: seen) (x :: acc) xs
  in
  loop [] [] values

let axis_product axes = prod (List.map snd axes)

let metal_wmma_helpers lang uops =
  let ctx = { lang; r = U.Tbl.create 0; lane_demand = U.Tbl.create 0 } in
  wmma_nodes uops
  |> dedup_by_key (fun ((info : U.wmma_info), dtype_out) ->
         (info.name, info.dtype_in, dtype_out))
  |> List.map (fun ((info : U.wmma_info), dtype_out) ->
         let in_dt =
           Dtype.Val.vec 2 (Dtype.Val.of_scalar info.dtype_in)
         in
         let out_dt = Dtype.Val.vec 2 (Dtype.Val.of_scalar dtype_out) in
         let dstr_in = render_dtype ctx (Dtype.Val in_dt) in
         let dstr_out = render_dtype ctx (Dtype.Val out_dt) in
         let scalar_in =
           render_dtype ctx (Dtype.Val (Dtype.Val.of_scalar info.dtype_in))
         in
         let scalar_out =
           render_dtype ctx (Dtype.Val (Dtype.Val.of_scalar dtype_out))
         in
         strf
           "%s __%s(%s a, %s b, %s c){\n\
           \  simdgroup_%s8x8 mat_a, mat_b; simdgroup_%s8x8 mat_c;\n\
           \  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n\
           \  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n\
           \  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n\
           \  return %s(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n\
            }"
           dstr_out info.name dstr_in dstr_in dstr_out scalar_in scalar_out
           dstr_out)

let metal_preamble lang uops =
  [ "#include <metal_stdlib>"; "using namespace metal;" ]
  @ metal_wmma_helpers lang uops

let metal_language : language =
  make_language
    ~kernel_typedef:"kernel void"
    ~buffer_prefix:"device "
    ~smem_prefix:"threadgroup __attribute__((aligned(16))) "
    ~arg_int_prefix:"constant int&"
    ~barrier:"threadgroup_barrier(mem_flags::mem_threadgroup);"
    ~float4:(Some "float4")
    ~code_for_workitem:metal_code_for_workitem
    ~extra_args:
      [
        "uint3 gid [[threadgroup_position_in_grid]]";
        "uint3 lid [[thread_position_in_threadgroup]]";
      ]
    ~type_map:metal_type_map
    ~code_for_op:metal_code_for_op
    ~string_rewrite:(metal_bitcast_rule :: base_rewrite)
    ~extra_matcher:metal_extra_matcher
    ~preamble:metal_preamble
    ()

(* CUDARenderer *)

let cuda_type_map : Dtype.scalar -> string option = function
  | Dtype.Bfloat16 -> Some "nv_bfloat16"
  | Dtype.Fp8e4m3 -> Some "__nv_fp8_e4m3"
  | Dtype.Fp8e5m2 -> Some "__nv_fp8_e5m2"
  | _ -> None

let is_half_or_bf16 dt =
  match Dtype.scalar dt with
  | Dtype.Float16 | Dtype.Bfloat16 -> true
  | _ -> false

let cuda_code_for_workitem name : string =
  let dim = workitem_name name in
  let a = Gpu_dim.axis dim in
  let c = Char.chr (120 + a) in
  match dim with
  | Gpu_dim.Group_id _ -> strf "blockIdx.%c" c
  | Gpu_dim.Local_id _ -> strf "threadIdx.%c" c
  | Gpu_dim.Global_idx _ ->
      strf "(blockIdx.%c*blockDim.%c+threadIdx.%c)" c c c

let cuda_code_for_op : code_for_op =
 fun op args dt ->
  let hfn name x =
    if is_half_or_bf16 dt then strf "h%s(%s)" name x
    else strf "%s(%s)" name x
  in
  match op, args with
  | Ops.Trunc, [ x ] -> hfn "trunc" x
  | Ops.Sin, [ x ] -> hfn "sin" x
  | Ops.Log2, [ x ] -> hfn "log2" x
  | Ops.Exp2, [ x ] -> hfn "exp2" x
  | Ops.Sqrt, [ x ] -> hfn "sqrt" x
  | Ops.Reciprocal, [ x ] ->
      if is_half_or_bf16 dt then strf "hrcp(%s)" x else strf "(1/%s)" x
  | _ -> base_code_for_op op args dt

let cuda_bitcast_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Bitcast,
    fun ctx bs _ ->
      let x = bs $ "x" in
      let srcs = U.src x in
      if Array.length srcs = 0 then None
      else
        Some
          (strf "tg_bitcast<%s>((%s)(%s))"
             (render_dtype ctx (U.dtype x))
             (render_dtype ctx (U.dtype srcs.(0)))
             (lookup ctx srcs.(0))) )

let cuda_extra_matcher (node : U.t) : U.t option =
  match
    create_non_native_float_pats ~casting:false
      [ Dtype.Fp8e4m3; Dtype.Fp8e5m2 ]
      node
  with
  | Some _ as r -> r
  | None -> (
      match U.op node, U.src node with
      | Ops.Cast, [| src |]
        when (Dtype.scalar (U.dtype node) = Dtype.Fp8e4m3
              || Dtype.scalar (U.dtype node) = Dtype.Fp8e5m2)
             && (Dtype.scalar (U.dtype src) = Dtype.Fp8e4m3
                 || Dtype.scalar (U.dtype src) = Dtype.Fp8e5m2)
             && Dtype.scalar (U.dtype node) <> Dtype.scalar (U.dtype src) ->
          Some
            (U.cast
               ~src:(U.cast ~src ~dtype:(Dtype.of_scalar Dtype.Float32))
               ~dtype:(U.dtype node))
      | _ -> extra_pm node)

let vector_elem_names n = List.init n vec_elem_letter

let cuda_vector_prefix lang dtype =
  let ctx = { lang; r = U.Tbl.create 0; lane_demand = U.Tbl.create 0 } in
  let vec = render_dtype ctx (Dtype.Val dtype) in
  let scal =
    render_dtype ctx (Dtype.Val (Dtype.Val.scalarize dtype))
  in
  let names = vector_elem_names (Dtype.Val.count dtype) in
  let elems = String.concat ", " names in
  let header = String.concat ", " (List.map (fun x -> scal ^ " " ^ x) names) in
  strf
    "struct __align__(%d) %s { %s %s; }; __device__ %s make_%s(%s) { %s r={%s}; return r; }"
    (Dtype.Val.itemsize dtype) vec scal elems vec vec header vec elems

let cuda_needs_vector_prefix dtype =
  match Dtype.Val.scalar dtype, Dtype.Val.count dtype with
  | (Dtype.Float16 | Dtype.Bfloat16), (4 | 8) -> true
  | (Dtype.Fp8e4m3 | Dtype.Fp8e5m2), (2 | 4 | 8 | 16) -> true
  | _ -> false

let cuda_wmma_type_name = function
  | Dtype.Float32 -> "tf32"
  | Dtype.Float16 -> "f16"
  | Dtype.Bfloat16 -> "bf16"
  | Dtype.Fp8e4m3 -> "e4m3"
  | Dtype.Fp8e5m2 -> "e5m2"
  | scalar -> Dtype.scalar_to_string scalar

let cuda_wmma_out_type_name = function
  | Dtype.Float32 -> "f32"
  | Dtype.Float16 -> "f16"
  | scalar -> Dtype.scalar_to_string scalar

let cuda_wmma_helpers lang uops =
  let ctx = { lang; r = U.Tbl.create 0; lane_demand = U.Tbl.create 0 } in
  wmma_nodes uops
  |> dedup_by_key (fun ((info : U.wmma_info), dtype_out) ->
         ( info.name,
           info.dims,
           info.dtype_in,
           dtype_out,
           info.device,
           info.threads,
           info.upcast_axes,
           info.reduce_axes ))
  |> List.map (fun ((info : U.wmma_info), dtype_out) ->
         let n, m, k = info.dims in
         let ua, ub, uc = info.upcast_axes in
         let upcast_sizes =
           [ axis_product ua; axis_product ub; axis_product uc ]
         in
         let wmma_dtypes =
           List.map2
             (fun dtype size ->
               render_dtype ctx
                 (Dtype.Val
                    (Dtype.Val.vec size (Dtype.Val.of_scalar dtype))))
             [ info.dtype_in; info.dtype_in; dtype_out ]
             upcast_sizes
         in
         let itemsize scalar =
           Dtype.Val.itemsize (Dtype.Val.of_scalar scalar)
         in
         let n_operands =
           List.map2
             (fun dtype size -> size * itemsize dtype / 4)
             [ info.dtype_in; info.dtype_in; dtype_out ]
             upcast_sizes
         in
         let n_a = List.nth n_operands 0 in
         let n_b = List.nth n_operands 1 in
         let n_c = List.nth n_operands 2 in
         let operands =
           List.init (n_a + n_b + n_c) (fun i -> strf "%%%d" i)
         in
         let take_range start count =
           List.init count (fun i -> List.nth operands (start + i))
           |> String.concat ", "
         in
         let c_ops = take_range 0 n_c in
         let a_ops = take_range n_c n_a in
         let b_ops = take_range (n_c + n_a) n_b in
         let output_constraints =
           List.init n_c (fun i -> strf "\"+r\"(c_pk[%d])" i)
           |> String.concat ", "
         in
         let input_constraints =
           (List.init n_a (fun i -> strf "\"r\"(a_pk[%d])" i)
            @ List.init n_b (fun i -> strf "\"r\"(b_pk[%d])" i))
           |> String.concat ", "
         in
         let dt_in = cuda_wmma_type_name info.dtype_in in
         let dt_out = cuda_wmma_out_type_name dtype_out in
         strf
           "__device__ %s __%s(%s a, %s b, %s c){\n\
           \  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);\n\
           \  asm(\"mma.sync.aligned.m%dn%dk%d.row.col.%s.%s.%s.%s\"\n\
           \      \"{%s}, {%s},\"\n\
           \      \"{%s}, {%s};\"\n\
           \    : %s\n\
           \    : %s);\n\
           \  return c;\n\
            }"
           (List.nth wmma_dtypes 2) info.name
           (List.nth wmma_dtypes 0) (List.nth wmma_dtypes 1)
           (List.nth wmma_dtypes 2)
           m n k dt_out dt_in dt_in dt_out c_ops a_ops b_ops c_ops
           output_constraints input_constraints)

let cuda_preamble _ctx uops =
  let used_dtypes = used_alu_dtypes uops in
  let has_used_scalar scalar =
    List.exists (fun dt -> Dtype.Val.scalar dt = scalar) used_dtypes
  in
  let prefix =
    [
      "#define INFINITY (__int_as_float(0x7f800000))";
      "#define NAN (__int_as_float(0x7fffffff))";
      "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }";
    ]
  in
  let prefix =
    if
      List.exists
        (fun dt ->
          match Dtype.Val.scalar dt with
          | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> true
          | _ -> false)
        used_dtypes
    then prefix @ [ "#include <cuda_fp8.h>" ]
    else prefix
  in
  let prefix =
    if has_used_scalar Dtype.Float16 then
      prefix @ [ "#include <cuda_fp16.h>" ]
    else prefix
  in
  let prefix =
    if has_used_scalar Dtype.Bfloat16 then
      prefix @ [ "#include <cuda_bf16.h>" ]
    else prefix
  in
  prefix
  @ List.map (cuda_vector_prefix _ctx)
      (List.filter cuda_needs_vector_prefix used_dtypes)
  @ cuda_wmma_helpers _ctx uops

let cuda_language : language =
  make_language
    ~kernel_typedef:
      "extern \"C\" __global__ void __launch_bounds__({launch_bounds})"
    ~smem_prefix:"__shared__ __align__(16) "
    ~smem_prefix_for_cast:false
    ~barrier:"__syncthreads();"
    ~float4:(Some "make_float4")
    ~gep_arr_threshold:8
    ~code_for_workitem:cuda_code_for_workitem
    ~type_map:cuda_type_map
    ~code_for_op:cuda_code_for_op
    ~string_rewrite:(cuda_bitcast_rule :: base_rewrite)
    ~extra_matcher:cuda_extra_matcher
    ~preamble:cuda_preamble
    ()

(* HIPRenderer (AMD) — minimal surface; full WMMA handling is deferred to
   the kernel preamble generator like in tinygrad. *)

let amd_type_map : Dtype.scalar -> string option = function
  | Dtype.Bfloat16 -> Some "hip_bfloat16"
  | Dtype.Fp8e4m3 -> Some "hip_fp8"
  | Dtype.Fp8e5m2 -> Some "hip_bf8"
  | _ -> None

let amd_is_cdna = function
  | Gpu_target.CDNA3 | Gpu_target.CDNA4 -> true
  | Gpu_target.RDNA3 | Gpu_target.RDNA4 -> false

let amd_fp8_index dt =
  match Dtype.scalar dt with
  | Dtype.Fp8e4m3 -> Some 0
  | Dtype.Fp8e5m2 -> Some 1
  | _ -> None

let ocml_call name (dt : Dtype.t) x =
  let bits =
    match Dtype.scalar dt with
    | Dtype.Float16 -> 16
    | Dtype.Float64 -> 64
    | _ -> 32
  in
  strf "__ocml_%s_f%d(%s)" name bits x

let amd_code_for_op : code_for_op =
 fun op args dt ->
  match op, args with
  | Ops.Trunc, [ x ] -> ocml_call "trunc" dt x
  | Ops.Sin, [ x ] -> ocml_call "sin" dt x
  | Ops.Log2, [ x ] -> ocml_call "log2" dt x
  | Ops.Exp2, [ x ] -> ocml_call "exp2" dt x
  | Ops.Sqrt, [ x ] -> ocml_call "sqrt" dt x
  | _ -> base_code_for_op op args dt

let amd_code_for_workitem name : string =
  let dim = workitem_name name in
  let a = Gpu_dim.axis dim in
  match dim with
  | Gpu_dim.Group_id _ -> strf "__ockl_get_group_id(%d)" a
  | Gpu_dim.Local_id _ -> strf "__ockl_get_local_id(%d)" a
  | Gpu_dim.Global_idx _ ->
      strf
        "(__ockl_get_group_id(%d)*__ockl_get_local_size(%d)+__ockl_get_local_id(%d))"
        a a a

let amd_vector_prefix lang dtype =
  let ctx = { lang; r = U.Tbl.create 0; lane_demand = U.Tbl.create 0 } in
  let vec = render_dtype ctx (Dtype.Val dtype) in
  let scal =
    render_dtype ctx (Dtype.Val (Dtype.Val.scalarize dtype))
  in
  let names = vector_elem_names (Dtype.Val.count dtype) in
  let header = String.concat ", " (List.map (fun x -> scal ^ " " ^ x) names) in
  let elems = String.concat ", " names in
  strf
    "typedef %s %s __attribute__((ext_vector_type(%d)));\n\
     static inline __attribute__((device)) %s make_%s(%s) { return { %s }; }"
    scal vec (Dtype.Val.count dtype) vec vec header elems

let has_const_nonfinite uops =
  List.exists
    (fun u ->
      match U.op u, const_view_of_uop u with
      | Ops.Const, Some c -> (
          match Const.view c with
          | Const.Float f -> not (Float.is_finite f)
          | Const.Bool _ | Const.Int _ | Const.Invalid -> false)
      | _ -> false)
    uops

let amd_ocml_decl op dtype =
  let method_name, attr =
    match op with
    | Ops.Exp2 -> ("exp2", "pure")
    | Ops.Log2 -> ("log2", "pure")
    | Ops.Sqrt -> ("sqrt", "const")
    | Ops.Sin -> ("sin", "")
    | Ops.Trunc -> ("trunc", "")
    | _ -> invalid_arg "amd_ocml_decl: unsupported op"
  in
  let scalar = Dtype.Val.scalarize dtype in
  let bits = Dtype.Val.bitsize scalar in
  let dtn = render_canonical_val scalar in
  strf "extern \"C\" __attribute__((device%s)) %s __ocml_%s_f%d(%s);"
    (if attr = "" then "" else ", " ^ attr)
    dtn method_name bits dtn

let amd_ocml_decls uops =
  let add decl acc = if List.mem decl acc then acc else decl :: acc in
  List.rev
    (List.fold_left
       (fun acc u ->
         match U.op u, stack_dtype u with
         | (Ops.Exp2 | Ops.Log2 | Ops.Sqrt | Ops.Sin | Ops.Trunc as op),
           Dtype.Val dtype -> (
             match Dtype.Val.scalar dtype with
             | Dtype.Float16 | Dtype.Float32 | Dtype.Float64 ->
                 add (amd_ocml_decl op dtype) acc
             | _ -> acc)
         | _ -> acc)
       [] uops)

let amd_fp8_const_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Const,
    fun ctx bs _ ->
      let x = bs $ "x" in
      match amd_fp8_index (U.dtype x), const_view_of_uop x with
      | Some fp8, Some c -> (
          match Const.view c with
          | Const.Float f when Float.is_nan f ->
              Some (strf "f32_to_fp8(%s, %d)" ctx.lang.nan fp8)
          | Const.Float f when f = Float.infinity ->
              Some (strf "f32_to_fp8(%s, %d)" ctx.lang.infinity fp8)
          | Const.Float f when f = Float.neg_infinity ->
              Some (strf "f32_to_fp8(-%s, %d)" ctx.lang.infinity fp8)
          | Const.Float f ->
              Some (strf "f32_to_fp8(%sf, %d)" (float_lit f) fp8)
          | Const.Bool _ | Const.Int _ | Const.Invalid -> None)
      | _ -> None )

let amd_fp8_cast_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Cast,
    fun ctx bs _ ->
      let x = bs $ "x" in
      match U.src x with
      | [| src |] -> (
          match amd_fp8_index (U.dtype x), Dtype.scalar (U.dtype src) with
          | Some fp8, Dtype.Float32 ->
              Some (strf "f32_to_fp8(%s, %d)" (lookup ctx src) fp8)
          | _ -> (
              match Dtype.scalar (U.dtype x), amd_fp8_index (U.dtype src) with
              | Dtype.Float32, Some 0 ->
                  Some
                    (strf
                       "__builtin_amdgcn_cvt_f32_fp8((unsigned int)%s, 0)"
                       (lookup ctx src))
              | Dtype.Float32, Some 1 ->
                  Some
                    (strf
                       "__builtin_amdgcn_cvt_f32_bf8((unsigned int)%s, 0)"
                       (lookup ctx src))
              | _ -> None))
      | _ -> None )

let amd_wmma_rule arch : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Wmma,
    fun ctx bs _ ->
      let x = bs $ "x" in
      match U.as_wmma x with
      | Some v when amd_is_cdna arch ->
          let _, _, k = v.info.dims in
          let a = lookup ctx v.a and b = lookup ctx v.b and c = lookup ctx v.c in
          if k = 128 then (
            match amd_fp8_index (U.dtype v.a) with
            | Some fp8 ->
                Some
                  (strf "__%s(%s, %s, %s, %d, %d, 0, 0, 0, 0)" v.info.name
                     a b c fp8 fp8)
            | None -> None)
          else Some (strf "__%s(%s, %s, %s, 0, 0, 0)" v.info.name a b c)
      | Some _ | None -> None )

let amd_string_rewrite arch =
  if amd_is_cdna arch then
    [ amd_wmma_rule arch; amd_fp8_const_rule; amd_fp8_cast_rule ] @ base_rewrite
  else base_rewrite

let amd_non_native_float_scalars =
  [
    Dtype.Bfloat16;
    Dtype.Fp8e4m3;
    Dtype.Fp8e5m2;
    Dtype.Fp8e4m3fnuz;
    Dtype.Fp8e5m2fnuz;
  ]

let amd_bf16_const_cast node =
  match U.op node, U.dtype node, const_view_of_uop node with
  | Ops.Const, Dtype.Val dtype, Some c
    when Dtype.Val.scalar dtype = Dtype.Bfloat16 -> (
      match Const.view c with
      | Const.Float f ->
          Some (cast_float_to_bf16 (U.const (Const.float Dtype.Val.float32 f)))
      | Const.Bool _ | Const.Int _ | Const.Invalid -> None)
  | _ -> None

let amd_fp8_wmma_bitcast node =
  match U.as_wmma node, U.dtype node with
  | Some v, Dtype.Val dtype
    when Dtype.Val.equal dtype (Dtype.Val.vec 4 Dtype.Val.float32)
         && (Dtype.equal (stack_dtype v.a)
               (Dtype.Val (Dtype.Val.vec 8 Dtype.Val.fp8e4m3))
             || Dtype.equal (stack_dtype v.a)
                  (Dtype.Val (Dtype.Val.vec 8 Dtype.Val.fp8e5m2))) ->
      Some
        (U.wmma
           ~a:(U.bitcast ~src:v.a ~dtype:Dtype.uint64)
           ~b:(U.bitcast ~src:v.b ~dtype:Dtype.uint64)
           ~c:v.c ~info:v.info ~dtype)
  | Some _, _ | None, _ -> None

let amd_extra_matcher arch node =
  match create_non_native_float_pats amd_non_native_float_scalars node with
  | Some _ as r -> r
  | None -> (
      match amd_fp8_wmma_bitcast node with
      | Some _ as r -> r
      | None -> (
          match amd_bf16_const_cast node with
          | Some _ as r -> r
          | None -> (
              match arch with
              | Gpu_target.CDNA4 -> extra_pm node
              | Gpu_target.RDNA3 | Gpu_target.RDNA4 | Gpu_target.CDNA3 -> (
                  match pm_manual_bf16_cast node with
                  | Some _ as r -> r
                  | None -> extra_pm node))))

let amd_type_map_name = function
  | Dtype.Bfloat16 -> "bf16"
  | Dtype.Float32 -> "f32"
  | Dtype.Float16 -> "f16"
  | Dtype.Fp8e4m3 -> "_fp8_fp8"
  | Dtype.Fp8e5m2 -> "_bf8_bf8"
  | scalar -> Dtype.scalar_to_string scalar

let amd_cdna_type_map_name dims scalar =
  match dims, scalar with
  | (16, 16, 16), Dtype.Bfloat16 -> "bf16_1k"
  | (16, 16, 32), Dtype.Bfloat16 -> "_bf16"
  | (16, 16, 32), Dtype.Float16 -> "_f16"
  | (16, 16, 128), (Dtype.Fp8e4m3 | Dtype.Fp8e5m2) -> "_f8f6f4"
  | _ -> amd_type_map_name scalar

let amd_wmma_prefix arch (info : U.wmma_info) dtype_out =
  let n, m, k = info.dims in
  let name = info.name in
  match arch with
  | Gpu_target.CDNA3 | Gpu_target.CDNA4 ->
      strf "#define __%s __builtin_amdgcn_mfma_%sf32_%dx%dx%d%s"
        name
        (if k = 128 then "scale_" else "")
        n m k (amd_cdna_type_map_name info.dims info.dtype_in)
  | Gpu_target.RDNA4 ->
      strf "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s_w32_gfx12"
        name (amd_type_map_name dtype_out) (amd_type_map_name info.dtype_in)
  | Gpu_target.RDNA3 when dtype_out = Dtype.Float32 ->
      strf "#define __%s __builtin_amdgcn_wmma_f32_16x16x16_%s_w32"
        name
        (match info.dtype_in with
         | Dtype.Float16 -> "f16"
         | _ -> "bf16")
  | Gpu_target.RDNA3 ->
      strf
        "static inline __attribute__((device)) half8 __%s(half16 a, half16 b, half8 c) {\n\
        \  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }\n\
        \  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);\n\
        \  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n\
         }"
        name

let amd_wmma_prefixes arch uops =
  let add info dtype_out acc =
    let prefix = amd_wmma_prefix arch info dtype_out in
    if List.mem prefix acc then acc else prefix :: acc
  in
  List.rev
    (List.fold_left
       (fun acc u ->
         match U.as_wmma u, U.dtype u with
         | Some v, Dtype.Val dtype ->
             add v.info (Dtype.Val.scalar dtype) acc
         | _ -> acc)
       [] uops)

let amd_preamble arch _lang uops =
  let used_dtypes = used_alu_dtypes uops in
  let has_used_scalar scalar =
    List.exists (fun dt -> Dtype.Val.scalar dt = scalar) used_dtypes
  in
  let prefix =
    if has_const_nonfinite uops then
      [ "#define INFINITY (__builtin_inff())"; "#define NAN (__builtin_nanf(\"\"))" ]
    else []
  in
  let prefix, ockl =
    if List.exists (fun u -> U.op u = Ops.Special) uops then
      ( prefix @ [ "typedef long unsigned int size_t;" ],
        [
          "extern \"C\" __attribute__((device, const)) unsigned int __ockl_get_local_id(size_t);";
          "extern \"C\" __attribute__((device, const)) unsigned int __ockl_get_group_id(size_t);";
          "extern \"C\" __attribute__((device, const)) unsigned int __ockl_get_local_size(size_t);";
        ] )
    else (prefix, [])
  in
  let bf16_typedef =
    if has_used_scalar Dtype.Bfloat16 then
      match arch with
      | Gpu_target.CDNA4 -> [ "typedef __bf16 hip_bfloat16;" ]
      | Gpu_target.RDNA3 | Gpu_target.RDNA4 | Gpu_target.CDNA3 ->
          [ "typedef unsigned short hip_bfloat16;" ]
    else []
  in
  let half_define =
    if has_used_scalar Dtype.Float16 then [ "#define half _Float16" ] else []
  in
  let fp8_typedefs =
    if has_used_scalar Dtype.Fp8e4m3 || has_used_scalar Dtype.Fp8e5m2 then
      [ "typedef unsigned char hip_bf8;"; "typedef unsigned char hip_fp8;" ]
    else []
  in
  let fp8_helper =
    if
      List.exists
        (fun u ->
          match U.op u, U.src u with
          | Ops.Const, _ -> Option.is_some (amd_fp8_index (U.dtype u))
          | Ops.Cast, [| src |] ->
              Option.is_some (amd_fp8_index (U.dtype u))
              && Dtype.scalar (U.dtype src) = Dtype.Float32
          | _ -> false)
        uops
    then
      [
        "static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {\n\
        \  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;\n\
        \  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));\n\
         }";
      ]
    else []
  in
  prefix @ bf16_typedef @ half_define @ fp8_typedefs @ fp8_helper @ ockl
  @ amd_ocml_decls uops
  @ List.map (amd_vector_prefix _lang) (used_vector_dtypes uops)
  @ amd_wmma_prefixes arch uops

let amd_language arch : language =
  make_language
    ~kernel_typedef:
      "extern \"C\" __attribute__((global)) void \
       __attribute__((amdgpu_flat_work_group_size(1, {launch_bounds})))"
    ~smem_prefix:"__attribute__((shared, aligned(16)))"
    ~smem_prefix_for_cast:false
    ~barrier:
      "__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");\
       __builtin_amdgcn_s_barrier();\
       __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");"
    ~float4:(Some "make_float4")
    ~code_for_workitem:amd_code_for_workitem
    ~type_map:amd_type_map
    ~code_for_op:amd_code_for_op
    ~string_rewrite:(amd_string_rewrite arch)
    ~extra_matcher:(amd_extra_matcher arch)
    ~preamble:(amd_preamble arch)
    ()

(* IntelRenderer: OpenCL variant with sub-group size. *)

let intel_bf16_cast_rule : ctx rule =
  let open Upat in
  ( op ~name:"x" Ops.Cast,
    fun ctx bs _ ->
      let x = bs $ "x" in
      let srcs = U.src x in
      if Array.length srcs = 0 then None
      else
        let src = srcs.(0) in
        match Dtype.scalar (U.dtype x), Dtype.scalar (U.dtype src) with
        | Dtype.Bfloat16, Dtype.Float32 ->
            Some
              (strf "intel_convert_bfloat16_as_ushort(%s)" (lookup ctx src))
        | Dtype.Float32, Dtype.Bfloat16 ->
            Some
              (strf "intel_convert_as_bfloat16_float(%s)" (lookup ctx src))
        | _ -> None )

let intel_language : language =
  make_language
    ~kernel_typedef:
      "__attribute__((intel_reqd_sub_group_size(8)))\n__kernel void"
    ~buffer_prefix:"__global "
    ~smem_align:"__attribute__ ((aligned (16))) "
    ~smem_prefix:"__local "
    ~barrier:"barrier(CLK_LOCAL_MEM_FENCE);"
    ~float4:(Some "(float4)")
    ~code_for_workitem:opencl_code_for_workitem
    ~type_map:opencl_type_map
    ~supports_images:true
    ~string_rewrite:
      (intel_bf16_cast_rule :: opencl_bitcast_rule :: opencl_bf16_const_rule
     :: base_rewrite)
    ~extra_matcher:opencl_extra_matcher
    ~preamble:opencl_preamble
    ()

let code_ops_base =
  Renderer.
    [
      Sqrt; Recip; Neg; Exp2; Log2; Sin; Trunc; And; Xor; Or; Add; Sub; Mul;
      Cmod; Cdiv; Cmpne; Shr; Shl; Cmplt; Where; Cmpeq;
    ]

let code_ops_clang =
  Renderer.
    [
      Sqrt; Neg; And; Xor; Or; Add; Sub; Mul; Cmod; Cdiv; Cmpne; Shr; Shl;
      Cmplt; Where; Cmpeq; Fdiv; Trunc;
    ]

let supports_not scalars dt =
  not (List.mem (Dtype.scalar dt) scalars)

let supports_opencl_dtype (arch : Gpu_target.opencl) dt =
  match Dtype.scalar dt with
  | Dtype.Float16 -> contains_substring arch "cl_khr_fp16"
  | Dtype.Float64 -> contains_substring arch "cl_khr_fp64"
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 | Dtype.Fp8e4m3fnuz
  | Dtype.Fp8e5m2fnuz ->
      false
  | _ -> true

let supports_qcom_dtype dt =
  match Dtype.scalar dt with
  | Dtype.Float16 ->
      Helpers.getenv "IMAGE" 0 <> 0 && Helpers.getenv "FLOAT16" 0 <> 0
  | Dtype.Bfloat16 | Dtype.Float64 | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz ->
      false
  | _ -> true

let supports_clang_dtype arch dt =
  match Dtype.scalar dt with
  | Dtype.Bfloat16 -> (
      match arch with
      | Gpu_target.X86_64 | Gpu_target.Arm64 -> true
      | Gpu_target.Riscv64 -> false)
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 | Dtype.Fp8e4m3fnuz
  | Dtype.Fp8e5m2fnuz ->
      false
  | _ -> true

let clang_emulated_floats arch =
  if supports_clang_dtype arch Dtype.bfloat16 then []
  else [ (Dtype.Bfloat16, Dtype.Float32) ]

let supports_metal_dtype arch dt =
  match Dtype.scalar dt with
  | Dtype.Bfloat16 -> (
      match arch with
      | Gpu_target.Apple family -> family >= 6
      | Gpu_target.Mac _ -> false)
  | Dtype.Float64 | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 | Dtype.Fp8e4m3fnuz
  | Dtype.Fp8e5m2fnuz ->
      false
  | _ -> true

let supports_cuda_dtype arch dt =
  match Dtype.scalar dt with
  | Dtype.Bfloat16 -> (
      match arch with
      | Gpu_target.SM75 -> false
      | Gpu_target.SM80 | Gpu_target.SM89 | Gpu_target.SM90 -> true)
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> (
      match arch with
      | Gpu_target.SM89 | Gpu_target.SM90 -> true
      | Gpu_target.SM75 | Gpu_target.SM80 -> false)
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> false
  | _ -> true

let supports_amd_dtype arch dt =
  match Dtype.scalar dt with
  | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> (
      match arch with
      | Gpu_target.CDNA4 -> true
      | Gpu_target.RDNA3 | Gpu_target.RDNA4 | Gpu_target.CDNA3 -> false)
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> false
  | _ -> true

let clang_no_abi arch =
  Renderer.make ~name:"clang" ~device:"CPU" ~has_local:false
    ~has_threads:(getenv "THREADS" 1 <> 0) ~has_shared:false
    ~shared_max:0 ~global_max:[ host_cpu_count (); 0; 0 ]
    ~local_max:[ 0; 0; 0 ]
    ~code_for_op:code_ops_clang ~extra_matcher:clang_language.extra_matcher
    ~supports_dtype:(supports_clang_dtype arch)
    ~emulated_floats:(clang_emulated_floats arch)
    ~render:(render clang_language) ()

let clang arch =
  Renderer.make ~name:"clang" ~device:"CPU" ~has_local:false
    ~has_threads:(getenv "THREADS" 1 <> 0) ~has_shared:false
    ~shared_max:0 ~global_max:[ host_cpu_count (); 0; 0 ]
    ~local_max:[ 0; 0; 0 ]
    ~code_for_op:code_ops_clang
    ~extra_matcher:clang_fixed_abi_language.extra_matcher
    ~supports_dtype:(supports_clang_dtype arch)
    ~emulated_floats:(clang_emulated_floats arch)
    ~render:(render clang_fixed_abi_language) ()

let opencl arch =
  Renderer.make ~name:"opencl" ~device:"CL" ~has_local:true
    ~has_shared:true ~shared_max:32768
    ~code_for_op:code_ops_base
    ~extra_matcher:opencl_language.extra_matcher
    ~supports_dtype:(supports_opencl_dtype arch)
    ?image_pitch_alignment:
      (arch_int_value ~prefix:"IMAGE_PITCH_ALIGNMENT=" arch)
    ~aux:opencl_aux
    ~render:(render opencl_language) ()

let intel arch =
  Renderer.make ~name:"intel" ~device:"CL" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~code_for_op:code_ops_base
    ~extra_matcher:intel_language.extra_matcher
    ~supports_dtype:(supports_opencl_dtype arch)
    ?image_pitch_alignment:
      (arch_int_value ~prefix:"IMAGE_PITCH_ALIGNMENT=" arch)
    ~render:(render intel_language) ()

let qcom =
  Renderer.make ~name:"qcom" ~device:"QCOM" ~has_local:true
    ~has_shared:true ~shared_max:32768
    ~code_for_op:code_ops_base
    ~extra_matcher:opencl_language.extra_matcher
    ~supports_dtype:supports_qcom_dtype
    ~image_pitch_alignment:64
    ~render:(render opencl_language) ()

let metal arch =
  let tensor_cores =
    match arch with
    | Gpu_target.Apple family when family >= 7 -> Tc.metal
    | Gpu_target.Apple _ | Gpu_target.Mac _ -> []
  in
  Renderer.make ~name:"metal" ~device:"METAL" ~has_local:true
    ~has_shared:true ~shared_max:32768 ~tensor_cores
    ~code_for_op:code_ops_base
    ~extra_matcher:metal_language.extra_matcher
    ~supports_dtype:(supports_metal_dtype arch)
    ~render:(render metal_language) ()

let cuda arch =
  let tensor_cores =
    match arch with
    | Gpu_target.SM75 -> Tc.cuda_sm75
    | Gpu_target.SM80 -> Tc.cuda_sm80
    | Gpu_target.SM89 | Gpu_target.SM90 -> Tc.cuda_sm89
  in
  Renderer.make ~name:"cuda" ~device:"CUDA" ~has_local:true
    ~has_shared:true
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~local_max:[ 1024; 1024; 64 ] ~shared_max:49152 ~tensor_cores
    ~code_for_op:code_ops_base
    ~extra_matcher:cuda_language.extra_matcher
    ~supports_dtype:(supports_cuda_dtype arch)
    ~render:(render cuda_language) ()

let amd arch =
  let lang = amd_language arch in
  let tensor_cores =
    match arch with
    | Gpu_target.RDNA3 -> Tc.amd_rdna3
    | Gpu_target.RDNA4 -> Tc.amd_rdna4
    | Gpu_target.CDNA3 -> Tc.amd_cdna3
    | Gpu_target.CDNA4 -> Tc.amd_cdna4
  in
  Renderer.make ~name:"amd" ~device:"AMD" ~has_local:true ~has_shared:true
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~global_prod_max:[ 0xFFFFFFFF; 0xFFFFFFFF; 0xFFFFFFFF ]
    ~shared_max:65536 ~tensor_cores
    ~code_for_op:code_ops_base
    ~extra_matcher:lang.extra_matcher
    ~supports_dtype:(supports_amd_dtype arch)
    ~render:(render lang) ()
