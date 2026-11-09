(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

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
  inline_prefix : string; (* the qualifier of a call body's function *)
  noinline : string; (* the attribute of a body compiled out of line *)
  suffix64 : string; (* the suffix of a 64-bit integer literal, after [u] *)
  bitcast : string -> string -> string -> string;
      (* [bitcast t s x] reads [x], of type [s], as a [t] *)
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
  masks : int Tbl.t; (* the lane width of each mask, in bytes *)
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
  (* A data type is a constant constructor, so identity is equality. *)
  let name =
    match List.assq_opt dt l.type_map with
    | Some name -> name
    | None -> invalid_arg (strf "the language has no type %s" (Dtype.name dt))
  in
  if sz > 1 then
    let element =
      match List.assq_opt dt l.vector_names with
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

let param_type l p =
  let volatile = match arg p with Param p -> p.volatile | _ -> false in
  (if volatile then "volatile " else "")
  ^ render_dtype l (dtype p) ~addrspace:(addrspace p)
      ~override_ptr:(addrspace p <> Some Dtype.Alu)

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
      strf "WMMA_%d_%d_%d_%s_%s" n m k (Dtype.name dtype_in)
        (Dtype.name (dtype u))
  | _ -> invalid_arg "a WMMA without its tensor core argument"

(* Base rewrite *)

let rule = Pattern_matcher.rule
let cast_of ?dtype ?name p = Upat.op ?dtype ?name ~src:[ p ] Op.Cast
let c = Upat.cvar "c"

let special l v =
  if Float.is_nan v then l.nan
  else if Float.sign_bit v then "-" ^ l.infinity
  else l.infinity

(* A NaN constant is a bitcast of its bits, which a conversion of the language's
   NaN would lose: its sign and payload, a signalling NaN's quiet bit included.
   A float64 NaN with bits 0x7ff0000020000000 is
   [__builtin_bit_cast(double, (unsigned long)(0x7ff0000020000000ul))]. *)
let nan_const ctx x v =
  let dt = dtype x in
  let u =
    List.find (fun u -> Dtype.itemsize u = Dtype.itemsize dt) Dtype.uints
  in
  let bits = Dtype.Value.to_z (Dtype.bitcast dt u (`Float v)) in
  let suffix =
    if Dtype.equal u Dtype.Uint64 then "u" ^ ctx.lang.suffix64 else "u"
  in
  ctx.lang.bitcast (render_type ctx.lang x) (render_scalar ctx.lang u)
    (strf "0x%Lx%s" (Bigint.to_int64 (Bigint.signed_extract bits 0 64)) suffix)

(* An integer constant is the value its type holds, the integer part of a float
   wrapped to the type's width, written as a literal of the type: a suffix
   states a 64-bit or unsigned type, a cast a narrower one, and the least int64
   is a difference, its magnitude being no signed literal. An infinity or a NaN
   has no integer value, and converts as the program runs. *)
let int_const ctx x (v : Dtype.value) =
  let dt = dtype x in
  match v with
  | `Float f when not (Float.is_finite f) ->
      strf "(%s)" (render_cast ctx x (special ctx.lang f))
  | _ -> (
      let n = Dtype.truncate dt (`Int (Dtype.Value.to_z v)) in
      let s = Format.asprintf "%a" Dtype.pp_const n in
      match dt with
      | Int64 when Dtype.Value.(n = Dtype.min Int64) ->
          strf "(-9223372036854775807%s-1)" ctx.lang.suffix64
      | Int64 -> s ^ ctx.lang.suffix64
      | Uint64 -> s ^ "u" ^ ctx.lang.suffix64
      | Uint32 -> s ^ "u"
      | Uint8 | Uint16 -> strf "(%s)" (render_cast ctx x (s ^ "u"))
      | Int8 | Int16 -> strf "(%s)" (render_cast ctx x s)
      | _ -> s)


(* The function of a call body: the name its lowering gave it. *)
let body_name body =
  match arg body with
  | String s -> s
  | _ -> invalid_arg "a call body without its name"

let associative = Op.Set.of_list Op.[ Add; Mul; Xor; Or; And ]

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
          | `Float v when Float.is_nan v -> Some (nan_const ctx (m "x") v)
          | `Float v when not (Float.is_finite v) ->
              Some (strf "(%s)" (render_cast ctx (m "x") (special ctx.lang v)))
          | _ -> None);
      r (cast_of ~dtype:[ Dtype.Float32 ] c) (fun _ m ->
          Some (const_str (m "c") ^ "f"));
      r (cast_of ~dtype:Dtype.ints ~name:"x" c) (fun ctx m ->
          Some (int_const ctx (m "x") (cval (m "c"))));
      r (cast_of ~dtype:[ Dtype.Bool ] c) (fun _ m ->
          Some (if Dtype.Value.to_bool (cval (m "c")) then "1" else "0"));
      (* consts are rendered to larger type and casted *)
      r
        (cast_of
           ~dtype:(Dtype.fp8s @ [ Dtype.Bfloat16; Dtype.Float16 ])
           ~name:"x" c)
        (fun ctx m ->
          Some (strf "(%s)" (render_cast ctx (m "x") (const_str (m "c") ^ "f"))));
      (* default const render *)
      r (cast_of c) (fun _ m -> Some (const_str (m "c")));
      (* casting *)
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
            (ctx.lang.bitcast (render_type ctx.lang x) (render_type ctx.lang s)
               ctx.%{s}));
      (* a buffer passed by its address *)
      r (Upat.op ~name:"x" Op.Refparam) (fun ctx m ->
          let x = m "x" in
          Some (strf "((%s)(%s))" (param_type ctx.lang x) ctx.%{nth x 1}));
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
          (* C groups a chain from the left, so only the first operand drops
             its parentheses: [(a*b)*c] is [a*b*c], and [a*(b*c)], which a
             float rounds and overflows differently, keeps them. *)
          let operand i v =
            if
              i = 0
              && is (op x) v
              && Op.Set.mem (op x) associative
              && not (Tbl.mem ctx.narrowed v)
            then Helpers.strip_parens ctx.%{v}
            else ctx.%{v}
          in
          Option.map
            (fun f -> f (List.mapi operand (src x)) (dtype x))
            (List.assoc_opt (op x) ctx.lang.code_for_op));
      (* a division is written whether or not the target lists it, which decides
         only whether code generation makes reciprocals divisions *)
      r
        (Upat.op ~src:[ Upat.var "a"; Upat.var "b" ] Op.Fdiv)
        (fun ctx m -> Some (strf "(%s/%s)" ctx.%{m "a"} ctx.%{m "b"}));
      (* call an external function by symbol: the CUSTOM_FUNCTION body names
         the callee, the sources are the arguments and set the types; code in
         the target's language is its format string of the arguments *)
      r
        (Upat.op ~name:"x" ~allow_any_len:true
           ~src:[ Upat.op ~name:"f" Op.Custom_function ]
           Op.Call)
        (fun ctx m ->
          let x = m "x" and l = ctx.lang in
          let args = List.tl (src x) in
          let types = List.map (render_type l) args in
          let values =
            List.map2 (fun t y -> strf "(%s)(%s)" t ctx.%{y}) types args
          in
          match arg (m "f") with
          | Custom_function f when f.fmt ->
              Some (format f.name (List.map (fun y -> ctx.%{y}) args))
          | Custom_function f ->
              Some
                (strf "((%s(*)(%s))(%s))(%s)%s"
                   (render_scalar l (dtype x))
                   (String.concat ", " types)
                   f.name
                   (String.concat ", " values)
                   (if Dtype.equal (dtype x) Dtype.Void then ";" else ""))
          | _ -> None);
      (* call a body of the program: its parameters, by slot, are the call's
         arguments *)
      r
        (Upat.op ~name:"x" ~allow_any_len:true
           ~src:[ Upat.op ~name:"body" Op.Linear ]
           Op.Call)
        (fun ctx m ->
          let x = m "x" and body = m "body" in
          let params =
            List.filter_map
              (fun u ->
                match arg u with
                | Param p when is Op.Param u -> Some (p.slot, u)
                | _ -> None)
              (src body)
          in
          let by_slot (s0, _) (s1, _) = Int.compare s0 s1 in
          let arg (slot, _) = ctx.%{nth x (slot + 1)} in
          Some
            (strf "%s(%s);" (body_name body)
               (String.concat ", "
                  (List.map arg (List.sort by_slot params)))));
    ])

(* Non-native floats *)

let alu_but_where = Op.Set.diff Op.Set.alu (Op.Set.of_list [ Op.Where ])

(* [x] computed on its sources cast to float32 *)
let on_floats x =
  Ops.v
    ~src:(List.map (fun v -> cast v Dtype.Float32) (src x))
    ~arg:(arg x) (op x)

(* C promotes an unsigned char or short to int, whose overflow is undefined:
   65535 * 65535 overflows it. The operations that wrap compute in an unsigned
   int and narrow after. *)
let pm_narrow_uint =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v
           ~op:(Op.Set.of_list Op.[ Add; Sub; Mul; Neg; Shl; Mulacc ])
           ~dtype:Dtype.[ Uint8; Uint16 ] ~name:"x" ())
        (fun m ->
          let x = m "x" in
          let wide = List.map (fun s -> cast s Dtype.Uint32) (src x) in
          Some (cast (replace x ~src:wide) (dtype x)));
    ])

(* C computes an operation on a signed integer in [int], or in [long long] for
   a 64-bit one, and leaves its overflow undefined, as it does a left shift of
   a negative value: a compiler takes either for impossible, and folds
   [x + y < x] to [y < 0]. Such an operation computes in the unsigned type of
   that width, which wraps, and converts back. One whose operands' bounds keep
   its exact result in the type and a shifted value from being negative, as an
   index's do, stays as it is. *)
let pm_wrap_signed =
  let computed dt =
    if Dtype.itemsize dt = 8 then (Dtype.Int64, Dtype.Uint64)
    else (Dtype.Int32, Dtype.Uint32)
  in
  let defined x =
    let open Dtype.Value in
    let c = fst (computed (dtype x)) in
    let inside v = Dtype.min c <= v && v <= Dtype.max c in
    let corners f (lo0, hi0) (lo1, hi1) =
      [ f lo0 lo1; f lo0 hi1; f hi0 lo1; f hi0 hi1 ]
    in
    match (op x, List.map (operand_bounds x) (src x)) with
    | Op.Add, [ p; q ] -> List.for_all inside (corners ( + ) p q)
    | Op.Sub, [ p; q ] -> List.for_all inside (corners ( - ) p q)
    | Op.Mul, [ p; q ] -> List.for_all inside (corners ( * ) p q)
    | Op.Neg, [ (lo, hi) ] -> inside (-lo) && inside (-hi)
    | Op.Mulacc, [ p; q; (lo, hi) ] ->
        List.for_all
          (fun v -> inside v && inside (v + lo) && inside (v + hi))
          (corners ( * ) p q)
    | Op.Shl, [ (lo, hi); _ ] ->
        (* a count of another type keeps its own bounds *)
        let k0 = vmin (nth x 1) and k1 = vmax (nth x 1) in
        of_int 0 <= lo
        && of_int 0 <= k0
        && k1 < of_int (Dtype.bitsize c)
        && inside (hi * `Int (Bigint.shift_left Bigint.one (to_int k1)))
    | _ -> false
  in
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v
           ~op:(Op.Set.of_list Op.[ Add; Sub; Mul; Neg; Shl; Mulacc ])
           ~dtype:Dtype.sints ~name:"x" ())
        (fun m ->
          let x = m "x" in
          if defined x then None
          else
            let u = snd (computed (dtype x)) in
            let wide = List.map (fun s -> cast s u) (src x) in
            Some (cast (replace x ~src:wide) (dtype x)));
    ])

let unsigned_of (dt : Dtype.t) =
  match dt with
  | Int8 -> Dtype.Uint8
  | Int16 -> Uint16
  | Int32 -> Uint32
  | Int64 -> Uint64
  | dt -> dt

(* C leaves a division by zero undefined, and a signed division of the least
   value by -1; x86 traps on both, on a vector lane of any width too. A shift by
   a count at or past the width of its operand, or of its lane, is undefined as
   well. tolk's quotient by zero is 0 and its remainder the dividend; the least
   value's quotient by -1 wraps to it and its remainder is 0; such a shift is
   the exact shift wrapped: 0, or -1 for a negative value shifted right. A
   division whose divisor's bounds reach 0 divides by [where (d <> 0) d 1]
   and selects its value at 0 after. One whose operands' bounds reach the
   least value and -1 computes on magnitudes in the unsigned type and takes
   its sign after: -2^31 / -1 is 2^31 / 1, negated to -2^31. A shift whose
   count's bounds leave the width shifts by the count's low bits and selects
   the result past it. *)
let pm_defined_div_shift =
  let zero = Dtype.Value.of_int 0 in
  let reaches v (lo, hi) = Dtype.Value.(lo <= v && v <= hi) in
  (* The constants a guard selects are stacks of the operation's lanes: a
     quotient by a constant 0, or a shift by a constant count at or past the
     width, folds to its value there, which a store then reads whole, and a
     store, unlike an elementwise operation or an index, does not read a
     scalar expanded to its lanes as the scalar. *)
  let not_zero d = alu d Op.Cmpne [ vconst_like d (`Int Bigint.zero) ] in
  let nonzero d = where (not_zero d) d (vconst_like d (`Int Bigint.one)) in
  (* [nonzero d], which bounds cannot tell from 0 *)
  let is_nonzero d =
    match (op d, src d) with
    | Op.Where, [ c; t; f ] ->
        Op.equal (op c) Op.Cmpne
        && nth c 0 == t
        && reaches zero (vmin (nth c 1), vmax (nth c 1))
        && Dtype.Value.(vmin (nth c 1) = vmax (nth c 1))
        && not (reaches zero (vmin f, vmax f))
    | _ -> false
  in
  let division x =
    let a = nth x 0 and b = nth x 1 and dt = dtype x in
    let bounds = operand_bounds x b in
    let by_zero = reaches zero bounds && not (is_nonzero b) in
    let overflow =
      List.mem dt Dtype.sints
      && reaches (Dtype.min dt) (operand_bounds x a)
      && reaches (Dtype.Value.of_int (-1)) bounds
    in
    if not (by_zero || overflow) then None
    else
      let r =
        if not overflow then alu a (op x) [ nonzero b ]
        else
          let negative v = O.(v < int 0) and minus v = alu v Op.Neg [] in
          let magnitude v =
            let w = cast v (unsigned_of dt) in
            where (negative v) (minus w) w
          in
          let d = magnitude b in
          let r =
            alu (magnitude a) (op x) [ (if by_zero then nonzero d else d) ]
          in
          let sign =
            if Op.equal (op x) Op.Truncdiv then
              alu (negative a) Op.Cmpne [ negative b ]
            else negative a
          in
          cast (where sign (minus r) r) dt
      in
      if not by_zero then Some r
      else
        let at_zero =
          if Op.equal (op x) Op.Truncdiv then vconst_like x (`Int Bigint.zero)
          else a
        in
        Some (where (not_zero b) r at_zero)
  in
  let shift x =
    let a = nth x 0 and k = nth x 1 and dt = dtype x in
    let w = Dtype.bitsize dt in
    if Dtype.Value.(of_int 0 <= vmin k && vmax k < of_int w) then None
    else
      let last = O.int (w - 1) in
      let inside = O.(k < int w) and low n = O.(n land last) in
      if Op.equal (op x) Op.Shr && List.mem dt Dtype.sints then
        Some (replace x ~src:[ a; low (where inside k last) ])
      else
        Some
          (where inside
             (replace x ~src:[ a; low k ])
             (vconst_like x (`Int Bigint.zero)))
  in
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v
           ~op:(Op.Set.of_list Op.[ Truncdiv; Truncmod ])
           ~dtype:Dtype.ints ~name:"x" ())
        (fun m -> division (m "x"));
      rule
        (Upat.v ~op:(Op.Set.of_list Op.[ Shl; Shr ]) ~dtype:Dtype.ints
           ~name:"x" ())
        (fun m -> shift (m "x"));
    ])

let create_non_native_float_pats ?(casting = true) dts =
  let in_dts u = Dtype.mem (dtype u) dts in
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
            else
              (* One rounding: a source more precise than a float32 reaches
                 it rounded to odd. *)
              Some
                (cast (Decomp_dtype.narrow x Dtype.Float32) (dtype (m "y"))));
        rule (Upat.op ~src:[ y ] ~name:"x" Op.Cast) (fun m ->
            let x = m "x" in
            if Dtype.equal (dtype x) Dtype.Float32 then None
            else Some (cast (cast (m "y") Dtype.Float32) (dtype x)));
      ])

(* bfloat16 is float32's top 16 bits, so a narrowing rounds the low 16 away, to
   nearest with ties to even. A carry past the fraction moves the exponent, as
   it must. A NaN sets its quiet bit first, since the bits that made it a NaN
   may be among those dropped. *)
let cast_float_to_bf16 x =
  let bits = bitcast x Dtype.Uint32 in
  let rounded =
    O.(
      where (x <> x)
        (bits lor int 0x400000)
        (bits + ((bits lsr int 16) land int 1) + int 0x7fff))
  in
  bitcast (cast O.(rounded lsr int 16) Dtype.Uint16) Dtype.Bfloat16

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

(* A node of a kernel's values. *)
let shaped_value u =
  match addrspace u with
  | (Some Dtype.Alu | None)
    when ((not (Dtype.equal (dtype u) Dtype.Void))
         && Option.is_some (shape_opt u))
         [@mutate off "the value nodes of a kernel are exactly its shaped ones"]
    ->
      true
  | _ -> false

let uops_to_dtypes uops =
  dedup
    (List.filter_map
       (fun u -> if shaped_value u then Some (dtype u, max_numel u) else None)
       uops)

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

(* Masks *)

(* A value, as opposed to memory or an address of it. *)
let is_value u =
  match addrspace u with
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> false
  | _ -> true

let is_mask u =
  Dtype.equal (dtype u) Dtype.Bool && max_numel u > 1 && is_value u

let mask_dtype = function
  | 1 -> Dtype.Int8
  | 2 -> Dtype.Int16
  | 4 -> Dtype.Int32
  | 8 -> Dtype.Int64
  | w -> invalid_arg (strf "no mask has lanes of %d bytes" w)

(* C computes a vector of booleans as a mask, each lane -1 or 0 as wide as the
   values it decides: a comparison or a conversion of numbers gives lanes of
   their width, logic and selections of masks the widest of their operands', and
   a stack of booleans lanes of one byte. [mask_widths uops] is each mask's lane
   width in bytes. *)
let mask_widths uops =
  let widths = Tbl.create 16 in
  let widest l =
    List.fold_left max 1 (List.filter_map (Tbl.find_opt widths) l)
  in
  let of_numbers u =
    match List.find_opt (fun s -> not (Dtype.is_bool (dtype s))) (src u) with
    | Some s -> Dtype.itemsize (dtype s)
    | None -> widest (src u)
  in
  List.iter
    (fun u ->
      if is_mask u then
        let w =
          match op u with
          | Op.Cmplt | Op.Cmpne | Op.Cmpeq | Op.Cast -> of_numbers u
          | _ -> widest (src u)
        in
        Tbl.replace widths u w)
    uops;
  widths

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

(* A multiply-add, rounded once: the function [name dt] of the three
   operands. *)
let fma name args dt =
  match args with
  | [ a; b; c ] -> strf "%s(%s,%s,%s)" (name dt) a b c
  | _ -> invalid_arg "a multiply-add takes three operands"

(* [double] for a double, [single] for a float. *)
let by_width ~double ~single dt =
  if Dtype.equal dt Dtype.Float64 then double else single

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
      (Recip, unary (fun x _ -> strf "(1/%s)" x));
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
      (Truncmod, infix "%");
      (Truncdiv, infix "/");
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
      (Mulacc, fma (fun _ -> "fma"));
    ]

let cstyle =
  {
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
    type_map =
      Dtype.
        [
          (Void, "void");
          (Bool, "bool");
          (Int8, "signed char");
          (Uint8, "unsigned char");
          (Int16, "short");
          (Uint16, "unsigned short");
          (Int32, "int");
          (Uint32, "unsigned int");
          (Int64, "long");
          (Uint64, "unsigned long");
          (Float16, "half");
          (Bfloat16, "__bf16");
          (Float32, "float");
          (Float64, "double");
        ];
    infinity = "INFINITY";
    nan = "NAN";
    inline_prefix = "";
    noinline = "__attribute__((noinline))";
    suffix64 = "l";
    bitcast = strf "__builtin_bit_cast(%s, (%s)(%s))";
    promoted = Dtype.[ Int8; Int16 ];
    vector_names = [];
    code_for_op;
    string_rewrite = base_rewrite;
  }

let param_name u =
  match arg u with
  | Param p ->
      let base =
        match p.name with
        | Some n -> String.map (function ':' -> '_' | c -> c) n
        | None -> strf "data%d" p.slot
      in
      base ^ "_" ^ String.concat "_" (List.map string_of_int (max_shape u))
  | _ -> invalid_arg "a parameter's name needs its ParamArg"

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
  let buftypes = List.map (fun (n, u) -> (n, buftype u)) bufs in
  let params = List.map (fun (n, t) -> t ^ " " ^ n) buftypes @ l.extra_args in
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
let held = Op.Set.of_list [ Op.Buffer; Op.Binary ]

type shape = Plain | Opens | Closes | Held

(* One statement of a program's body: its text, how it nests, the name and type
   it declares, and the operations it computes, those written into it
   included. *)
type statement = {
  text : string;
  shape : shape;
  declares : (string * string) option;
  ops : int;
}

let texts = List.map (fun s -> s.text)
let undeclared = Op.Set.of_list [ Op.Range; Op.Buffer; Op.Binary ]
let always_inlined =
  Op.Set.of_list [ Op.Index; Op.Shrink; Op.Refparam ]
let casts = Op.Set.of_list [ Op.Cast; Op.Bitcast ]

(* Code in the target's language, called as an expression: written into its
   reader. *)
let code_call u =
  is Op.Call u
  && match arg (nth u 0) with Custom_function f -> f.fmt | _ -> false

let inline_code u =
  code_call u && match arg u with Call c -> c.inline | _ -> false

(* Clang and Metal reject a statement nested more than 256 brackets deep. A
   value read once is written into its reader, so a chain of them nests without
   bound: a value whose expression nests past [max_nesting] is declared
   instead. The rest of the 256 holds its reader's brackets, a narrowing cast
   and the braces of the loops around it. *)
let max_nesting = 64

(* The deepest nesting of brackets, ( [ and {, in [s]. *)
let nesting s =
  let depth = ref 0 and deepest = ref 0 in
  for i = 0 to String.length s - 1 do
    match String.unsafe_get s i with
    | '(' | '[' | '{' ->
        incr depth;
        if !depth > !deepest then deepest := !depth
    | ')' | ']' | '}' -> decr depth
    | _ -> ()
  done;
  !deepest

let special_name u =
  match arg u with
  | String s -> s
  | _ -> invalid_arg "a SPECIAL without its name"

let render_uops l uops =
  let r = Tbl.create 256 and child_count = Tbl.create 256 in
  let user = Tbl.create 256 in
  let masks = mask_widths uops in
  let ctx = { lang = l; r; narrowed = Tbl.create 16; masks } in
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
    && Dtype.mem (dtype u) l.promoted
    && max_numel u = 1
    &&
    match Tbl.find_opt user u with
    | Some s -> not (is Op.Store s && nth s 1 == u)
    | None -> true
  in
  let bufs = ref []
  and kernel = ref []
  and depth = ref 1
  and name = ref "test" in
  (* The operations an inlined value computes, written into its reader. *)
  let inlined_ops = Tbl.create 256 in
  let ops_of u =
    List.fold_left
      (fun a v -> a + Option.value (Tbl.find_opt inlined_ops v) ~default:0)
      1 (src u)
  in
  let shape_of o =
    if Op.Set.mem o opens then Opens
    else if Op.Set.mem o closes then Closes
    else if Op.Set.mem o held then Held
    else Plain
  in
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
  let inlined u line =
    let o = op u in
    let cast = Op.equal o Op.Cast and casts = Op.Set.mem o casts in
    ((not cast) || max_numel u = 1)
    && ((cast && is Op.Const (nth u 0))
       || Op.Set.mem o always_inlined
       || inline_code u
       || Op.equal o Op.Load
          && addrspace (nth u 0) = Some Dtype.Reg
          && children u = 1
       || (casts && is_ptr (addrspace u))
       || (casts || Op.equal o Op.Stack || Op.Set.mem o alu_but_where)
          && children u = 1
          && not (Setting.value Setting.expand_ssa)
          && nesting line <= max_nesting)
  in
  let visit u =
    match op u with
    | Op.Noop | Op.Const | Op.Custom_function -> ()
    | Op.Stack when src u = [] -> ()
    | Op.After when Dtype.equal (dtype u) Dtype.Void -> ()
    | Op.After -> Tbl.replace r u ctx.%{nth u 0}
    | Op.Sink -> (
        match arg u with Kernel k -> name := function_name k | _ -> ())
    | Op.Param ->
        let n = param_name u in
        Tbl.replace r u n;
        bufs := (n, u) :: !bufs
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
        if inlined u line then begin
          Tbl.replace inlined_ops u (ops_of u);
          if narrowed u then begin
            Tbl.replace ctx.narrowed u ();
            Tbl.replace r u (strf "(%s)" (render_cast ctx u line))
          end
          else Tbl.replace r u line
        end
        else begin
          let declared =
            (not (Op.Set.mem o undeclared))
            && not (Dtype.equal (dtype u) Dtype.Void)
          in
          let ty () =
            match Tbl.find_opt ctx.masks u with
            | Some w -> render_dtype l (mask_dtype w) ~sz:(max_numel u)
            | None -> render_type l u
          in
          let line =
            if declared then
              strf "%s %s = %s%s" (ty ()) ctx.%{u} line
                (if Op.equal o Op.Special then "" else ";")
            else if Op.equal o Op.Cast then
              (* a discarded value renders as a bare statement *)
              line ^ ";"
            else line
          in
          let indent s = String.make (2 * !depth) ' ' ^ s in
          let text =
            String.concat "\n"
              (List.map indent (String.split_on_char '\n' line))
          in
          let declares =
            if declared then Some (ctx.%{u}, ty ())
            else
              match o with
              | Op.Range when not (Dtype.equal (dtype u) Dtype.Void) ->
                  Some (ctx.%{u}, render_scalar l (dtype u))
              | Op.Buffer -> Some (ctx.%{u}, render_dtype l (dtype u) ^ "*")
              | Op.Binary -> Some (ctx.%{u}, "const unsigned char*")
              | _ -> None
          in
          kernel :=
            { text; shape = shape_of o; declares; ops = ops_of u }
            :: !kernel;
          (* if it was used, increment *)
          Option.iter (fun p -> Hashtbl.replace counts p (count p + 1)) prefix
        end;
        if Op.Set.mem o opens then incr depth
  in
  List.iter visit uops;
  (!name, List.rev !kernel, List.rev !bufs)

(* The external functions and the call bodies the instructions [uops] call, in
   the order a toposort of the program meets their calls, bodies included. *)
let callees uops =
  let callee u =
    if not (is Op.Call u) || code_call u then None
    else
      let c = nth u 0 in
      if is Op.Custom_function c || is Op.Linear c then Some c else None
  in
  List.filter_map callee (toposort ~calls:Enter (sink uops))
  |> List.fold_left (fun a c -> if List.memq c a then a else c :: a) []
  |> List.rev

(* The bodies that a call compiled on its own calls ({!Ops.call_info}): a
   compiler keeps them out of line, so that it compiles each once however
   many calls it has. *)
let outlined uops =
  let found = ref [] in
  let rec scan us =
    List.iter
      (fun u ->
        if is Op.Call u && is Op.Linear (nth u 0) then begin
          (match arg u with
          | Call { precompile = true; _ } ->
              if not (List.memq (nth u 0) !found) then
                found := nth u 0 :: !found
          | _ -> ());
          scan (src (nth u 0))
        end)
      us
  in
  scan uops;
  !found

(* The program declares each external function it calls, which it links
   against, and defines a function for each call body, after the bodies it
   calls. *)
(* The program's parts: a declaration of each function it calls and a
   definition of each body, the kernel's name, statements and parameters, and
   the instructions its kernel and bodies hold. *)
let program l uops =
  let callees = callees uops in
  let outlined = outlined uops in
  let declare callee =
    match arg callee with
    | Custom_function f -> strf "extern void %s();" f.name
    | _ ->
        let _, statements, bufs = render_uops l (src callee) in
        let params =
          List.map (fun (n, p) -> strf "%s %s" (param_type l p) n) bufs
        in
        let qualifiers =
          if List.memq callee outlined then
            strf "static %s %s" l.inline_prefix l.noinline
          else strf "static inline %s" l.inline_prefix
        in
        strf "%s void %s(%s) {\n%s\n}" qualifiers (body_name callee)
          (String.concat ", " params)
          (String.concat "\n" (texts statements))
  in
  let name, kernel, bufs = render_uops l uops in
  let bodies = List.filter (is Op.Linear) callees in
  ( List.map declare callees,
    name,
    kernel,
    bufs,
    List.concat_map src bodies @ uops )

let prefix = function [] -> None | p -> Some p

let render render_kernel l uops =
  let parts, name, kernel, bufs, uops = program l uops in
  render_kernel ?prefix:(prefix parts) l ~name (texts kernel) bufs uops

(* Long kernels

   A compiler's passes grow faster than the function they read: Clang takes
   308 s for a kernel of 80 000 statements, in EarlyCSE and in the register
   allocator as much as anywhere, and no flag changes that. A long host kernel
   is written as functions of at most [piece] operations, which its body calls
   in order, so each stays short. Operations are counted with those written
   into their reader, since a chain of values read once is one statement.
   Statements keep their order, so the kernel computes the same bits. A kernel
   that declares a name twice, as two loops one after the other may, stays
   whole: a value is known by its name.

   A value read outside the function that declares it lives in a frame on the
   kernel's stack, written where it is declared and read where a function first
   uses it, since a value read early would be live across the function. The
   frame holds one array per C type, and a slot is reused once the last reader
   of its value has run, so that it is as large as the values alive at one
   time. *)

(* The operations each function holds, and those past which a kernel is cut. A
   kernel of fewer takes Clang seconds. *)
let piece_ops = 2000
let long_ops = 20_000

type item = One of statement | Block of statement * item list * statement

type piece =
  | Chunk of int * item list
  | Nest of int * statement * piece list * int * statement

let rec weight = function
  | One s -> s.ops
  | Block (o, body, c) ->
      List.fold_left (fun a i -> a + weight i) (o.ops + c.ops) body

let tree statements =
  let rec items acc = function
    | [] -> (List.rev acc, [])
    | { shape = Closes; _ } :: _ as rest -> (List.rev acc, rest)
    | ({ shape = Opens; _ } as o) :: rest -> (
        match items [] rest with
        | body, c :: rest -> items (Block (o, body, c) :: acc) rest
        | _, [] -> invalid_arg "a block that does not close")
    | s :: rest -> items (One s :: acc) rest
  in
  fst (items [] statements)

(* [items] cut into pieces of at most [budget] operations, in program order. A
   block past the budget stays in the kernel's body, and its statements are
   cut. Each piece and each end of a block takes a number in program order,
   from 1: 0 is the kernel's start. *)
let pieces budget items =
  let event = ref 0 in
  let next () =
    incr event;
    !event
  in
  let rec cut items =
    let flush run acc =
      if run = [] then acc else Chunk (next (), List.rev run) :: acc
    in
    let rec go run size acc = function
      | [] -> List.rev (flush run acc)
      | (Block (o, body, c) as b) :: rest when weight b > budget ->
          let acc = flush run acc in
          let first = next () in
          let inner = cut body in
          let last = next () in
          go [] 0 (Nest (first, o, inner, last, c) :: acc) rest
      | item :: rest ->
          let w = weight item in
          if size > 0 && size + w > budget then
            go [ item ] w (flush run acc) rest
          else go (item :: run) (size + w) acc rest
    in
    go [] 0 [] items
  in
  cut items

(* The names of [known] that [text] mentions, each once. A name of a rendered
   statement is a C identifier, so a word of the text is one only if whole. *)
let reads known text =
  let n = String.length text and found = Hashtbl.create 8 in
  let word = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
    | _ -> false
  in
  let i = ref 0 in
  while !i < n do
    if word (String.unsafe_get text !i) then begin
      let j = ref !i in
      while !j < n && word (String.unsafe_get text !j) do
        incr j
      done;
      let w = String.sub text !i (!j - !i) in
      if Hashtbl.mem known w then Hashtbl.replace found w ();
      i := !j
    end
    else incr i
  done;
  List.of_seq (Hashtbl.to_seq_keys found)

let rec iter_item f = function
  | One s -> f s
  | Block (o, body, c) ->
      f o;
      List.iter (iter_item f) body;
      f c

(* Calls [chunk event items] and [ends event s] for each end of a block left in
   the kernel's body, in program order. *)
let rec iter_piece ~chunk ~ends = function
  | Chunk (e, items) -> chunk e items
  | Nest (first, o, inner, last, c) ->
      ends first o;
      List.iter (iter_piece ~chunk ~ends) inner;
      ends last c

(* Each name's type and the event that declares it, [bufs] and the held
   statements at 0, and the events that read it. Raises [Exit] if a name is
   declared twice. *)
let declared_and_read ptype bufs held cut =
  let known = Hashtbl.create 1024 and read_at = Hashtbl.create 1024 in
  let declare e s =
    Option.iter
      (fun (n, t) ->
        if Hashtbl.mem known n then raise_notrace Exit;
        Hashtbl.replace known n (t, e))
      s.declares
  in
  List.iter (fun (n, p) -> Hashtbl.replace known n (ptype p, 0)) bufs;
  List.iter (declare 0) held;
  let each f =
    List.iter
      (iter_piece ~chunk:(fun e -> List.iter (iter_item (f e))) ~ends:f)
      cut
  in
  let read e s =
    List.iter
      (fun n ->
        let at = Option.value (Hashtbl.find_opt read_at n) ~default:[] in
        if not (List.mem e at) then Hashtbl.replace read_at n (e :: at))
      (reads known s.text)
  in
  each declare;
  each read;
  (known, read_at)

(* The nests of [cut], as the events of their ends. *)
let rec nests_of acc = function
  | Chunk _ -> acc
  | Nest (first, _, inner, last, _) ->
      List.fold_left nests_of ((first, last) :: acc) inner

(* The event after which [n] is dead. A value read in a nest it was declared
   outside lives to the nest's end: its next trip reads it again. *)
let last_read nests declared at =
  let reader e =
    List.fold_left
      (fun acc (first, last) ->
        if first < e && e <= last && not (first <= declared && declared <= last)
        then max acc last
        else acc)
      e nests
  in
  List.fold_left (fun a e -> max a (reader e)) declared at

(* The frame's slot of each of [names], from a linear scan of their values by
   the event that declares them: within one type, a slot whose value was last
   read before an event is free at it. Returns each name's slot, and each
   type's number and count of slots. *)
let slots nests known read_at names =
  let values =
    List.map
      (fun n ->
        let t, d = Hashtbl.find known n in
        (d, last_read nests d (Hashtbl.find read_at n), n, t))
      names
    |> List.sort Stdlib.compare
  in
  let slot_of = Hashtbl.create 1024 and types = Hashtbl.create 8 in
  let type_of t =
    match Hashtbl.find_opt types t with
    | Some entry -> entry
    | None ->
        let entry = (Hashtbl.length types, ref 0, ref []) in
        Hashtbl.replace types t entry;
        entry
  in
  let place (d, ended, n, t) =
    let id, count, busy = type_of t in
    let free, held = List.partition (fun (e, _) -> e < d) !busy in
    let slot, rest =
      match List.sort Stdlib.compare free with
      | (_, s) :: rest -> (s, rest)
      | [] ->
          incr count;
          (!count - 1, [])
    in
    busy := (ended, slot) :: (rest @ held);
    Hashtbl.replace slot_of n (strf "s%d[%d]" id slot)
  in
  List.iter place values;
  (slot_of, types)

(* The functions of [cut] and the kernel's body that calls them. *)
let write_cut ~attrs ~name bufs held cut (known, read_at) =
  let in_piece = Hashtbl.create 64 in
  List.iter
    (iter_piece
       ~chunk:(fun e _ -> Hashtbl.replace in_piece e ())
       ~ends:(fun _ _ -> ()))
    cut;
  let crossing =
    Hashtbl.fold
      (fun n (_, d) acc ->
        match Hashtbl.find_opt read_at n with
        | Some at when List.exists (( <> ) d) at -> n :: acc
        | _ -> acc)
      known []
  in
  let nests = List.fold_left nests_of [] cut in
  let slot_of, types = slots nests known read_at crossing in
  let slot n = Hashtbl.find slot_of n and frame n = Hashtbl.mem slot_of n in
  let type_of n = fst (Hashtbl.find known n) in
  let frame_type = name ^ "_frame" and piece_name e = strf "%s_%d" name e in
  let frame_def =
    Hashtbl.fold (fun t (id, count, _) acc -> (id, t, !count) :: acc) types []
    |> List.sort Stdlib.compare
    |> List.map (fun (id, t, count) -> strf "  %s s%d[%d];" t id count)
    |> String.concat "\n"
    |> strf "struct %s {\n%s\n};" frame_type
  in
  (* A piece reads a value from the frame before the item that first uses it,
     and writes one that crosses where it is declared. *)
  let piece e items =
    let loaded = Hashtbl.create 16 and body = Buffer.create 4096 in
    let line text =
      Buffer.add_string body text;
      Buffer.add_char body '\n'
    in
    let load s =
      List.iter
        (fun n ->
          if snd (Hashtbl.find known n) <> e && not (Hashtbl.mem loaded n)
          then begin
            Hashtbl.replace loaded n ();
            line (strf "  %s %s = f->%s;" (type_of n) n (slot n))
          end)
        (reads known s.text)
    in
    let emit s =
      line s.text;
      match s.declares with
      | Some (n, _) when frame n -> line (strf "  f->%s = %s;" (slot n) n)
      | _ -> ()
    in
    List.iter
      (fun i ->
        iter_item load i;
        iter_item emit i)
      items;
    strf "static __attribute__((noinline)) %s %s(struct %s* f) {\n%s}" attrs
      (piece_name e) frame_type (Buffer.contents body)
  in
  (* The kernel's body: the frame, the held statements, the blocks too long for
     a piece, which read the values of pieces into variables it declares, and a
     call of each piece. *)
  let statement text = { text; shape = Plain; declares = None; ops = 1 } in
  let parts = ref [] and locals = Hashtbl.create 16 and body = ref [] in
  let add s = body := s :: !body in
  let store n =
    if frame n then add (statement (strf "  f.%s = %s;" (slot n) n))
  in
  let declared s = Option.iter (fun (n, _) -> store n) s.declares in
  let load s =
    List.iter
      (fun n ->
        if Hashtbl.mem in_piece (snd (Hashtbl.find known n)) then begin
          Hashtbl.replace locals n ();
          add (statement (strf "  %s = f.%s;" n (slot n)))
        end)
      (reads known s.text)
  in
  let rec write = function
    | Chunk (e, items) ->
        parts := piece e items :: !parts;
        add (statement (strf "  %s(&f);" (piece_name e)))
    | Nest (_, o, inner, _, c) ->
        load o;
        add o;
        declared o;
        List.iter write inner;
        load c;
        add c
  in
  add (statement (strf "  struct %s f;" frame_type));
  List.iter (fun (n, _) -> store n) bufs;
  List.iter
    (fun s ->
      add s;
      declared s)
    held;
  let start = !body in
  body := [];
  List.iter write cut;
  let declarations =
    Hashtbl.fold
      (fun n () acc -> statement (strf "  %s %s;" (type_of n) n) :: acc)
      locals []
  in
  (frame_def :: List.rev !parts, List.rev start @ declarations @ List.rev !body)

(* [statements] as several functions of at most [piece] operations past [long],
   or whole: the program's parts that define them, and the kernel's body. *)
let cut_kernel ~piece ~long ~param_type:ptype ~attrs ~name bufs statements =
  let held, rest = List.partition (fun s -> s.shape = Held) statements in
  if List.fold_left (fun a s -> a + s.ops) 0 rest <= long then ([], statements)
  else
    let cut = pieces piece (tree rest) in
    match declared_and_read ptype bufs held cut with
    | exception Exit -> ([], statements)
    | names -> write_cut ~attrs ~name bufs held cut names

(* Clang *)

(* Clang computes on ext_vector_type values. A scalar operand of an infix
   operation is splat to every lane; a selection, a builtin or a conversion
   takes vectors, and masks as wide as the values they decide. A C cast between
   vectors reinterprets their bits: conversions are
   [__builtin_convertvector]. *)

let vector u = max_numel u > 1 && is_value u
let vector_type ctx dt n = render_dtype ctx.lang dt ~sz:n

(* [x] as [n] lanes of [dt]: a scalar is splat. *)
let splatted ctx x dt n =
  if max_numel x > 1 then ctx.%{x}
  else strf "((%s)(%s))" (vector_type ctx dt n) ctx.%{x}

let bool_literal x =
  match (op x, src x) with
  | Op.Cast, [ c ] when is Op.Const c -> Some (Dtype.Value.to_bool (cval c))
  | Op.Const, _ -> Some (Dtype.Value.to_bool (cval x))
  | _ -> None

(* The boolean [x] as a mask of lanes of [w] bytes: a scalar is the lane every
   lane of the vector it meets copies. *)
let as_mask ctx x w n =
  if max_numel x = 1 then
    match bool_literal x with
    | Some b -> if b then "-1" else "0"
    | None ->
        strf "(-(%s)(%s))" (render_scalar ctx.lang (mask_dtype w)) ctx.%{x}
  else if Tbl.find ctx.masks x = w then ctx.%{x}
  else
    strf "__builtin_convertvector(%s, %s)" ctx.%{x}
      (vector_type ctx (mask_dtype w) n)

let vector_mask ctx x w n =
  if max_numel x > 1 then as_mask ctx x w n
  else strf "((%s)(%s))" (vector_type ctx (mask_dtype w) n) (as_mask ctx x w n)

(* C gives a comparison of one-byte lanes as plain chars, which a target may
   make unsigned: the mask is read as signed chars, so that it widens to -1. *)
let byte_mask ctx x s =
  if is_mask x && Tbl.find ctx.masks x = 1 then
    strf "((%s)(%s))" (vector_type ctx Dtype.Int8 (max_numel x)) s
  else s

let mask_logic op a b =
  match op with
  | Op.And | Op.Mul -> Some (strf "(%s&%s)" a b)
  | Op.Or | Op.Max | Op.Add -> Some (strf "(%s|%s)" a b)
  | Op.Xor | Op.Cmpne -> Some (strf "(%s^%s)" a b)
  | Op.Cmpeq -> Some (strf "(~(%s^%s))" a b)
  | Op.Cmplt -> Some (strf "((~%s)&%s)" a b)
  | _ -> None

(* The most lanes a vector operation computes at once: Clang's time on a
   vector it puts together from loaded rows grows much faster than its lanes
   past 64, and vectors of 64 run as fast as wider ones. *)
let clang_lanes = 64

let clang_vectors =
  let r = rule_ctx in
  let vec ?(op = Op.Set.elementwise) () = Upat.v ~op ~name:"x" () in
  let boolean ops =
    Upat.v ~op:(Op.Set.of_list ops) ~name:"x"
      ~src:[ Upat.var ~dtype:[ Dtype.Bool ] "a"; Upat.var "b" ]
      ()
  in
  Pattern_matcher.fold (fun () ->
      [
        (* a lane of a mask is a boolean *)
        r
          (Upat.op ~name:"x" ~src:[ Upat.var "buf"; Upat.var "idx" ] Op.Index)
          (fun ctx m ->
            if not (is_mask (m "buf")) then None
            else
              Some (strf "((_Bool)(%s))" (render_index ctx (m "buf") (m "idx"))));
        (* logic and comparisons of masks *)
        r
          (boolean Op.[ And; Or; Xor; Max; Add; Mul; Cmpne; Cmpeq; Cmplt ])
          (fun ctx m ->
            let x = m "x" in
            if not (vector (m "a") || vector (m "b")) then None
            else
              let n = max_numel x in
              let w =
                List.fold_left max 1
                  (List.filter_map (Tbl.find_opt ctx.masks) (src x))
              in
              let a = as_mask ctx (m "a") w n and b = as_mask ctx (m "b") w n in
              match mask_logic (op x) a b with
              | Some s -> Some s
              | None ->
                  invalid_arg
                    (strf "no rendering of %s of masks" (Op.name (op x))));
        (* a comparison of numbers *)
        r (Upat.v ~op:Op.Set.comparison ~name:"x" ()) (fun ctx m ->
            let x = m "x" in
            if
              (not (vector x))
              || Dtype.is_bool (dtype (nth x 0))
              || Tbl.find ctx.masks x <> 1
            then None
            else
              Option.map
                (fun f ->
                  byte_mask ctx x
                    (f (List.map (fun u -> ctx.%{u}) (src x)) (dtype x)))
                (List.assoc_opt (op x) ctx.lang.code_for_op));
        (* a selection reads its condition at its arms' width *)
        r
          (Upat.op ~name:"x"
             ~src:[ Upat.var "c"; Upat.var "a"; Upat.var "b" ]
             Op.Where)
          (fun ctx m ->
            let x = m "x" and c = m "c" and a = m "a" and b = m "b" in
            if not (vector x) then None
            else
              let n = max_numel x in
              let w =
                match Tbl.find_opt ctx.masks x with
                | Some w -> w
                | None -> Dtype.itemsize (dtype x)
              in
              let arm u =
                if is_mask x then vector_mask ctx u w n
                else splatted ctx u (dtype x) n
              in
              if max_numel c = 1 then
                Some (strf "(%s?%s:%s)" ctx.%{c} (arm a) (arm b))
              else
                let a =
                  if vector a || vector b then
                    if is_mask x then as_mask ctx a w n else ctx.%{a}
                  else arm a
                in
                let b = if is_mask x then as_mask ctx b w n else ctx.%{b} in
                Some (strf "(%s?%s:%s)" (as_mask ctx c w n) a b));
        (* A cast between vectors reinterprets their bits, as a bitcast does,
           and Clang compiles it as one instruction: a long chain of
           [__builtin_bit_cast]s, each a copy through memory, takes its
           scalar replacement of aggregates a time that grows faster than
           the chain. *)
        r (Upat.op ~name:"x" Op.Bitcast) (fun ctx m ->
            let x = m "x" in
            let s = nth x 0 in
            if not (vector x && vector s) then None
            else
              Some
                (strf "((%s)((%s)(%s)))" (render_type ctx.lang x)
                   (render_type ctx.lang s) ctx.%{s}));
        (* conversions *)
        r
          (vec ~op:(Op.Set.of_list [ Op.Cast ]) ())
          (fun ctx m ->
            let x = m "x" in
            let s = nth x 0 in
            if not (vector x) then None
            else if Dtype.is_bool (dtype x) then
              Some (byte_mask ctx x (strf "(%s!=0)" ctx.%{s}))
            else
              let s =
                if Dtype.is_bool (dtype s) then strf "(%s&1)" ctx.%{s}
                else ctx.%{s}
              in
              Some
                (strf "__builtin_convertvector(%s, %s)" s
                   (vector_type ctx (dtype x) (max_numel x))));
        (* builtins take vectors *)
        r
          (vec ~op:(Op.Set.of_list Op.[ Sqrt; Trunc; Mulacc ]) ())
          (fun ctx m ->
            let x = m "x" in
            if not (vector x) then None
            else
              let name =
                match op x with
                | Op.Sqrt -> "sqrt"
                | Op.Trunc -> "trunc"
                | _ -> "fma"
              in
              let args =
                List.map
                  (fun u -> splatted ctx u (dtype x) (max_numel x))
                  (src x)
              in
              Some
                (strf "__builtin_elementwise_%s(%s)" name
                   (String.concat ", " args)));
        (* a stack of booleans is a mask of one-byte lanes *)
        r (Upat.op ~name:"x" Op.Stack) (fun ctx m ->
            let x = m "x" in
            if not (is_mask x) then None
            else
              Some
                (strf "(-(%s){%s})"
                   (vector_type ctx Dtype.Int8 (max_numel x))
                   (String.concat "," (List.map (fun y -> ctx.%{y}) (src x)))));
      ])

let clang_lang =
  let builtin name x dt =
    if Dtype.equal dt Dtype.Float64 then strf "__builtin_%s(%s)" name x
    else strf "__builtin_%sf(%s)" name x
  in
  let dropped = Op.[ Exp2; Sin; Log2; Recip ] in
  {
    cstyle with
    float4 = (fun t -> "(" ^ t ^ ")");
    float4_style = ("{", "}");
    gep_arr_threshold = 0;
    infinity = "__builtin_inff()";
    nan = "__builtin_nanf(\"\")";
    barrier = "__atomic_thread_fence(__ATOMIC_SEQ_CST);";
    buffer_suffix = " restrict";
    type_map =
      (Dtype.Bool, "_Bool") :: (Dtype.Float16, "__fp16") :: cstyle.type_map;
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
            ( Mulacc,
              fma (by_width ~double:"__builtin_fma" ~single:"__builtin_fmaf") );
          ];
    string_rewrite = Pattern_matcher.append clang_vectors base_rewrite;
  }

(* Backends lower a select of a value and a float zero by their comparison to
   an instruction or a call that may return the zero of the value's sign
   where the select returns the constant's: [(v < 0.0f) ? v : 0.0f] is -0.0
   at v = -0.0. clang's AArch64 backend makes it fminnm or fmaxnm, in clang
   19, Homebrew clang 22.1.7 and Apple clang 17, and its RISC-V backend fmin
   or fmax, in clang 19; its x86 backend, for a half, calls fminf or fmaxf,
   whose zero at a tie is the C library's. A maximum against a zero, rendered
   as such a select, meets it too, and an add of +0.0 after it is then folded
   away. The lowering needs the select's operands to be the comparison's: a
   zero of the types [zeros] that a comparison reads is rendered as a value
   the backend cannot see through, an empty asm statement over a register of
   class [reg] holding it, a narrow float held in a float. A select of a zero
   by any other condition, [c ? v : 0.0f], stays one the backend makes a mask
   of. *)
let float_zero zeros u =
  is Op.Cast u
  && Dtype.mem (dtype u) zeros
  &&
  match src u with
  | [ c ] when is Op.Const c -> (
      match value c with `Float v -> v = 0. | _ -> false)
  | _ -> false

let opaque ctx ~reg u =
  let wide = Dtype.equal (dtype u) Dtype.Float64 in
  let v =
    strf "({%s z = %s; __asm__(\"\" : \"+%s\"(z)); z;})"
      (if wide then "double" else "float")
      ctx.%{u} reg
  in
  if wide || Dtype.equal (dtype u) Dtype.Float32 then v
  else strf "(%s)" (render_cast ctx u v)

let opaque_zeros ~reg zeros =
  Pattern_matcher.fold (fun () ->
      [
        rule_ctx
          (Upat.v ~op:Op.Set.comparison ~name:"x" ())
          (fun ctx m ->
            let x = m "x" in
            let hidden = float_zero zeros in
            let operand u = if hidden u then opaque ctx ~reg u else ctx.%{u} in
            if not (List.exists hidden (src x)) then None
            else
              Option.map
                (fun f -> f (List.map operand (src x)) (dtype x))
                (List.assoc_opt (op x) ctx.lang.code_for_op));
      ])

(* Clang holds a [__fp16] as a storage format: C computes a selection between
   two of them, [c ? a : b], in float, and narrowing the result back quiets a
   signalling NaN, as x86's vcvtph2ps does, wherever LLVM's folds leave that
   round trip. A backend without bfloat16 registers may move a [__bf16]
   through float32 too: riscv64's, in clang 19, narrows a selected one with
   [__truncsfbf2]. A selection moves its value: a scalar one of the 16-bit
   floats [dts], and the merge of a gated load with its alternative, choose
   between the values' bits. Vectors are not promoted. *)

(* [bit_select l dt c a b] is [c ? a : b], of the 16-bit float [dt], chosen
   between the arms' bits. *)
let bit_select l dt c a b =
  let t = render_scalar l dt and u = render_scalar l Dtype.Uint16 in
  let bits v = l.bitcast u t v in
  l.bitcast t u (strf "(%s?%s:%s)" c (bits a) (bits b))

let bit_where dts l =
  let where = List.assoc Op.Where l.code_for_op in
  ( Op.Where,
    fun args dt ->
      match args with
      | [ c; a; b ] when Dtype.mem dt dts -> bit_select l dt c a b
      | _ -> where args dt )

let bit_loads dts =
  Pattern_matcher.fold (fun () ->
      [
        rule_ctx
          (Upat.op ~dtype:dts ~name:"x"
             ~src:[ Upat.var "bidx"; Upat.var "var"; Upat.var "gate" ]
             Op.Load)
          (fun ctx m ->
            let x = m "x" in
            if max_numel x > 1 then None
            else
              Some
                (bit_select ctx.lang (dtype x) ctx.%{m "gate"}
                   (render_access ctx (m "bidx"))
                   ctx.%{m "var"}));
      ])

(* LLVM legalizes a double to half cast on CPUs without native support (such as
   x86 without AVX512-FP16) into a compiler-rt libcall, so the cast goes through
   a float32, which the double reaches rounded to odd: the half is then the
   double rounded once. *)
let clang_extra_matcher =
  Pattern_matcher.concat
    [
      pm_narrow_uint;
      pm_wrap_signed;
      pm_defined_div_shift;
      Pattern_matcher.v
        (fun () -> [
          rule
            (Upat.cast (Upat.var ~dtype:[ Dtype.Float64 ] "x") Dtype.Float16)
            (fun m ->
              Some
                (cast (Decomp_dtype.narrow (m "x") Dtype.Float32) Dtype.Float16));
        ]);
      create_non_native_float_pats [ Dtype.Bfloat16 ];
      pm_manual_bf16_cast;
    ]

(* A vector on the CPU device is aligned to its element, so that it may be
   loaded from any element of a buffer; elsewhere to its size rounded down to a
   power of two, as Clang aligns it by default. *)
let clang_vector_prefix ~cpu l (dt, count) =
  let rec pow2_floor n p = if 2 * p > n then p else pow2_floor n (2 * p) in
  let alignment =
    if cpu then Dtype.itemsize dt
    else if Dtype.is_bool dt then 1
    else pow2_floor (Dtype.itemsize dt * count) 1
  in
  let vec = render_dtype l dt ~sz:count ~addrspace:(Some Dtype.Reg) in
  strf "typedef %s %s __attribute__((aligned(%d),ext_vector_type(%d)));"
    (render_scalar l dt) vec alignment count

let clang_kernel ~cpu ?prefix l ~name kernel bufs uops =
  let masks = mask_widths uops in
  (* A selection on vectors reads its condition as a mask of its arms' width. *)
  let types u =
    if not (shaped_value u) then []
    else
      let n = max_numel u in
      match Tbl.find_opt masks u with
      | Some w -> [ (mask_dtype w, n) ]
      | None when is Op.Where u && max_numel (nth u 0) > 1 ->
          [ (dtype u, n); (mask_dtype (Dtype.itemsize (dtype u)), n) ]
      | None -> [ (dtype u, n) ]
  in
  let vectors =
    List.filter
      (fun (_, count) -> count > 1)
      (dedup (List.concat_map types uops))
  in
  let defines =
    String.concat "\n" (List.map (clang_vector_prefix ~cpu l) vectors)
  in
  defines ^ "\n" ^ render_kernel ?prefix l ~name kernel bufs uops ^ "\n"

(* A program for the host, its kernel cut into functions of [piece] operations
   past [long]. *)
let render_host ~piece ~long l uops =
  let parts, name, kernel, bufs, uops = program l uops in
  let pieces, kernel =
    cut_kernel ~piece ~long ~param_type:(param_type l)
      ~attrs:(l.kernel_typedef 0) ~name bufs kernel
  in
  clang_kernel ~cpu:true ?prefix:(prefix (parts @ pieces)) l ~name
    (texts kernel) bufs uops

let clang_cut ~piece ~long (target : Helpers.Target.t) =
  let arch = target.arch in
  let native dt =
    ((not (Dtype.equal dt Dtype.Bfloat16))
    || String.starts_with ~prefix:"x86" arch
    || String.starts_with ~prefix:"arm" arch)
    && not (Dtype.mem dt Dtype.fp8s)
  in
  (* LLVM's x86 backend has no bfloat16 instructions short of AVX512-BF16, and
     legalises a bfloat16 value it only moves (a selection, a merge of a gated
     load with its alternative, a bitcast of stacked lanes) through float32:
     it aborts on a selection between vectors and calls compiler-rt's
     [__truncsfbf2], which no library of the process defines. Every bfloat16
     operation computes in float32 here, so on x86 a bfloat16 is held as its
     bits, an [unsigned short], as OpenCL holds it.

     A processor with AVX-512 that prefers 256-bit vectors, as Intel's do,
     makes 512-bit types illegal, so LLVM splits the kernel's wider vectors.
     LLVM 18 crashes splitting a negation, an absolute value or a copysign of
     more than 16 halves. A kernel is marked as one that uses 512-bit vectors,
     [min_vector_width(512)], which makes those types legal where the
     processor has them; code that uses none is unchanged.

     A float zero hidden from the backend is held in a register of the
     target's float class: [w] on arm64, [x] on x86 and [f] on riscv64. x86 and
     riscv64 compare a bfloat16 in float32. A selection of a half, and of a
     bfloat16 where it is a [__bf16], chooses between bits. *)
  let arm64 =
    String.starts_with ~prefix:"arm64" arch
    || String.starts_with ~prefix:"aarch64" arch
  in
  (* A half multiply-add, scalar or of lanes, is LLVM's fma of halves, IEEE
     754's fusedMultiplyAdd, which rounds once. With fp16, arm64's half
     arithmetic (FEAT_FP16), halves are legal and it is one fmadd or fmla;
     without, LLVM computes it in a wider type, and code generation rewrites
     it instead. *)
  let fp16 = arm64 && Compiler_cpu.has_feature arch "fp16" in
  let once dt =
    Dtype.mem dt Dtype.[ Float32; Float64 ]
    || (Dtype.equal dt Dtype.Float16 && fp16)
  in
  let lang =
    if arm64 then
      {
        clang_lang with
        string_rewrite =
          Pattern_matcher.append
            (opaque_zeros ~reg:"w"
               Dtype.[ Float16; Bfloat16; Float32; Float64 ])
            clang_lang.string_rewrite;
      }
    else if String.starts_with ~prefix:"x86" arch then
      {
        clang_lang with
        kernel_typedef =
          (fun _ -> "__attribute__((min_vector_width(512))) void");
        type_map = (Dtype.Bfloat16, "unsigned short") :: clang_lang.type_map;
        string_rewrite =
          Pattern_matcher.concat
            [
              opaque_zeros ~reg:"x" [ Dtype.Float16 ];
              pm_bf16_ushort_const;
              clang_lang.string_rewrite;
            ];
      }
    else
      {
        clang_lang with
        string_rewrite =
          Pattern_matcher.append
            (opaque_zeros ~reg:"f" Dtype.[ Float16; Float32; Float64 ])
            clang_lang.string_rewrite;
      }
  in
  (* On x86_64 Windows the object calls by Windows' convention, where a [long]
     has 32 bits ({!Compiler_cpu.clang}): a 64-bit integer is a [long long],
     and the kernel is called by System V's, as rig calls. *)
  let lang =
    if not (Compiler_cpu.win64 arch) then lang
    else
      {
        lang with
        kernel_typedef =
          (fun n -> "__attribute__((sysv_abi)) " ^ lang.kernel_typedef n);
        type_map =
          Dtype.(Int64, "long long")
          :: (Dtype.Uint64, "unsigned long long")
          :: lang.type_map;
        suffix64 = "ll";
      }
  in
  let lang =
    if not fp16 then lang
    else
      let name dt =
        if Dtype.equal dt Dtype.Float16 then "__builtin_fmaf16"
        else by_width ~double:"__builtin_fma" ~single:"__builtin_fmaf" dt
      in
      {
        lang with
        code_for_op = override lang.code_for_op [ (Op.Mulacc, fma name) ];
      }
  in
  let held =
    if String.starts_with ~prefix:"x86" arch then [ Dtype.Float16 ]
    else Dtype.[ Float16; Bfloat16 ]
  in
  let lang =
    {
      lang with
      code_for_op = override lang.code_for_op [ bit_where held lang ];
      string_rewrite =
        Pattern_matcher.append (bit_loads held) lang.string_rewrite;
    }
  in
  Renderer.v ~name:"ClangRenderer"
    ~registers:(fun () -> Compiler_cpu.registers arch)
    ~vector_lanes:clang_lanes ~has_local:false ~global_max:[ 1; 0; 0 ]
    ~extra_matcher:clang_extra_matcher ~packed_args:true
    ~code_for_op:lang.code_for_op ~native ~fma:once
    ~render:
      (if target.device = "CPU" then render_host ~piece ~long lang
       else render (clang_kernel ~cpu:false) lang)
    ~compiler:(Compiler_cpu.clang arch) target

let clang = clang_cut ~piece:piece_ops ~long:long_ops
let clang_in_pieces ~piece = clang_cut ~piece ~long:0

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
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    float4 = Fun.id;
    code_for_workitem = [ ('g', axis "gid"); ('l', axis "lid") ];
    extra_args =
      [
        "uint3 gid [[threadgroup_position_in_grid]]";
        "uint3 lid [[thread_position_in_threadgroup]]";
      ];
    type_map =
      (Dtype.Uint32, "uint") :: (Dtype.Bfloat16, "bfloat") :: cstyle.type_map;
    bitcast = strf "as_type<%s>((%s)(%s))";
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
                     (l.bitcast
                        (render_scalar l (dtype x))
                        (render_scalar l (dtype s))
                        ctx.%{s}));
           ]))
        base_rewrite;
  }

(* upcast to float32 the operations that have no bfloat16: Metal computes them
   in float, and a float does not convert to a bfloat implicitly *)
let metal_extra_matcher =
  Pattern_matcher.concat
    [
      pm_narrow_uint;
      pm_wrap_signed;
      pm_defined_div_shift;
      Pattern_matcher.v (fun () -> [
          rule
            (Upat.v
               ~op:(Op.Set.of_list Op.[ Sqrt; Exp2; Log2; Sin; Trunc ])
               ~dtype:[ Dtype.Bfloat16 ] ~name:"x" ())
            (fun m -> Some (cast (on_floats (m "x")) Dtype.Bfloat16));
        ]);
      pm_manual_bf16_cast;
    ]

(* Apple7's compiler service fails on a product of bfloat16 matrices with an
   operand it knows, such as literal zeros. A product into float32 multiplies
   them as float matrices: bfloat16 has float32's exponents, so the product
   of two bfloat16 values is exact in float32 while it stays in float32's
   normal range, and the sums are float32's, as before. A product into bfloat16
   stays one of bfloat16 matrices, whose sums widening would round
   elsewhere; its helper is never inlined, so the compiler does not see the
   operand. *)
let metal_wmma l (name, _, dtype_in, dtype_out, _) =
  let vec dt = render_dtype l dt ~sz:2 ~addrspace:(Some Dtype.Reg) in
  let out = vec dtype_out and in_ = vec dtype_in in
  let widened =
    Dtype.equal dtype_in Dtype.Bfloat16 && Dtype.equal dtype_out Dtype.Float32
  in
  let matrix = if widened then dtype_out else dtype_in in
  let opaque =
    if Dtype.equal dtype_in Dtype.Bfloat16 && not widened then
      "__attribute__((noinline)) "
    else ""
  in
  let read v i =
    if widened then strf "%s(%s[%d])" (render_scalar l matrix) v i
    else strf "%s[%d]" v i
  in
  let elements i =
    strf
      "  mat_a.thread_elements()[%d] = %s; mat_b.thread_elements()[%d] = %s; \
       mat_c.thread_elements()[%d] = c[%d];"
      i (read "a" i) i (read "b" i) i i
  in
  String.concat "\n"
    [
      strf "%s%s __%s(%s a, %s b, %s c){" opaque out name in_ in_ out;
      strf "  simdgroup_%s8x8 mat_a, mat_b; simdgroup_%s8x8 mat_c;"
        (render_scalar l matrix)
        (render_scalar l dtype_out);
      elements 0;
      elements 1;
      "  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);";
      strf
        "  return %s(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);"
        out;
      "}";
    ]

let metal_kernel ?(prefix = []) l ~name kernel bufs uops =
  let defines =
    [ "#include <metal_stdlib>"; "using namespace metal;" ]
    @ List.map (metal_wmma l) (wmma_args uops)
  in
  (* Its own definitions first, then the functions it calls. *)
  render_kernel l ~prefix:(defines @ prefix) ~name kernel bufs uops

let metal_with compiler (target : Helpers.Target.t) =
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
    && not (Dtype.mem dt (Dtype.Float64 :: Dtype.fp8s))
  in
  (* fma of halves is "Correctly rounded" from Apple GPU family 4, fast math
     off (MSL 4.1, Table 8.3; Compiler_metal); bfloat's table has no fma. *)
  let fma dt =
    Dtype.equal dt Dtype.Float32 || (Dtype.equal dt Dtype.Float16 && from 4)
  in
  (* Fitted on an M1 Max (Apple7, 32 GPU cores): a float32 product of 1024
     squared ran 1.18x faster keeping 32768 threads after its core's splits
     than 16384, and a half vector times 16384 by 8192 1.74x faster
     streaming with 16384 threads than 32768. *)
  Renderer.v ~name:"MetalRenderer" ~launch_locals:false ~core_threads:32768
    ~stream_threads:16384
    ~tensor_cores:(if from 7 then Tc.metal else [])
    ~extra_matcher:metal_extra_matcher ~packed_args:true
    ~code_for_op:metal_lang.code_for_op
    ~native ~fma
    ~render:(render metal_kernel metal_lang)
    ~compiler:(compiler ()) target

let metal target = metal_with Compiler_metal.compiler target

(* CUDA *)

let nms =
  List.map (String.make 1) (List.of_seq (String.to_seq "xyzwabcdefghijkl"))
  @ List.init 16 (fun i -> strf "v%d" (i + 16))

let rec take n = function x :: l when n > 0 -> x :: take (n - 1) l | _ -> []

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
    inline_prefix = "__device__";
    noinline = "__noinline__";
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
            ( Recip,
              unary (fun x dt ->
                  if half dt then strf "hrcp(%s)" x else strf "(1/%s)" x) );
            ( Mulacc,
              fma (fun dt ->
                  if half dt then "__hfma"
                  else by_width ~double:"__fma_rn" ~single:"__fmaf_rn" dt) );
          ];
    type_map =
      (Dtype.Uint32, "uint")
      :: (Dtype.Bfloat16, "nv_bfloat16")
      :: (Dtype.Fp8e4m3, "__nv_fp8_e4m3")
      :: (Dtype.Fp8e5m2, "__nv_fp8_e5m2")
      :: cstyle.type_map;
    bitcast = strf "tg_bitcast<%s>((%s)(%s))";
    string_rewrite = base_rewrite;
  }

(* Asynchronous copies

   A store into shared memory of a 16-byte load from global memory is PTX's
   [cp.async], which copies without the thread waiting for it; a gated load
   ignores its source where its gate fails and copies zeros (PTX ISA
   §9.7.10.28.3.1, [ignore-src]). A thread waits only for its own copies, so
   every barrier of a kernel holding one first waits for all of them
   ([cp.async.wait_all]), then for the workgroup (CUDA C++ Programming Guide
   §4.12.1: "if you use LDGSTS to prefetch some data that will be shared with
   other threads, a __syncthreads() is necessary after synchronizing with the
   LDGSTS completion mechanism"). *)

let zeros u =
  let zero c =
    (is Op.Const c || (is Op.Cast c && is Op.Const (nth c 0)))
    &&
    match value c with
    | `Int z -> Bigint.(equal z zero)
    | `Float f -> f = 0. && not (Float.sign_bit f)
    | _ -> false
  in
  zero u || (is Op.Stack u && List.for_all zero (src u))

let copies_async u =
  match (op u, src u) with
  | Op.Store, [ bidx; var ] when addrspace bidx = Some Dtype.Local -> (
      let from = if is Op.Load var then Some (nth var 0) else None in
      match (from, src var) with
      | Some from, ([ _ ] | [ _; _; _ ]) ->
          addrspace from = Some Dtype.Global
          && max_numel from = max_numel bidx
          && max_numel bidx * Dtype.itemsize (dtype bidx) = 16
          && (List.length (src var) = 1 || zeros (nth var 1))
      | _ -> false)
  | _ -> false

(* [.cg] caches the copy in L2 alone, the only level of a 16-byte copy that
   skips L1. *)
let cp_async =
  "__device__ __forceinline__ void tg_cp_async(void *d, const void *s, int \
   ignore) { asm volatile(\"{ .reg .pred p; setp.ne.b32 p, %2, 0; \
   cp.async.cg.shared.global [%0], [%1], 16, p; }\" :: \
   \"r\"((unsigned)__cvta_generic_to_shared(d)), \"l\"(s), \"r\"(ignore) : \
   \"memory\"); }"

let async_rewrite =
  Pattern_matcher.fold
    (fun () -> [
      rule_ctx
        (Upat.op ~name:"x" ~src:[ Upat.var "bidx"; Upat.var "var" ] Op.Store)
        (fun ctx m ->
          let x = m "x" and var = m "var" in
          if not (copies_async x) then None
          else
            let ignore =
              match src var with
              | [ _; _; gate ] -> strf "!(%s)" ctx.%{gate}
              | _ -> "0"
            in
            Some
              (strf "tg_cp_async(%s, %s, %s);" (render_ptr ctx (m "bidx"))
                 (render_ptr ctx (nth var 0)) ignore));
    ])

let cuda_async_lang =
  {
    cuda_lang with
    barrier =
      "asm volatile(\"cp.async.wait_all;\" ::: \"memory\"); __syncthreads();";
    string_rewrite =
      Pattern_matcher.append async_rewrite cuda_lang.string_rewrite;
  }

let cuda_extra_matcher =
  Pattern_matcher.concat
    [
      pm_narrow_uint;
      pm_wrap_signed;
      pm_defined_div_shift;
      create_non_native_float_pats ~casting:false Dtype.fp8s;
      Pattern_matcher.v (fun () -> [
          rule
            (Upat.op ~dtype:Dtype.fp8s
               ~src:[ Upat.var ~dtype:Dtype.fp8s "x" ]
               ~name:"y" Op.Cast)
            (fun m ->
              let x = m "x" and y = m "y" in
              if Dtype.equal (dtype x) (dtype y) then None
              else Some (cast (cast x Dtype.Float32) (dtype y)));
        ]);
    ]

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

let cuda_kernel ?(copies = false) ?(prefix = []) l ~name
    kernel bufs uops =
  let used = uops_to_dtypes uops in
  let uses p = List.exists (fun (dt, _) -> p dt) used in
  let is_fp8 dt = Dtype.mem dt Dtype.fp8s in
  (* CUDA has no vector of unsigned bytes or shorts under the names written
     here ([unsigned_short4]); a GPU that emulates a 16- or 8-bit float moves
     its lanes as such vectors of its storage. *)
  let vector (dt, count) =
    (List.mem count [ 4; 8 ] && Dtype.mem dt [ Dtype.Float16; Dtype.Bfloat16 ])
    || (List.mem count [ 2; 4; 8; 16 ] && is_fp8 dt)
    || List.mem count [ 2; 4; 8; 16 ]
       && Dtype.mem dt [ Dtype.Uint8; Dtype.Uint16 ]
  in
  let defines =
    [
      "typedef unsigned int uint;";
      "#define INFINITY (__int_as_float(0x7f800000))";
      "#define NAN (__int_as_float(0x7fffffff))";
      "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F \
       v) { union U { F f; T t; }; U u; u.f = v; return u.t; }";
    ]
    @ (if uses is_fp8 then [ "#include <cuda_fp8.h>" ] else [])
    @ (if uses (Dtype.equal Dtype.Float16) then [ "#include <cuda_fp16.h>" ]
       else [])
    @ (if uses (Dtype.equal Dtype.Bfloat16) then [ "#include <cuda_bf16.h>" ]
       else [])
    (* cuda_bf16.h defines the 2-lane vector [nv_bfloat162] and names its
       constructor [make_bfloat162]. *)
    @ (if List.exists (fun (dt, n) -> Dtype.equal dt Dtype.Bfloat16 && n = 2) used
       then
         [ "#define make_nv_bfloat162 make_bfloat162" ]
       else [])
    @ (if copies then [ cp_async ] else [])
    @ List.map (cuda_vector_prefix l) (List.filter vector used)
    @ List.map (cuda_wmma l) (wmma_args uops)
  in
  (* Its own definitions first, then the functions it calls. *)
  render_kernel l ~prefix:(defines @ prefix) ~name kernel bufs uops

let cuda_renderer ~sync compiler (target : Helpers.Target.t) =
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
    && ((not (Dtype.mem dt Dtype.fp8_ocp)) || ver >= 89)
    && not (Dtype.mem dt Dtype.fp8_fnuz)
  in
  (* __hfma "Performs half fused multiply-add in round-to-nearest-even mode
     ... rounding the result once", and likewise of nv_bfloat16 (cuda_fp16.h,
     cuda_bf16.h, CUDA 13.4), where the capability has the type. *)
  let fma dt =
    Dtype.mem dt Dtype.[ Float32; Float64 ]
    || (Dtype.equal dt Dtype.Float16 && ver >= 53)
    || (Dtype.equal dt Dtype.Bfloat16 && ver >= 80)
  in
  let compiler = compiler () in
  (* cp.async "Requires sm_80 or higher" (PTX ISA §9.7.10.28.3.1). A kernel
     that copies asynchronously renders its barriers to wait for the copies. *)
  let async_copy = ver >= 80 && not sync in
  let render uops =
    (* The program's choices read the bodies it calls too, whose loads and
       copies it renders. *)
    let all =
      List.concat_map
        (fun b -> if is Op.Linear b then src b else [])
        (callees uops)
      @ uops
    in
    let copies = async_copy && List.exists copies_async all in
    let l = if copies then cuda_async_lang else cuda_lang in
    render (cuda_kernel ~copies) l uops
  in
  let name = if sync then "CUDARenderer without cp.async" else "CUDARenderer" in
  (* NVRTC's time a call site of a launch falls to its least at 64 members
     (0.064 s, 1.0 ms a member, on fudge with CUDA 13) and grows past them:
     96 members 1.4 ms a member, 128 1.9 ms, 256 4.5 ms. *)
  Renderer.v ~name ~half8:true ~launch_members:64 ~launch_args:512
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~local_max:[ 1024; 1024; 64 ] ~shared_max:49152 ~async_copy ~tensor_cores
    ~extra_matcher:cuda_extra_matcher ~code_for_op:cuda_lang.code_for_op ~native
    ~fma ~render ~compiler target

(* A cubin on every device: the disk cache keeps the GPU's code, which the
   driver loads as it is, where it would translate PTX at each load and keep its
   work only in a cache of its own. NVRTC is given the device's architecture
   either way, and refuses one it does not know in either form. *)
let nvrtc (target : Helpers.Target.t) () =
  Compiler_cuda.nvrtc ~ptx:false
    ~cache_key:(String.lowercase_ascii target.device)
    target.arch

let cuda target = cuda_renderer ~sync:false (nvrtc target) target

(* HIP *)

let fp8_index dt =
  match dt with
  | Dtype.Fp8e4m3 | Dtype.Fp8e4m3fnuz -> 0
  | Dtype.Fp8e5m2 | Dtype.Fp8e5m2fnuz -> 1
  | dt -> invalid_arg (Format.asprintf "%a is not an 8-bit float" Dtype.pp dt)

let amd_fp8s = function
  | "gfx942" -> Dtype.fp8_fnuz
  | "gfx950" | "gfx1200" | "gfx1201" -> Dtype.fp8_ocp
  | _ -> []

let ocml op =
  unary (fun x dt ->
      let bits =
        match dt with Dtype.Float16 -> 16 | Dtype.Float64 -> 64 | _ -> 32
      in
      strf "__ocml_%s_f%d(%s)" op bits x)

let gpu arch = List.hd (String.split_on_char ':' arch)
let is_cdna arch = List.mem (gpu arch) [ "gfx942"; "gfx950" ]

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
    ])

(* the conversions of an architecture with 8-bit floats *)
let fp8_rewrite =
  Pattern_matcher.fold
    (fun () -> [
      rule_ctx (cast_of ~dtype:Dtype.fp8s ~name:"x" c) (fun ctx m ->
          let l = ctx.lang in
          let v =
            match cval (m "c") with
            | `Float v when Float.is_nan v -> None (* written from its bits *)
            | `Float v when v = Float.infinity -> Some l.infinity
            | `Float v when v = Float.neg_infinity -> Some ("-" ^ l.infinity)
            | _ -> Some (const_str (m "c") ^ "f")
          in
          Option.map
            (fun v -> strf "f32_to_fp8(%s, %d)" v (fp8_index (dtype (m "x"))))
            v);
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
  let rewrite =
    if amd_fp8s arch = [] then rewrite
    else Pattern_matcher.append fp8_rewrite rewrite
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
            ( Mulacc,
              fma (fun dt ->
                  if Dtype.equal dt Dtype.Float16 then "__builtin_fmaf16"
                  else by_width ~double:"__builtin_fma" ~single:"__builtin_fmaf" dt)
            );
          ];
    smem_prefix = "__attribute__((shared, aligned(16)))";
    smem_prefix_for_cast = false;
    inline_prefix = "__attribute__((device))";
    barrier =
      "__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");"
      ^ "__builtin_amdgcn_s_barrier();"
      ^ "__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");";
    float4 = (fun t -> "make_" ^ t);
    type_map =
      List.map
        (fun d -> (d, if fp8_index d = 0 then "hip_fp8" else "hip_bf8"))
        Dtype.fp8s
      @ cstyle.type_map;
    string_rewrite = rewrite;
  }

(* LLVM folds a float32 converted exactly from an integer and narrowed to
   bfloat16 into one conversion from the integer, through whatever float
   operations lie between them, and narrows a [zext] of an [i1], and an [or] or
   [and] of such [zext]s, to the [i1]. AMD's backend selects no conversion to
   bfloat16 from an [i1], and the compiler aborts. A conversion to or from
   bfloat16 is written on its bits, which leaves LLVM none to fold. *)
let hip_extra_matcher arch =
  Pattern_matcher.concat
    ([
       pm_narrow_uint;
       pm_wrap_signed;
       pm_defined_div_shift;
       create_non_native_float_pats (Dtype.Bfloat16 :: Dtype.fp8s);
       pm_manual_bf16_cast;
     ]
    @
    if amd_fp8s arch = [] then []
    else [ Tc.pm_wmma_fp8 (if is_cdna arch then Dtype.Uint64 else Dtype.Uint32) ])

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
  else if Dtype.equal dtype_out Dtype.Int32 then
    (* RDNA 4 takes 8 int8 values per lane in 2 registers, RDNA 3 16 in 4. *)
    let rdna4 = List.equal Tc.equal tensor_cores Tc.amd_rdna4 in
    let regs = if rdna4 then 2 else 4 in
    String.concat "\n"
      [
        strf "typedef int wmma_int%d __attribute__((ext_vector_type(%d)));" regs
          regs;
        strf
          "static inline __attribute__((device)) int8 __%s(signed_char%d a, \
           signed_char%d b, int8 c) {"
          name (regs * 4) (regs * 4);
        strf
          "  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32%s(true, \
           __builtin_bit_cast(wmma_int%d, a),"
          (if rdna4 then "_gfx12" else "")
          regs;
        strf "    true, __builtin_bit_cast(wmma_int%d, b), c, false);" regs;
        "}";
      ]
  else if List.equal Tc.equal tensor_cores Tc.amd_rdna4 then
    (* #define __WMMA_16_16_16_f16_f16
       __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12 *)
    strf "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s_w32_gfx12" name
      (tm dtype_out) (tm dtype_in)
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

let hip_kernel arch tensor_cores ?(prefix = []) l ~name kernel bufs uops =
  let used = uops_to_dtypes uops in
  let uses p = List.exists (fun (dt, _) -> p dt) used in
  let is_fp8 dt = Dtype.mem dt Dtype.fp8s in
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
    | Some (n, attr) when Dtype.mem dt Dtype.[ Float16; Float32; Float64 ] ->
        Some
          ( strf "__ocml_%s_f%d" n (Dtype.bitsize dt),
            render_dtype l dt,
            render_dtype l dt,
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
           (fun d ->
             ( d,
               (if is_cdna arch then "_" else "")
               ^ if fp8_index d = 0 then "fp8_fp8" else "bf8_bf8" ))
           Dtype.fp8s)
  in
  let defines =
    (if List.exists (fun u -> const_cast u && non_finite u) uops then
       [
         "#define INFINITY (__builtin_inff())";
         "#define NAN (__builtin_nanf(\"\"))";
       ]
     else [])
    @ (if specials then [ "typedef long unsigned int size_t;" ] else [])
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
  (* Its own definitions first, then the functions it calls. *)
  render_kernel l ~prefix:(defines @ prefix) ~name kernel bufs uops

let hip_with compiler (target : Helpers.Target.t) =
  (* gfx942 => MI300, gfx1100 => RX 7900, gfx1201 => RX 9700 *)
  let arch = target.arch in
  let tensor_cores = Tc.amd arch and lang = hip_lang arch in
  let native dt =
    (not (Dtype.mem dt Dtype.fp8s)) || Dtype.mem dt (amd_fp8s arch)
  in
  (* __builtin_fmaf16 is LLVM's fma of halves, V_FMA_F16 on every GPU here: "D =
     S0 * S1 + S2", fused (RDNA4 ISA). *)
  let fma dt = Dtype.mem dt Dtype.[ Float16; Float32; Float64 ] in
  (* the global limit is only really needed on gfx12, though gfx11 reports the
     same *)
  Renderer.v ~name:"HIPRenderer" ~half8:true ~shared_max:65536
    ~global_max:[ 2147483647; 65535; 65535 ]
    ~global_prod_max:[ 0xFFFFFFFF; 0xFFFFFFFF; 0xFFFFFFFF ]
    ~tensor_cores ~extra_matcher:(hip_extra_matcher arch)
    ~code_for_op:lang.code_for_op ~native ~fma
    ~render:(render (hip_kernel arch tensor_cores) lang)
    ~compiler:(compiler arch) target

let hip target = hip_with Compiler_amd.hip target

(* Renderers for tests *)

let source = Renderer.Compiler.v Fun.id
let cuda_sync target = cuda_renderer ~sync:true (fun () -> source) target

let without_compiler (target : Helpers.Target.t) =
  match target.renderer with
  | "CLANG" -> Renderer.with_compiler source (clang target)
  | "METAL" -> metal_with (fun () -> source) target
  | "CUDA" -> cuda_renderer ~sync:false (fun () -> source) target
  | "HIP" -> hip_with (fun _ -> source) target
  | r -> invalid_arg (strf "no renderer named %s" r)
