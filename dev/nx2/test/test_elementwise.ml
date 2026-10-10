(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's elementwise functions on the host: each computes every dtype its kind
   takes, which its doc states, as the host's kernel called once on the operands
   broadcast by hand, whose kinds the kernel suite holds against nx_kinds.h; a
   dtype the kind does not take raises naming the function. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog

type host = Nx.host

(* Arrays of any dtype *)

type case = Case : ('v, 's) A.t -> case

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_case ppf (Case a) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype a) L.pp (A.layout a)

(* A contiguous array of [dt] and shape [s] over drawn bytes, a boolean's [0] or
   [1]. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  A.v dt (L.contiguous s)
    (Rig.Buffer.of_string (if data = "" then "\000" else data))

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ])

let dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

(* [s] with some extents [1] and some leading axes dropped: a shape that
   broadcasts to [s]. *)
let narrower s =
  let open Gen in
  let* drop = int_range 0 (Array.length s) in
  let+ ones = array ~size:(constant (Array.length s - drop)) bool in
  Array.mapi (fun i one -> if one then 1 else s.(i + drop)) ones

(* Operands of one dtype: the first of [s], each other of [s] or a shape that
   broadcasts to it, [n] in all. *)
let operands n =
  let open Gen in
  let* (D.Any dt) = dtype in
  let* s = shape in
  let* first = drawn dt s in
  let+ rest =
    list
      ~size:(constant (n - 1))
      (let* s' = frequency [ (1, constant s); (1, narrower s) ] in
       drawn dt s')
  in
  (s, Case first, List.map (fun a -> Case (A.expect dt (A.Any a))) rest)

let pp_operands ppf (s, c, cs) =
  Format.fprintf ppf "%a: %a" pp_ints s
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_case)
    (c :: cs)

let unary_operands = Gen.with_pp pp_operands (operands 1)
let binary_operands = Gen.with_pp pp_operands (operands 2)
let ternary_operands = Gen.with_pp pp_operands (operands 3)

(* Bits *)

(* [a]'s bits in C order: one code per element of a sub-byte dtype, the bytes of
   any other. *)
let bits_of (type v s) (a : (v, s) A.t) =
  let a = A.copy a in
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

let value a : (_, _, host) Nx.t = Nx.Repr.of_array Nx.Host.v a
let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))
let result_bits x = bits_of (host_array x)

(* [a] broadcast to [s], as a view. *)
let stretched s a =
  if L.shape (A.layout a) = s then a else Option.get (A.move (M.Broadcast s) a)

let reference1 k dt s a =
  let dst = A.create Rig.host dt s in
  match Nx_cpu.apply1 k ~dst (stretched s a) with
  | A.Done -> Some (bits_of dst)
  | _ -> None

let reference2 k dt s a b =
  let dst = A.create Rig.host dt s in
  match Nx_cpu.apply2 k ~dst (stretched s a) (stretched s b) with
  | A.Done -> Some (bits_of dst)
  | _ -> None

let reference3 k s c a b =
  let dst = A.create Rig.host (A.dtype a) s in
  match
    Nx_cpu.apply3 k ~dst (stretched s c) (stretched s a) (stretched s b)
  with
  | A.Done -> Some (bits_of dst)
  | _ -> None

let names name f =
  raises_match (Exn.invalid_arg ~substring:("Nx." ^ name ^ ": ")) f

(* The law: where its kind takes the dtype, [f] computes, giving the reference's
   bits; elsewhere it raises naming itself. *)
let agrees name ~accepted reference f =
  if accepted then begin
    cover "computed" true;
    match reference () with
    | Some want -> equal ~msg:name (array int) want (f ())
    | None -> fail (name ^ ": the host's kernel declined a dtype its kind takes")
  end
  else begin
    cover "raised" true;
    names name (fun () -> ignore (f ()))
  end

(* Unary *)

type unary = {
  name : string;
  kind : P.unary;
  f : 'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t;
}

let unaries =
  [
    { name = "neg"; kind = Neg; f = Nx.neg };
    { name = "recip"; kind = Recip; f = Nx.recip };
    { name = "abs"; kind = Abs; f = Nx.abs };
    { name = "sign"; kind = Sign; f = Nx.sign };
    { name = "sqrt"; kind = Sqrt; f = Nx.sqrt };
    { name = "exp"; kind = Exp; f = Nx.exp };
    { name = "exp2"; kind = Exp2; f = Nx.exp2 };
    { name = "expm1"; kind = Expm1; f = Nx.expm1 };
    { name = "log"; kind = Log; f = Nx.log };
    { name = "log2"; kind = Log2; f = Nx.log2 };
    { name = "log1p"; kind = Log1p; f = Nx.log1p };
    { name = "sin"; kind = Sin; f = Nx.sin };
    { name = "cos"; kind = Cos; f = Nx.cos };
    { name = "tan"; kind = Tan; f = Nx.tan };
    { name = "asin"; kind = Asin; f = Nx.asin };
    { name = "acos"; kind = Acos; f = Nx.acos };
    { name = "atan"; kind = Atan; f = Nx.atan };
    { name = "sinh"; kind = Sinh; f = Nx.sinh };
    { name = "cosh"; kind = Cosh; f = Nx.cosh };
    { name = "tanh"; kind = Tanh; f = Nx.tanh };
    { name = "floor"; kind = Floor; f = Nx.floor };
    { name = "ceil"; kind = Ceil; f = Nx.ceil };
    { name = "round"; kind = Round; f = Nx.round };
    { name = "trunc"; kind = Trunc; f = Nx.trunc };
  ]

let law_unary (s, Case a, _) =
  let dt = A.dtype a in
  List.iter
    (fun u ->
      agrees u.name
        ~accepted:(P.accepts1 (Unary u.kind) dt dt)
        (fun () -> reference1 (Unary u.kind) dt s a)
        (fun () -> result_bits (u.f (value a))))
    unaries

(* [erf] takes floats alone, by its type. *)
let law_erf (s, Case a, _) =
  let dt = A.dtype a in
  match D.kind dt with
  | D.Float ->
      agrees "erf"
        ~accepted:(P.accepts1 (Unary Erf) dt dt)
        (fun () -> reference1 (Unary Erf) dt s a)
        (fun () -> result_bits (Nx.erf (value a)))
  | D.Signed | D.Unsigned | D.Boolean | D.Complex -> cover "computed" true

(* Binary *)

type binary = {
  name : string;
  kind : D.any -> P.binary;
  f : 'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t;
}

let fixed k _ = k

let binaries =
  [
    { name = "add"; kind = fixed P.Add; f = Nx.add };
    { name = "sub"; kind = fixed P.Sub; f = Nx.sub };
    { name = "mul"; kind = fixed P.Mul; f = Nx.mul };
    {
      name = "div";
      kind =
        (fun (D.Any dt) ->
          match D.kind dt with D.Signed | D.Unsigned -> P.Idiv | _ -> P.Fdiv);
      f = Nx.div;
    };
    { name = "mod_"; kind = fixed P.Mod; f = Nx.mod_ };
    { name = "pow"; kind = fixed P.Pow; f = Nx.pow };
    { name = "atan2"; kind = fixed P.Atan2; f = Nx.atan2 };
    { name = "maximum"; kind = fixed P.Maximum; f = Nx.maximum };
    { name = "minimum"; kind = fixed P.Minimum; f = Nx.minimum };
    { name = "bitwise_and"; kind = fixed P.And; f = Nx.bitwise_and };
    { name = "bitwise_or"; kind = fixed P.Or; f = Nx.bitwise_or };
    { name = "bitwise_xor"; kind = fixed P.Xor; f = Nx.bitwise_xor };
  ]

let law_binary (s, Case a, rest) =
  let dt = A.dtype a in
  let b = match rest with [ Case b ] -> A.expect dt (A.Any b) | _ -> a in
  cover "broadcast" (L.shape (A.layout b) <> s);
  cover "complex" (D.is D.Complex dt);
  List.iter
    (fun (o : binary) ->
      let k = P.Binary (o.kind (D.Any dt)) in
      agrees o.name ~accepted:(P.accepts2 k dt)
        (fun () -> reference2 k dt s a b)
        (fun () -> result_bits (o.f (value a) (value b))))
    binaries

(* Comparisons *)

type comparison = {
  name : string;
  kind : P.compare;
  swapped : bool;
  f :
    'v 's.
    ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> (bool, D.bool_elt, host) Nx.t;
}

let comparisons =
  [
    { name = "equal"; kind = Equal; swapped = false; f = Nx.equal };
    { name = "not_equal"; kind = Not_equal; swapped = false; f = Nx.not_equal };
    { name = "less"; kind = Less; swapped = false; f = Nx.less };
    {
      name = "less_equal";
      kind = Less_equal;
      swapped = false;
      f = Nx.less_equal;
    };
    { name = "greater"; kind = Less; swapped = true; f = Nx.greater };
    {
      name = "greater_equal";
      kind = Less_equal;
      swapped = true;
      f = Nx.greater_equal;
    };
  ]

let law_compare (s, Case a, rest) =
  let dt = A.dtype a in
  cover "complex" (D.is D.Complex dt);
  let b = match rest with [ Case b ] -> A.expect dt (A.Any b) | _ -> a in
  List.iter
    (fun (o : comparison) ->
      let k = P.Compare o.kind in
      let x, y = if o.swapped then (b, a) else (a, b) in
      agrees o.name ~accepted:(P.accepts2 k dt)
        (fun () -> reference2 k D.Bool s x y)
        (fun () -> result_bits (o.f (value a) (value b))))
    comparisons

(* Three operands *)

let law_fma (s, Case a, rest) =
  let dt = A.dtype a in
  let b, c =
    match rest with
    | [ Case b; Case c ] -> (A.expect dt (A.Any b), A.expect dt (A.Any c))
    | _ -> (a, a)
  in
  agrees "fma" ~accepted:(P.accepts3 Fma dt dt)
    (fun () -> reference3 Fma s a b c)
    (fun () -> result_bits (Nx.fma (value a) (value b) (value c)))

let law_where ((s, Case a, rest), flags) =
  let dt = A.dtype a in
  let b = match rest with Case b :: _ -> A.expect dt (A.Any b) | [] -> a in
  let n = Array.fold_left ( * ) 1 s in
  let c =
    A.of_array D.Bool s (Array.init n (fun i -> List.nth flags (i mod 8)))
  in
  agrees "where" ~accepted:true
    (fun () -> reference3 Where s c a b)
    (fun () -> result_bits (Nx.where (value c) (value a) (value b)))

(* Bitcast *)

let law_bitcast_same (_, Case a, _) =
  let x = value a in
  equal bool true (Nx.bitcast (A.dtype a) x == x)

let elementwise =
  group "elementwise"
    [
      prop "each unary function is its kind" unary_operands law_unary;
      prop "each binary function is its kind, broadcast" binary_operands
        law_binary;
      prop "each comparison is its kind, broadcast" binary_operands law_compare;
      prop "fma is its kind, broadcast" ternary_operands law_fma;
      prop "where is its kind, broadcast"
        (Gen.pair ternary_operands (Gen.list ~size:(Gen.constant 8) Gen.bool))
        law_where;
      prop "erf is its kind" unary_operands law_erf;
      prop "a bitcast to its own dtype is the value itself" unary_operands
        law_bitcast_same;
    ]

let () = exit (run "nx elementwise" [ elementwise ])
