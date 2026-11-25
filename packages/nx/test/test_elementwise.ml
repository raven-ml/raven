(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's functions of one kind on the host, in nx.mli's order: each computes
   every dtype its doc states, as the host's kernel called once on the operands
   broadcast by hand, whose kinds the kernel suites hold against nx_kinds.h; a
   dtype it does not take raises naming the function. The values nx.mli states
   for integers, and the conversions, against their rules computed in OCaml. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

type host = Nx.host

let names name f = raises_match (Exn.invalid_arg ~sub:("Nx." ^ name ^ ": ")) f

(* Arrays of any dtype *)

type case = Case : ('v, 's) A.t -> case

let pp_case ppf (Case a) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype a) L.pp (A.layout a)

(* A contiguous array of [dt] and shape [s] over [data], a boolean's bytes [0]
   or [1]. *)
let over (type v s) (dt : (v, s) D.t) s data : (v, s) A.t =
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  A.v dt
    (L.contiguous (Iarray.of_array s))
    (Rig.Buffer.of_string (if data = "" then "\000" else data))

let drawn (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let n = Array.fold_left ( * ) 1 s in
  Gen.map (over dt s)
    (Gen.string_of ~size:(Gen.constant (D.bytes dt n)) Gen.char)

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ])

(* Every dtype, the complex ones, 2 of [D.all]'s, drawn a quarter of the time so
   that each run reaches the laws' complex cases. *)
let dtype =
  let complex = List.filter (fun (D.Any dt) -> D.is D.Complex dt) D.all in
  Gen.with_pp
    (fun ppf (D.Any dt) -> D.pp ppf dt)
    (Gen.frequency [ (5, Gen.of_list D.all); (1, Gen.of_list complex) ])

(* [s] with some extents [1] and some leading axes dropped: a shape that
   broadcasts to [s]. *)
let narrower s =
  let open Gen in
  let* drop = int_range 0 (Array.length s) in
  let+ ones = array ~size:(constant (Array.length s - drop)) bool in
  Array.mapi (fun i one -> if one then 1 else s.(i + drop)) ones

type operands = int array * case * case list

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
  ((s, Case first, List.map (fun a -> Case (A.expect dt (A.Any a))) rest)
    : operands)

let pp_operands ppf (s, c, cs) =
  Format.fprintf ppf "%a: %a" pp_ints s
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_case)
    (c :: cs)

(* [n] operands of shape [[5]] at every dtype, over fixed bytes that hold zeros,
   ones, NaNs and extremes: each dtype runs on every run. *)
let every_dtype n =
  let bytes =
    "\000\001\255\127\128\192\003\124\126\255\000\128\064\063\200\017"
  in
  List.map
    (fun (D.Any dt) ->
      let data k =
        String.init (D.bytes dt 5) (fun i -> bytes.[(i + (3 * k)) mod 16])
      in
      let a k = over dt [| 5 |] (data k) in
      (([| 5 |], Case (a 0), List.init (n - 1) (fun k -> Case (a (k + 1))))
        : operands))
    D.all

let unary_operands = Gen.with_pp pp_operands (operands 1)
let binary_operands = Gen.with_pp pp_operands (operands 2)
let ternary_operands = Gen.with_pp pp_operands (operands 3)
let unary name ~f = prop ~examples:(every_dtype 1) name unary_operands f
let binary name ~f = prop ~examples:(every_dtype 2) name binary_operands f
let ternary name ~f = prop ~examples:(every_dtype 3) name ternary_operands f

(* The dtypes nx.mli says a function takes. *)

let every_dtype_but_booleans (D.Any dt) = not (D.is D.Boolean dt)

let floats_and_integers (D.Any dt) =
  D.is D.Float dt || D.is D.Signed dt || D.is D.Unsigned dt

let floats_and_complex (D.Any dt) = D.is D.Float dt || D.is D.Complex dt
let floats (D.Any dt) = D.is D.Float dt
let complex (D.Any dt) = D.is D.Complex dt

let integers_and_booleans (D.Any dt) =
  D.is D.Signed dt || D.is D.Unsigned dt || D.is D.Boolean dt

let every_dtype_of _ = true

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
let host_array x = Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x))
let result_bits x = bits_of (host_array x)

(* [a] broadcast to [s], as a view. *)
let stretched s a =
  if Iarray.to_array (L.shape (A.layout a)) = s then a
  else Option.get (A.move (M.Broadcast (Iarray.of_array s)) a)

(* nx.cpu, whose one-node maps are the references. *)
let cpu = (module Nx_cpu : Nx_kernel.S)

let reference1 k dt s a =
  let dst = A.create Rig.host dt (Iarray.of_array s) in
  match Support.map1 cpu k ~dst (stretched s a) with
  | A.Done -> Some (bits_of dst)
  | _ | (exception Invalid_argument _) -> None

let reference2 k dt s a b =
  let dst = A.create Rig.host dt (Iarray.of_array s) in
  match Support.map2 cpu k ~dst (stretched s a) (stretched s b) with
  | A.Done -> Some (bits_of dst)
  | _ | (exception Invalid_argument _) -> None

let reference3 k s c a b =
  let dst = A.create Rig.host (A.dtype a) (Iarray.of_array s) in
  match
    Support.map3 cpu k ~dst (stretched s c) (stretched s a) (stretched s b)
  with
  | A.Done -> Some (bits_of dst)
  | _ | (exception Invalid_argument _) -> None

(* The law: where nx.mli says [f] takes the dtype, [f] computes, giving the
   reference's bits, a NaN for a NaN where the kind leaves its bits unspecified,
   whatever they are: they follow the layouts and the operands' addresses;
   elsewhere it raises naming itself. [dt] is the result's dtype. *)
let agrees name ~takes ~kind dt reference f =
  let bits =
    if Kernels_cases.any_nan kind then Kernels_cases.same_values (D.Any dt)
    else array int
  in
  if takes then
    match reference () with
    | Some want -> equal ~msg:name bits want (f ())
    | None ->
        fail (name ^ ": the host's kernel declined a dtype nx.mli says it takes")
  else names name (fun () -> ignore (f ()))

let other (s, a, rest) =
  match rest with Case b :: _ -> (s, a, Case b) | [] -> (s, a, a)

(* [f] is the unary kind [u] where it takes the dtype. *)
let kind1 name u ~takes (f : 'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    (s, Case a, _) =
  let dt = A.dtype a in
  agrees name ~takes:(takes (D.Any dt)) ~kind:(Support.K1 (Unary u)) dt
    (fun () -> reference1 (Unary u) dt s a)
    (fun () -> result_bits (f (value a)))

(* [f] is the binary kind [k dt] where it takes the dtype, broadcast. *)
let kind2 name k ~takes
    (f :
      'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    operands =
  let s, Case a, Case b = other operands in
  let dt = A.dtype a and b = A.expect (A.dtype a) (A.Any b) in
  cover "broadcast" (Iarray.to_array (L.shape (A.layout b)) <> s);
  let k = P.Binary (k (D.Any dt)) in
  agrees name ~takes:(takes (D.Any dt)) ~kind:(Support.K2 k) dt
    (fun () -> reference2 k dt s a b)
    (fun () -> result_bits (f (value a) (value b)))

let fixed k _ = k

(* A comparison is its kind, [greater] and [greater_equal] [less] and
   [less_equal] swapped, at every dtype. *)
let compare name k ~swapped
    (f :
      'v 's.
      ('v, 's, host) Nx.t ->
      ('v, 's, host) Nx.t ->
      (bool, D.bool_elt, host) Nx.t) operands =
  let s, Case a, Case b = other operands in
  let b = A.expect (A.dtype a) (A.Any b) in
  cover "broadcast" (Iarray.to_array (L.shape (A.layout b)) <> s);
  let k = P.Compare k in
  let x, y = if swapped then (b, a) else (a, b) in
  agrees name ~takes:true ~kind:(Support.K2 k) D.Bool
    (fun () -> reference2 k D.Bool s x y)
    (fun () -> result_bits (f (value a) (value b)))

(* Integers, as nx.mli states their values *)

let integers =
  List.filter (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt) D.all

let integer_id (D.Any dt) = D.name dt

(* A value of an integer dtype from its value modulo [2^w]. *)
let wrapped (type v s) (dt : (v, s) D.t) (x : int64) : v =
  let sext bits =
    Int64.shift_right (Int64.shift_left x (64 - bits)) (64 - bits)
  in
  let zext bits = Int64.logand x (Int64.sub (Int64.shift_left 1L bits) 1L) in
  match dt with
  | D.Int64 -> x
  | D.Uint64 -> x
  | D.Int32 -> Int64.to_int32 x
  | D.Uint32 -> Int64.to_int32 x
  | D.Int16 -> Int64.to_int (sext 16)
  | D.Uint16 -> Int64.to_int (zext 16)
  | D.Int8 -> Int64.to_int (sext 8)
  | D.Uint8 -> Int64.to_int (zext 8)
  | D.Int4 -> Int64.to_int (sext 4)
  | D.Uint4 -> Int64.to_int (zext 4)
  | _ -> invalid_arg ("wrapped: " ^ D.name dt)

(* An integer element's value as an int64; uint64's less [2^64] where its top
   bit is set. *)
let int_value (type v s) (dt : (v, s) D.t) (v : v) : int64 =
  match dt with
  | D.Int64 -> v
  | D.Uint64 -> v
  | D.Int32 -> Int64.of_int32 v
  | D.Uint32 -> Int64.logand (Int64.of_int32 v) 0xffff_ffffL
  | D.Int16 -> Int64.of_int v
  | D.Uint16 -> Int64.of_int v
  | D.Int8 -> Int64.of_int v
  | D.Uint8 -> Int64.of_int v
  | D.Int4 -> Int64.of_int v
  | D.Uint4 -> Int64.of_int v
  | _ -> invalid_arg ("int_value: " ^ D.name dt)

(* The least and greatest values of an integer dtype, as int64s. *)
let range (type v s) (dt : (v, s) D.t) =
  let w = D.bits dt in
  if D.is D.Unsigned dt then
    (0L, if w = 64 then -1L else Int64.pred (Int64.shift_left 1L w))
  else
    let least = Int64.shift_left (-1L) (w - 1) in
    (least, Int64.lognot least)

(* [f] of rows [(a, b)] of [dt], against [expected a b], computed modulo the
   width, each row named. *)
let rows2 (type v s) (dt : (v, s) D.t) f expected pairs =
  let n = List.length pairs in
  let col sel =
    Nx.create dt [| n |]
      (Array.of_list (List.map (fun p -> wrapped dt (sel p)) pairs))
  in
  let got = Nx.to_array (f (col fst) (col snd)) in
  List.iteri
    (fun i (a, b) ->
      equal
        ~msg:(Printf.sprintf "%s at %Ld, %Ld" (D.name dt) a b)
        (elements dt)
        [ wrapped dt (expected a b) ]
        [ got.(i) ])
    pairs

let test_div (D.Any dt) =
  let least, _ = range dt in
  let signed = D.is D.Signed dt in
  let pairs =
    [ (7L, 2L); (5L, 0L); (0L, 0L); (6L, 3L) ]
    @
    if signed then
      [ (-7L, 2L); (7L, -2L); (-7L, -2L); (least, -1L); (least, 0L) ]
    else []
  in
  (* Truncated toward zero, 0 by 0, the least value by -1 itself. *)
  let expected a b =
    if b = 0L then 0L else if b = -1L then Int64.neg a else Int64.div a b
  in
  rows2 dt Nx.div expected pairs

let test_recip (D.Any dt) =
  let least, greatest = range dt in
  let xs =
    [ 0L; 1L; 2L; greatest ]
    @ if D.is D.Signed dt then [ -1L; -2L; least ] else []
  in
  let signed = D.is D.Signed dt in
  let expected a _ = if a = 1L || (signed && a = -1L) then a else 0L in
  rows2 dt (fun a _ -> Nx.recip a) expected (List.map (fun x -> (x, 0L)) xs)

(* [a^b] modulo [2^64], [b >= 0]. *)
let rec power a b =
  if b = 0L then 1L
  else
    let h = power (Int64.mul a a) (Int64.shift_right_logical b 1) in
    if Int64.logand b 1L = 1L then Int64.mul a h else h

let test_pow (D.Any dt) =
  let _, greatest = range dt in
  let signed = D.is D.Signed dt in
  let pairs =
    [ (0L, 0L); (2L, 3L); (3L, 40L); (greatest, 2L); (1L, 0L) ]
    @
    if signed then
      [ (0L, -1L); (1L, -5L); (-1L, -3L); (-1L, -4L); (2L, -1L); (-2L, 3L) ]
    else []
  in
  (* Modulo the width, [1] at [b = 0]; a negative [b] gives [0] but at the bases
     [1] and [-1]. *)
  let expected a b =
    if b >= 0L then power a b
    else if a = 1L then 1L
    else if a = -1L then if Int64.logand b 1L = 1L then -1L else 1L
    else 0L
  in
  (* [pow 3 40] wraps at every width; read the dtype's own values back. *)
  let pairs =
    List.map
      (fun (a, b) -> (int_value dt (wrapped dt a), int_value dt (wrapped dt b)))
      pairs
  in
  rows2 dt Nx.pow expected pairs

(* Integer operands of one dtype, with a divisor of [0], [-1] or [1] a third of
   the time. *)
let integer_pair =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) integers in
  let* n = int_range 0 17 in
  let* a = drawn dt [| n |] in
  let+ b = drawn dt [| n |]
  and+ small =
    array ~size:(constant n)
      (frequency
         [ (2, constant None); (1, map Option.some (of_list [ 0L; -1L; 1L ])) ])
  in
  let bs =
    Array.mapi
      (fun i v -> match small.(i) with Some s -> wrapped dt s | None -> v)
      (A.to_array b)
  in
  (Case a, Case (A.of_array dt (Iarray.of_array [| n |]) bs))

let pp_pair ppf (a, b) = Format.fprintf ppf "%a, %a" pp_case a pp_case b

(* Each integer dtype over its least and greatest values, [0], [±1] and [±7],
   against [0], [±1] and [2]. *)
let integer_examples =
  List.map
    (fun (D.Any dt) ->
      let least, greatest = range dt in
      let xs = [ least; greatest; 0L; 1L; -1L; 7L; -7L ] in
      let ds = [ 0L; 1L; -1L; 2L ] in
      let a = List.concat_map (fun x -> List.map (fun _ -> x) ds) xs in
      let b = List.concat_map (fun _ -> ds) xs in
      let arr l =
        A.of_array dt
          (Iarray.of_array [| List.length l |])
          (Array.of_list (List.map (wrapped dt) l))
      in
      (Case (arr a), Case (arr b)))
    integers

(* [a = b (a / b) + mod_ a b], and [mod_ a b] is zero or of [a]'s sign. *)
let law_mod (Case a, Case b) =
  let dt = A.dtype a and b = A.expect (A.dtype a) (A.Any b) in
  let bs = A.to_array b in
  cover "a divisor of 0" (Array.exists (fun v -> int_value dt v = 0L) bs);
  cover "a divisor of -1" (Array.exists (fun v -> int_value dt v = -1L) bs);
  let x = value a and y = value b in
  let back = Nx.add (Nx.mul y (Nx.div x y)) (Nx.mod_ x y) in
  equal (elements dt)
    (Array.to_list (A.to_array a))
    (Array.to_list (Nx.to_array back));
  if D.is D.Signed dt then
    Array.iteri
      (fun i r ->
        let r = int_value dt r and v = int_value dt (A.to_array a).(i) in
        if not (r = 0L || Int64.compare r 0L = Int64.compare v 0L) then
          failf "mod_ %Ld %Ld is %Ld, not of the dividend's sign" v
            (int_value dt bs.(i))
            r)
      (Nx.to_array (Nx.mod_ x y))

(* Rounding: halves away from zero *)

let test_round () =
  let check (type s) (dt : (float, s) D.t) below =
    let xs = [| 0.5; -0.5; 1.5; -1.5; 2.5; -2.5; below; -.below |] in
    equal ~msg:(D.name dt) (array float_exact)
      [| 1.; -1.; 2.; -2.; 3.; -3.; 0.; -0. |]
      (Nx.to_array (Nx.round (Nx.create dt [| 8 |] xs)))
  in
  (* The greatest float below a half. *)
  check Nx.float32 0x1.fffffep-2;
  check Nx.float64 0x1.fffffffffffffp-2

(* Shapes that do not broadcast *)

let refusals =
  let f32 n = Nx.zeros Nx.float32 [| n |] in
  let i32 n = Nx.zeros Nx.int32 [| n |] in
  let b n = Nx.zeros Nx.bool [| n |] in
  let two name f = (name, fun () -> ignore (f (f32 2) (f32 3))) in
  [
    two "add" Nx.add;
    two "sub" Nx.sub;
    two "mul" Nx.mul;
    two "div" Nx.div;
    two "mod_" Nx.mod_;
    two "pow" Nx.pow;
    two "atan2" Nx.atan2;
    two "hypot" Nx.hypot;
    two "maximum" Nx.maximum;
    two "minimum" Nx.minimum;
    two "equal" Nx.equal;
    two "not_equal" Nx.not_equal;
    two "less" Nx.less;
    two "less_equal" Nx.less_equal;
    two "greater" Nx.greater;
    two "greater_equal" Nx.greater_equal;
    ("bitwise_and", fun () -> ignore (Nx.bitwise_and (i32 2) (i32 3)));
    ("bitwise_or", fun () -> ignore (Nx.bitwise_or (i32 2) (i32 3)));
    ("bitwise_xor", fun () -> ignore (Nx.bitwise_xor (i32 2) (i32 3)));
    ("fma", fun () -> ignore (Nx.fma (f32 2) (f32 3) (f32 2)));
    ("lerp", fun () -> ignore (Nx.lerp (f32 2) (f32 2) (f32 3)));
    ("where", fun () -> ignore (Nx.where (b 2) (f32 3) (f32 3)));
  ]

(* Positions and layouts. On the host, a binary function's and fma's results are
   a function of the operands' values: every element of a row, which runs
   through a vector loop's body and its scalar tail, at every layout, broadcast
   and strided operands included, is the one-element result, a NaN for a NaN,
   whose sign and payload are unspecified. *)

(* A float dtype and the codes of its parts drawn: 0, 1, inf and a NaN of each
   sign. *)
type float_dtype = Float_dtype : ('v, 's) D.t * int64 list -> float_dtype

let float_dtypes =
  let f32 = [ 0L; 0x3F80_0000L; 0x7F80_0000L; 0x7FC0_0001L; 0xFFC0_0002L ] in
  let f64 =
    [
      0L;
      0x3FF0_0000_0000_0000L;
      0x7FF0_0000_0000_0000L;
      0x7FF8_0000_0000_0001L;
      0xFFF8_0000_0000_0002L;
    ]
  in
  D.
    [
      Float_dtype (Float16, [ 0L; 0x3C00L; 0x7C00L; 0x7E01L; 0xFE02L ]);
      Float_dtype (Bfloat16, [ 0L; 0x3F80L; 0x7F80L; 0x7FC1L; 0xFFC2L ]);
      Float_dtype (Float32, f32);
      Float_dtype (Float64, f64);
      Float_dtype (Complex64, f32);
      Float_dtype (Complex128, f64);
    ]

(* A part's code [c] as [w] bytes, little-endian. *)
let part_bytes w c =
  String.init w (fun i ->
      let b = Int64.shift_right_logical c (8 * i) in
      Char.chr (Int64.to_int (Int64.logand b 0xFFL)))

(* [n] elements of [dt], each the bytes [elt]. *)
let filled dt n elt =
  value (over dt [| n |] (String.concat "" (List.init n (fun _ -> elt))))

(* A row with a vector body and a scalar tail at every dtype. *)
let row = 37

(* [f] over operands of [row] elements of [elts], all contiguous, all strided,
   and each broadcast in turn, by name. *)
let layouts dt f elts =
  let f xs = f (Array.of_list xs) in
  let one e = Nx.broadcast_to [| row |] (filled dt 1 e) in
  let every e = Nx.slice [ Nx.Rs (0, 2 * row, 2) ] (filled dt (2 * row) e) in
  let broadcast i _ =
    let xs = List.mapi (fun j e -> if i = j then one e else filled dt row e) in
    (Printf.sprintf "operand %d broadcast" i, f (xs elts))
  in
  ("contiguous", f (List.map (filled dt row) elts))
  :: ("strided", f (List.map every elts))
  :: List.mapi broadcast elts

let check_layouts (Float_dtype (dt, codes)) =
  let w = D.bytes dt 1 / if D.is D.Complex dt then 2 else 1 in
  let elts =
    let part = List.map (part_bytes w) codes in
    if not (D.is D.Complex dt) then part
    else List.concat_map (fun re -> List.map (fun im -> re ^ im) part) part
  in
  let values = Kernels_cases.same_values (D.Any dt) in
  let check name f operands =
    let msg = Format.asprintf "%s %a" name D.pp dt in
    let ones = Array.of_list (List.map (filled dt 1) operands) in
    let one = result_bits (f ones) in
    let want = Array.concat (List.init row (fun _ -> one)) in
    List.iter
      (fun (layout, got) ->
        equal ~msg:(msg ^ ", " ^ layout) values want (result_bits got))
      (layouts dt f operands)
  in
  let each f = List.iter f elts in
  let binaries =
    [
      ("add", P.Add, fun a b -> Nx.add a b);
      ("sub", P.Sub, fun a b -> Nx.sub a b);
      ("mul", P.Mul, fun a b -> Nx.mul a b);
      ("div", P.Fdiv, fun a b -> Nx.div a b);
      ("mod_", P.Mod, fun a b -> Nx.mod_ a b);
      ("pow", P.Pow, fun a b -> Nx.pow a b);
      ("atan2", P.Atan2, fun a b -> Nx.atan2 a b);
      ("maximum", P.Maximum, fun a b -> Nx.maximum a b);
      ("minimum", P.Minimum, fun a b -> Nx.minimum a b);
    ]
  in
  List.iter
    (fun (name, k, f) ->
      if P.accepts2 (P.Binary k) dt then
        each (fun x ->
            each (fun y -> check name (fun a -> f a.(0) a.(1)) [ x; y ])))
    binaries;
  if P.accepts3 Fma dt dt then
    each (fun x ->
        each (fun y ->
            each (fun z ->
                check "fma" (fun a -> Nx.fma a.(0) a.(1) a.(2)) [ x; y; z ])))

let test_layouts () = List.iter check_layouts float_dtypes

(* fma and where *)

let law_fma (s, Case a, rest) =
  let dt = A.dtype a in
  let b, c =
    match rest with
    | [ Case b; Case c ] -> (A.expect dt (A.Any b), A.expect dt (A.Any c))
    | _ -> (a, a)
  in
  agrees "fma"
    ~takes:(floats_and_integers (D.Any dt))
    ~kind:(Support.K3 Fma) dt
    (fun () -> reference3 Fma s a b c)
    (fun () -> result_bits (Nx.fma (value a) (value b) (value c)))

let law_where ((s, Case a, rest), flags) =
  let dt = A.dtype a in
  let b = match rest with Case b :: _ -> A.expect dt (A.Any b) | [] -> a in
  let n = Array.fold_left ( * ) 1 s in
  let c =
    A.of_array D.Bool (Iarray.of_array s)
      (Array.init n (fun i -> List.nth flags (i mod 8)))
  in
  agrees "where" ~takes:true ~kind:(Support.K3 Where) dt
    (fun () -> reference3 Where s c a b)
    (fun () -> result_bits (Nx.where (value c) (value a) (value b)))

(* A narrow float computes at float32 and rounds once. *)

type narrow = Narrow : (float, 's) D.t -> narrow

let narrows =
  Gen.of_list
    ~pp:(fun ppf (Narrow dt) -> D.pp ppf dt)
    D.
      [
        Narrow Float16;
        Narrow Bfloat16;
        Narrow Float8_e4m3fn;
        Narrow Float8_e5m2;
        Narrow Float4_e2m1fn;
      ]

(* fma at a float narrower than float32 is its exact value rounded once, as
   Exact's integers round it, where a float32 would round first. *)
let law_narrow_fma (Narrow dt, seed) =
  let f = Exact.format dt in
  let a, b, c = Kernels_cases.Narrow_fma.operands f seed in
  Kernels_cases.Narrow_fma.covers (D.name dt) f a b c;
  let v x = Nx.create dt [| Array.length x |] x in
  let got = Nx.to_array (Nx.cast D.Float64 (Nx.fma (v a) (v b) (v c))) in
  equal (list string) [] (Kernels_cases.Narrow_fma.misses f a b c got)

(* lerp *)

(* lerp's two forms at float64, as nx.mli states them, each fma rounded once
   (Exact.fma: Float.fma rounds some subnormal results twice under MinGW). *)
let lerp_reference a b w =
  let d = b -. a in
  if w < 0.5 then Exact.fma w d a else Exact.fma (w -. 1.) d b

let weight =
  Gen.frequency
    [
      (3, Gen.float_range (-2.) 3.);
      (2, Gen.of_list [ 0.; -0.; 1.; 0.5; -0.5; Float.pred 0.5; 2. ]);
      (1, Gen.any_float);
    ]

let ends =
  Gen.frequency
    [
      (3, Gen.float_range (-1e3) 1e3);
      (1, Gen.of_list [ 0.; -0. ]);
      (1, Gen.any_float);
    ]

(* lerp's ends and weight, printed in hexadecimal. *)
let lerp_operands =
  Gen.with_pp
    (fun ppf ((a, b), w) -> Format.fprintf ppf "a %h, b %h, w %h" a b w)
    (Gen.pair (Gen.pair ends ends) weight)

(* Equal as numbers: [-0.] and [+0.] alike, as nx.mli states the ends. *)
let number =
  Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%h" x) ~equal:Float.equal

let law_lerp ((a, b), w) =
  cover "an end" (w = 0. || w = 1.);
  cover "the form from b" (w >= 0.5);
  cover "below -1/2" (w <= -0.5);
  let s = Nx.scalar Nx.float64 in
  let got = (Nx.to_array (Nx.lerp (s a) (s b) (s w))).(0) in
  equal ~msg:"the form nx.mli states" float_exact (lerp_reference a b w) got;
  if Float.is_finite (b -. a) && w = 0. then equal ~msg:"w = 0" number a got;
  if Float.is_finite (b -. a) && w = 1. then equal ~msg:"w = 1" number b got

(* The ends at signed zeros, each at both weights. *)
let lerp_examples =
  List.concat_map
    (fun ab -> [ (ab, 0.); (ab, 1.) ])
    [ (-0., 1.); (1., -0.); (-0., -0.); (0., -0.); (-0., 0.) ]

let law_lerp_narrow (Narrow dt, ((a, b), w)) =
  let s x = Nx.scalar dt x in
  let a = s a and b = s b and w = s w in
  let wide x = Nx.cast Nx.float32 x in
  let want = Nx.cast dt (Nx.lerp (wide a) (wide b) (wide w)) in
  (* Each element's bits, through the exact cast to float64, [None] for a
     NaN. *)
  let bits x =
    Array.map
      (fun v -> if Float.is_nan v then None else Some (Int64.bits_of_float v))
      (Nx.to_array (Nx.cast Nx.float64 x))
  in
  equal (array (option int64)) (bits want) (bits (Nx.lerp a b w))

let test_lerp_shapes () =
  let a = Nx.zeros Nx.float32 [| 2; 1 |] and b = Nx.ones Nx.float32 [| 3 |] in
  let y = Nx.lerp a b (Nx.scalar Nx.float32 0.25) in
  equal (array int) [| 2; 3 |] (Nx.shape y);
  equal (array float_exact) (Array.make 6 0.25) (Nx.to_array y)

(* atan2's range ends: [-π] at [y = -0.], [x < 0], and [π] rounded to the dtype,
   3.25 in float8_e4m3fn. *)
let test_atan2_ends () =
  let at dt y x = Nx.atan2 (Nx.scalar dt y) (Nx.scalar dt x) in
  let first x = (A.to_array (host_array x)).(0) in
  equal float_exact (-.Float.pi) (first (at D.Float64 (-0.) (-1.)));
  equal float_exact Float.pi (first (at D.Float64 0. (-1.)));
  equal float_exact 3.25 (first (at D.Float8_e4m3fn 0. (-1.)))

(* square *)

let law_square (_, Case a, _) =
  let dt = A.dtype a in
  if every_dtype_but_booleans (D.Any dt) then
    equal
      (Kernels_cases.same_values (D.Any dt))
      (result_bits (Nx.mul (value a) (value a)))
      (result_bits (Nx.square (value a)))
  else names "square" (fun () -> ignore (Nx.square (value a)))

let square_examples : operands list =
  let one (type v s) (dt : (v, s) D.t) (xs : v array) =
    let a = A.of_array dt (Iarray.of_array [| Array.length xs |]) xs in
    (([| Array.length xs |], Case a, []) : operands)
  in
  [
    one D.Int8 [| 12; -128; -1 |];
    one D.Float32 [| -0.; 3.; 1e20 |];
    one D.Complex64 [| { Complex.re = 1.; im = 2. } |];
  ]

(* Conversion *)

(* How a value is read: whole, reversed (a view of negative steps), at a step of
   two (strided), or broadcast along a new leading axis (a step of zero). *)
type view = Whole | Flipped | Strided | Broadcast

let view_name = function
  | Whole -> "whole"
  | Flipped -> "flipped"
  | Strided -> "strided"
  | Broadcast -> "broadcast"

let viewed v (x : ('v, 's, host) Nx.t) =
  match v with
  | Whole -> x
  | Flipped -> Nx.flip x
  | Strided ->
      (* Each element beside a copy of itself, read at a step of two. *)
      let s = Nx.shape x in
      let r = Array.length s in
      let column = Nx.reshape (Array.append s [| 1 |]) x in
      Nx.slice
        (List.init r (fun _ -> Nx.A) @ [ Nx.I 0 ])
        (Nx.concatenate ~axis:r [ column; column ])
  | Broadcast -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape x)) x

(* Whether [x] lies halfway between two floats of [into]. *)
let is_tie (type r) (into : (float, r) D.t) x =
  let r = D.of_float into x in
  Float.is_finite x && Float.is_finite r && r <> x
  && Float.abs (x -. r) = Nx_support.ulp into x /. 2.

(* The tie of [into] above [x] rounded to [into], where [dt], wider, holds
   it. *)
let tie (type s r) (dt : (float, s) D.t) (into : (float, r) D.t) x =
  let r = D.of_float into x in
  if
    (not (Float.is_finite r)) || Float.abs r >= (D.float_format into).max_finite
  then None
  else
    let t = r +. (Nx_support.ulp into r /. 2.) in
    if D.of_float dt t = t then Some t else None

(* A value to cast: its dtype, the one to cast into, and how it is read. *)
type cast =
  | Cast : ('v, 's) D.t * ('w, 'r) D.t * ('v, 's, host) Nx.t * view -> cast

let pp_cast ppf (Cast (dt, into, x, v)) =
  Format.fprintf ppf "%a %a %s into %a" D.pp dt pp_ints (Nx.shape x)
    (view_name v) D.pp into

let any_dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

(* A length of no element, one, a few, and past nx.cpu's vector bodies of eight
   and sixteen lanes with a tail. *)
let length = Gen.of_list [ 0; 1; 3; 17; 37 ]

(* Floats that break a store: NaNs of both signs, infinities, zeros of both
   signs, halves, the formats' largest finite values and their rounding
   midpoints, integer range ends, subnormals and values past them. *)
let breaking =
  [|
    Float.nan;
    Float.neg Float.nan;
    infinity;
    neg_infinity;
    0.;
    -0.;
    0.5;
    -0.5;
    1.5;
    2.5;
    -2.5;
    7.;
    9.;
    65504.;
    65519.;
    65520.;
    448.;
    464.;
    57344.;
    61440.;
    1e10;
    -1e10;
    300.;
    -300.;
    2.7;
    -2.7;
    4294967295.;
    2147483648.;
    -2147483649.;
    1e39;
    1e-310;
    1e-40;
    6e-8;
    0x1p53 +. 2.;
    9.2e18;
    -9.3e18;
    1.8e19;
    3e38;
  |]

(* [xs] as a value of [dt], with the ties of [into] around them where [into] is
   a narrower float. *)
let float_value (type s w r) (dt : (float, s) D.t) (into : (w, r) D.t) xs =
  let ties =
    match D.kind into with
    | D.Float when D.bits into < D.bits dt ->
        List.filter_map (tie dt into) (Array.to_list xs)
    | _ -> []
  in
  let xs = Array.append xs (Array.of_list ties) in
  Nx.create dt [| Array.length xs |] (Array.map (D.of_float dt) xs)

let drawn_floats n =
  let open Gen in
  array ~size:(constant n)
    (frequency
       [
         (2, of_list (Array.to_list breaking));
         (1, map Int64.float_of_bits int64);
         (1, float_range (-1000.) 1000.);
       ])

(* [n] elements of [dt] to cast into [into]: floats that break a store, with
   [into]'s ties, or drawn bytes. *)
let cast_value (type v s w r) (dt : (v, s) D.t) (into : (w, r) D.t) n :
    (v, s, host) Nx.t Gen.t =
  match D.kind dt with
  | D.Float -> Gen.map (float_value dt into) (drawn_floats n)
  | _ -> Gen.map value (drawn dt [| n |])

let casts =
  let open Gen in
  with_pp pp_cast
    (let* (D.Any dt) = any_dtype in
     let* (D.Any into) = any_dtype in
     let* n = length in
     let* v = of_list [ Whole; Flipped; Strided; Broadcast ] in
     let+ x = cast_value dt into n in
     Cast (dt, into, viewed v x, v))

(* [breaking], with [into]'s ties, or 16 elements of fixed bytes, as [dt]. *)
let fixed_value (type v s w r) (dt : (v, s) D.t) (into : (w, r) D.t) :
    (v, s, host) Nx.t =
  match D.kind dt with
  | D.Float -> float_value dt into breaking
  | _ ->
      value
        (over dt [| 16 |]
           (String.init (D.bytes dt 16) (fun i ->
                Char.chr (((i * 73) + 17) land 255))))

(* Every pair of dtypes over the breaking floats or fixed bytes, strided. *)
let cast_examples =
  List.concat_map
    (fun (D.Any dt) ->
      List.map
        (fun (D.Any into) ->
          let x = fixed_value dt into in
          Cast (dt, into, viewed Strided x, Strided))
        D.all)
    D.all

(* [m], read unsigned, rounded to [p] significant bits, ties to even: the exact
   value an integer rounds to in a format of precision [p] whose exponent range
   is unbounded above, as a float, which holds it exactly. *)
let round_bits m p =
  let rec length m n =
    if m = 0L then n else length (Int64.shift_right_logical m 1) (n + 1)
  in
  let n = length m 0 in
  if n <= p then Int64.to_float m
  else
    let shift = n - p in
    let q = Int64.shift_right_logical m shift in
    let rest = Int64.logand m (Int64.sub (Int64.shift_left 1L shift) 1L) in
    let c = Int64.unsigned_compare rest (Int64.shift_left 1L (shift - 1)) in
    let q =
      if c > 0 || (c = 0 && Int64.logand q 1L = 1L) then Int64.succ q else q
    in
    Float.ldexp (Int64.to_float q) shift

(* The precision of [into]'s float format, a complex dtype's parts'. *)
let precision (type w r) (into : (w, r) D.t) =
  match D.kind into with
  | D.Float -> (D.float_format into).fraction_bits + 1
  | D.Complex -> if D.equal into D.Complex64 then 24 else 53
  | D.Signed | D.Unsigned | D.Boolean -> 64

(* The element a cast of [x], of [dt], into [into] stores by nx.mli's rule. An
   integer into a float rounds once from its exact value: to the format's
   precision in Int64, then [Dtype.of_float], which keeps that value or
   overflows it as a store does. *)
let cast_ref (type v s w r) (dt : (v, s) D.t) (into : (w, r) D.t) (x : v) : w =
  match D.kind dt with
  | D.Float -> D.of_float into x
  | D.Boolean -> D.of_float into (if x then 1. else 0.)
  | D.Complex -> (
      let nonzero = x.Complex.re <> 0. || x.im <> 0. in
      match into with
      | D.Complex64 ->
          {
            Complex.re = D.of_float D.Float32 x.re;
            im = D.of_float D.Float32 x.im;
          }
      | D.Complex128 -> x
      | D.Bool -> nonzero
      | D.Bit -> nonzero
      | _ -> D.of_float into x.re)
  | D.Signed | D.Unsigned -> (
      let w = int_value dt x in
      let negative = D.is D.Signed dt && Int64.compare w 0L < 0 in
      match D.kind into with
      | D.Signed | D.Unsigned -> wrapped into w
      | D.Boolean -> w <> 0L
      | D.Float | D.Complex ->
          let m =
            round_bits (if negative then Int64.neg w else w) (precision into)
          in
          D.of_float into (if negative then -.m else m))

(* The float elements of [xs], none where [dt] is no float. *)
let float_elements (type v s) (dt : (v, s) D.t) (xs : v array) : float array =
  match D.kind dt with D.Float -> xs | _ -> [||]

(* Whether [x] lies past [into]'s finite range, as a store reads it. *)
let past (type w r) (into : (w, r) D.t) x =
  Float.is_finite x
  &&
  match D.kind into with
  | D.Float -> Float.abs x > (D.float_format into).max_finite
  | D.Signed | D.Unsigned ->
      let w = Float.of_int (D.bits into) in
      if D.is D.Unsigned into then x >= Float.pow 2. w || x <= -1.
      else x >= Float.pow 2. (w -. 1.) || x <= -.Float.pow 2. (w -. 1.) -. 1.
  | D.Complex | D.Boolean -> false

let tie_of (type w r) (into : (w, r) D.t) x =
  match D.kind into with D.Float -> is_tie into x | _ -> false

let law_cast (Cast (dt, into, x, v)) =
  let xs = Nx.to_array x in
  let fs = float_elements dt xs in
  let has p = Array.exists p fs in
  cover "a NaN" (has Float.is_nan);
  cover "an infinity" (has (fun e -> Float.abs e = infinity));
  cover "-0." (has (fun e -> e = 0. && Float.sign_bit e));
  cover "past the destination's range" (has (past into));
  cover "a tie of the destination" (has (tie_of into));
  cover "a view" (v <> Whole && Array.length xs > 1);
  let got = Nx.to_array (Nx.cast into x) in
  Array.iteri
    (fun i e ->
      equal
        ~msg:(Format.asprintf "element %d, %a" i (D.pp_value dt) e)
        (elements into)
        [ cast_ref dt into e ]
        [ got.(i) ])
    xs

(* The rows nx.mli and Dtype's stores state. *)
let test_cast_rows () =
  let one (type v s w r) (dt : (v, s) D.t) (into : (w, r) D.t) (x : v) : w =
    (Nx.to_array (Nx.cast into (Nx.create dt [| 1 |] [| x |]))).(0)
  in
  let f64 = D.Float64 in
  let floats =
    [
      ( "65519. into float16 is its largest float",
        one f64 D.Float16 65519.,
        65504. );
      ("65520. into float16 is inf", one f64 D.Float16 65520., infinity);
      ("1e39 into float32 is inf", one f64 D.Float32 1e39, infinity);
      ("inf into float8_e4m3fn is 448.", one f64 D.Float8_e4m3fn infinity, 448.);
      ( "-inf into float8_e4m3fn is -448.",
        one f64 D.Float8_e4m3fn neg_infinity,
        -448. );
      ("500. into float8_e4m3fn saturates", one f64 D.Float8_e4m3fn 500., 448.);
      ("7. into float4_e2m1fn saturates", one f64 D.Float4_e2m1fn 7., 6.);
      ("NaN into float4_e2m1fn is +0.", one f64 D.Float4_e2m1fn nan, 0.);
      ( "int64 2^53 + 1 into float64 ties to even",
        one D.Int64 f64 0x20_0000_0000_0001L,
        0x1p53 );
      (* Through a float64, [2^53 + 2^29], a tie of float32 that rounds to
         [2^53]. *)
      ( "int64 2^53 + 2^29 + 1 into float32 rounds once from its value",
        one D.Int64 D.Float32 0x20_0000_2000_0001L,
        0x1p53 +. 0x1p30 );
      ( "uint64 2^64 - 1 into float32 is 2^64",
        one D.Uint64 D.Float32 (-1L),
        0x1p64 );
    ]
  in
  List.iter (fun (msg, got, want) -> equal ~msg float_exact want got) floats;
  let ints =
    [
      ("-0.5 into uint8 is 0", one f64 D.Uint8 (-0.5), 0);
      ("-1e10 into int8 saturates", one f64 D.Int8 (-1e10), -128);
      ("300. into uint8 saturates", one f64 D.Uint8 300., 255);
      ("-2.7 into int8 truncates", one f64 D.Int8 (-2.7), -2);
      ( "int16 300 into int8 is kept modulo the width",
        one D.Int16 D.Int8 300,
        44 );
      ("int32 9 into int4 is -7", one D.Int32 D.Int4 9l, -7);
      ("true into uint8 is 1", one D.Bool D.Uint8 true, 1);
    ]
  in
  List.iter (fun (msg, got, want) -> equal ~msg int want got) ints;
  equal ~msg:"inf into int32 saturates" int32 Int32.max_int
    (one f64 D.Int32 infinity);
  equal ~msg:"-inf into int32 saturates" int32 Int32.min_int
    (one f64 D.Int32 neg_infinity);
  equal ~msg:"NaN into int32 is 0" int32 0l (one f64 D.Int32 nan);
  equal ~msg:"uint32 0xffffffff into int32 is -1" int32 (-1l)
    (one D.Uint32 D.Int32 (-1l));
  equal ~msg:"-0. into bool is false" bool false (one f64 D.Bool (-0.));
  equal ~msg:"NaN into bool is true" bool true (one f64 D.Bool nan);
  equal ~msg:"0 + 1i into bool is true" bool true
    (one D.Complex64 D.Bool { Complex.re = 0.; im = 1. });
  equal ~msg:"3 + 4i into float32 is its real part" float_exact 3.
    (one D.Complex128 D.Float32 { Complex.re = 3.; im = 4. });
  let z = one f64 D.Complex64 2.5 in
  equal ~msg:"2.5 into complex64 is 2.5 + 0i"
    (pair float_exact float_exact)
    (2.5, 0.) (z.re, z.im);
  let z = one D.Complex128 D.Complex64 { Complex.re = 1e300; im = 1e-300 } in
  equal ~msg:"complex128 into complex64 part by part"
    (pair float_exact float_exact)
    (infinity, 0.) (z.re, z.im)

(* bitcast *)

(* [a]'s bits in memory order, each element's from its low bit. *)
let bit_string (type v s) (a : (v, s) A.t) =
  let w = match D.bits (A.dtype a) with 1 -> 1 | 4 -> 4 | _ -> 8 in
  Array.concat
    (List.map
       (fun c -> Array.init w (fun i -> (c lsr i) land 1))
       (Array.to_list (bits_of a)))

(* A value to bitcast, its dtype, the one to read it in, and how it is read: of
   shape [[n]], or [[n; k]] where the other dtype is [k] times wider. *)
type bitcast =
  | Bitcast :
      ('v, 's) D.t * ('w, 'r) D.t * ('v, 's, host) Nx.t * view
      -> bitcast

let pp_bitcast ppf (Bitcast (dt, into, x, v)) =
  Format.fprintf ppf "%a %a %s into %a" D.pp dt pp_ints (Nx.shape x)
    (view_name v) D.pp into

(* A uint8 [[3; 2]] reversed read as uint16: no view widens it, so the bitcast
   runs a program. *)
let program_path =
  let x = Nx.flip (Nx.reshape [| 3; 2 |] (Nx.arange Nx.uint8 1 7 1)) in
  [ Bitcast (D.Uint8, D.Uint16, x, Flipped) ]

let bitcasts =
  let open Gen in
  with_pp pp_bitcast
    (let* (D.Any dt) = any_dtype in
     let* (D.Any into) = any_dtype in
     let* n = of_list [ 0; 1; 3; 5; 17 ] in
     let* v = of_list [ Whole; Flipped; Strided; Broadcast ] in
     let k = D.bits into / D.bits dt in
     let s = if k > 1 then [| n; k |] else [| n |] in
     let+ a = drawn dt s in
     Bitcast (dt, into, viewed v (value a), v))

(* [bitcast into x]'s shape: [x]'s at one width, [k] appended where [into] is
   [k] times narrower, the last axis dropped where it is wider. *)
let bitcast_shape dt into s =
  let a = D.bits dt and b = D.bits into in
  if a = b then s
  else if a > b then Array.append s [| a / b |]
  else Array.sub s 0 (Array.length s - 1)

let law_bitcast (Bitcast (dt, into, x, v)) =
  cover "narrower" (D.bits dt > D.bits into);
  cover "wider" (D.bits dt < D.bits into);
  cover "a view" (v <> Whole && Nx.numel x > 1);
  cover "a sub-byte dtype" (D.bits dt < 8 || D.bits into < 8);
  let y = Nx.bitcast into x in
  equal (array int) (bitcast_shape dt into (Nx.shape x)) (Nx.shape y);
  equal (array int) (bit_string (host_array x)) (bit_string (host_array y))

let law_bitcast_back (Bitcast (dt, into, x, _)) =
  let back = Nx.bitcast dt (Nx.bitcast into x) in
  equal (array int) (Nx.shape x) (Nx.shape back);
  equal (array int) (bit_string (host_array x)) (bit_string (host_array back))

let law_itself (_, Case a, _) =
  let x = value a in
  let dt = A.dtype a in
  satisfies ~claim:"x itself" pass (fun y -> y == x) (Nx.bitcast dt x);
  satisfies ~claim:"x itself" pass (fun y -> y == x) (Nx.cast dt x)

let test_bitcast_refusals () =
  names "bitcast" (fun () ->
      ignore (Nx.bitcast Nx.uint16 (Nx.zeros Nx.uint8 [| 3 |])));
  names "bitcast" (fun () ->
      ignore (Nx.bitcast Nx.uint4 (Nx.zeros Nx.uint8 (Array.make L.max_rank 1))))

let elementwise =
  [
    group "arithmetic"
      [
        binary "add is Add, broadcast, on every dtype but booleans"
          ~f:(kind2 "add" (fixed P.Add) ~takes:every_dtype_but_booleans Nx.add);
        binary "sub is Sub, broadcast, on every dtype but booleans"
          ~f:(kind2 "sub" (fixed P.Sub) ~takes:every_dtype_but_booleans Nx.sub);
        binary "mul is Mul, broadcast, on every dtype but booleans"
          ~f:(kind2 "mul" (fixed P.Mul) ~takes:every_dtype_but_booleans Nx.mul);
        binary "div is Fdiv or Idiv, broadcast, on every dtype but booleans"
          ~f:
            (kind2 "div"
               (fun (D.Any dt) ->
                 match D.kind dt with
                 | D.Signed | D.Unsigned -> P.Idiv
                 | _ -> P.Fdiv)
               ~takes:every_dtype_but_booleans Nx.div);
        cases
          "div of integers truncates toward zero, is 0 by 0 and the least by -1"
          ~id:integer_id integers test_div;
        binary "mod_ is Mod, broadcast, on floats and integers"
          ~f:(kind2 "mod_" (fixed P.Mod) ~takes:floats_and_integers Nx.mod_);
        prop ~examples:integer_examples
          "mod_ of integers keeps a = b (a / b) + mod_ a b, of a's sign"
          (Gen.with_pp pp_pair integer_pair)
          law_mod;
        binary "pow is Pow, broadcast, on every dtype but booleans"
          ~f:(kind2 "pow" (fixed P.Pow) ~takes:every_dtype_but_booleans Nx.pow);
        cases
          "pow of integers wraps, and a negative exponent gives 0 but at 1 and \
           -1"
          ~id:integer_id integers test_pow;
        ternary "fma is Fma, broadcast, on floats and integers" ~f:law_fma;
        prop "fma rounds once at floats narrower than float32" ~count:20
          ~examples:
            D.
              [
                (Narrow Float16, 1);
                (Narrow Bfloat16, 1);
                (Narrow Float8_e4m3fn, 1);
                (Narrow Float8_e5m2, 1);
                (Narrow Float4_e2m1fn, 1);
              ]
          (Gen.pair narrows Gen.int) law_narrow_fma;
        prop ~examples:lerp_examples "lerp is one fma, from a below w = 1/2"
          lerp_operands law_lerp;
        prop "lerp of a narrow float is lerp at float32, rounded once"
          (Gen.pair narrows (Gen.pair (Gen.pair ends ends) weight))
          law_lerp_narrow;
        test "lerp broadcasts" test_lerp_shapes;
        binary "maximum is Maximum, broadcast, on every dtype"
          ~f:
            (kind2 "maximum" (fixed P.Maximum) ~takes:every_dtype_of Nx.maximum);
        binary "minimum is Minimum, broadcast, on every dtype"
          ~f:
            (kind2 "minimum" (fixed P.Minimum) ~takes:every_dtype_of Nx.minimum);
        unary "neg is Neg on every dtype but booleans"
          ~f:(kind1 "neg" Neg ~takes:every_dtype_but_booleans Nx.neg);
        prop
          ~examples:(square_examples @ every_dtype 1)
          "square x is mul x x, bit for bit, on every dtype but booleans"
          unary_operands law_square;
        unary "recip is Recip on every dtype but booleans"
          ~f:(kind1 "recip" Recip ~takes:every_dtype_but_booleans Nx.recip);
        cases "recip of an integer is x at 1 and -1 and 0 elsewhere"
          ~id:integer_id integers test_recip;
        unary "abs is Abs on floats and integers"
          ~f:(kind1 "abs" Abs ~takes:floats_and_integers Nx.abs);
        unary "sign is Sign on every dtype but booleans"
          ~f:(kind1 "sign" Sign ~takes:every_dtype_but_booleans Nx.sign);
        cases
          "operations of several operands refuse shapes that do not broadcast"
          ~id:fst refusals (fun (name, f) -> names name f);
      ];
    group "powers, exponentials and trigonometry"
      [
        unary "sqrt is Sqrt on floats and complex numbers"
          ~f:(kind1 "sqrt" Sqrt ~takes:floats_and_complex Nx.sqrt);
        unary "exp is Exp on floats and complex numbers"
          ~f:(kind1 "exp" Exp ~takes:floats_and_complex Nx.exp);
        unary "exp2 is Exp2 on floats and complex numbers"
          ~f:(kind1 "exp2" Exp2 ~takes:floats_and_complex Nx.exp2);
        unary "expm1 is Expm1 on floats"
          ~f:(kind1 "expm1" Expm1 ~takes:floats Nx.expm1);
        unary "log is Log on floats and complex numbers"
          ~f:(kind1 "log" Log ~takes:floats_and_complex Nx.log);
        unary "log2 is Log2 on floats and complex numbers"
          ~f:(kind1 "log2" Log2 ~takes:floats_and_complex Nx.log2);
        unary "log1p is Log1p on floats"
          ~f:(kind1 "log1p" Log1p ~takes:floats Nx.log1p);
        unary "sin is Sin on floats and complex numbers"
          ~f:(kind1 "sin" Sin ~takes:floats_and_complex Nx.sin);
        unary "cos is Cos on floats and complex numbers"
          ~f:(kind1 "cos" Cos ~takes:floats_and_complex Nx.cos);
        unary "tan is Tan on floats and complex numbers"
          ~f:(kind1 "tan" Tan ~takes:floats_and_complex Nx.tan);
        unary "asin is Asin on floats and complex numbers"
          ~f:(kind1 "asin" Asin ~takes:floats_and_complex Nx.asin);
        unary "acos is Acos on floats and complex numbers"
          ~f:(kind1 "acos" Acos ~takes:floats_and_complex Nx.acos);
        unary "atan is Atan on floats and complex numbers"
          ~f:(kind1 "atan" Atan ~takes:floats_and_complex Nx.atan);
        binary "atan2 is Atan2, broadcast, on floats"
          ~f:(kind2 "atan2" (fixed P.Atan2) ~takes:floats Nx.atan2);
        test "atan2 reaches its range's ends, rounded" test_atan2_ends;
        unary "sinh is Sinh on floats and complex numbers"
          ~f:(kind1 "sinh" Sinh ~takes:floats_and_complex Nx.sinh);
        unary "cosh is Cosh on floats and complex numbers"
          ~f:(kind1 "cosh" Cosh ~takes:floats_and_complex Nx.cosh);
        unary "tanh is Tanh on floats and complex numbers"
          ~f:(kind1 "tanh" Tanh ~takes:floats_and_complex Nx.tanh);
        unary "asinh of a complex number is Asinh"
          ~f:(fun ((_, Case a, _) as o) ->
            if complex (D.Any (A.dtype a)) then
              kind1 "asinh" Asinh ~takes:complex (fun x -> Nx.asinh x) o);
        unary "acosh of a complex number is Acosh"
          ~f:(fun ((_, Case a, _) as o) ->
            if complex (D.Any (A.dtype a)) then
              kind1 "acosh" Acosh ~takes:complex (fun x -> Nx.acosh x) o);
        unary "atanh of a complex number is Atanh"
          ~f:(fun ((_, Case a, _) as o) ->
            if complex (D.Any (A.dtype a)) then
              kind1 "atanh" Atanh ~takes:complex (fun x -> Nx.atanh x) o);
        unary "erf is Erf on floats" ~f:(fun ((s, Case a, _) as o) ->
            match D.kind (A.dtype a) with
            | D.Float ->
                let dt = A.dtype a in
                agrees "erf" ~takes:true ~kind:(Support.K1 (Unary Erf)) dt
                  (fun () -> reference1 (Unary Erf) dt s a)
                  (fun () -> result_bits (Nx.erf (value a)))
            | _ -> ignore o);
      ];
    group "rounding"
      [
        unary "floor is Floor on floats and integers"
          ~f:(kind1 "floor" Floor ~takes:floats_and_integers Nx.floor);
        unary "ceil is Ceil on floats and integers"
          ~f:(kind1 "ceil" Ceil ~takes:floats_and_integers Nx.ceil);
        unary "round is Round on floats and integers"
          ~f:(kind1 "round" Round ~takes:floats_and_integers Nx.round);
        test "round takes halves away from zero" test_round;
        unary "trunc is Trunc on floats and integers"
          ~f:(kind1 "trunc" Trunc ~takes:floats_and_integers Nx.trunc);
      ];
    group "comparisons and bits"
      [
        binary "equal is Equal, broadcast, on every dtype"
          ~f:(compare "equal" Equal ~swapped:false Nx.equal);
        binary "not_equal is Not_equal, broadcast, on every dtype"
          ~f:(compare "not_equal" Not_equal ~swapped:false Nx.not_equal);
        binary "less is Less, broadcast, on every dtype"
          ~f:(compare "less" Less ~swapped:false Nx.less);
        binary "less_equal is Less_equal, broadcast, on every dtype"
          ~f:(compare "less_equal" Less_equal ~swapped:false Nx.less_equal);
        binary "greater a b is less b a, broadcast, on every dtype"
          ~f:(compare "greater" Less ~swapped:true Nx.greater);
        binary "greater_equal a b is less_equal b a, broadcast, on every dtype"
          ~f:(compare "greater_equal" Less_equal ~swapped:true Nx.greater_equal);
        binary "bitwise_and is And, broadcast, on integers and booleans"
          ~f:
            (kind2 "bitwise_and" (fixed P.And) ~takes:integers_and_booleans
               Nx.bitwise_and);
        binary "bitwise_or is Or, broadcast, on integers and booleans"
          ~f:
            (kind2 "bitwise_or" (fixed P.Or) ~takes:integers_and_booleans
               Nx.bitwise_or);
        binary "bitwise_xor is Xor, broadcast, on integers and booleans"
          ~f:
            (kind2 "bitwise_xor" (fixed P.Xor) ~takes:integers_and_booleans
               Nx.bitwise_xor);
        prop
          ~examples:
            (List.map
               (fun o ->
                 (o, [ true; false; true; true; false; false; true; false ]))
               (every_dtype 3))
          "where is Where, broadcast, on every dtype"
          (Gen.pair ternary_operands (Gen.list ~size:(Gen.constant 8) Gen.bool))
          law_where;
      ];
    group "conversion"
      [
        prop ~examples:cast_examples
          "cast stores each element by nx.mli's rule, from any dtype into any"
          casts law_cast;
        test "cast stores the values nx.mli and Dtype state" test_cast_rows;
        unary "cast and bitcast into the value's own dtype are the value itself"
          ~f:law_itself;
        prop ~examples:program_path
          "bitcast reads x's bits in the dtype, from the low bits" bitcasts
          law_bitcast;
        prop ~examples:program_path "bitcast back into x's dtype gives x's bits"
          bitcasts law_bitcast_back;
        test
          "bitcast refuses a last axis of other than k, and the greatest rank"
          test_bitcast_refusals;
      ];
    group "the host's kernels"
      [
        test "binary functions and fma give the one-element result everywhere"
          test_layouts;
      ];
  ]

let () = exit (run "nx elementwise" elementwise)
