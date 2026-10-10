(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's reductions and scans on the host. On int32 each is the fold of its
   elements, an order-free reference; on every dtype each computes the dtypes
   its doc states, as the host's kernel over the same axes (the narrower floats
   at float32, rounded once), and raises naming itself elsewhere. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module S = Nx_kernel.Spec

type host = Nx.host

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let value a : (_, _, host) Nx.t = Nx.Repr.of_array Nx.Host.v a
let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))

let names name f =
  raises_match (Exn.invalid_arg ~substring:("Nx." ^ name ^ ": ")) f

(* Shapes and axes *)

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ])

(* Axes of a value of rank [r]: [None] for all, or some, a few negative. *)
let axes_of r =
  let open Gen in
  if r = 0 then constant None
  else
    let* picks = array ~size:(constant r) bool in
    let+ negative = bool in
    let axes =
      List.concat
        (List.mapi
           (fun a p -> if p then [ (if negative then a - r else a) ] else [])
           (Array.to_list picks))
    in
    if axes = [] then None else Some axes

let normal r axes =
  match axes with
  | None -> List.init r Fun.id
  | Some l ->
      List.sort Int.compare (List.map (fun a -> if a < 0 then a + r else a) l)

let reduced s axes =
  Array.of_list
    (List.filteri (fun a _ -> not (List.mem a axes)) (Array.to_list s))

let kept s axes = Array.mapi (fun a n -> if List.mem a axes then 1 else n) s

(* Indices of [s] in C order. *)
let indices s =
  let n = Array.fold_left ( * ) 1 s in
  List.init n (fun i ->
      let idx = Array.make (Array.length s) 0 and k = ref i in
      for a = Array.length s - 1 downto 0 do
        idx.(a) <- !k mod s.(a);
        k := !k / s.(a)
      done;
      idx)

(* int32: the fold of the elements *)

let int32_case =
  let open Gen in
  with_pp
    (fun ppf (s, axes, keepdims, _) ->
      Format.fprintf ppf "%a over %s%s" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l))
        (if keepdims then ", kept" else ""))
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let* keepdims = bool in
     let n = Array.fold_left ( * ) 1 s in
     let+ xs =
       array ~size:(constant n)
         (frequency
            [
              (4, map Int32.of_int (int_range (-5) 5));
              ( 1,
                of_list
                  ~pp:(fun ppf x -> Format.fprintf ppf "%ld" x)
                  [ Int32.max_int; Int32.min_int ] );
            ])
     in
     (s, axes, keepdims, xs))

(* The fold of [f] from [init] of [xs] of shape [s] along [axes], in C order of
   the result. *)
let fold f init s axes xs =
  let out = reduced s axes in
  let acc = Hashtbl.create 16 in
  List.iteri
    (fun i idx ->
      let key =
        Array.of_list
          (List.filteri (fun a _ -> not (List.mem a axes)) (Array.to_list idx))
      in
      let prev = Option.value ~default:init (Hashtbl.find_opt acc key) in
      Hashtbl.replace acc key (f prev xs.(i)))
    (indices s);
  Array.of_list
    (List.map
       (fun idx -> Option.value ~default:init (Hashtbl.find_opt acc idx))
       (indices out))

let law_int32 (s, axes, keepdims, xs) =
  let r = Array.length s in
  let ax = normal r axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  cover "kept" keepdims;
  let x = value (A.of_array D.Int32 s xs) in
  let check name f init g =
    let y = host_array (g ?axes ~keepdims x) in
    equal ~msg:(name ^ ": shape") (array int)
      (if keepdims then kept s ax else reduced s ax)
      (L.shape (A.layout y));
    equal ~msg:name (array int32) (fold f init s ax xs) (A.to_array y)
  in
  check "sum" Int32.add 0l (fun ?axes ~keepdims x -> Nx.sum ?axes ~keepdims x);
  check "prod" Int32.mul 1l (fun ?axes ~keepdims x -> Nx.prod ?axes ~keepdims x);
  if List.exists (fun a -> s.(a) = 0) ax then begin
    names "max" (fun () -> Nx.max ?axes x);
    names "min" (fun () -> Nx.min ?axes x)
  end
  else begin
    check "max" Stdlib.max Int32.min_int (fun ?axes ~keepdims x ->
        Nx.max ?axes ~keepdims x);
    check "min" Stdlib.min Int32.max_int (fun ?axes ~keepdims x ->
        Nx.min ?axes ~keepdims x)
  end

(* The running fold along [axis]. *)
let running f s axis xs =
  let out = Array.copy xs in
  List.iteri
    (fun i idx ->
      if idx.(axis) > 0 then begin
        let prev = Array.copy idx in
        prev.(axis) <- idx.(axis) - 1;
        let j =
          Array.fold_left ( + ) 0
            (Array.mapi
               (fun a v ->
                 v
                 * Array.fold_left ( * ) 1
                     (Array.sub s (a + 1) (Array.length s - a - 1)))
               prev)
        in
        out.(i) <- f out.(j) xs.(i)
      end)
    (indices s);
  out

let law_int32_scan (s, _, _, xs) =
  let r = Array.length s in
  let n = Array.length xs in
  let x = value (A.of_array D.Int32 s xs) in
  List.iter
    (fun (name, f, (g : ?axis:int -> _ -> _)) ->
      let flat = host_array (g x) in
      equal ~msg:(name ^ ": shape") (array int) s (L.shape (A.layout flat));
      equal ~msg:(name ^ " in C order") (array int32) (running f [| n |] 0 xs)
        (A.to_array flat);
      if r > 0 then
        equal
          ~msg:(name ^ " along the last axis")
          (array int32)
          (running f s (r - 1) xs)
          (A.to_array (host_array (g ~axis:(-1) x))))
    [
      ("cumsum", Int32.add, fun ?axis x -> Nx.cumsum ?axis x);
      ("cumprod", Int32.mul, fun ?axis x -> Nx.cumprod ?axis x);
      ("cummax", Stdlib.max, fun ?axis x -> Nx.cummax ?axis x);
      ("cummin", Stdlib.min, fun ?axis x -> Nx.cummin ?axis x);
    ]

(* Every dtype: the host's kernel *)

type case = Case : ('v, 's) A.t -> case

let dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

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

let any_case =
  let open Gen in
  with_pp
    (fun ppf (Case a, axes) ->
      Format.fprintf ppf "%a %a over %s" D.pp (A.dtype a) pp_ints
        (L.shape (A.layout a))
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l)))
    (let* (D.Any dt) = dtype in
     let* s = shape in
     let* a = drawn dt s in
     let+ axes = axes_of (Array.length s) in
     (Case a, axes))

let bits_of (type v s) (a : (v, s) A.t) =
  let a = A.copy a in
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

(* [a] at [dt], as nx.cpu casts. *)
let cast (type v s w r) (dt : (w, r) D.t) (a : (v, s) A.t) : (w, r) A.t =
  let dst = A.create Rig.host dt (L.shape (A.layout a)) in
  match Nx_cpu.apply1 Cast ~dst a with
  | A.Done -> dst
  | _ -> failwith "the host's kernel declined a cast"

let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

(* The host kernel's reduction [m] of [a] along [axes]: at float32 for the
   floats narrower than 32 bits, at the byte-wide dtype for the sub-byte ones,
   rounded back once; a complex sum over the parts. [None] where the kernel
   declines. *)
let rec reference : type v s.
    S.monoid -> (v, s) A.t -> int array -> (v, s) A.t option =
 fun m a axes ->
  let dt = A.dtype a in
  let through (type w r) (acc : (w, r) D.t) =
    Option.map (cast dt) (reference m (cast acc a) axes)
  in
  let pairs (type w r) (part : (w, r) D.t) =
    Option.map
      (fun p -> A.copy (Option.get (A.bitcast dt (A.copy p))))
      (reference m (Option.get (A.bitcast part (A.copy a))) axes)
  in
  match (dt, m) with
  | ( ( D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
      | D.Float4_e2m1fn ),
      _ ) ->
      through D.Float32
  | D.Int4, _ -> through D.Int8
  | D.Uint4, _ -> through D.Uint8
  | D.Bit, _ -> through D.Bool
  | D.Complex64, S.Sum -> pairs D.Float32
  | D.Complex128, S.Sum -> pairs D.Float64
  | _ -> (
      let s = L.shape (A.layout a) in
      let out =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let dst = A.create Rig.host dt out in
      let spec =
        S.reduce (identity dt) ~loads:[| S.Plain |] ~axes
          [| (S.Monoid m, 0, D.Any dt) |]
      in
      match Nx_cpu.reduce spec ~dsts:[| A.Any dst |] [| A.Any a |] with
      | A.Done -> Some dst
      | _ -> None)

(* What each reduction's doc states it takes. *)
let documented (type v s) (m : S.monoid) (dt : (v, s) D.t) =
  match (D.kind dt, m) with
  | (D.Float | D.Signed | D.Unsigned), _ -> true
  | D.Boolean, (Max | Min) -> true
  | D.Complex, Sum -> true
  | D.Boolean, (Sum | Prod | Logsumexp)
  | D.Complex, (Prod | Max | Min | Logsumexp) ->
      false

let reductions =
  [
    ("sum", S.Sum, fun ?axes x -> Nx.sum ?axes x);
    ("prod", S.Prod, fun ?axes x -> Nx.prod ?axes x);
    ("max", S.Max, fun ?axes x -> Nx.max ?axes x);
    ("min", S.Min, fun ?axes x -> Nx.min ?axes x);
  ]

let law_any (Case a, axes) =
  let dt = A.dtype a in
  let s = L.shape (A.layout a) in
  let ax = Array.of_list (normal (Array.length s) axes) in
  let empty = Array.exists (fun i -> s.(i) = 0) ax in
  List.iter
    (fun (name, m, f) ->
      let f () = bits_of (host_array (f ?axes (value a))) in
      if documented m dt && not (empty && (m = S.Max || m = S.Min)) then begin
        cover "computed" true;
        match reference m a ax with
        | Some want -> equal ~msg:name (array int) (bits_of want) (f ())
        | None -> fail (name ^ ": the host's kernel declined a stated dtype")
      end
      else begin
        cover "raised" true;
        names name (fun () -> ignore (f ()))
      end)
    reductions

(* Errors *)

let test_errors () =
  let x = value (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  names "sum" (fun () -> Nx.sum ~axes:[ 2 ] x);
  names "sum" (fun () -> Nx.sum ~axes:[ 0; -2 ] x);
  names "max" (fun () -> Nx.max (value (A.of_array D.Float32 [| 0 |] [||])));
  names "cumsum" (fun () -> Nx.cumsum ~axis:2 x);
  names "sum" (fun () ->
      Nx.sum (value (A.of_array D.Bool [| 2 |] [| true; false |])));
  names "mean" (fun () ->
      Nx.mean (value (A.of_array D.Int32 [| 2 |] [| 1l; 2l |])));
  names "cumsum" (fun () ->
      Nx.cumsum (value (A.of_array D.Bool [| 2 |] [| true; false |])))

let test_mean () =
  let x = value (A.of_array D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 6. |]) in
  equal (array float_exact) [| 2.; 4. |]
    (A.to_array (host_array (Nx.mean ~axes:[ 0 ] x)));
  let all = host_array (Nx.mean ~keepdims:true x) in
  equal (array int) [| 1; 1 |] (L.shape (A.layout all));
  equal (array float_exact) [| 3. |] (A.to_array all);
  let none = value (A.of_array D.Float64 [| 0 |] [||]) in
  equal bool true (Float.is_nan (A.to_array (host_array (Nx.mean none))).(0))

(* Scans through their expansions: a complex sum adds the parts, and a narrow
   float accumulates at float32, rounding each prefix once. *)
let test_scan_expansions () =
  let z =
    A.of_array D.Complex64 [| 3 |]
      Complex.
        [|
          { re = 1.; im = -1. }; { re = 2.; im = 0.5 }; { re = -4.; im = 2. };
        |]
  in
  equal ~msg:"complex64 cumsum"
    (array
       (Testable.make
          ~pp:(fun ppf (c : Complex.t) -> Format.fprintf ppf "%g%+gi" c.re c.im)
          ~equal:( = )))
    Complex.
      [|
        { re = 1.; im = -1. }; { re = 3.; im = -0.5 }; { re = -1.; im = 1.5 };
      |]
    (A.to_array (host_array (Nx.cumsum (value z))));
  let xs = Array.init 300 (fun i -> 1. +. (Float.of_int i /. 7.)) in
  let b = cast D.Bfloat16 (A.of_array D.Float32 [| 300 |] xs) in
  let wide = A.to_array (host_array (Nx.cumsum (value (cast D.Float32 b)))) in
  let want =
    A.to_array (cast D.Bfloat16 (A.of_array D.Float32 [| 300 |] wide))
  in
  equal ~msg:"bfloat16 cumsum, rounded once" (array float_exact) want
    (A.to_array (host_array (Nx.cumsum (value b))))

let reductions_group =
  group "reductions"
    [
      prop "int32 reductions are the folds of their elements" int32_case
        law_int32;
      prop "int32 scans are the running folds, in C order or along an axis"
        int32_case law_int32_scan;
      prop "each reduction computes its stated dtypes as the host's kernel"
        any_case law_any;
      test
        "bad axes, empty extremes and refused dtypes raise naming the function"
        test_errors;
      test "mean divides the sum by the count" test_mean;
      test "scans of complex numbers and narrow floats expand"
        test_scan_expansions;
    ]

let () = exit (run "nx reduce" [ reductions_group ])
