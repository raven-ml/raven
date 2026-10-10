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

(* Logsumexp, Moments and Arg through Nx.Prim, on a set whose kernels compute
   the core reductions alone, against references over each result's terms. *)

(* nx.cpu, declining every reduction and scan but one Sum, Prod, Max or Min, as
   a library that computes the core alone may. *)
module Core_only = struct
  include Nx_cpu

  let name = "nx.core"

  let core s =
    match S.reductions s with
    | [| (S.Monoid (Sum | Prod | Max | Min), _, _) |] -> true
    | _ -> false

  let reduce s ~dsts ops =
    if core s then Nx_cpu.reduce s ~dsts ops else A.Declined

  let scan s ~dsts ops = if core s then Nx_cpu.scan s ~dsts ops else A.Declined
end

module Core =
  (val Nx.devices ~kernels:(module Core_only) [ Nx_support.memory 1 ])

let core a = Nx.place Core.on (value a)

let reduce1 r axes x =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Nx.shape x);
           axes;
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
         })
  in
  y

let read x = A.to_array (host_array x)
let bits = Int64.bits_of_float
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* Each result's terms, in C order of the results and of the reduced indices. *)
let terms s axes xs =
  Array.map List.rev (fold (fun acc x -> x :: acc) [] s axes xs)

let float64_case =
  let open Gen in
  let element =
    frequency
      [
        (24, float_range (-30.) 30.);
        (1, constant Float.neg_infinity);
        (1, constant Float.infinity);
        (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
      ]
  in
  with_pp
    (fun ppf (s, axes, _) ->
      Format.fprintf ppf "%a over %s" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l)))
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
     (s, axes, xs))

let law_logsumexp (s, axes, xs) =
  let ax = normal (Array.length s) axes in
  let x = core (A.of_array D.Float64 s xs) in
  let got =
    read (reduce1 (Monoid (Logsumexp, 0, Nx.float64)) (Array.of_list ax) x)
  in
  Array.iteri
    (fun i ts ->
      let g = got.(i) in
      match List.find_opt Float.is_nan ts with
      | Some nan ->
          cover "a NaN term" true;
          equal ~msg:"the first NaN, its bits" int64 (bits nan) (bits g)
      | None ->
          let m = List.fold_left Float.max Float.neg_infinity ts in
          if Float.is_finite m then begin
            cover "finite" true;
            let s = List.fold_left (fun s t -> s +. exp (t -. m)) 0. ts in
            let want = m +. log s in
            let n = Float.of_int (List.length ts) in
            at_most
              ~msg:(Printf.sprintf "%h against %h" g want)
              float_exact
              ~than:((n +. 8.) *. epsilon_float *. (1. +. Float.abs want))
              (Float.abs (g -. want))
          end
          else begin
            cover "no term, or an infinite maximum" true;
            equal ~msg:"an infinity" int64 (bits m) (bits g)
          end)
    (terms s ax xs)

(* float32 terms near an offset [c], whose large ratio of mean to spread a
   formula reading [Σ x²] would lose. *)
let moments_case =
  let open Gen in
  with_pp
    (fun ppf (s, axes, c, _) ->
      Format.fprintf ppf "%a over %s near %g" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l))
        c)
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let* c = of_list [ 0.; 1e3; 1e5 ] in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (map
            (fun d -> Int32.float_of_bits (Int32.bits_of_float (c +. d)))
            (float_range (-10.) 10.))
     in
     (s, axes, c, xs))

(* The bounds nx's expansion of Moments states, at float32's unit roundoff: the
   mean within [γ(n) Σ|x| / n], the variance [V] within [γ(n + 3) V + (1 + γ(n +
   3)) γ(n)² (Σ|x| / n)²]. The references are float64 two-pass sums, whose own
   error is the slack. *)
let law_moments (s, axes, _, xs) =
  let ax = normal (Array.length s) axes in
  let x = core (A.of_array D.Float32 s xs) in
  let mean, var = reduce1 (Moments (0, Nx.float32)) (Array.of_list ax) x in
  let mean = read mean and var = read var in
  let u = ldexp 1. (-24) in
  let gamma k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  Array.iteri
    (fun i ts ->
      let n = List.length ts in
      if n = 0 then begin
        cover "no term" true;
        equal ~msg:"mean" bool true (Float.is_nan mean.(i));
        equal ~msg:"variance" bool true (Float.is_nan var.(i))
      end
      else begin
        cover "terms" true;
        let nf = Float.of_int n in
        let m = List.fold_left ( +. ) 0. ts /. nf in
        let v =
          List.fold_left (fun a t -> a +. ((t -. m) *. (t -. m))) 0. ts /. nf
        in
        let abs_mean =
          List.fold_left (fun a t -> a +. Float.abs t) 0. ts /. nf
        in
        let slack =
          4. *. nf *. epsilon_float *. (v +. (abs_mean *. abs_mean))
        in
        at_most ~msg:"mean" float_exact
          ~than:((gamma n *. abs_mean) +. slack)
          (Float.abs (mean.(i) -. m));
        at_most ~msg:"variance" float_exact
          ~than:
            ((gamma (n + 3) *. v)
            +. (1. +. gamma (n + 3))
               *. gamma n *. gamma n *. abs_mean *. abs_mean
            +. slack)
          (Float.abs (var.(i) -. v))
      end)
    (terms s ax xs)

(* The extreme and the first position of its bits, by the reference's strict
   [better]. *)
let first_extreme better ts =
  let best = ref 0 in
  List.iteri (fun i t -> if better t (List.nth ts !best) then best := i) ts;
  (List.nth ts !best, Int64.of_int !best)

let law_arg (s, axes, _, xs) =
  let ax = normal (Array.length s) axes in
  if not (List.exists (fun a -> s.(a) = 0) ax) then begin
    let x = core (A.of_array D.Int32 s xs) in
    List.iter
      (fun (name, e, better) ->
        let v, p = reduce1 (Arg (e, 0, Nx.int32)) (Array.of_list ax) x in
        let want = Array.map (first_extreme better) (terms s ax xs) in
        equal ~msg:(name ^ " extremes") (array int32) (Array.map fst want)
          (read v);
        equal ~msg:(name ^ " positions") (array int64) (Array.map snd want)
          (read p))
      [
        ("max", (Max : S.extreme), fun t b -> Int32.compare t b > 0);
        ("min", Min, fun t b -> Int32.compare t b < 0);
      ]
  end

let vector dt xs = core (A.of_array dt [| Array.length xs |] xs)

let test_arg_cases () =
  let arg (type v s) e (dt : (v, s) D.t) axes shape (xs : v array) =
    let v, p = reduce1 (Arg (e, 0, dt)) axes (core (A.of_array dt shape xs)) in
    (read v, read p)
  in
  let f64 e xs =
    let v, p = arg e D.Float64 [| 0 |] [| Array.length xs |] xs in
    (bits v.(0), p.(0))
  in
  let found = pair int64 int64 in
  equal ~msg:"max of signed zeros: +0, the first one" found
    (bits 0., 1L)
    (f64 Max [| -0.; 0.; -0.; 0. |]);
  equal ~msg:"min of signed zeros: -0, the first one" found
    (bits (-0.), 0L)
    (f64 Min [| -0.; 0.; -0. |]);
  let a = nan_with 7L and b = nan_with 9L in
  equal ~msg:"a NaN is the extreme: the first, its bits" found
    (bits a, 1L)
    (f64 Max [| 1.; a; 5.; b |]);
  equal ~msg:"and for min" found (bits a, 1L) (f64 Min [| 1.; a; -5.; b |]);
  equal ~msg:"ties: the first" found (bits 7., 1L) (f64 Max [| 2.; 7.; 7. |]);
  let grid = [| 1.; 9.; 3.; 9.; 0.; 2. |] in
  equal ~msg:"over both axes: positions in C order of the reduced indices"
    (pair (array float_exact) (array int64))
    ([| 9. |], [| 1L |])
    (arg Max D.Float64 [| 0; 1 |] [| 2; 3 |] grid);
  equal ~msg:"along axis 0"
    (pair (array float_exact) (array int64))
    ([| 9.; 9.; 3. |], [| 1L; 0L; 0L |])
    (arg Max D.Float64 [| 0 |] [| 2; 3 |] grid);
  equal ~msg:"int4"
    (pair (array int) (array int64))
    ([| 7 |], [| 2L |])
    (arg Max D.Int4 [| 0 |] [| 4 |] [| 3; -8; 7; 7 |]);
  equal ~msg:"bool"
    (pair (array bool) (array int64))
    ([| true |], [| 1L |])
    (arg Max D.Bool [| 0 |] [| 3 |] [| false; true; true |]);
  equal ~msg:"float16 signed zeros" (array int64) [| 1L |]
    (snd (arg Max D.Float16 [| 0 |] [| 3 |] [| -0.; 0.; 0. |]));
  equal ~msg:"float4_e2m1fn signed zeros" (array int64) [| 2L |]
    (snd (arg Min D.Float4_e2m1fn [| 0 |] [| 3 |] [| 1.; 0.; -0. |]))

let test_moments_cases () =
  let mean, var =
    reduce1 (Moments (0, Nx.float64)) [| 0 |] (vector D.Float64 [||])
  in
  equal ~msg:"no term: NaN" (pair bool bool) (true, true)
    (Float.is_nan (read mean).(0), Float.is_nan (read var).(0));
  let a = nan_with 7L in
  let mean, var =
    reduce1
      (Moments (0, Nx.float64))
      [| 0 |]
      (vector D.Float64 [| 1.; a; nan_with 9L |])
  in
  equal ~msg:"a NaN term: the first, its bits" (pair int64 int64)
    (bits a, bits a)
    (bits (read mean).(0), bits (read var).(0));
  let mean, var =
    reduce1
      (Moments (0, Nx.float16))
      [| 0 |]
      (vector D.Float16 [| 1.; 2.; 3.; 4. |])
  in
  equal ~msg:"float16"
    (pair float_exact float_exact)
    (2.5, 1.25)
    ((read mean).(0), (read var).(0));
  (* 3000 terms cross a block of the sum's lane order. *)
  let xs = Array.init 3000 (fun i -> 1e4 +. Float.of_int (i mod 7)) in
  let mean, var =
    reduce1 (Moments (0, Nx.float64)) [| 0 |] (vector D.Float64 xs)
  in
  let m = Array.fold_left ( +. ) 0. xs /. 3000. in
  let v =
    Array.fold_left (fun a x -> a +. ((x -. m) *. (x -. m))) 0. xs /. 3000.
  in
  equal ~msg:"3000 terms"
    (pair (float 1e-9) (float 1e-9))
    (m, v)
    ((read mean).(0), (read var).(0))

let test_logsumexp_cases () =
  let lse (type v s) (dt : (v, s) D.t) (xs : v array) =
    read (reduce1 (Monoid (Logsumexp, 0, dt)) [| 0 |] (vector dt xs))
  in
  equal ~msg:"no term" int64 (bits Float.neg_infinity)
    (bits (lse D.Float64 [||]).(0));
  equal ~msg:"terms all -inf" int64 (bits Float.neg_infinity)
    (bits (lse D.Float64 [| Float.neg_infinity; Float.neg_infinity |]).(0));
  equal ~msg:"+inf beside -inf" int64 (bits Float.infinity)
    (bits (lse D.Float64 [| Float.neg_infinity; Float.infinity; 1. |]).(0));
  let wide = A.of_array D.Float32 [| 1 |] [| Float.log 2. |] in
  equal ~msg:"bfloat16 at float32, rounded once" (array float_exact)
    (A.to_array (cast D.Bfloat16 wide))
    (lse D.Bfloat16 [| 0.; 0. |])

let test_optional_rules () =
  let x = vector D.Int32 [| 1l; 2l |] in
  raises (Invalid_argument "t: Moments does not take int32") (fun () ->
      reduce1 (Moments (0, Nx.int32)) [| 0 |] x);
  raises (Invalid_argument "t: Arg Max of no term") (fun () ->
      reduce1 (Arg (Max, 0, Nx.int32)) [| 0 |] (vector D.Int32 [||]))

let optional_group =
  group "optional reductions"
    [
      prop "logsumexp is m + log Σ exp (x - m), its specials exact" float64_case
        law_logsumexp;
      prop "moments are within their stated bounds" moments_case law_moments;
      prop "arg is the extreme and its first position" int32_case law_arg;
      test "arg's cases" test_arg_cases;
      test "moments' cases" test_moments_cases;
      test "logsumexp's cases" test_logsumexp_cases;
      test "their rules raise naming the caller" test_optional_rules;
    ]

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

let () = exit (run "nx reduce" [ reductions_group; optional_group ])
