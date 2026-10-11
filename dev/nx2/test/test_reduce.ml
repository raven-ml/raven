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
    Nx_kernel.monoid -> (v, s) A.t -> int array -> (v, s) A.t option =
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
  | D.Complex64, Nx_kernel.Sum -> pairs D.Float32
  | D.Complex128, Nx_kernel.Sum -> pairs D.Float64
  | _ -> (
      let s = L.shape (A.layout a) in
      let out =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let dst = A.create Rig.host dt out in
      match
        Nx_cpu.reduce (identity dt) ~axes
          [| (Monoid m, 0) |]
          ~dsts:[| A.Any dst |]
          [| Plain a |]
      with
      | A.Done -> Some dst
      | _ -> None)

(* What each reduction's doc states it takes. *)
let documented (type v s) (m : Nx_kernel.monoid) (dt : (v, s) D.t) =
  match (D.kind dt, m) with
  | (D.Float | D.Signed | D.Unsigned), _ -> true
  | D.Boolean, (Max | Min) -> true
  | D.Complex, Sum -> true
  | D.Boolean, (Sum | Prod | Logsumexp)
  | D.Complex, (Prod | Max | Min | Logsumexp) ->
      false

let reductions =
  [
    ("sum", (Sum : Nx_kernel.monoid), fun ?axes x -> Nx.sum ?axes x);
    ("prod", Prod, fun ?axes x -> Nx.prod ?axes x);
    ("max", Max, fun ?axes x -> Nx.max ?axes x);
    ("min", Min, fun ?axes x -> Nx.min ?axes x);
  ]

let law_any (Case a, axes) =
  let dt = A.dtype a in
  let s = L.shape (A.layout a) in
  let ax = Array.of_list (normal (Array.length s) axes) in
  let empty = Array.exists (fun i -> s.(i) = 0) ax in
  List.iter
    (fun (name, m, f) ->
      let f () = bits_of (host_array (f ?axes (value a))) in
      let extreme =
        match (m : Nx_kernel.monoid) with Max | Min -> true | _ -> false
      in
      if documented m dt && not (empty && extreme) then begin
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

  let core : Nx_kernel.reduction -> bool = function
    | Monoid (Sum | Prod | Max | Min) -> true
    | _ -> false

  let reduce p ~axes rs ~dsts loads =
    match rs with
    | [| (r, _) |] when core r -> Nx_cpu.reduce p ~axes rs ~dsts loads
    | _ -> A.Declined

  let scan p ~axis ((r, _) as rk) ~dsts loads =
    if core r then Nx_cpu.scan p ~axis rk ~dsts loads else A.Declined
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

(* Moments scans: by the expansion on a set whose kernels decline them, and
   by nx.cpu on the host, each within its stated bounds of float64 two-pass
   references over each prefix. *)

let scan1 r axis x =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Nx.shape x);
         axis;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
       })

(* Where a Moments scan computes: by the expansion on {!Core}, by nx.cpu on
   the host. *)
type path = Expansion | Cpu

let path_name = function Expansion -> "expansion" | Cpu -> "nx.cpu"

(* The Moments scan of [a] along [axis] by [path], read. *)
let scan_moments (type v s) path axis (a : (v, s) A.t) =
  let dt = A.dtype a in
  match path with
  | Expansion ->
      let m, v = scan1 (Moments (0, dt)) axis (core a) in
      (read m, read v)
  | Cpu ->
      let m, v = scan1 (Moments (0, dt)) axis (value a) in
      (read m, read v)

(* float32 terms near an offset [c], as {!moments_case}'s, along one axis. *)
let moments_scan_case =
  let open Gen in
  with_pp
    (fun ppf (s, axis, c, _) ->
      Format.fprintf ppf "%a along %d near %g" pp_ints s axis c)
    (let* s = shape in
     let s = if Array.length s = 0 then [| 3 |] else s in
     let* axis = int_range 0 (Array.length s - 1) in
     let* c = of_list [ 0.; 1e3; 1e5 ] in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (map
            (fun d -> Int32.float_of_bits (Int32.bits_of_float (c +. d)))
            (float_range (-10.) 10.))
     in
     (s, axis, c, xs))

(* Each prefix of each slice of [s] along [axis]: its terms' positions in C
   order, the last one the result's. *)
let prefixes s axis =
  let outer = Array.mapi (fun a n -> if a = axis then 1 else n) s in
  List.concat_map
    (fun idx ->
      let at j =
        let i = Array.copy idx in
        i.(axis) <- j;
        Array.fold_left (fun p (k, n) -> (p * n) + k) 0
          (Array.map2 (fun k n -> (k, n)) i s)
      in
      List.init s.(axis) (fun j -> List.init (j + 1) at))
    (indices outer)

(* The expansion's bounds: the mean within [γ(n + 2) Σ|x - c| / n + u|μ|], the
   variance [V] within [γ(3n + 7) (V + (μ - c)²)], [c] the slice's first
   term; nx.cpu's: the mean within [(n + 9) u X / 2], the variance within
   [(2n + 21) u X²], [X] the terms' largest magnitude. *)
let law_moments_scan (s, axis, _, xs) =
  let x = A.of_array D.Float32 s xs in
  let u = ldexp 1. (-24) in
  let gamma k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  let expansion n ~m ~v ~c ~spread ~big:_ =
    let nf = Float.of_int n in
    ( (gamma (n + 2) *. spread /. nf) +. (u *. Float.abs m),
      gamma ((3 * n) + 7) *. (v +. ((m -. c) *. (m -. c))) )
  in
  let welford n ~m:_ ~v:_ ~c:_ ~spread:_ ~big =
    let nf = Float.of_int n in
    ((nf +. 9.) /. 2. *. u *. big, ((2. *. nf) +. 21.) *. u *. big *. big)
  in
  List.iter
    (fun (path, bound) ->
      let by = path_name path in
      let mean, var = scan_moments path axis x in
      List.iter
        (fun ps ->
          let ts = List.map (fun p -> xs.(p)) ps in
          let at = List.nth ps (List.length ps - 1) in
          let n = List.length ts in
          let nf = Float.of_int n in
          let m = List.fold_left ( +. ) 0. ts /. nf in
          let v =
            List.fold_left (fun a t -> a +. ((t -. m) *. (t -. m))) 0. ts /. nf
          in
          let c = List.hd ts in
          let spread =
            List.fold_left (fun a t -> a +. Float.abs (t -. c)) 0. ts
          in
          let big =
            List.fold_left (fun a t -> Float.max a (Float.abs t)) 0. ts
          in
          let bm, bv = bound n ~m ~v ~c ~spread ~big in
          let slack = 4. *. nf *. epsilon_float *. (v +. (big *. big)) in
          cover "one term" (n = 1);
          cover "terms" (n > 1);
          at_most ~msg:(by ^ " mean") float_exact ~than:(bm +. slack)
            (Float.abs (mean.(at) -. m));
          at_most ~msg:(by ^ " variance") float_exact ~than:(bv +. slack)
            (Float.abs (var.(at) -. v)))
        (prefixes s axis))
    [ (Expansion, expansion); (Cpu, welford) ]

(* Both paths: no result along an empty axis; one term's own mean and a
   variance of [+0]; a NaN's bits from it on; an infinity's mean and a NaN
   variance; a narrow float at float32, rounded once. *)
let test_moments_scan_cases () =
  let both name f =
    List.iter (fun p -> f (name ^ ", " ^ path_name p) p) [ Expansion; Cpu ]
  in
  let moments p dt s xs =
    scan_moments p (Array.length s - 1) (A.of_array dt s xs)
  in
  both "an empty axis" (fun msg p ->
      let mean, var = moments p D.Float64 [| 2; 0 |] [||] in
      equal ~msg (pair int int) (0, 0) (Array.length mean, Array.length var));
  both "one term" (fun msg p ->
      let mean, var = moments p D.Float64 [| 3; 1 |] [| 2.5; -0.; 1e300 |] in
      equal ~msg (pair (array float_exact) (array float_exact))
        ([| 2.5; 0.; 1e300 |], [| 0.; 0.; 0. |])
        (mean, var));
  both "a NaN" (fun msg p ->
      let a = nan_with 7L in
      let xs = [| 1.; a; nan_with 9L; 2. |] in
      let mean, var = moments p D.Float64 [| 4 |] xs in
      equal ~msg
        (pair (array int64) (array int64))
        ( [| bits 1.; bits a; bits a; bits a |],
          [| bits 0.; bits a; bits a; bits a |] )
        (Array.map bits mean, Array.map bits var));
  both "infinities" (fun msg p ->
      let xs = [| 1.; Float.infinity; 2.; Float.neg_infinity |] in
      let mean, var = moments p D.Float64 [| 4 |] xs in
      equal ~msg:(msg ^ ", mean") (array float_exact)
        [| 1.; Float.infinity; Float.infinity |]
        (Array.sub mean 0 3);
      equal ~msg:(msg ^ ", NaNs") (list bool)
        [ true; true; true; true ]
        (List.map Float.is_nan [ mean.(3); var.(1); var.(2); var.(3) ]));
  both "float16" (fun msg p ->
      let mean, var = moments p D.Float16 [| 4 |] [| 1.; 2.; 3.; 4. |] in
      let want = A.of_array D.Float32 [| 4 |] [| 0.; 0.25; 2. /. 3.; 1.25 |] in
      equal ~msg (pair (array float_exact) (array float_exact))
        ([| 1.; 1.5; 2.; 2.5 |], A.to_array (cast D.Float16 want))
        (mean, var))

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
        ("max", (Max : Nx_kernel.extreme), fun t b -> Int32.compare t b > 0);
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

(* Logsumexp and Arg scans: by the expansion on {!Core} and on the host, whose
   kernels decline them both, against references over each prefix. *)

(* [a] by [on], read through a transposed view where [strided]: the elements
   of [a], held in a layout that runs its axes backwards. *)
let transposed on strided a =
  let s = L.shape (A.layout a) in
  let r = Array.length s in
  if (not strided) || r < 2 then on a
  else
    let back = Array.init r (fun i -> s.(r - 1 - i)) in
    let xs = A.to_array a in
    let at idx =
      Array.fold_left (fun p a -> (p * s.(a)) + idx.(r - 1 - a)) 0
        (Array.init r Fun.id)
    in
    let held = Array.of_list (List.map (fun i -> xs.(at i)) (indices back)) in
    Nx.transpose (on (A.of_array (A.dtype a) back held))

(* [f] on {!Core} then on the host, given the path's name and its placement of
   an array. *)
let each_path
    (f :
      'd.
      string -> ('v 's. bool -> ('v, 's) A.t -> ('v, 's, 'd) Nx.t) -> unit) =
  f "expansion" (fun strided a -> transposed core strided a);
  f "nx.cpu" (fun strided a -> transposed value strided a)

(* Each element's bits, as the integers of its bytes. *)
let element_bits (type v s) (a : (v, s) A.t) =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let b = bits_of a in
  Array.init (Array.length b / w) (fun i -> Array.sub b (i * w) w)

(* A shape of rank at least one and an axis of it, at times long enough for
   several passes of a doubling. *)
let along =
  let open Gen in
  let* s = shape in
  let s = if Array.length s = 0 then [| 3 |] else s in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ n = frequency [ (3, constant s.(axis)); (1, int_range 5 40) ] in
  let s = Array.copy s in
  s.(axis) <- n;
  (s, axis)

(* Covers what a scan's law needs: a long axis, a strided view. *)
let cover_scan s axis strided =
  cover "an axis of more than 4" (s.(axis) > 4);
  cover "transposed" (strided && Array.length s > 1)

type float_case =
  | Float_case :
      (float, 's) D.t * int array * int * bool * float array
      -> float_case

(* A float dtype and terms of it along an axis, spread over a range wide
   enough that a slice's early prefixes sit far below its later maximum, with
   infinities and NaNs, read through a transposed view or not. *)
let float_scan_case =
  let open Gen in
  let floats (type s) (dt : (float, s) D.t) wide =
    let element =
      frequency
        [
          (12, float_range (-.wide) wide);
          (6, float_range (-3.) 3.);
          (1, constant (-0.));
          (1, constant Float.neg_infinity);
          (1, constant Float.infinity);
          (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
        ]
    in
    let* s, axis = along in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Float_case (dt, s, axis, strided, xs)
  in
  with_pp
    (fun ppf (Float_case (dt, s, axis, strided, _)) ->
      Format.fprintf ppf "%a %a along %d%s" D.pp dt pp_ints s axis
        (if strided then ", transposed" else ""))
    (one_of
       [
         floats D.Float64 1000.;
         floats D.Float32 1000.;
         floats D.Float16 1000.;
         floats D.Bfloat16 1000.;
         floats D.Float8_e4m3fn 100.;
         floats D.Float8_e5m2 1000.;
         floats D.Float4_e2m1fn 4.;
       ])

(* [v] rounded to [dt], as a store rounds. *)
let rounded (type s) (dt : (float, s) D.t) v =
  (A.to_array (A.of_array dt [| 1 |] [| v |])).(0)

(* The expansion's bound: at [n] terms of largest finite magnitude [X], within
   [⌈log₂ n⌉ (u (X + ln n + 1) + ε)] in the accumulator, [ε] the error [Exp]
   and [Log1p] add, at most [(2 + 1/2) 2u] and [(2 + 1/2) 2u ln 2]; then
   rounded once, so between the roundings of the bound's ends. The reference,
   float64 over each prefix from its maximum, adds its own [2n] units. *)
let law_logsumexp_scan (Float_case (dt, s, axis, strided, xs)) =
  let u = match dt with D.Float64 -> ldexp 1. (-53) | _ -> ldexp 1. (-24) in
  let a = A.of_array dt s xs in
  cover_scan s axis strided;
  each_path (fun by place ->
      let x = place strided a in
      let ts = read x and tb = element_bits (host_array x) in
      let y = scan1 (Monoid (Logsumexp, 0, dt)) axis x in
      let got = read y and gb = element_bits (host_array y) in
      List.iter
        (fun ps ->
          let at = List.nth ps (List.length ps - 1) in
          let terms = List.map (fun p -> ts.(p)) ps in
          let g = got.(at) in
          match List.find_opt (fun p -> Float.is_nan ts.(p)) ps with
          | Some p ->
              cover "a NaN term" true;
              equal
                ~msg:(by ^ ": the first NaN, its bits")
                (array int) tb.(p) gb.(at)
          | None ->
              let m = List.fold_left Float.max Float.neg_infinity terms in
              if not (Float.is_finite m) then begin
                cover "an infinite maximum" true;
                equal ~msg:(by ^ ": an infinity") float_exact m g
              end
              else begin
                cover "finite" true;
                cover "a term far below the maximum"
                  (List.exists (fun t -> t < m -. 100.) terms);
                let n = Float.of_int (List.length terms) in
                let sum =
                  List.fold_left (fun a t -> a +. exp (t -. m)) 0. terms
                in
                let want = m +. log sum in
                let big =
                  List.fold_left
                    (fun a t ->
                      if Float.is_finite t then Float.max a (Float.abs t)
                      else a)
                    0. terms
                in
                let passes = Float.ceil (Float.log2 n) in
                let bound =
                  (passes *. u *. (big +. log n +. 1. +. 8.5))
                  +. (2. *. n *. epsilon_float *. (Float.abs want +. 1.))
                in
                let msg = Printf.sprintf "%s: %h against %h" by g want in
                at_least ~msg float_exact ~than:(rounded dt (want -. bound)) g;
                at_most ~msg float_exact ~than:(rounded dt (want +. bound)) g
              end)
        (prefixes s axis))

let test_logsumexp_scan_cases () =
  each_path (fun by place ->
      let lse (type s) (dt : (float, s) D.t) s xs =
        let x = place false (A.of_array dt s xs) in
        read (scan1 (Monoid (Logsumexp, 0, dt)) (Array.length s - 1) x)
      in
      let msg m = m ^ ", " ^ by in
      let inf = Float.infinity and ninf = Float.neg_infinity in
      equal ~msg:(msg "an empty axis") int 0
        (Array.length (lse D.Float64 [| 2; 0 |] [||]));
      equal
        ~msg:(msg "one term: itself, -0 as +0")
        (array int64)
        (Array.map bits [| 2.5; 0.; ninf |])
        (Array.map bits (lse D.Float64 [| 3; 1 |] [| 2.5; -0.; ninf |]));
      let ln2 = Float.log 2. in
      equal
        ~msg:(msg "prefixes far below the later maximum, float32")
        (array (float 1e-3))
        [| -1000.; -1000. +. ln2; 0.; 1. +. Float.log1p (exp (-1.)) |]
        (lse D.Float32 [| 4 |] [| -1000.; -1000.; 0.; 1. |]);
      equal
        ~msg:(msg "prefixes far below the later maximum, float64")
        (array (float 1e-12))
        [| -2000.; -2000. +. ln2; 0. |]
        (lse D.Float64 [| 3 |] [| -2000.; -2000.; 0. |]);
      let a = nan_with 7L in
      equal
        ~msg:(msg "a NaN: the first, its bits, from it on")
        (array int64)
        [| bits 1.; bits a; bits a; bits a |]
        (Array.map bits
           (lse D.Float64 [| 4 |] [| 1.; a; nan_with 9L; inf |]));
      equal ~msg:(msg "infinities") (array float_exact)
        [| ninf; ninf; 1.; inf; inf |]
        (lse D.Float64 [| 5 |] [| ninf; ninf; 1.; inf; ninf |]);
      let wide = A.of_array D.Float32 [| 2 |] [| 0.; ln2 |] in
      equal
        ~msg:(msg "float16 at float32, rounded once")
        (array float_exact)
        (A.to_array (cast D.Float16 wide))
        (lse D.Float16 [| 2 |] [| 0.; 0. |]))

type arg_case =
  | Arg_case : {
      dt : ('v, 's) D.t;
      key : 'v -> float;
      s : int array;
      axis : int;
      strided : bool;
      xs : 'v array;
    }
      -> arg_case

(* Terms of a dtype Arg takes along an axis, from few values so that ties are
   common, with signed zeros, infinities and NaNs where the dtype has them,
   read through a transposed view or not. [key] orders them as Max does. *)
let arg_scan_case =
  let open Gen in
  let make (type v s) (dt : (v, s) D.t) (key : v -> float) (element : v Gen.t)
      =
    let* s, axis = along in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Arg_case { dt; key; s; axis; strided; xs }
  in
  let float =
    frequency
      [
        (8, map Float.of_int (int_range (-3) 3));
        (1, constant (-0.));
        (1, constant Float.infinity);
        (1, constant Float.neg_infinity);
        (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
      ]
  in
  let extremes small least most =
    frequency
      [ (6, map small (int_range (-3) 3)); (1, of_list [ least; most ]) ]
  in
  with_pp
    (fun ppf (Arg_case { dt; s; axis; strided; _ }) ->
      Format.fprintf ppf "%a %a along %d%s" D.pp dt pp_ints s axis
        (if strided then ", transposed" else ""))
    (one_of
       [
         make D.Float64 Fun.id float;
         make D.Float32 Fun.id float;
         make D.Float16 Fun.id float;
         make D.Int32 Int32.to_float
           (extremes Int32.of_int Int32.min_int Int32.max_int);
         make D.Int64 Int64.to_float
           (extremes Int64.of_int Int64.min_int Int64.max_int);
         make D.Uint8 Float.of_int (of_list [ 0; 1; 2; 255 ]);
         make D.Int4 Float.of_int (int_range (-8) 7);
         make D.Bool (fun b -> if b then 1. else 0.) bool;
       ])

(* Whether [t] is past [b] in Max's order, or Min's where [flip]: [-0] below
   [+0], and a NaN past every number either way. *)
let beats flip t b =
  let nan = Float.is_nan in
  if nan t || nan b then nan t && not (nan b)
  else
    let c =
      if t = b then Bool.compare (Float.sign_bit b) (Float.sign_bit t)
      else Float.compare t b
    in
    if flip then c < 0 else c > 0

let law_arg_scan (Arg_case { dt; key; s; axis; strided; xs }) =
  let a = A.of_array dt s xs in
  cover_scan s axis strided;
  each_path (fun by place ->
      let x = place strided a in
      let keys = Array.map key (read x) and tb = element_bits (host_array x) in
      List.iter
        (fun (name, e, flip) ->
          let by = by ^ " " ^ name in
          let v, p = scan1 (Arg (e, 0, dt)) axis x in
          let vb = element_bits (host_array v) and p = read p in
          List.iter
            (fun ps ->
              let at = List.nth ps (List.length ps - 1) in
              let _, want =
                first_extreme (beats flip) (List.map (fun i -> keys.(i)) ps)
              in
              let first = List.nth ps (Int64.to_int want) in
              cover "a tie"
                (List.exists (fun i -> i <> first && tb.(i) = tb.(first)) ps);
              equal ~msg:(by ^ ": position") int64 want p.(at);
              equal
                ~msg:(by ^ ": the extreme's bits")
                (array int) tb.(first) vb.(at))
            (prefixes s axis))
        [ ("max", (Max : Nx_kernel.extreme), false); ("min", Min, true) ])

let test_arg_scan_cases () =
  each_path (fun by place ->
      let msg m = m ^ ", " ^ by in
      let arg (type v s) e (dt : (v, s) D.t) s (xs : v array) =
        let x = place false (A.of_array dt s xs) in
        let v, q = scan1 (Arg (e, 0, dt)) (Array.length s - 1) x in
        (read v, read q)
      in
      let f64 e xs =
        let v, q = arg e D.Float64 [| Array.length xs |] xs in
        (Array.map bits v, q)
      in
      let found = pair (array int64) (array int64) in
      let v, q = arg Max D.Int32 [| 2; 0 |] [||] in
      equal ~msg:(msg "an empty axis") (pair int int) (0, 0)
        (Array.length v, Array.length q);
      equal
        ~msg:(msg "max of signed zeros: +0, the first one")
        found
        (Array.map bits [| -0.; 0.; 0.; 0. |], [| 0L; 1L; 1L; 1L |])
        (f64 Max [| -0.; 0.; -0.; 0. |]);
      equal
        ~msg:(msg "min of signed zeros: -0, the first one")
        found
        (Array.map bits [| 0.; -0.; -0.; -0. |], [| 0L; 1L; 1L; 1L |])
        (f64 Min [| 0.; -0.; 0.; -0. |]);
      let a = nan_with 7L and b = nan_with 9L in
      let first_nan =
        (Array.map bits [| 1.; a; a; a |], [| 0L; 1L; 1L; 1L |])
      in
      equal
        ~msg:(msg "a NaN is the max: the first, its bits")
        found first_nan
        (f64 Max [| 1.; a; 5.; b |]);
      equal
        ~msg:(msg "and the min")
        found first_nan
        (f64 Min [| 1.; a; -5.; b |]);
      equal ~msg:(msg "ties: the first") found
        (Array.map bits [| 2.; 7.; 7. |], [| 0L; 1L; 1L |])
        (f64 Max [| 2.; 7.; 7. |]);
      let grid = [| 1.; 9.; 3.; 9.; 0.; 2. |] in
      let v, q =
        scan1
          (Arg (Max, 0, Nx.float64))
          0
          (place false (A.of_array D.Float64 [| 2; 3 |] grid))
      in
      equal ~msg:(msg "along axis 0")
        (pair (array float_exact) (array int64))
        ([| 1.; 9.; 3.; 9.; 9.; 3. |], [| 0L; 0L; 0L; 1L; 0L; 0L |])
        (read v, read q);
      equal ~msg:(msg "int4")
        (pair (array int) (array int64))
        ([| 3; 3; 7; 7 |], [| 0L; 0L; 2L; 2L |])
        (arg Max D.Int4 [| 4 |] [| 3; -8; 7; 7 |]);
      equal ~msg:(msg "bool")
        (pair (array bool) (array int64))
        ([| false; true; true |], [| 0L; 1L; 1L |])
        (arg Max D.Bool [| 3 |] [| false; true; true |]))

let test_optional_rules () =
  let x = vector D.Int32 [| 1l; 2l |] in
  raises (Invalid_argument "t: Logsumexp does not take int32") (fun () ->
      scan1 (Monoid (Logsumexp, 0, Nx.int32)) 0 x);
  raises (Invalid_argument "t: Moments does not take int32") (fun () ->
      reduce1 (Moments (0, Nx.int32)) [| 0 |] x);
  raises (Invalid_argument "t: Moments does not take int32") (fun () ->
      scan1 (Moments (0, Nx.int32)) 0 x);
  raises (Invalid_argument "t: Arg Max of no term") (fun () ->
      reduce1 (Arg (Max, 0, Nx.int32)) [| 0 |] (vector D.Int32 [||]))

let optional_group =
  group "optional reductions"
    [
      prop "logsumexp is m + log Σ exp (x - m), its specials exact" float64_case
        law_logsumexp;
      prop "moments are within their stated bounds" moments_case law_moments;
      prop "moment scans are within their stated bounds" moments_scan_case
        law_moments_scan;
      prop "arg is the extreme and its first position" int32_case law_arg;
      test "arg's cases" test_arg_cases;
      test "moments' cases" test_moments_cases;
      test "moment scans' cases" test_moments_scan_cases;
      test "logsumexp's cases" test_logsumexp_cases;
      prop "logsumexp scans are within their stated bound, specials exact"
        float_scan_case law_logsumexp_scan;
      test "logsumexp scans' cases" test_logsumexp_scan_cases;
      prop "arg scans are each prefix's extreme and its first position"
        arg_scan_case law_arg_scan;
      test "arg scans' cases" test_arg_scan_cases;
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
