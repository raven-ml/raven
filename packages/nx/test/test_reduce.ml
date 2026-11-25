(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's reductions and scans on the host, in nx.mli's order. Over int32 each is
   the fold of its elements, an order-free reference; at every dtype each
   computes the dtypes its doc states as nx.cpu's kernel does over the same axes
   (the narrower floats at float32, rounded once, a NaN result any NaN) and
   raises naming itself elsewhere; a NaN term makes every result it reaches a
   NaN. Then each function's own values, laws and bounds: the float bounds
   against references far more exact than them, a compensated float64 sum for
   float32 and correctly rounded rows of gen/special.py for float64. Variance
   and standard deviation are test_spread's. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

type host = Nx.host

let value a : (_, _, host) Nx.t = Nx.Repr.of_array Nx.Host.v a
let host_array x = Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x))
let read x = A.to_array (host_array x)
let names name f = raises_match (Exn.invalid_arg ~sub:("Nx." ^ name ^ ": ")) f
let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs
let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs
let item x = Nx.item [] x
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

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

let pp_axes ppf = function
  | None -> Format.pp_print_string ppf "all"
  | Some l ->
      Format.pp_print_string ppf (String.concat "," (List.map string_of_int l))

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

(* Each result's terms, in C order of the results and of the reduced indices. *)
let terms s axes xs =
  Array.map List.rev (fold (fun acc x -> x :: acc) [] s axes xs)

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

(* int32: the fold of the elements *)

let int32_case =
  let open Gen in
  with_pp
    (fun ppf (s, axes, keepdims, _) ->
      Format.fprintf ppf "%a over %a%s" pp_ints s pp_axes axes
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

(* An empty reduced axis, which a draw reaches about one case in seven. *)
let int32_examples = [ ([| 2; 0 |], Some [ 1 ], false, [||]) ]

let law_int32 (s, axes, keepdims, xs) =
  let r = Array.length s in
  let ax = normal r axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  cover "kept" keepdims;
  let x = value (A.of_array D.Int32 (Iarray.of_array s) xs) in
  let check name f init g =
    let y = host_array (g ?axes ~keepdims x) in
    equal ~msg:(name ^ ": shape") (array int)
      (if keepdims then kept s ax else reduced s ax)
      (Iarray.to_array (L.shape (A.layout y)));
    equal ~msg:name (array int32) (fold f init s ax xs) (A.to_array y)
  in
  check "sum" Int32.add 0l (fun ?axes ~keepdims x -> Nx.sum ?axes ~keepdims x);
  check "prod" Int32.mul 1l (fun ?axes ~keepdims x -> Nx.prod ?axes ~keepdims x);
  let elements = Array.fold_left ( * ) 1 (reduced s ax) in
  if List.exists (fun a -> s.(a) = 0) ax && elements > 0 then begin
    names "max" (fun () -> Nx.max ?axes x);
    names "min" (fun () -> Nx.min ?axes x)
  end
  else begin
    check "max" Stdlib.max Int32.min_int (fun ?axes ~keepdims x ->
        Nx.max ?axes ~keepdims x);
    check "min" Stdlib.min Int32.max_int (fun ?axes ~keepdims x ->
        Nx.min ?axes ~keepdims x)
  end

(* Each scan in C order, and along the first axis drawn, or the last. *)
let law_int32_scan (s, axes, _, xs) =
  let r = Array.length s in
  let n = Array.length xs in
  let axis = match axes with Some (a :: _) -> a | _ -> -1 in
  cover "a negative axis" (axis < 0 && r > 0);
  cover "an axis other than the last" (r > 1 && (axis + r) mod r <> r - 1);
  let x = value (A.of_array D.Int32 (Iarray.of_array s) xs) in
  List.iter
    (fun (name, f, (g : ?axis:int -> _ -> _)) ->
      let flat = host_array (g x) in
      equal ~msg:(name ^ ": shape") (array int) s
        (Iarray.to_array (L.shape (A.layout flat)));
      equal ~msg:(name ^ " in C order") (array int32) (running f [| n |] 0 xs)
        (A.to_array flat);
      if r > 0 then
        equal
          ~msg:(Printf.sprintf "%s along %d" name axis)
          (array int32)
          (running f s ((axis + r) mod r) xs)
          (read (g ~axis x)))
    [
      ("cumsum", Int32.add, fun ?axis x -> Nx.cumsum ?axis x);
      ("cumprod", Int32.mul, fun ?axis x -> Nx.cumprod ?axis x);
      ("cummax", Stdlib.max, fun ?axis x -> Nx.cummax ?axis x);
      ("cummin", Stdlib.min, fun ?axis x -> Nx.cummin ?axis x);
    ]

(* Every dtype: nx.cpu's kernel *)

type case = Case : ('v, 's) A.t -> case

let dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

(* An array of [dt] of shape [s] from drawn bytes, a byte zero one time in [1 +
   zeros], booleans 0 or 1. *)
let drawn ?(zeros = 0) (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data =
    string_of
      ~size:(constant (D.bytes dt n))
      (frequency [ (1, char); (zeros, constant '\000') ])
  in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  A.v dt
    (L.contiguous (Iarray.of_array s))
    (Rig.Buffer.of_string (if data = "" then "\000" else data))

let pp_case ppf (Case a, axes) =
  Format.fprintf ppf "%a %a over %a" D.pp (A.dtype a) pp_ints
    (Iarray.to_array (L.shape (A.layout a)))
    pp_axes axes

let any_case ?zeros () =
  let open Gen in
  with_pp pp_case
    (let* (D.Any dt) = dtype in
     let* s = shape in
     let* a = drawn ?zeros dt s in
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
  match Nx_kernels_support.map1 (module Nx_cpu) Cast ~dst a with
  | A.Done -> dst
  | _ -> failwith "the host's kernel declined a cast"

(* Each element's bits, as the integers of its bytes. *)
let element_bits (type v s) (a : (v, s) A.t) =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let b = bits_of a in
  Array.init (Array.length b / w) (fun i -> Array.sub b (i * w) w)

(* The NaN-ness of each part of each element of [a]: one part for a real dtype,
   two for a complex one, none NaN for integers and booleans. *)
let nan_parts (type v s) (a : (v, s) A.t) : bool array array =
  match D.kind (A.dtype a) with
  | D.Float ->
      Array.map (fun x -> [| Float.is_nan x |]) (A.to_array (cast D.Float64 a))
  | D.Complex ->
      Array.map
        (fun (c : Complex.t) -> [| Float.is_nan c.re; Float.is_nan c.im |])
        (A.to_array (cast D.Complex128 a))
  | D.Signed | D.Unsigned | D.Boolean ->
      Array.make (L.numel (A.layout a)) [| false |]

let nans a = Array.map (Array.exists Fun.id) (nan_parts a)

(* Each element's bits, or [None] for a NaN: a NaN result promises a NaN, never
   which one. Under [~parts], a complex number's parts are each so, as a sum
   adds them apart. *)
let rec blind : type v s. ?parts:bool -> (v, s) A.t -> int array option array =
 fun ?(parts = false) a ->
  match A.dtype a with
  | D.Complex64 when parts ->
      blind (Option.get (A.bitcast D.Float32 (A.copy a)))
  | D.Complex128 when parts ->
      blind (Option.get (A.bitcast D.Float64 (A.copy a)))
  | _ ->
      let n = nans a in
      Array.mapi (fun i e -> if n.(i) then None else Some e) (element_bits a)

let blind_bits = array (option (array int))
let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

(* nx.cpu's reduction [m] of [a] along [axes], or its scan along [axes]'s one
   axis where [scan]: at float32 for the floats narrower than 32 bits, at the
   byte-wide dtype for the sub-byte ones, rounded back once; a complex sum over
   the parts. [None] where the kernel declines. *)
let rec kernel : type v s.
    scan:bool ->
    Nx_kernel.monoid ->
    (v, s) A.t ->
    int array ->
    (v, s) A.t option =
 fun ~scan m a axes ->
  let dt = A.dtype a in
  let through (type w r) (acc : (w, r) D.t) =
    Option.map (cast dt) (kernel ~scan m (cast acc a) axes)
  in
  let pairs (type w r) (part : (w, r) D.t) =
    Option.map
      (fun p -> A.copy (Option.get (A.bitcast dt (A.copy p))))
      (kernel ~scan m (Option.get (A.bitcast part (A.copy a))) axes)
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
        if scan then s
        else
          Iarray.of_array
            (Array.of_list
               (List.filteri
                  (fun i _ -> not (Array.mem i axes))
                  (Iarray.to_list s)))
      in
      let dst = A.create Rig.host dt out in
      let answer =
        if scan then
          Nx_cpu.scan (identity dt) ~axis:axes.(0) (Monoid m, 0)
            ~dsts:[| A.Any dst |] [| Plain a |]
        else
          Nx_cpu.reduce (identity dt) ~shape:s ~axes:(Iarray.of_array axes)
            [| (Monoid m, 0) |]
            ~dsts:[| A.Any dst |] [| Plain a |]
      in
      match answer with A.Done -> Some dst | _ -> None)

(* What each reduction's doc states it takes. *)
let documented (type v s) (m : Nx_kernel.monoid) (dt : (v, s) D.t) =
  match (D.kind dt, m) with
  | (D.Float | D.Signed | D.Unsigned | D.Complex), _ -> true
  | D.Boolean, (Max | Min) -> true
  | D.Boolean, (Sum | Prod) -> false

let reductions =
  [
    ("sum", (Sum : Nx_kernel.monoid), fun ?axes x -> Nx.sum ?axes x);
    ("prod", Prod, fun ?axes x -> Nx.prod ?axes x);
    ("max", Max, fun ?axes x -> Nx.max ?axes x);
    ("min", Min, fun ?axes x -> Nx.min ?axes x);
  ]

let scans =
  [
    ("cumsum", (Sum : Nx_kernel.monoid), fun ~axis x -> Nx.cumsum ~axis x);
    ("cumprod", Prod, fun ~axis x -> Nx.cumprod ~axis x);
    ("cummax", Max, fun ~axis x -> Nx.cummax ~axis x);
    ("cummin", Min, fun ~axis x -> Nx.cummin ~axis x);
  ]

(* A boolean sum, which a draw reaches about one case in eleven. *)
let refused_example =
  [ (Case (A.of_array D.Bool [| 2 |] [| true; false |]), None) ]

let law_host (Case a, axes) =
  let dt = A.dtype a in
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let ax = Array.of_list (normal (Array.length s) axes) in
  (* An element of the result of no term: an extreme there raises. *)
  let no_term =
    Array.exists (fun i -> s.(i) = 0) ax
    && Array.fold_left ( * ) 1 (reduced s (Array.to_list ax)) > 0
  in
  let computes (m : Nx_kernel.monoid) =
    let extreme = match m with Max | Min -> true | Sum | Prod -> false in
    documented m dt && not (no_term && extreme)
  in
  cover "computed" (List.exists (fun (_, m, _) -> computes m) reductions);
  cover "refused" (List.exists (fun (_, m, _) -> not (computes m)) reductions);
  List.iter
    (fun (name, m, f) ->
      let f () = host_array (f ?axes (value a)) in
      let parts = m = Nx_kernel.Sum in
      if computes m then
        match kernel ~scan:false m a ax with
        | Some want ->
            equal ~msg:name blind_bits (blind ~parts want) (blind ~parts (f ()))
        | None -> fail (name ^ ": the host's kernel declined a stated dtype")
      else names name (fun () -> ignore (f ())))
    reductions;
  if Array.length ax > 0 then
    List.iter
      (fun (name, m, f) ->
        let f () = host_array (f ~axis:ax.(0) (value a)) in
        let parts = m = Nx_kernel.Sum in
        if documented m dt then
          match kernel ~scan:true m a [| ax.(0) |] with
          | Some want ->
              equal ~msg:name blind_bits (blind ~parts want)
                (blind ~parts (f ()))
          | None -> fail (name ^ ": the host's kernel declined a stated dtype")
        else names name (fun () -> ignore (f ())))
      scans

(* The reduction of a transposed or reversed view has the bits of the same
   values laid out contiguously: nx.cpu's association depends on the shape
   alone. *)
let law_views (Case a, axes) =
  let dt = A.dtype a in
  let r = Array.length (Iarray.to_array (L.shape (A.layout a))) in
  let x = value a in
  let view = if r >= 2 then Nx.transpose x else Nx.flip x in
  let copy = Nx.copy view in
  let axes = Option.map (List.map (fun a -> if a < 0 then a + r else a)) axes in
  cover "a transposed view" (r >= 2);
  cover "a reversed view" (r = 1);
  List.iter
    (fun (name, m, f) ->
      let parts = m = Nx_kernel.Sum in
      if documented m dt then
        match f ?axes copy with
        | exception Invalid_argument _ -> ()
        | want ->
            equal ~msg:name blind_bits
              (blind ~parts (host_array want))
              (blind ~parts (host_array (f ?axes view))))
    reductions

(* A NaN term reaches every result after it *)

(* Rows of terms of a float or complex dtype, as their bytes. *)
type nan_case =
  | Nan_case : {
      dt : ('v, 's) D.t;
      rows : int;
      n : int;
      bytes : string;
    }
      -> nan_case

let string_of_bytes b = String.init (Array.length b) (fun i -> Char.chr b.(i))

let part_bytes w c =
  String.init w (fun j ->
      Char.chr (Int64.to_int (Int64.shift_right_logical c (8 * j)) land 0xFF))

(* The bytes of each of [xs] stored in [dt]. *)
let encoded (type s) (dt : (float, s) D.t) xs =
  let a = A.of_array dt [| List.length xs |] (Array.of_list xs) in
  Array.to_list (Array.map string_of_bytes (element_bits a))

let numbers = [ 0.; -0.; 1.; -1.; 2.; 0.5; 3.; infinity; neg_infinity ]

(* A float's or a complex part's bytes: numbers, and NaNs signalling and quiet,
   of both signs and with payloads, from their bits. *)
let pools (type v s) (dt : (v, s) D.t) =
  let nans w l = List.map (part_bytes w) l in
  match dt with
  | D.Float64 | D.Complex128 ->
      ( encoded D.Float64 numbers,
        nans 8
          [
            0x7FF0000000000001L;
            0xFFF4000000000005L;
            0x7FF8000000000123L;
            0xFFF8000000000000L;
          ] )
  | D.Float32 | D.Complex64 ->
      ( encoded D.Float32 numbers,
        nans 4 [ 0x7F800001L; 0xFFA00005L; 0x7FC00123L; 0xFFC00000L ] )
  | D.Float16 ->
      (encoded D.Float16 numbers, nans 2 [ 0x7C01L; 0xFD05L; 0x7E23L; 0xFE00L ])
  | D.Bfloat16 ->
      (encoded D.Bfloat16 numbers, nans 2 [ 0x7F81L; 0xFFA5L; 0x7FC3L; 0xFFC0L ])
  | D.Float8_e5m2 ->
      (encoded D.Float8_e5m2 numbers, nans 1 [ 0x7DL; 0xFDL; 0x7EL; 0xFFL ])
  | D.Float8_e4m3fn ->
      ( encoded D.Float8_e4m3fn [ 0.; -0.; 1.; -1.; 2.; 0.5; 3.; 448.; -448. ],
        nans 1 [ 0x7FL; 0xFFL ] )
  | _ -> invalid_arg "pools: not a float or complex dtype"

let nan_dtypes =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e5m2;
      Any Float8_e4m3fn;
      Any Float32;
      Any Float64;
      Any Complex64;
      Any Complex128;
    ]

let parts_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Complex -> 2 | _ -> 1

let nan_array (type v s) (dt : (v, s) D.t) rows n bytes : (v, s) A.t =
  A.v dt
    (L.contiguous [| rows; n |])
    (Rig.Buffer.of_string (if bytes = "" then "\000" else bytes))

let pp_nan_case ppf (Nan_case { dt; rows; n; bytes }) =
  Format.fprintf ppf "%a [%d; %d] %s" D.pp dt rows n
    (String.concat " "
       (List.init (String.length bytes) (fun i ->
            Printf.sprintf "%02x" (Char.code bytes.[i]))))

(* Rows of one to nine terms, a third of their floats or parts NaNs. *)
let nan_case =
  let open Gen in
  with_pp pp_nan_case
    (let* (D.Any dt) = of_list nan_dtypes in
     let numbers, nans = pools dt in
     let* rows = int_range 1 3 in
     let* n = int_range 1 9 in
     let part = frequency [ (2, of_list numbers); (1, of_list nans) ] in
     let+ ps = list ~size:(constant (rows * n * parts_of dt)) part in
     Nan_case { dt; rows; n; bytes = String.concat "" ps })

(* [r] of [x] along its rows' axis, through Nx.Prim. *)
let reduce_along (type v s d a) (x : (v, s, d) Nx.t)
    (r : (d, a) Nx.Prim.reduction) : a =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Iarray.of_array (Nx.shape x));
           axes = [| 1 |];
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
           at = None;
         })
  in
  y

(* [r] of [x] running along its rows' axis, through Nx.Prim. *)
let scan_along (type v s d a) (x : (v, s, d) Nx.t)
    (r : (d, a) Nx.Prim.reduction) : a =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Iarray.of_array (Nx.shape x));
         axis = 1;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
         at = None;
       })

(* The operations nx.mli's reductions section promises a NaN of where a NaN term
   reaches: each result per row is a NaN where its row holds one, each running
   result where a term up to it does, each position the first NaN's; a complex
   number with a NaN part is a NaN, but to a sum, whose parts fold apart. *)
let law_nans (Nan_case { dt; rows; n; bytes }) =
  let a = nan_array dt rows n bytes in
  let raw = Array.map string_of_bytes (element_bits a) in
  let quiet = Array.map string_of_bytes (element_bits (Nx_support.quieted a)) in
  let parts = nan_parts a in
  let any i = Array.exists Fun.id parts.(i) in
  let first r p =
    List.find_opt (fun j -> p ((r * n) + j)) (List.init n Fun.id)
  in
  let firsts = List.filter_map (fun r -> first r any) (List.init rows Fun.id) in
  cover "a signalling NaN first in a row"
    (List.exists
       (fun r ->
         match first r any with
         | Some j -> raw.((r * n) + j) <> quiet.((r * n) + j)
         | None -> false)
       (List.init rows Fun.id));
  cover "a quiet NaN first in a row"
    (List.exists
       (fun r ->
         match first r any with
         | Some j -> raw.((r * n) + j) = quiet.((r * n) + j)
         | None -> false)
       (List.init rows Fun.id));
  cover "a row of no NaN" (List.length firsts < rows);
  cover "a narrow float" (D.bits dt < 32);
  cover "a complex number" (parts_of dt = 2);
  cover "a complex number of one NaN part"
    (parts_of dt = 2 && Array.exists (fun p -> p.(0) <> p.(1)) parts);
  (* Whether a NaN reaches result [k] of row [r] from the terms up to [upto], in
     part [p] where the parts fold apart, else in any part. *)
  let reached ~apart p r upto =
    match first r (fun i -> if apart then parts.(i).(p) else any i) with
    | Some j -> j <= upto
    | None -> false
  in
  let nan_where ~msg ~apart ~result y =
    let got = nan_parts (host_array y) in
    let want =
      Array.mapi
        (fun k ps ->
          let r, upto = result k in
          if apart then Array.mapi (fun p g -> g || reached ~apart p r upto) ps
          else [| Array.exists Fun.id ps || reached ~apart 0 r upto |])
        got
    in
    let got =
      if apart then got
      else Array.map (fun ps -> [| Array.exists Fun.id ps |]) got
    in
    equal ~msg (array (array bool)) want got
  in
  let per_row ?(apart = false) msg y =
    nan_where ~msg ~apart ~result:(fun r -> (r, n - 1)) y
  in
  let per_prefix ?(apart = false) msg y =
    nan_where ~msg ~apart ~result:(fun k -> (k / n, k mod n)) y
  in
  let first_nan msg y =
    let got = read y in
    let want =
      Array.mapi
        (fun r g ->
          match first r any with Some j -> Int64.of_int j | None -> g)
        got
    in
    equal ~msg (array int64) want got
  in
  let x = value a in
  let apart = parts_of dt = 2 in
  per_row ~apart "sum" (Nx.sum ~axes:[ 1 ] x);
  per_row "prod" (Nx.prod ~axes:[ 1 ] x);
  per_row "max" (Nx.max ~axes:[ 1 ] x);
  per_row "min" (Nx.min ~axes:[ 1 ] x);
  per_prefix ~apart "cumsum" (Nx.cumsum ~axis:1 x);
  per_prefix "cumprod" (Nx.cumprod ~axis:1 x);
  per_prefix "cummax" (Nx.cummax ~axis:1 x);
  per_prefix "cummin" (Nx.cummin ~axis:1 x);
  first_nan "argmax" (Nx.argmax ~axis:1 x);
  first_nan "argmin" (Nx.argmin ~axis:1 x);
  per_row "Arg Max's extreme" (fst (reduce_along x (Arg (Max, 0, Nx.dtype x))));
  per_row "Arg Min's extreme" (fst (reduce_along x (Arg (Min, 0, Nx.dtype x))));
  per_prefix "running Arg Max's extreme"
    (fst (scan_along x (Arg (Max, 0, Nx.dtype x))));
  per_prefix "running Arg Min's extreme"
    (fst (scan_along x (Arg (Min, 0, Nx.dtype x))));
  let floats (type s) (x : (float, s, host) Nx.t) =
    per_row "mean" (Nx.mean ~axes:[ 1 ] x);
    per_row "var" (Nx.var ~axes:[ 1 ] x);
    per_row "std" (Nx.std ~axes:[ 1 ] x);
    per_row "logsumexp" (Nx.logsumexp ~axes:[ 1 ] x);
    per_prefix "logcumsumexp" (Nx.logcumsumexp ~axis:1 x);
    per_prefix "running mean" (fst (scan_along x (Moments (0, Nx.dtype x))));
    per_prefix "running variance" (snd (scan_along x (Moments (0, Nx.dtype x))))
  in
  match Nx.dtype x with
  | D.Float64 -> floats x
  | D.Float32 -> floats x
  | D.Float16 -> floats x
  | D.Bfloat16 -> floats x
  | D.Float8_e4m3fn -> floats x
  | D.Float8_e5m2 -> floats x
  | _ -> ()

(* Refusals *)

let mem = Nx_support.memory

module Cpu3 = (val Nx.devices [ mem 0; mem 1; mem 3 ])

let raise_rows =
  let x = Nx.create Nx.float32 [| 2; 3 |] (Array.make 6 1.) in
  let m = Nx.create Nx.bool [| 2; 3 |] (Array.make 6 true) in
  let v = f64 [| 1. |] and b = Nx.create Nx.bool [| 2 |] [| true; false |] in
  let row name what f = (name ^ ", " ^ what, name, fun () -> ignore (f ())) in
  [
    row "sum" "an axis past the rank" (fun () -> Nx.sum ~axes:[ 2 ] x);
    row "sum" "an axis repeated" (fun () -> Nx.sum ~axes:[ 0; -2 ] x);
    row "sum" "booleans" (fun () -> Nx.sum b);
    row "prod" "an axis before the rank" (fun () -> Nx.prod ~axes:[ -3 ] x);
    row "prod" "booleans" (fun () -> Nx.prod b);
    row "max" "an axis past the rank" (fun () -> Nx.max ~axes:[ 2 ] x);
    row "min" "an axis repeated" (fun () -> Nx.min ~axes:[ 0; 0 ] x);
    row "mean" "an axis past the rank" (fun () -> Nx.mean ~axes:[ 2 ] x);
    row "mean" "booleans" (fun () -> Nx.mean b);
    row "all" "an axis past the rank" (fun () -> Nx.all ~axes:[ 2 ] m);
    row "any" "an axis before the rank" (fun () -> Nx.any ~axes:[ -3 ] m);
    row "count" "an axis repeated" (fun () -> Nx.count ~axes:[ 0; 0 ] m);
    row "argmax" "an axis past the rank" (fun () -> Nx.argmax ~axis:2 x);
    row "argmin" "an axis before the rank" (fun () -> Nx.argmin ~axis:(-3) x);
    row "var" "an axis past the rank" (fun () -> Nx.var ~axes:[ 2 ] x);
    row "std" "an axis past the rank" (fun () -> Nx.std ~axes:[ 1 ] v);
    row "logsumexp" "an axis past the rank" (fun () ->
        Nx.logsumexp ~axes:[ 2 ] x);
    row "logmeanexp" "an axis repeated" (fun () ->
        Nx.logmeanexp ~axes:[ 0; 0 ] x);
    row "cumsum" "an axis past the rank" (fun () -> Nx.cumsum ~axis:2 x);
    row "cumsum" "booleans" (fun () -> Nx.cumsum b);
    row "cumprod" "an axis before the rank" (fun () -> Nx.cumprod ~axis:(-3) x);
    row "cumprod" "booleans" (fun () -> Nx.cumprod b);
    row "cummax" "an axis past the rank" (fun () -> Nx.cummax ~axis:2 x);
    row "cummin" "an axis past the rank" (fun () -> Nx.cummin ~axis:2 x);
    row "logcumsumexp" "an axis past the rank" (fun () ->
        Nx.logcumsumexp ~axis:2 x);
    row "ewma" "an axis past the rank" (fun () -> Nx.ewma ~axis:2 ~alpha:0.5 x);
  ]

(* A refused axis is named as the caller passed it, a negative one as negative,
   not as normalised (51d269f8e1). *)
let test_axis_printed () =
  let x = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.make 24 1.) in
  let names_axis a f =
    raises_match
      (fun e ->
        Exn.invalid_arg ~sub:"Nx.sum: " e && Exn.invalid_arg ~sub:(a ^ " ") e)
      f
  in
  names_axis "-5" (fun () -> Nx.sum ~axes:[ -5 ] x);
  names_axis "axis -1" (fun () -> Nx.sum ~axes:[ 2; -1 ] x)

(* sum, prod *)

let z re im = { Complex.re; im }

let test_sum_from_zero () =
  let bits = array float_exact in
  equal ~msg:"float32 [-0.; -0.]" bits [| 0. |]
    (Nx.to_array (Nx.sum ~keepdims:true (f32 [| -0.; -0. |])));
  equal ~msg:"float64 [-0.; -0.]" bits [| 0. |]
    (Nx.to_array (Nx.sum ~keepdims:true (f64 [| -0.; -0. |])));
  equal ~msg:"float16 [-0.; -0.]" bits [| 0. |]
    (Nx.to_array
       (Nx.cast Nx.float64
          (Nx.sum ~keepdims:true (Nx.cast Nx.float16 (f64 [| -0.; -0. |])))));
  equal ~msg:"float64 of none" bits [| 0. |]
    (Nx.to_array (Nx.sum ~keepdims:true (f64 [||])));
  let parts (c : Complex.t) = [| c.re; c.im |] in
  equal ~msg:"complex64 [-0. - 0.i; -0. - 0.i]" bits [| 0.; 0. |]
    (parts
       (item
          (Nx.sum
             (Nx.create Nx.complex64 [| 2 |] [| z (-0.) (-0.); z (-0.) (-0.) |]))));
  equal ~msg:"int32 of none" int32 0l
    (item (Nx.sum (Nx.zeros Nx.int32 [| 0 |])))

(* Complex products, extremes and their first positions, reduced and scanned
   through Nx on the host, against folds of their terms. A number with a NaN
   part is a NaN. A product's terms are Gaussian integers whose parts are 0 or ±
   a power of two, so every order of multiplication gives the exact product but
   for the sign of a zero part: products compare by value, extremes by bits, and
   a NaN as any NaN. *)

type zcase =
  | Zcase : {
      dt : (Complex.t, 's) D.t;
      s : int array;
      axes : int list option;
      strided : bool;
      xs : Complex.t array;
    }
      -> zcase

let znan (c : Complex.t) = Float.is_nan c.re || Float.is_nan c.im
let bits = Int64.bits_of_float
let zbits (c : Complex.t) = (bits c.re, bits c.im)
let pp_z ppf (c : Complex.t) = Format.fprintf ppf "%h%+hi" c.re c.im

let z_bits =
  Testable.make ~pp:pp_z ~equal:(fun a b ->
      zbits a = zbits b || (znan a && znan b))

let z_value =
  Testable.make ~pp:pp_z ~equal:(fun (a : Complex.t) b ->
      if znan a || znan b then znan a && znan b else a.re = b.re && a.im = b.im)

(* A NaN of payload [k] in bits float32 keeps. *)
let payload k = nan_with (Int64.shift_left (Int64.of_int k) 29)

(* A NaN in one part or both. *)
let nan_term =
  let open Gen in
  let* k = int_range 1 1000 in
  let+ which = int_range 0 2 in
  match which with
  | 0 -> { Complex.re = payload k; im = 1. }
  | 1 -> { Complex.re = -0.; im = payload k }
  | _ -> { Complex.re = payload k; im = payload (k + 1) }

(* [xs] with at most [most] of its elements replaced by draws of [rare]. *)
let sprinkle xs ~most rare =
  let open Gen in
  let n = Array.length xs in
  if n = 0 then constant xs
  else
    let+ picks =
      list ~size:(int_range 0 most) (pair (int_range 0 (n - 1)) rare)
    in
    let xs = Array.copy xs in
    List.iter (fun (i, x) -> xs.(i) <- x) picks;
    xs

(* A value of [m] elements read through a transposed view where [strided]: the
   elements of [a] held in a layout that runs its axes backwards. *)
let transposed strided a =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let r = Array.length s in
  if (not strided) || r < 2 then value a
  else
    let back = Array.init r (fun i -> s.(r - 1 - i)) in
    let xs = A.to_array a in
    let at idx =
      Array.fold_left
        (fun p a -> (p * s.(a)) + idx.(r - 1 - a))
        0 (Array.init r Fun.id)
    in
    let held = Array.of_list (List.map (fun i -> xs.(at i)) (indices back)) in
    Nx.transpose (value (A.of_array (A.dtype a) (Iarray.of_array back) held))

(* Terms of a product: units, zeros where [zeros], at most 40 of [1 ± i] and
   [2], and NaNs where [nans]. Extremes: parts of a few values, so that ties are
   common, signed zeros and infinities, and NaNs where [nans]. Shapes of a few
   elements; the long ones are examples. *)
let zcase ~product =
  let open Gen in
  let make (type s) (dt : (Complex.t, s) D.t) =
    let* s = shape in
    let* axes = axes_of (Array.length s) in
    let* strided = bool in
    let* nans = bool in
    let n = Array.fold_left ( * ) 1 s in
    let* xs =
      if product then
        let* zeros = bool in
        let unit = of_list [ z 1. 0.; z (-1.) 0.; z 0. 1.; z 0. (-1.) ] in
        let* xs =
          array ~size:(constant n)
            (frequency
               [ (30, unit); ((if zeros then 1 else 0), constant (z 0. 0.)) ])
        in
        sprinkle xs ~most:40 (of_list [ z 1. 1.; z 1. (-1.); z 2. 0. ])
      else
        let part =
          frequency
            [
              (8, of_list [ -1.; -0.; 0.; 1.; 2. ]);
              (1, of_list [ Float.infinity; Float.neg_infinity ]);
            ]
        in
        array ~size:(constant n)
          (let* re = part in
           let+ im = part in
           z re im)
    in
    let+ xs = if nans then sprinkle xs ~most:3 nan_term else constant xs in
    (* The terms as stored. *)
    let xs = A.to_array (A.of_array dt (Iarray.of_array s) xs) in
    Zcase { dt; s; axes; strided; xs }
  in
  with_pp
    (fun ppf (Zcase { dt; s; axes; strided; _ }) ->
      Format.fprintf ppf "%a %a over %a%s" D.pp dt pp_ints s pp_axes axes
        (if strided then ", transposed" else ""))
    (let* c128 = bool in
     if c128 then make D.Complex128 else make D.Complex64)

(* Long terms: an output of more than a block's 1024 terms, a scan longer than a
   chunk's 4096, columns of many terms read transposed. *)
let long_zcases ~product =
  let term i =
    let k = i * 7919 mod 97 in
    if product then
      if k = 0 then z 1. 1.
      else [| z 1. 0.; z (-1.) 0.; z 0. 1.; z 0. (-1.) |].(k mod 4)
    else z (Float.of_int ((k mod 5) - 2)) (Float.of_int ((k / 5 mod 3) - 1))
  in
  let make s axes strided =
    let n = Array.fold_left ( * ) 1 s in
    let xs = Array.init n term in
    let xs =
      if n > 10 then (
        xs.(n / 3) <- z (payload 5) 1.;
        xs)
      else xs
    in
    Zcase { dt = D.Complex128; s; axes; strided; xs }
  in
  [ make [| 5000 |] None false; make [| 3; 1100 |] (Some [ 1 ]) true ]

(* A product starts from its first term, and is 1 for none. *)
let product ts =
  match (List.find_opt znan ts, ts) with
  | Some t, _ -> t
  | None, [] -> Complex.one
  | None, t :: ts -> List.fold_left Complex.mul t ts

let zprod acc t =
  if znan acc then acc else if znan t then t else Complex.mul acc t

let along_of s axes =
  match axes with
  | Some (a :: _) -> Some (if a < 0 then a + Array.length s else a)
  | _ -> None

let law_complex_prod (Zcase { dt; s; axes; strided; xs }) =
  let ts = terms s (normal (Array.length s) axes) xs in
  cover "more than a block of terms"
    (Array.exists (fun t -> List.length t > 1024) ts);
  cover "a NaN result" (Array.exists (fun t -> List.exists znan t) ts);
  cover "a product other than a unit"
    (Array.exists (fun t -> Complex.norm (product t) > 1.) ts);
  cover "transposed" (strided && Array.length s > 1);
  let x = transposed strided (A.of_array dt (Iarray.of_array s) xs) in
  equal (array z_value) (Array.map product ts) (read (Nx.prod ?axes x))

let law_complex_cumprod (Zcase { dt; s; axes; strided; xs }) =
  let along = along_of s axes in
  let n = Array.length xs in
  cover "a scan longer than a chunk" (n > 4096 && along = None);
  cover "transposed" (strided && Array.length s > 1);
  let x = transposed strided (A.of_array dt (Iarray.of_array s) xs) in
  let want =
    match along with
    | None -> running zprod [| n |] 0 xs
    | Some a -> running zprod s a xs
  in
  equal (array z_value) want (read (Nx.cumprod ?axis:along x))

let test_prod_cases () =
  let v dt xs = Nx.create dt [| Array.length xs |] xs in
  equal ~msg:"a complex product of no term is 1 + 0i" (array z_bits)
    [| z 1. 0. |]
    (Nx.to_array (Nx.reshape [| 1 |] (Nx.prod (v Nx.complex64 [||]))));
  equal ~msg:"a float product of no term is 1" float_exact 1.
    (item (Nx.prod (f32 [||])));
  equal ~msg:"an int32 product along an empty axis is 1" (array int32)
    [| 1l; 1l; 1l |]
    (Nx.to_array (Nx.prod ~axes:[ 1 ] (Nx.zeros Nx.int32 [| 3; 0 |])));
  (* (∞ + 0i)(0 + i) is NaN + ∞i, as Mul's textbook product gives it. *)
  equal ~msg:"a NaN product of numbers is a NaN" (array z_bits)
    [| z nan Float.infinity |]
    (Nx.to_array
       (Nx.reshape [| 1 |]
          (Nx.prod (v Nx.complex128 [| z Float.infinity 0.; z 0. 1. |]))));
  (* 1 + 0i times either would make a NaN part or flip a zero's sign. *)
  equal ~msg:"a product of one term is that term" (array z_bits)
    [| z Float.infinity 0.; z 3. (-0.) |]
    (Nx.to_array
       (Nx.prod ~axes:[ 1 ]
          (Nx.create Nx.complex64 [| 2; 1 |]
             [| z Float.infinity 0.; z 3. (-0.) |])));
  equal ~msg:"and so is a scan's first result" (array z_bits)
    [| z (-0.) (-0.); z 0. (-0.) |]
    (Nx.to_array (Nx.cumprod (v Nx.complex128 [| z (-0.) (-0.); z 1. 0. |])))

(* max, min *)

(* Whether [t] is past [b] in Max's order, or Min's where [flip]: real parts,
   then imaginary parts, [-0] below [+0], and a NaN past every number. *)
let zbeats flip (t : Complex.t) (b : Complex.t) =
  if znan t || znan b then znan t && not (znan b)
  else
    let part x y =
      if x = y then Bool.compare (Float.sign_bit y) (Float.sign_bit x)
      else Float.compare x y
    in
    let c = part t.re b.re in
    let c = if c <> 0 then c else part t.im b.im in
    if flip then c < 0 else c > 0

(* The first extreme of [ts] and its position, in one pass. *)
let zextreme flip ts =
  let _, best, at =
    List.fold_left
      (fun (i, best, at) t ->
        if i = 0 || zbeats flip t best then (i + 1, t, i) else (i + 1, best, at))
      (0, Complex.zero, 0) ts
  in
  (best, Int64.of_int at)

let zrunning beats acc t =
  if znan acc then acc else if beats t acc then t else acc

let extremes =
  [
    ("max", false, fun ?axes x -> Nx.max ?axes x);
    ("min", true, fun ?axes x -> Nx.min ?axes x);
  ]

let law_complex_extremes (Zcase { dt; s; axes; strided; xs }) =
  let r = Array.length s in
  let ax = normal r axes in
  let ts = terms s ax xs in
  let x = transposed strided (A.of_array dt (Iarray.of_array s) xs) in
  let empty = List.exists (fun a -> s.(a) = 0) ax in
  cover "a NaN result" (Array.exists (fun t -> List.exists znan t) ts);
  cover "transposed" (strided && r > 1);
  cover "more than a block of terms"
    ((not empty) && Array.exists (fun t -> List.length t > 1024) ts);
  if not empty then
    List.iter
      (fun (name, flip, f) ->
        equal ~msg:name (array z_bits)
          (Array.map (fun t -> fst (zextreme flip t)) ts)
          (read (f ?axes x)))
      extremes

let law_complex_args (Zcase { dt; s; axes; strided; xs }) =
  let along = along_of s axes in
  let x = transposed strided (A.of_array dt (Iarray.of_array s) xs) in
  let lanes =
    match along with
    | None -> [| Array.to_list xs |]
    | Some a -> terms s [ a ] xs
  in
  let any_terms =
    Option.fold ~none:(xs <> [||]) ~some:(fun a -> s.(a) > 0) along
  in
  cover "a NaN lane" (Array.exists (List.exists znan) lanes);
  cover "along an axis" (along <> None);
  if Array.length xs > 0 && any_terms then begin
    equal ~msg:"argmax" (array int64)
      (Array.map (fun t -> snd (zextreme false t)) lanes)
      (read (Nx.argmax ?axis:along x));
    equal ~msg:"argmin" (array int64)
      (Array.map (fun t -> snd (zextreme true t)) lanes)
      (read (Nx.argmin ?axis:along x))
  end

let law_complex_scans (Zcase { dt; s; axes; strided; xs }) =
  let along = along_of s axes in
  let x = transposed strided (A.of_array dt (Iarray.of_array s) xs) in
  cover "a NaN term" (Array.exists znan xs);
  cover "along an axis" (along <> None);
  List.iter
    (fun (name, flip, scan) ->
      let step = zrunning (zbeats flip) in
      let want =
        match along with
        | None -> running step [| Array.length xs |] 0 xs
        | Some a -> running step s a xs
      in
      equal ~msg:name (array z_bits) want (read (scan ?axis:along x)))
    [
      ("cummax", false, fun ?axis x -> Nx.cummax ?axis x);
      ("cummin", true, fun ?axis x -> Nx.cummin ?axis x);
    ]

let test_extreme_cases () =
  let ws =
    Nx.create Nx.complex128 [| 4 |]
      [| z 1. 2.; z 2. (-5.); z 2. (-7.); z 2. (-5.) |]
  in
  equal ~msg:"max orders real parts, then imaginary parts" z_bits (z 2. (-5.))
    (item (Nx.max ws));
  equal ~msg:"min: a -0 part below +0" z_bits (z 1. (-0.))
    (item (Nx.min (Nx.create Nx.complex64 [| 2 |] [| z 1. 0.; z 1. (-0.) |])));
  equal ~msg:"max of floats: +0. above -0." float_exact 0.
    (item (Nx.max (f64 [| -0.; 0.; -0. |])));
  let b xs = Nx.create Nx.bool [| Array.length xs |] xs in
  equal ~msg:"max of booleans is any" bool true
    (item (Nx.max (b [| false; true |])));
  equal ~msg:"min of booleans is all" bool false
    (item (Nx.min (b [| true; false |])))

(* An extreme along an empty axis is refused only where its result has an
   element: one with none is the empty result, on one device or cut over three
   along either axis (57e9ba1593). *)
let empty_results =
  let empty s = value (A.of_array D.Float32 (Iarray.of_array s) [||]) in
  let shape_of x = Iarray.to_array (L.shape (A.layout (host_array x))) in
  let cut axis = Nx.place (Cpu3.split ~axis) (empty [| 0; 0 |]) in
  [
    ("max", [| 0 |], fun () -> shape_of (Nx.max ~axes:[ 1 ] (empty [| 0; 0 |])));
    ( "min, kept",
      [| 0; 1 |],
      fun () -> shape_of (Nx.min ~axes:[ 1 ] ~keepdims:true (empty [| 0; 0 |]))
    );
    ( "argmax",
      [| 0 |],
      fun () -> shape_of (Nx.argmax ~axis:1 (empty [| 0; 0 |])) );
    ( "argmin",
      [| 3; 0 |],
      fun () -> shape_of (Nx.argmin ~axis:1 (empty [| 3; 0; 0 |])) );
    ( "max, cut along 0",
      [| 0 |],
      fun () -> shape_of (Nx.max ~axes:[ 1 ] (cut 0)) );
    ( "max, cut along 1",
      [| 0 |],
      fun () -> shape_of (Nx.max ~axes:[ 1 ] (cut 1)) );
    ( "argmin, cut along 0",
      [| 0 |],
      fun () -> shape_of (Nx.argmin ~axis:1 (cut 0)) );
    ( "argmin, cut along 1",
      [| 0 |],
      fun () -> shape_of (Nx.argmin ~axis:1 (cut 1)) );
  ]

let empty_refusals =
  let empty s = Nx.zeros Nx.float32 s in
  let row id name f = (id, name, fun () -> ignore (f ())) in
  [
    row "max along an empty axis" "max" (fun () ->
        Nx.max ~axes:[ 1 ] (empty [| 2; 0 |]));
    row "min of none" "min" (fun () -> Nx.min (empty [| 0; 0 |]));
    row "argmax along an empty axis" "argmax" (fun () ->
        Nx.argmax ~axis:1 (empty [| 2; 0 |]));
    row "argmin of none" "argmin" (fun () -> Nx.argmin (empty [| 0; 0 |]));
    row "max of no complex number" "max" (fun () ->
        Nx.max (Nx.create Nx.complex64 [| 0 |] [||]));
    row "argmin of no complex number" "argmin" (fun () ->
        Nx.argmin (Nx.create Nx.complex128 [| 0 |] [||]));
  ]

(* mean *)

(* A float or complex mean: the sum in the carrier over the count, rounded once
   to the dtype, as nx.mli states. *)
let float_mean_case =
  let open Gen in
  with_pp pp_case
    (let* (D.Any dt) =
       of_list
         ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt)
         (List.filter
            (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt)
            D.all)
     in
     let* s = shape in
     let* a = drawn dt s in
     let+ axes = axes_of (Array.length s) in
     (Case a, axes))

let law_float_mean (Case a, axes) =
  let dt = A.dtype a in
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let n =
    List.fold_left (fun n i -> n * s.(i)) 1 (normal (Array.length s) axes)
  in
  cover "a narrow float" (D.is D.Float dt && D.bits dt < 32);
  cover "a complex number" (D.is D.Complex dt);
  cover "no element" (n = 0);
  let k = Float.of_int n in
  let check (type v s w r) (x : (v, s, host) Nx.t) (w : (w, r) D.t)
      (divide : (w, r, host) Nx.t -> (w, r, host) Nx.t) =
    let want = Nx.cast (Nx.dtype x) (divide (Nx.sum ?axes (Nx.cast w x))) in
    equal blind_bits
      (blind ~parts:true (host_array want))
      (blind ~parts:true (host_array (Nx.mean ?axes x)))
  in
  let floats (type s) (x : (float, s, host) Nx.t) =
    match Nx.dtype x with
    | D.Float64 -> check x D.Float64 (fun y -> Nx.div_s y k)
    | _ -> check x D.Float32 (fun y -> Nx.div_s y k)
  in
  let x = value a in
  match Nx.dtype x with
  | D.Float64 -> floats x
  | D.Float32 -> floats x
  | D.Float16 -> floats x
  | D.Bfloat16 -> floats x
  | D.Float8_e4m3fn -> floats x
  | D.Float8_e5m2 -> floats x
  | D.Float4_e2m1fn -> floats x
  | D.Complex64 -> check x D.Complex64 (fun y -> Nx.div_s y (z k 0.))
  | D.Complex128 -> check x D.Complex128 (fun y -> Nx.div_s y (z k 0.))
  | _ -> invalid_arg "law_float_mean: not a float or complex dtype"

(* A mean of no element, which a draw reaches about one case in eight. *)
let mean_examples =
  [ (Case (A.of_array D.Float16 [| 0; 2 |] [||]), Some [ 0 ]) ]

let test_mean_values () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 6. |] in
  equal ~msg:"along 0" (array float_exact) [| 2.; 4. |]
    (Nx.to_array (Nx.mean ~axes:[ 0 ] x));
  let all = Nx.mean ~keepdims:true x in
  equal ~msg:"kept, shape" (array int) [| 1; 1 |] (Nx.shape all);
  equal ~msg:"kept" (array float_exact) [| 3. |] (Nx.to_array all);
  equal ~msg:"of none" float_exact Float.nan (item (Nx.mean (f64 [||])))

(* Integer means *)

(* An integer's bits: an unsigned 32-bit one zero-extended. *)
let to_int64 (type v s) (dt : (v, s) D.t) (v : v) : int64 =
  match dt with
  | D.Int64 -> v
  | D.Uint64 -> v
  | D.Int32 -> Int64.of_int32 v
  | D.Uint32 -> Int64.logand (Int64.of_int32 v) 0xFFFF_FFFFL
  | D.Int16 -> Int64.of_int v
  | D.Uint16 -> Int64.of_int v
  | D.Int8 -> Int64.of_int v
  | D.Uint8 -> Int64.of_int v
  | D.Int4 -> Int64.of_int v
  | D.Uint4 -> Int64.of_int v
  | _ -> invalid_arg "to_int64: not an integer dtype"

(* The value of [dt] whose low bits are [x]'s. *)
let of_int64 (type v s) (dt : (v, s) D.t) (x : int64) : v =
  let b = D.bits dt in
  let x =
    if b = 64 then x
    else if D.is D.Signed dt then
      Int64.shift_right (Int64.shift_left x (64 - b)) (64 - b)
    else Int64.shift_right_logical (Int64.shift_left x (64 - b)) (64 - b)
  in
  match dt with
  | D.Int64 -> x
  | D.Uint64 -> x
  | D.Int32 -> Int64.to_int32 x
  | D.Uint32 -> Int64.to_int32 x
  | D.Int16 -> Int64.to_int x
  | D.Uint16 -> Int64.to_int x
  | D.Int8 -> Int64.to_int x
  | D.Uint8 -> Int64.to_int x
  | D.Int4 -> Int64.to_int x
  | D.Uint4 -> Int64.to_int x
  | _ -> invalid_arg "of_int64: not an integer dtype"

(* An integer's value as limbs [(hi, lo)], [hi * 2^32 + lo], [lo] in [[0,
   2^32)]: sums of them are exact in [int]. *)
let limbs (type v s) (dt : (v, s) D.t) (v : v) =
  let x = to_int64 dt v in
  let hi =
    if D.is D.Signed dt then Int64.shift_right x 32
    else Int64.shift_right_logical x 32
  in
  (Int64.to_int hi, Int64.to_int (Int64.logand x 0xFFFF_FFFFL))

let add_limbs (h, l) (h', l') = (h + h', l + l')

(* The sum [(h, l)] of [n] integers divided by [n], truncated toward zero, by
   long division of its magnitude in base 2^32: its bits, and whether it is
   negative and inexact. *)
let divided n (h, l) =
  let h = h + (l asr 32) and l = l land 0xFFFF_FFFF in
  let negative = h < 0 in
  let a, b =
    if not negative then (h, l)
    else if l = 0 then (-h, 0)
    else (-h - 1, 0x1_0000_0000 - l)
  in
  let low = ((a mod n) lsl 32) + b in
  let q = Int64.(add (shift_left (of_int (a / n)) 32) (of_int (low / n))) in
  ((if negative then Int64.neg q else q), negative && low mod n <> 0)

let integer_dtypes =
  List.filter (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt) D.all

let int_case =
  let open Gen in
  with_pp
    (fun ppf (D.Any dt, s, axes, _) ->
      Format.fprintf ppf "%a %a over %a" D.pp dt pp_ints s pp_axes axes)
    (let* (D.Any dt) =
       of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) integer_dtypes
     in
     let* s = shape in
     let* axes = axes_of (Array.length s) in
     let extremes =
       [ to_int64 dt (D.min_value dt); to_int64 dt (D.max_value dt) ]
     in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (frequency
            [
              (3, map Int64.of_int (int_range (-5) 5));
              (2, of_list extremes);
              (1, int64);
            ])
     in
     (D.Any dt, s, axes, xs))

let law_int_mean (D.Any dt, s, axes, xs) =
  let ax = normal (Array.length s) axes in
  let xs = Array.map (of_int64 dt) xs in
  let x = value (A.of_array dt (Iarray.of_array s) xs) in
  let n = List.fold_left (fun n a -> n * s.(a)) 1 ax in
  let refused = n = 0 && Array.fold_left ( * ) 1 (reduced s ax) > 0 in
  let want =
    if refused then [||]
    else
      let sums = fold add_limbs (0, 0) s ax (Array.map (limbs dt) xs) in
      Array.map (divided n) sums
  in
  cover "64 bits" ((not refused) && D.bits dt = 64);
  cover "a negative mean rounded toward zero" (Array.exists snd want);
  if refused then names "mean" (fun () -> Nx.mean ?axes x)
  else
    equal (array int64) (Array.map fst want)
      (Array.map (to_int64 dt) (read (Nx.mean ?axes x)))

(* Means the spec gives: the exact mean rounded toward zero, between the least
   and greatest element. *)
let test_int_means () =
  let mean dt xs =
    to_int64 dt (item (Nx.mean (Nx.create dt [| Array.length xs |] xs)))
  in
  equal ~msg:"int32 3.5" int64 3L (mean Nx.int32 [| 3l; 4l |]);
  equal ~msg:"int32 -3.5" int64 (-3L) (mean Nx.int32 [| -3l; -4l |]);
  equal ~msg:"int32 -1/3" int64 0L (mean Nx.int32 [| -1l; 0l; 0l |]);
  equal ~msg:"int8 -127.5" int64 (-127L) (mean Nx.int8 [| -128; -127 |]);
  equal ~msg:"int4 -7.5" int64 (-7L) (mean Nx.int4 [| -8; -7 |]);
  equal ~msg:"uint4 14.5" int64 14L (mean Nx.uint4 [| 15; 14 |]);
  equal ~msg:"uint8 254.5" int64 254L (mean Nx.uint8 [| 255; 254 |]);
  equal ~msg:"uint32 2^32 - 1.5" int64 0xFFFF_FFFEL
    (mean Nx.uint32 [| -1l; -2l |]);
  let hi = Int64.max_int and lo = Int64.min_int in
  equal ~msg:"int64 max" int64 hi (mean Nx.int64 [| hi; hi |]);
  equal ~msg:"int64 min" int64 lo (mean Nx.int64 [| lo; lo |]);
  equal ~msg:"int64 (max + min) / 2 = -1/2" int64 0L
    (mean Nx.int64 [| hi; lo |]);
  equal ~msg:"int64 (2 min - 1) / 3" int64 (-6148914691236517205L)
    (mean Nx.int64 [| lo; lo; -1L |]);
  equal ~msg:"uint64 2^64 - 1" int64 (-1L) (mean Nx.uint64 [| -1L; -1L |]);
  equal ~msg:"uint64 (2^64 - 1) / 2" int64 Int64.max_int
    (mean Nx.uint64 [| -1L; 0L |]);
  equal ~msg:"uint64 (2^65 - 1) / 3" int64 (-6148914691236517206L)
    (mean Nx.uint64 [| -1L; -1L; 1L |]);
  let x = Nx.zeros Nx.int32 [| 2; 0 |] in
  equal ~msg:"an empty axis of an empty result" (array int) [| 0 |]
    (Nx.shape (Nx.mean ~axes:[ 0 ] x));
  let x = Nx.create Nx.int16 [| 2; 2 |] [| 1; 2; 4; 7 |] in
  let kept = Nx.mean ~axes:[ 1 ] ~keepdims:true x in
  equal ~msg:"kept, shape" (array int) [| 2; 1 |] (Nx.shape kept);
  equal ~msg:"kept" (array int) [| 1; 5 |] (Nx.to_array kept)

(* Over [2^31] elements at 64 bits, [2^(64 - b)] at [b] bits, broadcast from
   one, so that nothing of that size is held. Below 16 bits the limit passes the
   [2^53] elements an array can have. *)
let int_mean_refusals =
  let many dt one n = Nx.broadcast_to [| n |] (Nx.scalar dt one) in
  let row name f = (name, "mean", fun () -> ignore (f ())) in
  [
    row "an empty axis with a result element" (fun () ->
        Nx.mean ~axes:[ 1 ] (Nx.zeros Nx.int32 [| 2; 0 |]));
    row "2^31 int64 elements" (fun () -> Nx.mean (many Nx.int64 1L (1 lsl 31)));
    row "2^31 uint64 elements" (fun () ->
        Nx.mean (many Nx.uint64 1L (1 lsl 31)));
    row "2^32 int32 elements" (fun () -> Nx.mean (many Nx.int32 1l (1 lsl 32)));
    row "2^48 int16 elements" (fun () -> Nx.mean (many Nx.int16 1 (1 lsl 48)));
    row "2^48 uint16 elements" (fun () -> Nx.mean (many Nx.uint16 1 (1 lsl 48)));
  ]

(* all, any, count *)

(* Bytes zero two times in three, so that both answers come often. *)
let truth_case = any_case ~zeros:2 ()

let truth_examples =
  [
    (Case (A.of_array D.Float32 [| 2; 0 |] [||]), Some [ 1 ]);
    (Case (A.of_array D.Float64 [| 2 |] [| Float.nan; -0. |]), None);
    (Case (A.of_array D.Complex64 [| 2 |] [| z 0. 1.; z (-0.) (-0.) |]), None);
  ]

(* Each element read as cast into bool reads it. *)
let truths (type v s) (a : (v, s) A.t) =
  (A.to_array (cast D.Bool a) : bool array)

let law_all (Case a, axes) =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let ax = normal (Array.length s) axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  cover "false" (Array.exists not (truths a));
  equal (array bool)
    (fold ( && ) true s ax (truths a))
    (Nx.to_array (Nx.all ?axes (value a)))

let law_any (Case a, axes) =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let ax = normal (Array.length s) axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  cover "true" (Array.exists Fun.id (truths a));
  equal (array bool)
    (fold ( || ) false s ax (truths a))
    (Nx.to_array (Nx.any ?axes (value a)))

let law_count (Case a, axes) =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let ax = normal (Array.length s) axes in
  let m = Nx.cast Nx.bool (value a) in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  equal (array int64)
    (fold (fun n t -> if t then Int64.succ n else n) 0L s ax (truths a))
    (Nx.to_array (Nx.count ?axes m))

let test_truth_values () =
  equal ~msg:"all: a NaN is non-zero" bool true
    (item (Nx.all (f64 [| nan; -1. |])));
  equal ~msg:"any: -0. is zero" bool false (item (Nx.any (f64 [| -0.; 0. |])));
  equal ~msg:"all of none" bool true (item (Nx.all (f64 [||])));
  equal ~msg:"any of none" bool false (item (Nx.any (f64 [||])));
  let c = Nx.create Nx.complex128 [| 2 |] [| z 0. 1.; Complex.zero |] in
  equal ~msg:"any: a complex number non-zero in a part" (array bool) [| true |]
    (Nx.to_array (Nx.any ~keepdims:true c));
  equal ~msg:"all, kept" (array bool) [| false |]
    (Nx.to_array (Nx.all ~keepdims:true c));
  let m =
    Nx.create Nx.bool [| 2; 3 |] [| true; false; true; true; true; true |]
  in
  equal ~msg:"count along 1" (array int64) [| 2L; 3L |]
    (Nx.to_array (Nx.count ~axes:[ 1 ] m));
  equal ~msg:"count of bits" (array int64) [| 5L |]
    (Nx.to_array (Nx.count ~keepdims:true (Nx.flatten (Nx.cast Nx.bit m))));
  equal ~msg:"count of none" int64 0L
    (item (Nx.count (Nx.zeros Nx.bool [| 0 |])));
  equal ~msg:"all along 1, kept" (array int) [| 2; 1 |]
    (Nx.shape (Nx.all ~axes:[ 1 ] ~keepdims:true m))

(* argmax, argmin *)

let specials = [ -1.; 0.; -0.; 1.; 2.; nan; infinity; neg_infinity ]

let arg_case =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 1 4) in
  let* axis =
    if Array.length s = 0 then constant None
    else option (int_range 0 (Array.length s - 1))
  in
  let+ xs =
    array ~size:(constant (Array.fold_left ( * ) 1 s)) (of_list specials)
  in
  (s, axis, xs)

(* [a] before [b] in [maximum]'s order, NaN above all. *)
let above a b =
  if Float.is_nan b then false
  else if Float.is_nan a then true
  else a > b || (a = b && Float.sign_bit b && not (Float.sign_bit a))

(* The first element of [ts] that no later one is [better] than. *)
let first better ts =
  let best = ref 0 in
  Array.iteri (fun i t -> if better t ts.(!best) then best := i) ts;
  Int64.of_int !best

let law_arg_order (s, axis, xs) =
  let x = Nx.create Nx.float64 s xs in
  let lanes =
    match axis with
    | None -> [| xs |]
    | Some a -> Array.map Array.of_list (terms s [ a ] xs)
  in
  cover "a NaN" (Array.exists Float.is_nan xs);
  cover "along an axis" (axis <> None);
  equal ~msg:"argmax" (array int64)
    (Array.map (first above) lanes)
    (Nx.to_array (Nx.argmax ?axis x));
  let below a b =
    if Float.is_nan a || Float.is_nan b then above a b else above b a
  in
  equal ~msg:"argmin" (array int64)
    (Array.map (first below) lanes)
    (Nx.to_array (Nx.argmin ?axis x))

(* At every dtype the element at argmax has max's bits, the first such, or is
   the first NaN where max is one; argmin so of min. *)
let law_arg_extreme (Case a, axes) =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let r = Array.length s in
  let axis = Option.map (fun l -> (List.hd l + r) mod r) axes in
  let n = Array.length (element_bits a) in
  let lanes =
    match axis with
    | None -> [| List.init n Fun.id |]
    | Some a -> terms s [ a ] (Array.init n Fun.id)
  in
  let empty = Array.exists (fun l -> l = []) lanes || n = 0 in
  cover "along an axis" (axis <> None);
  cover "a NaN lane" ((not empty) && Array.exists (fun b -> b) (nans a));
  if not empty then
    List.iter
      (fun (name, arg, extreme) ->
        let eb = element_bits a and nn = nans a in
        let m =
          host_array
            (extreme ?axes:(Option.map (fun a -> [ a ]) axis) (value a))
        in
        let mb = element_bits m and mn = nans m in
        let want =
          Array.mapi
            (fun k lane ->
              let at i = if mn.(k) then nn.(i) else eb.(i) = mb.(k) in
              let j = List.find (fun i -> at i) lane in
              Int64.of_int (List.length (List.filter (fun i -> i < j) lane)))
            lanes
        in
        equal ~msg:name (array int64) want (read (arg ?axis (value a))))
      [
        ( "argmax",
          (fun ?axis x -> Nx.argmax ?axis x),
          fun ?axes x -> Nx.max ?axes x );
        ( "argmin",
          (fun ?axis x -> Nx.argmin ?axis x),
          fun ?axes x -> Nx.min ?axes x );
      ]

(* NaN lanes, which drawn bytes rarely give: a NaN first and a NaN after a
   number, along an axis and over the elements. *)
let arg_examples =
  [
    ( Case (A.of_array D.Float32 [| 2; 3 |] [| 1.; nan; 2.; nan; nan; 0. |]),
      Some [ 1 ] );
    (Case (A.of_array D.Float64 [| 3 |] [| -0.; nan; 0. |]), None);
  ]

let test_arg_values () =
  let at f x = item (f x) in
  equal ~msg:"argmax [-0.; 0.]" int64 1L (at Nx.argmax (f64 [| -0.; 0. |]));
  equal ~msg:"argmin [0.; -0.]" int64 1L (at Nx.argmin (f64 [| 0.; -0. |]));
  equal ~msg:"argmin, the first NaN" int64 1L
    (at Nx.argmin (f64 [| 1.; nan; nan; -5. |]));
  equal ~msg:"argmax of uint8" int64 0L
    (at Nx.argmax (Nx.create Nx.uint8 [| 2 |] [| 200; 1 |]));
  equal ~msg:"argmax of booleans, the first true" int64 1L
    (at Nx.argmax (Nx.create Nx.bool [| 3 |] [| false; true; true |]));
  equal ~msg:"argmax of complex numbers, the first tie" int64 1L
    (at Nx.argmax
       (Nx.create Nx.complex128 [| 3 |] [| Complex.one; z 1. 2.; z 1. 2. |]));
  let m = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 9l; 3l; 7l; 2l; 9l |] in
  equal ~msg:"without axis, the position in C order" int64 1L (at Nx.argmax m);
  equal ~msg:"without axis, rank 0" (array int) [||] (Nx.shape (Nx.argmax m));
  equal ~msg:"without axis, kept" (array int) [| 1; 1 |]
    (Nx.shape (Nx.argmax ~keepdims:true m));
  equal ~msg:"along 0" (array int64) [| 1L; 0L; 1L |]
    (Nx.to_array (Nx.argmax ~axis:0 m));
  equal ~msg:"along -1, kept" (array int) [| 2; 1 |]
    (Nx.shape (Nx.argmin ~axis:(-1) ~keepdims:true m));
  equal ~msg:"of a scalar" int64 0L (at Nx.argmax (Nx.scalar Nx.int32 4l));
  equal ~msg:"an empty axis of an empty result" (array int) [| 2; 0 |]
    (Nx.shape (Nx.argmax ~axis:0 (Nx.zeros Nx.int8 [| 0; 2; 0 |])))

(* logsumexp, logmeanexp *)

let ln = Float.log

(* nx.cpu's bound on a Logsumexp [r] of [n] terms (nx_cpu.mli): blocks of [L]
   roundings, a tree of depth [D], [(L + D + 5 + 6 ln n) u + u|r|]. *)
let lse_bound u n r =
  let l = ((Stdlib.min n 1024 + 15) / 16) + Stdlib.min 3 (n - 1) in
  let rec depth b d = if b <= 1 then d else depth ((b + 1) / 2) (d + 1) in
  let ld = Float.of_int (l + depth ((n + 1023) / 1024) 0) in
  ((ld +. 5. +. (6. *. ln (Float.of_int n))) *. u) +. (u *. Float.abs r)

(* [lse_ref m terms] is [m + log Σ exp (t - m)] over [terms] of the finite
   maximum [m], at float64, and a bound on its error: each [t - m] split exactly
   into its rounding and a remainder (two-sum), each [exp] taken of the rounding
   and corrected by the remainder, and the exps summed compensated (Neumaier),
   so the sum's relative error is a few units whatever the count. The bound adds
   a unit for each [exp], two for the sum, one for the [log] and its own [ulp],
   and half one for the last addition, libm's [exp] and [log] within one [ulp]:
   near [2^-50], far below a float32 bound. *)
let lse_ref m terms =
  let sum = ref 0. and c = ref 0. in
  let add e =
    let s = !sum +. e in
    let lost =
      if Float.abs !sum >= Float.abs e then !sum -. s +. e else e -. s +. !sum
    in
    c := !c +. lost;
    sum := s
  in
  List.iter
    (fun t ->
      let d = t -. m in
      if Float.is_finite d then begin
        let back = d -. t in
        let rest = t -. (d -. back) +. (-.m -. back) in
        add (exp d *. (1. +. rest))
      end)
    terms;
  let s = !sum +. !c in
  let ls = log s in
  let r = m +. ls in
  let err =
    (epsilon_float *. (4. +. Float.abs ls)) +. (ldexp 1. (-53) *. Float.abs r)
  in
  (r, err)

let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* float32 lanes about 0 and far from it, of spreads of a unit to past float32's
   [exp] range. *)
let lse_case =
  let open Gen in
  let* n = int_range 1 40 in
  let* offset = of_list [ 0.; 1e3; -1e4 ] in
  let* spread = of_list [ 1.; 30.; 80. ] in
  let+ xs = array ~size:(constant n) (float_range (-.spread) spread) in
  Array.map (fun x -> fl32 (offset +. x)) xs

let near ~msg ~tol want got =
  equal ~msg (float (Float.max tol Float.min_float)) want got

(* logmeanexp takes [log n], within 2 ulps and so [4u ln n], from logsumexp
   before the one rounding, which adds [u|r|]. *)
let law_lse xs =
  let n = Array.length xs in
  cover "far from 0" (Array.exists (fun x -> Float.abs x > 500.) xs);
  cover "terms whose exp overflows" (Array.exists (fun x -> x > 89.) xs);
  let fn = Float.of_int n in
  let m = Array.fold_left Float.max neg_infinity xs in
  let r, err = lse_ref m (Array.to_list xs) in
  let u = 0x1p-24 in
  near ~msg:"logsumexp"
    ~tol:(lse_bound u n r +. err)
    r
    (item (Nx.logsumexp (f32 xs)));
  let lme = r -. ln fn in
  near ~msg:"logmeanexp"
    ~tol:(lse_bound u n r +. (4. *. u *. ln fn) +. (u *. Float.abs lme) +. err)
    lme
    (item (Nx.logmeanexp (f32 xs)))

(* Rows of eight terms, each correctly rounded by gen/special.py: the reference
   errs by half an ulp of [r] at most, within [u|r|]. *)
let lse_rows name =
  List.map (fun r -> ("f32", r)) (Nx_support.golden name D.Float32)
  @ List.map (fun r -> ("f64", r)) (Nx_support.golden name D.Float64)

let row_id (d, (r : Nx_support.row)) =
  d ^ " "
  ^ String.concat " " (Array.to_list (Array.map (Printf.sprintf "%h") r.args))

let row_lane (d, (r : Nx_support.row)) =
  if d = "f32" then (0x1p-24, Nx.cast Nx.float64 (f32 r.args))
  else (0x1p-53, f64 r.args)

let check_row ~f ~extra row =
  let _, (r : Nx_support.row) = row in
  let u, x = row_lane row in
  let n = Array.length r.args in
  let got = item (Nx.cast Nx.float64 (f u x)) in
  if not (Float.is_finite r.value) then equal float_exact r.value got
  else
    near ~msg:"against the row"
      ~tol:
        (lse_bound u n r.value +. extra u n r.value +. (u *. Float.abs r.value))
      r.value got

(* The rows' lanes are of [dtype]: computed there, read as float64. *)
let at_row_dtype u (f : 's. (float, 's, host) Nx.t -> (float, 's, host) Nx.t) x
    =
  if u = 0x1p-24 then Nx.cast Nx.float64 (f (Nx.cast Nx.float32 x)) else f x

let test_lse_values () =
  let lse xs = item (Nx.logsumexp (f64 xs)) in
  let lme xs = item (Nx.logmeanexp (f64 xs)) in
  let u = 0x1p-53 in
  equal ~msg:"of none" float_exact neg_infinity (lse [||]);
  equal ~msg:"of -∞ alone" float_exact neg_infinity
    (lse [| neg_infinity; neg_infinity |]);
  equal ~msg:"+∞ beside a number" float_exact infinity (lse [| infinity; 1. |]);
  equal ~msg:"a NaN beside +∞" float_exact nan (lse [| nan; infinity |]);
  let r = 1000. +. ln 2. in
  near ~msg:"[1000; 1000], whose exp overflows"
    ~tol:(lse_bound u 2 r +. (u *. r))
    r
    (lse [| 1000.; 1000. |]);
  equal ~msg:"logmeanexp of none" float_exact nan (lme [||]);
  near ~msg:"logmeanexp [1000; 1000]"
    ~tol:(lse_bound u 2 r +. (4. *. u *. ln 2.) +. (u *. 1000.))
    1000.
    (lme [| 1000.; 1000. |]);
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 0.; 0.; 1.; 1. |] in
  let got = Nx.to_array (Nx.logsumexp ~axes:[ -1 ] m) in
  let u = 0x1p-24 in
  near ~msg:"row 0" ~tol:(lse_bound u 2 (ln 2.) +. (u *. ln 2.)) (ln 2.) got.(0);
  near ~msg:"row 1"
    ~tol:(lse_bound u 2 (1. +. ln 2.) +. (u *. (1. +. ln 2.)))
    (1. +. ln 2.)
    got.(1)

(* At a float narrower than float32, [f] computes at float32 and rounds once:
   over every code, in rows of 16. *)
let narrow name (f : 's. (float, 's, host) Nx.t -> (float, 's, host) Nx.t) =
  cases
    (name ^ " at a narrow float is float32's, rounded once")
    ~id:(fun (D.Any dt) -> D.name dt)
    (List.filter (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32) D.all)
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          let n = 1 lsl D.bits dt in
          let codes =
            match D.bits dt with
            | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 n 1)
            | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 n 1)
            | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 n 1)
          in
          let x = Nx.place Nx.Host.on (Nx.reshape [| n / 16; 16 |] codes) in
          equal (array float_exact)
            (Nx.to_array
               (Nx.cast Nx.float64 (Nx.cast dt (f (Nx.cast Nx.float32 x)))))
            (Nx.to_array (Nx.cast Nx.float64 (f x)))
      | _ -> ())

(* cumsum *)

(* Float or complex terms from drawn bytes in rows of up to 40, and a prefix of
   the rows to keep. *)
let prefix_case =
  let open Gen in
  let scanned =
    List.filter (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt) D.all
  in
  with_pp
    (fun ppf (Case a, k) ->
      Format.fprintf ppf "%a %a, prefix %d" D.pp (A.dtype a) pp_ints
        (Iarray.to_array (L.shape (A.layout a)))
        k)
    (let* (D.Any dt) =
       of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) scanned
     in
     let* rows = int_range 1 3 in
     let* n = int_range 0 40 in
     let* a = drawn dt [| rows; n |] in
     let+ k = int_range 0 n in
     (Case a, k))

(* Past a scan chunk's 4096 terms, a prefix that ends inside the second. *)
let prefix_examples =
  [
    ( Case
        (A.of_array D.Float32 [| 1; 4100 |]
           (Array.init 4100 (fun i -> Float.of_int (i * 7919 mod 1000) /. 7.))),
      4098 );
  ]

let law_cumsum_prefix (Case a, k) =
  let n = (Iarray.to_array (L.shape (A.layout a))).(1) in
  cover "a proper prefix" (k > 0 && k < n);
  cover "a complex number" (D.is D.Complex (A.dtype a));
  let x = value a in
  let prefix y = Nx.slice [ Nx.A; Nx.R (0, k) ] y in
  equal blind_bits
    (blind ~parts:true (host_array (prefix (Nx.cumsum ~axis:1 x))))
    (blind ~parts:true (host_array (Nx.cumsum ~axis:1 (prefix x))))

let test_complex_cumsum () =
  let x =
    Nx.create Nx.complex64 [| 3 |] [| z 1. (-1.); z 2. 0.5; z (-4.) 2. |]
  in
  equal (array z_bits)
    [| z 1. (-1.); z 3. (-0.5); z (-1.) 1.5 |]
    (Nx.to_array (Nx.cumsum x))

let test_narrow_cumsum () =
  let xs = Array.init 300 (fun i -> 1. +. (Float.of_int i /. 7.)) in
  let b = Nx.cast Nx.bfloat16 (f32 xs) in
  let wide = Nx.cumsum (Nx.cast Nx.float32 b) in
  equal (array float_exact)
    (Nx.to_array (Nx.cast Nx.float64 (Nx.cast Nx.bfloat16 wide)))
    (Nx.to_array (Nx.cast Nx.float64 (Nx.cumsum b)))

(* logcumsumexp *)

(* Each slice of [s] along [axis], in C order of the other axes, as the position
   in C order of its index [j]. *)
let slices s axis =
  let n = s.(axis) in
  let r = Array.length s in
  let after = Array.sub s (axis + 1) (r - axis - 1) in
  let stride = Array.fold_left ( * ) 1 after in
  let total = Array.fold_left ( * ) 1 s in
  if n = 0 then []
  else
    List.init (total / n) (fun o ->
        let base = (o / stride * n * stride) + (o mod stride) in
        fun j -> base + (j * stride))

(* Each prefix of each slice of [s] along [axis]: its terms' positions in C
   order, the last one the result's. *)
let prefixes s axis =
  List.concat_map
    (fun at -> List.init s.(axis) (fun j -> List.init (j + 1) at))
    (slices s axis)

(* The most combinations output [i] of a scan's network passes through, [D_i =
   max(0, 2⌈log₂(i+1)⌉ - 1)], for [n = i + 1] terms. *)
let depth n =
  let rec log2_up b = if 1 lsl b >= n then b else log2_up (b + 1) in
  Float.of_int (Int.max 0 ((2 * log2_up 0) - 1))

(* [v] rounded to [dt], as a store rounds. *)
let rounded (type s) (dt : (float, s) D.t) v =
  (A.to_array (A.of_array dt [| 1 |] [| v |])).(0)

(* nx.cpu's bound: at [n] terms through at most [D] combinations, within [(7 D +
   6 ln n) u + u |r|] in the accumulator. *)
let lcse_bound u n r =
  (((7. *. depth n) +. (6. *. ln (Float.of_int n))) *. u) +. (u *. Float.abs r)

(* A shape of rank at least one and an axis of it, at times long enough for
   several levels of the network. *)
let along =
  let open Gen in
  let* s = shape in
  let s = if Array.length s = 0 then [| 3 |] else s in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ n = frequency [ (3, constant s.(axis)); (1, int_range 5 40) ] in
  let s = Array.copy s in
  s.(axis) <- n;
  (s, axis)

type lcse_case =
  | Lcse_case :
      (float, 's) D.t * int array * int option * bool * float array
      -> lcse_case

(* A float dtype and moderate terms, at times infinite, [-0.] or a NaN, or in
   half the cases mostly those, -inf the most, in 5 to 8 rows of 2 to 6, so that
   a case holds prefixes of each kind; along an axis or over the elements, read
   through a transposed view or not. *)
let lcse_case =
  let open Gen in
  let floats (type s) (dt : (float, s) D.t) =
    let element ~moderate ~neg_inf =
      frequency
        [
          (moderate, float_range (-5.) 5.);
          (1, constant (-0.));
          (neg_inf, constant Float.neg_infinity);
          (1, constant Float.infinity);
          (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
        ]
    in
    let* special = bool in
    let* s, axis =
      if not special then along
      else
        let+ rows = int_range 5 8
        and+ cols = int_range 2 6
        and+ axis = frequency [ (2, constant 0); (1, constant 1) ] in
        ([| rows; cols |], axis)
    in
    let element =
      if special then element ~moderate:2 ~neg_inf:3
      else element ~moderate:40 ~neg_inf:1
    in
    let* flat =
      if special then bool
      else frequency [ (3, constant false); (1, constant true) ]
    in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Lcse_case (dt, s, (if flat then None else Some axis), strided, xs)
  in
  with_pp
    (fun ppf (Lcse_case (dt, s, axis, strided, _)) ->
      Format.fprintf ppf "%a %a %s%s" D.pp dt pp_ints s
        (match axis with
        | Some a -> Printf.sprintf "along %d" a
        | None -> "in C order")
        (if strided then ", transposed" else ""))
    (frequency
       [
         (2, floats D.Float64);
         (2, floats D.Float32);
         (1, floats D.Float16);
         (1, floats D.Bfloat16);
         (1, floats D.Float8_e4m3fn);
         (1, floats D.Float8_e5m2);
         (1, floats D.Float4_e2m1fn);
       ])

let lcse_value (type s) (dt : (float, s) D.t) s strided xs =
  transposed strided (A.of_array dt (Iarray.of_array s) xs)

(* Each prefix's terms as stored, and its result. *)
let lcse_prefixes (Lcse_case (dt, s, axis, strided, xs)) =
  let x = lcse_value dt s strided xs in
  let n = Array.length xs in
  let y = Nx.logcumsumexp ?axis x in
  let ts = read x and got = read y in
  let ps =
    match axis with Some a -> prefixes s a | None -> prefixes [| n |] 0
  in
  List.map
    (fun ps ->
      (List.map (fun p -> ts.(p)) ps, got.(List.nth ps (List.length ps - 1))))
    ps

let has_nan terms = List.exists Float.is_nan terms
let greatest terms = List.fold_left Float.max Float.neg_infinity terms

(* Below float64, within nx.cpu's bound of a float64 log-sum-exp over each
   prefix of finite greatest term, then rounded once, so between the roundings
   of the bound's ends; the reference ({!lse_ref}) adds its own error, near
   [2^-50]. *)
let law_lcse_bound (Lcse_case (dt, s, axis, _, _) as c) =
  let u = 0x1p-24 in
  cover "a float narrower than float32" (D.bits dt < 32);
  cover "over the elements" (axis = None && Array.length s > 1);
  cover "an axis of more than 4"
    (match axis with Some a -> s.(a) > 4 | None -> false);
  let finite =
    List.filter
      (fun (t, _) -> (not (has_nan t)) && Float.is_finite (greatest t))
      (lcse_prefixes c)
  in
  cover "a -inf term among finite ones"
    (List.exists (fun (t, _) -> List.mem Float.neg_infinity t) finite);
  if D.bits dt < 64 then
    List.iter
      (fun (terms, g) ->
        let want, err = lse_ref (greatest terms) terms in
        let bound = lcse_bound u (List.length terms) want +. err in
        let msg = Printf.sprintf "%h against %h" g want in
        at_least ~msg float_exact ~than:(rounded dt (want -. bound)) g;
        at_most ~msg float_exact ~than:(rounded dt (want +. bound)) g)
      finite

let law_lcse_specials c =
  let ps = lcse_prefixes c in
  let some p = List.exists (fun (t, _) -> p t) ps in
  cover "a NaN term" (some has_nan);
  cover "+inf"
    (some (fun t -> (not (has_nan t)) && greatest t = Float.infinity));
  cover "every term -inf"
    (some (fun t -> (not (has_nan t)) && greatest t = Float.neg_infinity));
  List.iter
    (fun (terms, g) ->
      if has_nan terms then equal ~msg:"a NaN" float_exact nan g
      else if not (Float.is_finite (greatest terms)) then
        equal ~msg:"an infinity" float_exact (greatest terms) g)
    ps

(* The logcumsumexp of a prefix of the axis is the prefix of the logcumsumexp,
   bit for bit or both NaN; over the elements, it is the logcumsumexp of the
   elements in C order. Cut over three devices along another axis than the one
   it scans, it has the host's bits. It reads the axis it scans whole, the
   elements' order included, so a cut along it raises where the value has
   elements, and placed whole as the refusal writes, it has the host's bits. *)
let law_lcse_bits (Lcse_case (dt, s, axis, strided, xs), cut, keep) =
  let x = lcse_value dt s strided xs in
  let r = Array.length s in
  let y = Nx.logcumsumexp ?axis x in
  let bits_of y = blind (host_array y) in
  let k = match axis with Some a -> keep mod (s.(a) + 1) | None -> 0 in
  let along p = match axis with Some a -> p s.(a) | None -> false in
  cover "a proper prefix" (along (fun d -> k < d));
  cover "an empty prefix" (along (fun d -> k = 0 && d > 0));
  (match axis with
  | Some a ->
      let prefix t =
        Nx.slice (List.init r (fun i -> if i = a then Nx.R (0, k) else Nx.A)) t
      in
      equal ~msg:"the prefix" blind_bits
        (bits_of (prefix y))
        (bits_of (Nx.logcumsumexp ~axis:a (prefix x)))
  | None ->
      let flat = Nx.reshape [| Array.length xs |] x in
      equal ~msg:"C order" blind_bits
        (bits_of (Nx.reshape [| Array.length xs |] y))
        (bits_of (Nx.logcumsumexp ~axis:0 flat)));
  let c = cut mod r in
  cover "cut along the scanned axis" (axis = Some c);
  cover "cut along another axis" (axis <> None && axis <> Some c);
  cover "cut, in C order" (axis = None && Array.length xs > 0);
  let placed = Nx.place (Cpu3.split ~axis:c) x in
  let placed =
    if (axis = None || axis = Some c) && Array.length xs > 0 then begin
      raises_match
        (Exn.invalid_arg
           ~sub:"and is read at on [m0; m1; m3]; Nx.place (Placement.on s) x")
        (fun () -> Nx.logcumsumexp ?axis placed);
      Nx.place Cpu3.on placed
    end
    else placed
  in
  equal ~msg:"cut over three devices" blind_bits (bits_of y)
    (bits_of (Nx.logcumsumexp ?axis placed))

(* A case, the axis to cut, and the prefix to keep, empty one time in four or
   more. *)
let lcse_bits_case =
  Gen.(
    triple lcse_case (int_range 0 7)
      (frequency
         [ (1, constant ~pp:Format.pp_print_int 0); (3, int_range 0 40) ]))

let test_lcse_values () =
  let lcse x = Nx.to_array (Nx.logcumsumexp x) in
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let big = [| 1000.; 1000.; 999. |] in
  equal ~msg:"the premise: log (cumsum (exp x)) overflows at 1000"
    (array float_exact) [| inf; inf; inf |]
    (Nx.to_array (Nx.log (Nx.cumsum (Nx.exp (f64 big)))));
  let got = lcse (f64 big) and u = 0x1p-53 in
  List.iteri
    (fun i want ->
      near
        ~msg:(Printf.sprintf "float64 terms of 1000, %d" i)
        ~tol:(lcse_bound u (i + 1) want +. (u *. want))
        want got.(i))
    [ 1000.; 1000. +. ln 2.; 1000. +. ln (2. +. exp (-1.)) ];
  let got = lcse (f32 [| 100.; 100. |]) and u = 0x1p-24 in
  near ~msg:"float32 terms of 100, whose exp overflows"
    ~tol:(lcse_bound u 2 (100. +. ln 2.) +. (u *. 100.))
    (100. +. ln 2.)
    got.(1);
  List.iter
    (fun (s, axis) ->
      equal ~msg:"an empty value's shape" (array int) s
        (Nx.shape
           (Nx.logcumsumexp ?axis
              (value (A.of_array D.Float32 (Iarray.of_array s) [||])))))
    [ ([| 2; 0 |], Some 1); ([| 0; 3 |], Some 1); ([| 0 |], None) ];
  equal ~msg:"one element along the axis: itself, -0 as +0" (array float_exact)
    [| 2.5; 0.; ninf |]
    (Nx.to_array
       (Nx.logcumsumexp ~axis:1
          (Nx.create Nx.float64 [| 3; 1 |] [| 2.5; -0.; ninf |])));
  equal ~msg:"-inf terms" (array float_exact) [| ninf; ninf; 0.; 0. |]
    (lcse (f64 [| ninf; ninf; 0.; ninf |]));
  equal ~msg:"+inf from it on" (array float_exact) [| 1.; inf; inf; inf |]
    (lcse (f64 [| 1.; inf; 2.; ninf |]));
  equal ~msg:"a NaN, from the first on" (array float_exact)
    [| 1.; nan; nan; nan |]
    (lcse (f64 [| 1.; nan_with 7L; nan_with 9L; inf |]))

(* The last of eight results, a logsumexp of the row's terms, within nx.cpu's
   bound at the network's depth for eight. *)
let check_lcse_row row =
  let _, (r : Nx_support.row) = row in
  let u, x = row_lane row in
  let n = Array.length r.args in
  let got =
    (Nx.to_array (at_row_dtype u (fun x -> Nx.logcumsumexp x) x)).(n - 1)
  in
  if not (Float.is_finite r.value) then equal float_exact r.value got
  else
    near ~msg:"against the row"
      ~tol:(lcse_bound u n r.value +. (u *. Float.abs r.value))
      r.value got

(* ewma *)

(* Pairs [hi + lo] of twice float64's digits. *)
let two_sum a b =
  let s = a +. b in
  let v = s -. a in
  (s, a -. (s -. v) +. (b -. v))

let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  fast_two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p = ah *. bh in
  fast_two_sum p (Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh))

(* The recurrence in pairs, from the [alpha] the format holds: far within either
   format's bound. *)
let recurrence alpha xs =
  let one_minus = two_sum 1. (-.alpha) in
  let y = ref (0., 0.) in
  Array.mapi
    (fun i x ->
      y :=
        if i = 0 then (x, 0.)
        else dd_add (dd_mul one_minus !y) (dd_mul (alpha, 0.) (x, 0.));
      fst !y +. snd !y)
    xs

let float_pp ppf v = Format.fprintf ppf "%h" v

let series =
  let open Gen in
  let* is64 = bool in
  let* n = int_range 0 70 in
  let* xs = array ~size:(constant n) (float_range (-1e3) 1e3) in
  let* alpha =
    of_list ~pp:float_pp
      [ 1.; 0.5; 0.9; 0.7; 0.1; 1e-3; 0x1.8p-13; 0x1p-13; 0x1.4p-18 ]
  in
  let+ k = int_range 0 n in
  (is64, xs, alpha, k)

(* Within nx.mli's [γ(i δ + 2D + 1) Mᵢ], [δ] 0 for an alpha of the format of at
   least 1/2, else 1; the exact average is of [alpha] rounded into the format,
   and the call takes it unrounded. *)
let law_ewma (is64, xs, alpha, _) =
  let u, store, x =
    if is64 then (0x1p-53, Fun.id, f64 xs)
    else (0x1p-24, fl32, Nx.cast Nx.float64 (f32 xs))
  in
  let held = store alpha and xs = Array.map store xs in
  let delta = if held >= 0.5 then 0. else 1. in
  cover "alpha below 1/2" (delta = 1.);
  cover "1 - alpha a float below 1/2"
    (delta = 1.
    && store (1. -. held) = 1. -. held
    && snd (two_sum 1. (-.held)) = 0.);
  cover "alpha not a float32" ((not is64) && held <> alpha);
  cover "float32" (not is64);
  let got =
    if is64 then Nx.to_array (Nx.ewma ~alpha x)
    else
      Nx.to_array (Nx.cast Nx.float64 (Nx.ewma ~alpha (Nx.cast Nx.float32 x)))
  in
  let want = recurrence held xs in
  let m = ref 0. in
  Array.iteri
    (fun i w ->
      m := Float.max !m (Float.abs xs.(i));
      let k = (Float.of_int i *. delta) +. (2. *. depth (i + 1)) +. 1. in
      near
        ~msg:(Printf.sprintf "element %d" i)
        ~tol:(k *. u /. (1. -. (k *. u)) *. !m)
        w got.(i))
    want

(* Small alphas with [1 - alpha] a float, over long lanes of ones, where the
   rounded products of slopes weigh every term alike. *)
let ewma_examples =
  [
    (false, Array.make 1024 1., 0x1.8p-13, 0);
    (true, Array.make 1024 1., 0x1.8p-13, 0);
  ]

let law_ewma_prefix (is64, xs, alpha, k) =
  let n = Array.length xs in
  cover "a proper prefix" (k > 0 && k < n);
  let average xs =
    if is64 then Nx.to_array (Nx.ewma ~alpha (f64 xs))
    else Nx.to_array (Nx.cast Nx.float64 (Nx.ewma ~alpha (f32 xs)))
  in
  equal (array float_exact)
    (Array.sub (average xs) 0 k)
    (average (Array.sub xs 0 k))

let test_ewma_values () =
  let xs = [| -0.; 2.; -3.; 1e-300 |] in
  equal ~msg:"alpha 1 is x" (array float_exact) xs
    (Nx.to_array (Nx.ewma ~alpha:1. (f64 xs)));
  equal ~msg:"the recurrence at alpha 0.5" (array float_exact)
    [| 0.; 2.; 3.; 1.5 |]
    (Nx.to_array (Nx.ewma ~alpha:0.5 (f64 [| 0.; 4.; 4.; 0. |])));
  equal ~msg:"a NaN reaches every later element" (array float_exact)
    [| 1.; nan; nan; nan |]
    (Nx.to_array (Nx.ewma ~alpha:0.5 (f64 [| 1.; nan; 2.; 3. |])));
  equal ~msg:"an infinity as the recurrence carries it" (array float_exact)
    [| infinity; infinity; nan |]
    (Nx.to_array (Nx.ewma ~alpha:0.5 (f64 [| infinity; 1.; neg_infinity |])));
  let x = Array.make 4096 0. in
  x.(2047) <- 0x1p1023;
  equal ~msg:"a term whose composed weight underflows drops out" float_exact 0.
    (Nx.to_array (Nx.ewma ~alpha:0.5 (f64 x))).(4095);
  let m = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  equal ~msg:"along an axis, the transpose's" (array float_exact)
    (Nx.to_array (Nx.transpose (Nx.ewma ~axis:1 ~alpha:0.25 (Nx.transpose m))))
    (Nx.to_array (Nx.ewma ~axis:0 ~alpha:0.25 m));
  equal ~msg:"without an axis, in C order" (array float_exact)
    (Nx.to_array (Nx.ewma ~alpha:0.25 (Nx.flatten m)))
    (Nx.to_array (Nx.ewma ~alpha:0.25 m));
  equal ~msg:"in x's shape" (array int) [| 2; 3 |]
    (Nx.shape (Nx.ewma ~alpha:0.25 m))

let alpha_rows =
  List.map
    (fun alpha ->
      ( Printf.sprintf "%g" alpha,
        fun () -> ignore (Nx.ewma ~alpha (f64 [| 1. |])) ))
    [ 0.; -1.; 1.5; nan ]

(* A donated operand that another operation consumed is dead: ewma names itself,
   with and without an axis, at a narrow float too, which it computes at float32
   (4815ee9a7a). *)
let test_ewma_dead () =
  let dead () =
    let x = Nx.create Nx.float16 [| 3 |] [| 1.; 2.; 3. |] in
    ignore (Nx.neg (Nx.donate x));
    x
  in
  names "ewma" (fun () -> Nx.ewma ~axis:0 ~alpha:0.5 (dead ()));
  names "ewma" (fun () -> Nx.ewma ~alpha:0.5 (dead ()))

(* Each device reads the axis whole. *)
let test_ewma_cut () =
  let x =
    Nx.create Nx.float64 [| 4; 6 |] (Array.init 24 (fun i -> Float.of_int i))
  in
  let reads_whole f =
    raises_match
      (fun e ->
        Exn.invalid_arg ~sub:"Nx.ewma: " e
        && Exn.invalid_arg ~sub:"Nx.place (" e)
      f
  in
  reads_whole (fun () ->
      Nx.ewma ~axis:1 ~alpha:0.5 (Nx.place (Cpu3.split ~axis:1) x));
  reads_whole (fun () -> Nx.ewma ~alpha:0.5 (Nx.place (Cpu3.split ~axis:0) x));
  equal ~msg:"along an axis no device cuts" (array float_exact)
    (Nx.to_array (Nx.ewma ~axis:1 ~alpha:0.5 x))
    (Nx.to_array
       (Nx.place Nx.Host.on
          (Nx.ewma ~axis:1 ~alpha:0.5 (Nx.place (Cpu3.split ~axis:0) x))))

(* Nx.Prim's reductions and scans *)

let reduce1 r axes x =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Iarray.of_array (Nx.shape x));
           axes;
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
           at = None;
         })
  in
  y

let scan1 r axis x =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Iarray.of_array (Nx.shape x));
         axis;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
         at = None;
       })

(* Logsumexp and Moments take floats alone; an extreme of no term is refused,
   each naming the caller. *)
let test_prim_refusals () =
  let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
  let says sub f =
    raises_match
      (fun e -> Exn.invalid_arg ~sub:"t: " e && Exn.invalid_arg ~sub e)
      f
  in
  says "Logsumexp" (fun () -> scan1 (Logsumexp (0, Nx.int32)) 0 x);
  says "Moments" (fun () -> reduce1 (Moments (0, Nx.int32)) [| 0 |] x);
  says "Moments" (fun () -> scan1 (Moments (0, Nx.int32)) 0 x);
  says "no term" (fun () ->
      reduce1 (Arg (Max, 0, Nx.int32)) [| 0 |] (Nx.zeros Nx.int32 [| 0 |]))

(* A scan with a carry continues each slice from the carry's element. Over a
   value cut along its axis, each window's results take the terms before it, so
   it raises naming the place that reads it whole, and placed so it has the bits
   of the scan on one device (bdbf157b0f). *)
let test_carried_cut () =
  let s : int iarray = [| 5; 3 |] in
  let xs = Array.init 15 (fun i -> Int32.of_int ((i * 7) - 40)) in
  let carried x c : (int32, D.int32_elt, _) Nx.t =
    Nx.Prim.eval ~by:"t"
      (Scan
         {
           layout = L.contiguous s;
           axis = 0;
           prog = identity D.Int32;
           reduction = Monoid (Sum, 0, D.Int32);
           loads = [| Plain x |];
           carry = Some (Nx.Prim.Any c);
           at = None;
         })
  in
  let x = value (A.of_array D.Int32 s xs) in
  let c = value (A.of_array D.Int32 [| 1; 3 |] [| 100l; -7l; 2l |]) in
  let want = read (carried x c) in
  let cut = Nx.place (Cpu3.split ~axis:0) x and on = Nx.place Cpu3.on c in
  raises_match (Exn.invalid_arg ~sub:"Nx.place (Placement.on s) x") (fun () ->
      carried cut on);
  equal (array int32) want (read (carried (Nx.place Cpu3.on cut) on))

let () =
  exit
    (run "nx reduce"
       [
         group "reductions and scans"
           [
             prop
               "over int32, sum, prod, max and min are the folds of their \
                elements, axes dropped or kept"
               ~examples:int32_examples int32_case law_int32;
             prop
               "over int32, the scans are the running folds along an axis or \
                in C order"
               int32_case law_int32_scan;
             prop
               "on the host each computes the dtypes it states as nx.cpu does, \
                narrow floats at float32, and refuses the others"
               ~examples:refused_example (any_case ()) law_host;
             prop "on nx.cpu a reduction's bits depend on its shape alone"
               (any_case ()) law_views;
             prop "a NaN term makes every float result it reaches a NaN"
               nan_case law_nans;
             cases
               "each raises naming itself for an axis outside the rank or \
                repeated, or a dtype it does not take"
               ~id:(fun (id, _, _) -> id)
               raise_rows
               (fun (_, name, f) -> names name f);
             test "a refused axis is named as the caller passed it"
               test_axis_printed;
           ];
         group "sum"
           [
             test "adds from +0: -0. terms give +0., none gives 0"
               test_sum_from_zero;
           ];
         group "prod"
           [
             prop "a complex product is its terms' product from the first"
               ~examples:(long_zcases ~product:true)
               (zcase ~product:true) law_complex_prod;
             test "of none is 1, of one term that term" test_prod_cases;
           ];
         group "max and min"
           [
             prop
               "complex extremes order real parts, then imaginary parts, -0 \
                below +0, a NaN past every number"
               ~examples:(long_zcases ~product:false)
               (zcase ~product:false) law_complex_extremes;
             test "have the values their order gives" test_extreme_cases;
             cases
               "an empty reduced axis gives the empty result where it has no \
                element"
               ~id:(fun (id, _, _) -> id)
               empty_results
               (fun (_, want, f) -> equal (array int) want (f ()));
             cases
               "an empty reduced axis raises where the result has an element"
               ~id:(fun (id, _, _) -> id)
               empty_refusals
               (fun (_, name, f) -> names name f);
           ];
         group "mean"
           [
             prop
               "a float or complex mean is the carrier's sum over the count, \
                rounded once"
               ~examples:mean_examples float_mean_case law_float_mean;
             test "has the values its definition gives" test_mean_values;
             prop ~count:300
               "an integer mean is the exact mean truncated toward zero"
               int_case law_int_mean;
             test "integer means of extremes are exact" test_int_means;
             cases
               "an integer mean raises over an empty axis and past the \
                elements it computes exactly"
               ~id:(fun (id, _, _) -> id)
               int_mean_refusals
               (fun (_, name, f) -> names name f);
           ];
         group "all, any and count"
           [
             prop "all is true where every element is non-zero as cast reads it"
               ~examples:truth_examples truth_case law_all;
             prop "any is true where some element is non-zero"
               ~examples:truth_examples truth_case law_any;
             prop "count is the number of trues" ~examples:truth_examples
               truth_case law_count;
             test "have the values their definition gives" test_truth_values;
           ];
         group "argmax and argmin"
           [
             prop "are the first extreme of float64 in maximum's order" arg_case
               law_arg_order;
             prop
               "the element at argmax has max's bits, the first such, or is \
                the first NaN"
               ~examples:arg_examples (any_case ()) law_arg_extreme;
             prop "complex positions are the first extremes'"
               ~examples:(long_zcases ~product:false)
               (zcase ~product:false) law_complex_args;
             test "have the values their definition gives" test_arg_values;
           ];
         group "logsumexp and logmeanexp"
           [
             prop "float32 is within nx.cpu's bound of the definition" lse_case
               law_lse;
             cases
               "logsumexp is within nx.cpu's bound of correctly rounded rows"
               ~id:row_id (lse_rows "logsumexp")
               (check_row
                  ~f:(fun u x -> at_row_dtype u (fun x -> Nx.logsumexp x) x)
                  ~extra:(fun _ _ _ -> 0.));
             cases
               "logmeanexp is within logsumexp's bound of correctly rounded \
                rows, with log n's and one rounding"
               ~id:row_id (lse_rows "logmeanexp")
               (check_row
                  ~f:(fun u x -> at_row_dtype u (fun x -> Nx.logmeanexp x) x)
                  ~extra:(fun u n r ->
                    (4. *. u *. ln (Float.of_int n)) +. (u *. Float.abs r)));
             test "have the values their definition gives" test_lse_values;
             narrow "logsumexp" (fun x -> Nx.logsumexp ~axes:[ 1 ] x);
             narrow "logmeanexp" (fun x -> Nx.logmeanexp ~axes:[ 1 ] x);
           ];
         group "cumsum"
           [
             prop
               "the cumsum of a prefix of the axis is the prefix of the \
                cumsum, bit for bit"
               ~examples:prefix_examples prefix_case law_cumsum_prefix;
             test "a complex cumsum adds the parts apart" test_complex_cumsum;
             test
               "a narrow float's cumsum accumulates at float32 and rounds each \
                result once"
               test_narrow_cumsum;
           ];
         group "cumprod"
           [
             prop "a complex cumprod is its terms' running product"
               ~examples:(long_zcases ~product:true)
               (zcase ~product:true) law_complex_cumprod;
           ];
         group "cummax and cummin"
           [
             prop "complex running extremes keep the first extreme and a NaN"
               ~examples:(long_zcases ~product:false)
               (zcase ~product:false) law_complex_scans;
           ];
         group "logcumsumexp"
           [
             prop "below float64 is within nx.cpu's bound over each prefix"
               lcse_case law_lcse_bound;
             prop
               "is NaN from the first NaN on, +∞ from a +∞, -∞ while every \
                term is -∞"
               lcse_case law_lcse_specials;
             cases
               "float32 and float64 are within nx.cpu's bound of correctly \
                rounded rows"
               ~id:row_id (lse_rows "logsumexp") check_lcse_row;
             prop "prefixes, C order and cuts over devices keep its bits"
               lcse_bits_case law_lcse_bits;
             test "has the values its definition gives" test_lcse_values;
           ];
         group "ewma"
           [
             prop "is within nx.mli's bound of the recurrence"
               ~examples:ewma_examples series law_ewma;
             prop "a prefix's average is the average's prefix, bit for bit"
               series law_ewma_prefix;
             test "has the values the recurrence gives" test_ewma_values;
             cases "raises naming itself unless 0 < alpha <= 1" ~id:fst
               alpha_rows (fun (_, f) -> names "ewma" f);
             test "a dead operand raises naming ewma" test_ewma_dead;
             test "reads its axis whole on each device" test_ewma_cut;
             narrow "ewma" (fun x -> Nx.ewma ~axis:1 ~alpha:0.3 x);
           ];
         group "Prim reductions and scans"
           [
             test "Logsumexp, Moments and Arg's rules raise naming the caller"
               test_prim_refusals;
             test
               "a carried scan over a value cut along its axis raises, and \
                placed whole is the whole's"
               test_carried_cut;
           ];
       ])
