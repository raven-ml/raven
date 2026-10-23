(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's filters on the host. Each is its reduction of every window gathered
   by its positions, which index arithmetic gives: the same terms in the same
   order, so the same bits, over every dtype a filter takes, views, windows
   past and at their axes' extents, and NaN and signed zeros. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

type host = Nx.host

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let value a : (_, _, host) Nx.t = Nx.Repr.of_array Nx.Host.v a
let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))

let bits_of (type v s) (a : (v, s) A.t) =
  let a = A.copy a in
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

(* Elements *)

(* Floats a window's extreme or mean turns on: NaNs of both signs, signed
   zeros, infinities, and numbers within every float format's range. *)
let specials =
  [|
    Float.nan; -.Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity; 1.;
    -1.; 0.5; -1.5; 3.; -6.;
  |]

let float_gen =
  let open Gen in
  frequency
    [
      ( 1,
        map (fun i -> specials.(i)) (int_range 0 (Array.length specials - 1)) );
      (1, float_range (-8.) 8.);
    ]

(* [a] at [dt], as nx.cpu casts. *)
let cast (type v s w r) (dt : (w, r) D.t) (a : (v, s) A.t) : (w, r) A.t =
  let dst = A.create Rig.host dt (L.shape (A.layout a)) in
  match Nx_kernels_support.map1 (module Nx_cpu) Cast ~dst a with
  | A.Done -> dst
  | _ -> failwith "the host's kernel declined a cast"

(* [s]'s elements of [dt] of random bytes, booleans 0 or 1. *)
let raw : type v s. (v, s) D.t -> int array -> (v, s) A.t Gen.t =
 fun dt s ->
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

(* [n] elements of [dt]: floats and complex parts of [float_gen], others of
   random bytes. *)
let elements : type v s. (v, s) D.t -> int array -> (v, s) A.t Gen.t =
 fun dt s ->
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let size = constant n in
  let complex () =
    let+ parts = array ~size (pair float_gen float_gen) in
    Array.map (fun (re, im) -> { Complex.re; im }) parts
  in
  match dt with
  | D.Complex64 -> map (A.of_array dt s) (complex ())
  | D.Complex128 -> map (A.of_array dt s) (complex ())
  | _ -> (
      match D.kind dt with
      | D.Float ->
          let+ xs = array ~size float_gen in
          cast dt (A.of_array D.Float64 s xs)
      | _ -> raw dt s)

(* Views *)

(* How an operand is viewed: as made, flipped along an axis, with its axes
   reversed, or broadcast along its first axis from one element. *)
type view = Plain | Flipped of int | Reversed | Broadcast

let pp_view ppf = function
  | Plain -> Format.pp_print_string ppf "plain"
  | Flipped a -> Format.fprintf ppf "flipped along %d" a
  | Reversed -> Format.pp_print_string ppf "axes reversed"
  | Broadcast -> Format.pp_print_string ppf "broadcast"

(* [x] of shape [s] through [v]. *)
let viewed (type v s) (dt : (v, s) D.t) s v : (v, s, host) Nx.t Gen.t =
  let open Gen in
  let r = Array.length s in
  match v with
  | Plain -> map value (elements dt s)
  | Flipped a -> map (fun x -> Nx.flip ~axes:[ a ] (value x)) (elements dt s)
  | Reversed ->
      let rev = Array.init r (fun i -> s.(r - 1 - i)) in
      map (fun x -> Nx.transpose (value x)) (elements dt rev)
  | Broadcast ->
      let one = Array.mapi (fun i e -> if i = 0 then 1 else e) s in
      map (fun x -> Nx.broadcast_to s (value x)) (elements dt one)

let view_of r =
  let open Gen in
  if r = 0 then constant Plain
  else
    frequency
      [
        (3, constant Plain);
        (1, map (fun a -> Flipped a) (int_range 0 (r - 1)));
        (1, constant Reversed);
        (1, constant Broadcast);
      ]

(* Windows *)

(* A filter's operand, kernel and stride, [None] where it defaults. *)
type 'c case = {
  x : 'c;
  kernel : int array;
  stride : int array option;
  view : view;
}

(* Up to four axes of up to seven elements, an empty one at times; up to
   three filtered, each kernel extent 1, the axis's extent, one past it or
   between; each stride 1, the kernel extent, one past it or two past. *)
let windows_gen =
  let open Gen in
  let* s =
    array ~size:(frequency [ (1, constant 0); (6, int_range 1 4) ])
      (frequency [ (1, constant 0); (8, int_range 1 7) ])
  in
  let r = Array.length s in
  let* k =
    frequency
      [ (1, constant 0); (6, int_range (Stdlib.min r 1) (Stdlib.min r 3)) ]
  in
  let last = Array.sub s (r - k) k in
  let* kernel =
    Gen.map Array.of_list
      (Gen.map List.rev
         (Array.fold_left
            (fun acc d ->
              let* l = acc in
              let+ e =
                frequency
                  [
                    (2, constant 1); (2, constant (Stdlib.max 1 d));
                    (1, constant (d + 1)); (3, int_range 1 (Stdlib.max 1 d));
                  ]
              in
              e :: l)
            (constant []) last))
  in
  let* stride =
    frequency
      [
        (1, constant None);
        ( 3,
          map Option.some
            (Gen.map Array.of_list
               (Array.fold_left
                  (fun acc e ->
                    let* l = acc in
                    let+ st =
                      of_list ~pp:Format.pp_print_int [ 1; e; e + 1; e + 2 ]
                    in
                    l @ [ st ])
                  (constant []) kernel)) );
      ]
  in
  let+ view = view_of r in
  (s, kernel, stride, view)

let pp_case pp_x ppf c =
  Format.fprintf ppf "%a, kernel %a, stride %s, %a" pp_x c.x pp_ints c.kernel
    (match c.stride with
    | None -> "default"
    | Some s -> Format.asprintf "%a" pp_ints s)
    pp_view c.view

let pp_value ppf x =
  Format.fprintf ppf "%a %a" D.pp (Nx.dtype x) pp_ints (Nx.shape x)

(* The positions in [flatten x] of each window's elements, [[out…; taps]],
   the taps in C order of their places in the window. *)
let positions s ~kernel ~stride =
  let r = Array.length s and k = Array.length kernel in
  let lead = r - k in
  let count i =
    let d = s.(lead + i) and e = kernel.(i) in
    if e > d then 0 else ((d - e) / stride.(i)) + 1
  in
  let out = Array.append (Array.sub s 0 lead) (Array.init k count) in
  let taps = Array.fold_left ( * ) 1 kernel in
  Nx.init D.Int64 (Array.append out [| taps |]) (fun i ->
      let t = ref i.(r) and tap = Array.make k 0 in
      for a = k - 1 downto 0 do
        tap.(a) <- !t mod kernel.(a);
        t := !t / kernel.(a)
      done;
      let p = ref 0 in
      for a = 0 to r - 1 do
        let j =
          if a < lead then i.(a)
          else (i.(a) * stride.(a - lead)) + tap.(a - lead)
        in
        p := (!p * s.(a)) + j
      done;
      Int64.of_int !p)

let stride_of c = Option.value c.stride ~default:c.kernel

let cover_windows s c ~nan ~zeros =
  let r = Array.length s and k = Array.length c.kernel in
  let st = stride_of c in
  let along f = List.exists Fun.id (List.init k f) in
  cover "no filtered axis" (k = 0);
  cover "three filtered axes" (k = 3);
  let extent i = s.(r - k + i) in
  cover "a kernel extent of an axis's"
    (along (fun i -> c.kernel.(i) = extent i));
  cover "a kernel extent past an axis's"
    (along (fun i -> c.kernel.(i) > extent i));
  cover "one-element windows"
    (k > 0 && Array.for_all (fun e -> e = 1) c.kernel);
  cover "overlapping windows" (along (fun i -> st.(i) < c.kernel.(i)));
  cover "a stride past its kernel extent"
    (along (fun i -> st.(i) > c.kernel.(i)));
  cover "the stride defaulted" (c.stride = None);
  cover "an empty operand" (Array.mem 0 s);
  cover "a view" (c.view <> Plain);
  cover "a NaN in a window" nan;
  cover "both zeros in a window" zeros

(* [a]'s elements where they are floats. *)
let floats (type v s) (dt : (v, s) D.t) (a : v array) : float array =
  match dt with
  | D.Float16 -> a
  | D.Bfloat16 -> a
  | D.Float32 -> a
  | D.Float64 -> a
  | D.Float8_e4m3fn -> a
  | D.Float8_e5m2 -> a
  | D.Float4_e2m1fn -> a
  | _ -> [||]

(* Whether some window of the gathered [g], [[out…; taps]], holds a NaN,
   and some both zeros. *)
let specials_in (type v s) (g : (v, s, host) Nx.t) =
  let s = Nx.shape g in
  let a = floats (Nx.dtype g) (Nx.to_array g) in
  let taps = s.(Array.length s - 1) in
  let windows = if taps = 0 then 0 else Array.length a / taps in
  let some f = List.exists f (List.init windows Fun.id) in
  let has w p = Array.exists p (Array.sub a (w * taps) taps) in
  let neg z = z = 0. && 1. /. z < 0. and pos z = z = 0. && 1. /. z > 0. in
  (some (fun w -> has w Float.is_nan), some (fun w -> has w neg && has w pos))

(* Extremes *)

type any = Any : ('v, 's, host) Nx.t -> any

let any_case =
  let open Gen in
  with_pp (pp_case (fun ppf (Any x) -> pp_value ppf x))
  @@ let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) D.all in
     let* s, kernel, stride, view = windows_gen in
     let+ x = viewed dt s view in
     { x = Any x; kernel; stride; view }

(* [s]'s elements of [dt] of fixed bytes, booleans 0 or 1. *)
let fixed (type v s) (dt : (v, s) D.t) s : (v, s) A.t =
  let n = Array.fold_left ( * ) 1 s in
  let byte i =
    match dt with
    | D.Bool -> Char.chr (i land 1)
    | _ -> Char.chr ((i * 37) land 0xff)
  in
  let bytes = String.init (D.bytes dt n) byte in
  A.v dt (L.contiguous s) (Rig.Buffer.of_string bytes)

(* A float32 case of NaNs and signed zeros in windows, one of three
   filtered axes, and one of each dtype: every run reaches them. *)
let extreme_examples =
  let float32 xs s kernel stride =
    {
      x = Any (value (A.of_array D.Float32 s xs));
      kernel; stride; view = Plain;
    }
  in
  [
    float32 [| 1.; Float.nan; -0.; 0.; -.Float.nan; 2. |] [| 6 |] [| 2 |] None;
    float32 [| 0.; -0.; 3.; -0.; 0.; 5. |] [| 2; 3 |] [| 2; 2 |]
      (Some [| 1; 1 |]);
    float32 (Array.init 24 Float.of_int) [| 2; 3; 4 |] [| 2; 2; 3 |]
      (Some [| 1; 2; 1 |]);
  ]
  @ List.map
      (fun (D.Any dt) ->
        let a = fixed dt [| 3; 5 |] in
        {
          x = Any (value a);
          kernel = [| 2; 2 |];
          stride = Some [| 1; 2 |];
          view = Plain;
        })
      D.all

let law_extremes c =
  let (Any x) = c.x in
  let s = Nx.shape x and stride = stride_of c in
  let g = Nx.take (positions s ~kernel:c.kernel ~stride) x in
  let nan, zeros = specials_in g in
  cover_windows s c ~nan ~zeros;
  List.iter
    (fun (D.Any d) -> cover (D.name d) (D.name d = D.name (Nx.dtype x)))
    D.all;
  let last = [ Array.length s ] in
  let check name filter reduce =
    let got = filter ~kernel_size:c.kernel ?stride:c.stride x in
    let want = reduce ~axes:last g in
    equal ~msg:(name ^ "'s shape") (array int) (Nx.shape want) (Nx.shape got);
    equal ~msg:name (array int)
      (bits_of (host_array want))
      (bits_of (host_array got))
  in
  check "maximum_filter" Nx.maximum_filter (fun ~axes g -> Nx.max ~axes g);
  check "minimum_filter" Nx.minimum_filter (fun ~axes g -> Nx.min ~axes g)

(* Means *)

type floats = Floats : (float, 's, host) Nx.t -> floats

type float_dtype = F : (float, 's) D.t -> float_dtype

let float_dtypes =
  [
    F D.Float16; F D.Bfloat16; F D.Float32; F D.Float64; F D.Float8_e4m3fn;
    F D.Float8_e5m2; F D.Float4_e2m1fn;
  ]

let float_case =
  let open Gen in
  with_pp (pp_case (fun ppf (Floats x) -> pp_value ppf x))
  @@ let* (F dt) = of_list ~pp:(fun ppf (F d) -> D.pp ppf d) float_dtypes in
     let* s, kernel, stride, view = windows_gen in
     let+ x = viewed dt s view in
     { x = Floats x; kernel; stride; view }

let mean_examples =
  let case (F dt) xs s kernel stride =
    let x = value (cast dt (A.of_array D.Float64 s xs)) in
    { x = Floats x; kernel; stride; view = Plain }
  in
  [
    case (F D.Float32) [| 1.; Float.nan; -0.; -0.; -.Float.nan; 2. |] [| 6 |]
      [| 2 |] None;
    case (F D.Float32) [| 0.; -0.; 3.; -0.; 0.; 5. |] [| 2; 3 |] [| 2; 2 |]
      (Some [| 1; 1 |]);
    case (F D.Float32) (Array.init 24 Float.of_int) [| 2; 3; 4 |]
      [| 2; 2; 3 |] (Some [| 1; 2; 1 |]);
  ]
  @ List.map
      (fun d ->
        case d [| 1.; 2.; -0.5; 3.; 0.; -1. |] [| 6 |] [| 4 |] (Some [| 1 |]))
      float_dtypes

let law_mean c =
  let (Floats x) = c.x in
  let s = Nx.shape x and stride = stride_of c in
  let g = Nx.take (positions s ~kernel:c.kernel ~stride) x in
  let nan, zeros = specials_in g in
  cover_windows s c ~nan ~zeros;
  List.iter
    (fun (F d) -> cover (D.name d) (D.name d = D.name (Nx.dtype x)))
    float_dtypes;
  let got = Nx.uniform_filter ~kernel_size:c.kernel ?stride:c.stride x in
  let want = Nx.mean ~axes:[ Array.length s ] g in
  equal ~msg:"the shape" (array int) (Nx.shape want) (Nx.shape got);
  equal (array int) (bits_of (host_array want)) (bits_of (host_array got))

(* Errors *)

let test_errors () =
  let x = value (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  let names name f =
    raises_match (Exn.invalid_arg ~substring:("Nx." ^ name ^ ": ")) f
  in
  List.iter
    (fun (name, f) ->
      let raises kernel_size stride =
        names name (fun () -> ignore (f ~kernel_size ?stride x))
      in
      raises [| 1; 1; 1 |] None;
      raises [| 1; 1 |] (Some [| 1 |]);
      raises [| 1; 1 |] (Some [| 1; 1; 1 |]);
      raises [| 0; 1 |] None;
      raises [| 1 |] (Some [| 0 |]))
    [
      ( "maximum_filter",
        fun ~kernel_size ?stride x -> Nx.maximum_filter ~kernel_size ?stride x
      );
      ( "minimum_filter",
        fun ~kernel_size ?stride x -> Nx.minimum_filter ~kernel_size ?stride x
      );
      ( "uniform_filter",
        fun ~kernel_size ?stride x -> Nx.uniform_filter ~kernel_size ?stride x
      );
    ]

let () =
  exit
    (run "nx filters"
       [
         group "filters"
           [
             prop ~examples:extreme_examples
               "maximum_filter and minimum_filter are each window's max and min"
               any_case law_extremes;
             prop ~examples:mean_examples "uniform_filter is each window's mean"
               float_case law_mean;
             test
               "a rank below the kernel's, a stride of another length, or a \
                kernel or stride below 1, raises"
               test_errors;
           ];
       ])
