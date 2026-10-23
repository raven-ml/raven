(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters through every kernel library the host runs: a gather and
   an add-scatter are adjoint, and a scatter into its own operand gives the
   fresh result. nx.cpu's group checks what Nx_kernel.S states, element by
   element against a reference built here from the elements' bits: zero reads
   and dropped writes outside the axis, a target's updates in C order, a narrow
   float's sum in float32 rounded once, the first NaN of an extreme quieted;
   at every dtype and through strided, reversed, permuted and broadcast
   views. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support
open Elements
(* Cases *)

let all_dtypes = D.all

type gather_case = { axis : int; idx : A.any; x : A.any; gviews : string list }

let pp_gather ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "gather along %d of %a %a by %a (%s)" c.axis D.pp
    (A.dtype x) pp_ints (shape_of c.x) pp_ints (shape_of c.idx)
    (String.concat ", " c.gviews)

(* Shapes of rank 1 to 4, extents 0 to 5, now and then a long one. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 5) in
  let long =
    let* n = int_range 100 3000 in
    let+ before = array ~size:(int_range 0 1) (int_range 1 3) in
    Array.append before [| n |]
  in
  frequency [ (6, small); (1, long) ]

let gather_gen ?(dtypes = all_dtypes) ?(shapes = shape_gen) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* xs = shapes in
  let r = Array.length xs in
  let* axis = int_range 0 (r - 1) in
  let* n = frequency [ (4, int_range 0 6); (1, int_range 7 40) ] in
  let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
  (* CR: Value views here, in scatter_gen and test_sort never broadcast.
     Draw their broadcast axes explicitly; keep the reference laws. Seed
     broadcasts on and off each operation's axis: in-range gathers, scatters
     with an update and an untouched target, sorts keeping at least one value.
     Cover nonempty actual layouts with an extent > 1 and stride 0. *)
  let* xv = view_of r in
  let* bc =
    array ~size:(constant r)
      (frequency [ (3, constant false); (1, constant true) ])
  in
  let broadcast =
    List.filter (fun i -> bc.(i) && i <> axis) (List.init r Fun.id)
  in
  let* iv = view_of ~broadcast r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let x = operand d xs xv (fun _ -> element d rs) in

  {
    axis;
    idx = int64s is iv (fun () -> position rs xs.(axis));
    x;
    gviews = view_names xv @ List.map (( ^ ) "idx ") (view_names iv);
  }

(* Gathers *)

let gather_on (b : Support.backend) ~axis idx x =
  let module K = (val b.kernels) in
  let (A.Any x) = x in
  let idx = A.expect D.Int64 idx in
  let dst = A.create Rig.host (A.dtype x) (L.shape (A.layout idx)) in
  match K.gather ~axis ~dst idx x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let gather_expected c =
  let xs = shape_of c.x and is = shape_of c.idx in
  let es = elements c.x and ps = positions c.idx in
  let z = zero (dtype_of c.x) in
  Array.init (total is) (fun k ->
      let p = ps.(k) in
      if p < 0L || p >= Int64.of_int xs.(c.axis) then z
      else
        let i = index_of is k in
        i.(c.axis) <- Int64.to_int p;
        es.(flat xs i))

let gather_agrees (b : Support.backend) c =
  match gather_on b ~axis:c.axis c.idx c.x with
  | None -> failf "%s declined a gather" b.name
  | Some y -> equal (array string) (gather_expected c) (elements y)

let law_gather_reference (b : Support.backend) c =
  let d = Int64.of_int (shape_of c.x).(c.axis) in
  cover "out of range"
    (Array.exists (fun p -> p < 0L || p >= d) (positions c.idx));
  cover "broadcast positions" (List.mem "idx broadcast" c.gviews);
  gather_agrees b c

(* Scatters *)

type scatter_case = {
  combine : Nx_kernel.combine;
  saxis : int;
  into : A.any;
  sidx : A.any;
  updates : A.any;
  sviews : string list;
}

let combine_name = function
  | Nx_kernel.Set -> "Set"
  | Add -> "Add"
  | Max -> "Max"
  | Min -> "Min"

let pp_scatter ppf c =
  let (A.Any x) = c.into in
  Format.fprintf ppf "scatter %s along %d into %a %a of %a (%s)"
    (combine_name c.combine) c.saxis D.pp (A.dtype x) pp_ints (shape_of c.into)
    pp_ints (shape_of c.updates)
    (String.concat ", " c.sviews)

let takes_add (D.Any dt) = not (D.is D.Boolean dt)

let combine_gen d =
  Gen.of_list
    ~pp:(fun ppf c -> Format.pp_print_string ppf (combine_name c))
    (if takes_add d then Nx_kernel.[ Set; Add; Max; Min ]
     else Nx_kernel.[ Set; Max; Min ])

let scatter_gen ?(dtypes = all_dtypes) ?combines ?(draw = element)
    ?(shapes = shape_gen)
    ?(ms = Gen.frequency [ (4, Gen.int_range 0 8); (1, Gen.int_range 9 60) ]) ()
    =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* combine =
    match combines with
    | None -> combine_gen d
    | Some cs ->
        of_list
          ~pp:(fun ppf c -> Format.pp_print_string ppf (combine_name c))
          cs
  in
  let* ts = shapes in
  let r = Array.length ts in
  let* axis = int_range 0 (r - 1) in
  let* m = ms in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  let* tv = view_of r in
  let* uv = view_of r in
  let* bc =
    array ~size:(constant r)
      (frequency [ (3, constant false); (1, constant true) ])
  in
  let broadcast =
    List.filter (fun i -> bc.(i) && i <> axis) (List.init r Fun.id)
  in
  let* iv = view_of ~broadcast r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let into = operand d ts tv (fun _ -> draw d rs) in
  let updates = operand d us uv (fun _ -> draw d rs) in

  {
    combine;
    saxis = axis;
    into;
    sidx = int64s us iv (fun () -> position rs ts.(axis));
    updates;
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv)
      @ List.map (( ^ ) "idx ") (view_names iv);
  }

let scatter_on (b : Support.backend) ?(in_place = false) c =
  let module K = (val b.kernels) in
  let (A.Any into) = c.into in
  let u = A.expect (A.dtype into) c.updates in
  let idx = A.expect D.Int64 c.sidx in
  let dst, into =
    if in_place then
      let d = A.copy into in
      (d, d)
    else (A.create Rig.host (A.dtype into) (L.shape (A.layout into)), into)
  in
  match K.scatter c.combine ~unique:false ~axis:c.saxis ~dst ~into idx u with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Whether an extreme of a and b keeps a: the first NaN, else the greater
   (lesser), a on a tie. *)
let keeps combine ~nan_a ~nan_b cmp =
  if nan_a then true
  else if nan_b then false
  else if combine = Nx_kernel.Max then cmp >= 0
  else cmp <= 0

(* An expected element: its bits, or any NaN of a narrow float. *)
type want = Bits of string | Some_nan

(* The combine of the updates [us] into [x], at [d], in order. *)
let combined (D.Any dt as d) combine x us =
  match us with
  | [] -> Bits x
  | _ -> (
      let w = String.length x in
      match combine with
      | Nx_kernel.Set -> Bits (List.nth us (List.length us - 1))
      | Add when is_narrow d ->
          let s =
            List.fold_left
              (fun s u ->
                let y = decode d u in
                if Float.is_nan s then s
                else if Float.is_nan y then y
                else
                  let t = round32 (s +. y) in
                  if t = 0. then 0. else t)
              (let v = decode d x in
               if Float.is_nan v then v else if v = 0. then 0. else v)
              us
          in
          if Float.is_nan s then Some_nan else Bits (encode d s)
      | Add when D.is D.Float dt ->
          let z = if w = 4 then f32 0. else f64 0. in
          Bits (List.fold_left (add_bits ~w) (add_bits ~w z x) us)
      | Add when D.is D.Complex dt ->
          let h = w / 2 in
          let part e k = String.sub e (k * h) h in
          let z = if h = 4 then f32 0. else f64 0. in
          let sum k =
            List.fold_left
              (fun s u -> add_bits ~w:h s (part u k))
              (add_bits ~w:h z (part x k))
              us
          in
          Bits (sum 0 ^ sum 1)
      | Add ->
          Bits
            (int_bits d w
               (List.fold_left
                  (fun s u -> Int64.add s (int_value d u))
                  (int_value d x) us))
      | (Max | Min) when D.is D.Boolean dt ->
          (* Or and And of truths, stored as 0 or 1. *)
          let op = if combine = Nx_kernel.Max then ( || ) else ( && ) in
          let truth e = e <> "\000" in
          let r = List.fold_left (fun t u -> op t (truth u)) (truth x) us in
          Bits (if r then "\001" else "\000")
      | Max | Min ->
          let pick a b =
            let keep =
              if D.is D.Complex dt then
                let h = w / 2 in
                let get = if h = 4 then get32 else get64 in
                let ra = get a 0
                and ia = get a h
                and rb = get b 0
                and ib = get b h in
                let c = compare_floats ra rb in
                let c = if c = 0 then compare_floats ia ib else c in
                keeps combine
                  ~nan_a:(Float.is_nan ra || Float.is_nan ia)
                  ~nan_b:(Float.is_nan rb || Float.is_nan ib)
                  c
              else if D.is D.Float dt then
                let get e =
                  if is_narrow d then decode d e
                  else if w = 4 then get32 e 0
                  else get64 e 0
                in
                let x = get a and y = get b in
                keeps combine ~nan_a:(Float.is_nan x) ~nan_b:(Float.is_nan y)
                  (compare_floats x y)
              else
                let x = int_value d a and y = int_value d b in
                let c =
                  if D.is D.Signed dt then Int64.compare x y
                  else Int64.unsigned_compare x y
                in
                keeps combine ~nan_a:false ~nan_b:false c
            in
            if keep then a else b
          in
          Bits (quiet d (List.fold_left pick x us)))

(* The updates that land on each target of [c], in C order. *)
let lands c =
  let ts = shape_of c.into and us = shape_of c.updates in
  let ues = elements c.updates in
  let lands = Array.make (total ts) [] in
  Array.iteri
    (fun k p ->
      if p >= 0L && p < Int64.of_int ts.(c.saxis) then begin
        let i = index_of us k in
        i.(c.saxis) <- Int64.to_int p;
        let t = flat ts i in
        lands.(t) <- ues.(k) :: lands.(t)
      end)
    (positions c.sidx);
  Array.map List.rev lands

let scatter_expected c =
  let d = dtype_of c.into in
  let ls = lands c in
  Array.mapi (fun t x -> combined d c.combine x ls.(t)) (elements c.into)

let is_nan_element d e = is_narrow d && Float.is_nan (decode d e)

let check_scatter c y =
  let d = dtype_of c.into in
  let got = elements y in
  let want = scatter_expected c in
  let wrong =
    List.filter_map Fun.id
      (List.mapi
         (fun t w ->
           match w with
           | Bits b when b = got.(t) -> None
           | Some_nan when is_nan_element d got.(t) -> None
           | _ -> Some t)
         (Array.to_list want))
  in
  let show t =
    Printf.sprintf "%d: %s, not %s" t
      (String.concat ""
         (List.map
            (fun ch -> Printf.sprintf "%02x" (Char.code ch))
            (List.of_seq (String.to_seq got.(t)))))
      (match want.(t) with
      | Some_nan -> "a NaN"
      | Bits b ->
          String.concat ""
            (List.map
               (fun ch -> Printf.sprintf "%02x" (Char.code ch))
               (List.of_seq (String.to_seq b))))
  in
  equal (list string) [] (List.map show (List.filteri (fun i _ -> i < 5) wrong))

let scatter_agrees (b : Support.backend) c =
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> check_scatter c y

(* Whether two of [c]'s updates share a target, and whether one lands outside
   the axis. *)
let sharing c =
  let ps = positions c.sidx in
  let d = (shape_of c.into).(c.saxis) in
  let inside =
    List.filter (fun p -> p >= 0L && p < Int64.of_int d) (Array.to_list ps)
  in
  ( List.length (List.sort_uniq compare inside) < List.length inside,
    List.length inside < Array.length ps )

let law_scatter_reference (b : Support.backend) c =
  let shared, dropped = sharing c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  cover "narrow float add" (c.combine = Nx_kernel.Add && is_narrow (dtype_of c.into));
  scatter_agrees b c

(* A signalling NaN of [d] from its bits, in one part of a complex number or
   both: the exponent's ones, the quiet bit clear, a payload below it and a
   sign drawn. float8_e4m3fn's NaNs are quiet: it draws its one NaN of each
   sign. *)
let signalling (D.Any dt as d) rs =
  let part, parts =
    match dt with
    | D.Complex64 -> (D.Any D.Float32, 2)
    | D.Complex128 -> (D.Any D.Float64, 2)
    | _ -> (d, 1)
  in
  let nan () =
    match quiet_fields part with
    | None -> String.make 1 (if Random.State.bool rs then '\x7F' else '\xFF')
    | Some (exp, frac) ->
        let below = Int64.(sub (shift_left 1L (frac - 1)) 1L) in
        let payload = Int64.succ (Random.State.int64 rs below) in
        let ones = Int64.(shift_left (sub (shift_left 1L exp) 1L) frac) in
        let sign =
          if Random.State.bool rs then Int64.shift_left 1L (exp + frac) else 0L
        in
        let v = Int64.(logor sign (logor ones payload)) in
        String.init
          ((1 + exp + frac) / 8)
          (fun j ->
            Char.chr (Int64.to_int (Int64.shift_right_logical v (8 * j)) land 0xFF))
  in
  if parts = 1 then nan ()
  else
    match Random.State.int rs 3 with
    | 0 -> nan () ^ element part rs
    | 1 -> element part rs ^ nan ()
    | _ -> nan () ^ nan ()

(* An element of [d], a signalling NaN one time in three. *)
let signals_often d rs =
  if Random.State.int rs 3 = 0 then signalling d rs else element d rs

let nan_dtypes =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float32;
      Any Float64;
      Any Complex64;
      Any Complex128;
    ]

let nan_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:nan_dtypes ~combines:Nx_kernel.[ Max; Min ]
       ~draw:signals_often ())

(* A Max or Min scatter is the first NaN among a target's element and its
   updates, quieted; a target no update reaches keeps its bits, a signalling
   NaN included. *)
let law_nan_extremes (b : Support.backend) c =
  let d = dtype_of c.into in
  let signals e = quiet d e <> e in
  let xs = elements c.into and ls = lands c in
  let some f = Array.exists Fun.id (Array.mapi f xs) in
  cover "a signalling NaN reaches a target"
    (some (fun t x -> ls.(t) <> [] && List.exists signals (x :: ls.(t))));
  cover "a signalling NaN no update reaches"
    (some (fun t x -> ls.(t) = [] && signals x));
  scatter_agrees b c

(* Laws for every library *)

(* Σ gather x p · y = Σ x · scatter Add 0 p y, in wrapping int64: a position
   outside the axis reads zero in one and drops its write in the other. *)
let law_adjoint (b : Support.backend) c =
  let d = D.Any D.Int64 in
  let rs = Random.State.make [| 3 |] in
  let ints s =
    operand d s
      (plain (Array.length s))
      (fun _ -> int_bits d 8 (Int64.of_int (Random.State.int rs 2001 - 1000)))
  in
  let xs = shape_of c.x and is = shape_of c.idx in
  let x = ints xs and y = ints is in
  let module K = (val b.kernels) in
  match gather_on b ~axis:c.axis c.idx x with
  | None -> ()
  | Some g -> (
      let into = A.of_array D.Int64 xs (Array.make (total xs) 0L) in
      let s =
        scatter_on b
          {
            combine = Add;
            saxis = c.axis;
            into = A.Any into;
            sidx = c.idx;
            updates = y;
            sviews = [];
          }
      in
      match s with
      | None -> ()
      | Some s ->
          let dot a b' =
            let a = A.to_array (A.expect D.Int64 a)
            and b' = A.to_array (A.expect D.Int64 b') in
            let acc = ref 0L in
            Array.iteri
              (fun i v -> acc := Int64.add !acc (Int64.mul v b'.(i)))
              a;
            !acc
          in
          equal int64 (dot x s) (dot g y))

(* A scatter into its own operand, donated, gives the fresh result. *)
let law_in_place (b : Support.backend) c =
  match (scatter_on b c, scatter_on b ~in_place:true c) with
  | Some fresh, Some own -> equal (array string) (elements fresh) (elements own)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

(* Set keeps each target's last update in C order: its bits. *)
let law_set_last (b : Support.backend) c =
  let shared, dropped = sharing c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  match scatter_on b c with None -> () | Some y -> check_scatter c y

(* Each operand as a C-contiguous copy of its view gives the same bits, on
   a second run too. *)
let law_add_bits (b : Support.backend) c =
  let plain (A.Any x) = A.Any (A.copy x) in
  let c' =
    { c with into = plain c.into; sidx = plain c.sidx; updates = plain c.updates }
  in
  cover "shared targets" (fst (sharing c));
  cover "through views" (c.sviews <> []);
  match (scatter_on b c, scatter_on b c, scatter_on b c') with
  | Some y, Some y', Some y'' ->
      equal ~msg:"again" (array string) (elements y) (elements y');
      equal ~msg:"contiguous" (array string) (elements y) (elements y'')
  | None, None, None -> ()
  | _ -> failf "%s declined some of three" b.name

(* A target's bits are those of a scatter of only its updates, in C order,
   into its element: the association is a function of their count. Checks
   the first 20 targets reached. *)
let law_add_alone (b : Support.backend) c =
  let d = dtype_of c.into in
  let ls = lands c in
  cover "a target several updates reach"
    (Array.exists (fun l -> List.length l > 1) ls);
  cover "a narrow float" (is_narrow d);
  cover "through views" (c.sviews <> []);
  let alone x us =
    let us = Array.of_list us and one = plain 1 in
    let n = Array.length us in
    let c =
      {
        combine = Nx_kernel.Add;
        saxis = 0;
        into = operand d [| 1 |] one (fun _ -> x);
        sidx = int64s [| n |] one (fun () -> 0L);
        updates = operand d [| n |] one (fun k -> us.(k));
        sviews = [];
      }
    in
    match scatter_on b c with
    | Some y -> (elements y).(0)
    | None -> failf "%s declined a scatter of one target" b.name
  in
  match scatter_on b c with
  | None -> ()
  | Some y ->
      let got = elements y and xs = elements c.into in
      let reached =
        List.filter (fun t -> ls.(t) <> []) (List.init (Array.length ls) Fun.id)
      in
      let first = List.filteri (fun i _ -> i < 20) reached in
      equal (list string)
        (List.map (fun t -> alone xs.(t) ls.(t)) first)
        (List.map (fun t -> got.(t)) first)

(* Bounds of Add *)

(* [x + y] and its rounding error. *)
let two_sum x y =
  let s = x +. y in
  let z = s -. x in
  (s, x -. (s -. z) +. (y -. z))

(* The sum of [xs] within 2^-100 of the sum of their magnitudes, and that
   sum: the running sum held in two parts. *)
let exact xs =
  let hi = ref 0. and lo = ref 0. and mag = ref 0. in
  List.iter
    (fun x ->
      let s, t = two_sum !hi x in
      hi := s;
      lo := !lo +. t;
      mag := !mag +. Float.abs x)
    xs;
  (!hi +. !lo, !mag)

(* The largest finite value of a narrow float. *)
let most_finite (D.Any dt) =
  match dt with
  | D.Float16 -> 65504.
  | D.Bfloat16 -> 0x1.FEp127
  | D.Float8_e4m3fn -> 448.
  | D.Float8_e5m2 -> 57344.
  | _ -> 6.

(* A finite element of a float or complex dtype: a draw of (-1, 1) times
   2^e, e over 31 binades below 2^30 or a narrow float's largest, so that
   sums of a few round at every width; float32's and a narrow float's
   rounded to their nearest. *)
let finite (D.Any dt as d) r =
  let top =
    if is_narrow d then Float.to_int (Float.log2 (most_finite d)) else 30
  in
  let v () =
    Float.ldexp (Random.State.float r 2. -. 1.) (top - Random.State.int r 31)
  in
  match dt with
  | D.Float32 -> f32 (round32 (v ()))
  | D.Float64 -> f64 (v ())
  | D.Complex64 -> f32 (round32 (v ())) ^ f32 (round32 (v ()))
  | D.Complex128 -> f64 (v ()) ^ f64 (v ())
  | _ -> encode d (v ())

let float_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt)
    all_dtypes

(* The parts of an element of [d] as values: one, or a complex number's
   two. *)
let parts (D.Any dt as d) e =
  let w = String.length e in
  if is_narrow d then [ decode d e ]
  else if D.is D.Complex dt then
    let h = w / 2 in
    let get = if h = 4 then get32 else get64 in
    [ get e 0; get e h ]
  else if w = 4 then [ get32 e 0 ]
  else [ get64 e 0 ]

(* Why a target's Add [got] of [x] and [us] is past
   [γ(n + 1) (|x| + Σ |ui|)] of the exact sum, part by part, [u] the unit
   roundoff of float32 for float32, narrow floats and complex64 and of
   float64 otherwise; a narrow float's sum is rounded once from a float32
   within it, so [got] lies between the roundings of its ends. Each part's
   magnitudes bound the complex number's moduli. A narrow sum that may
   overflow, a partial float32 sum past float32's largest finite value or
   the sum rounding past the dtype's, is not checked: [None]. *)
let past (D.Any dt as d) x us got =
  let n = Float.of_int (List.length us + 1) in
  let part = if D.is D.Complex dt then D.bits dt / 2 else D.bits dt in
  let u = if part = 64 then 0x1p-53 else 0x1p-24 in
  let g = n *. u /. (1. -. (n *. u)) in
  let terms = List.map (parts d) (x :: us) in
  let wrong =
    List.concat
      (List.mapi
         (fun k got ->
           let want, mag = exact (List.map (fun t -> List.nth t k) terms) in
           let bound = (g *. mag) +. (0x1p-100 *. mag) in
           if is_narrow d then
             if mag +. bound > 0x1.FFFFFEp127
                || Float.abs want +. bound > most_finite d
             then []
             else
               let lo = decode d (encode d (want -. bound))
               and hi = decode d (encode d (want +. bound)) in
               if lo <= got && got <= hi then []
               else [ Printf.sprintf "%h is not in [%h, %h]" got lo hi ]
           else if Float.abs (got -. want) <= bound then []
           else
             [ Printf.sprintf "%h is %h from %h, past %h" got (got -. want) want bound ])
         (parts d got))
  in
  match wrong with [] -> None | w -> Some (String.concat "; " w)

(* Whether [got] differs from the exact sum of [x] and [us] in a part. *)
let rounds d x us got =
  let terms = List.map (parts d) (x :: us) in
  List.exists Fun.id
    (List.mapi
       (fun k g -> fst (exact (List.map (fun t -> List.nth t k) terms)) <> g)
       (parts d got))

(* A float Add is within its bound of the exact sum at each target an
   update reaches. *)
let law_add_bound (b : Support.backend) c =
  let d = dtype_of c.into in
  let ls = lands c in
  let reached = Array.exists (fun l -> l <> []) ls in
  cover "a target several updates reach" (Array.exists (fun l -> List.length l > 1) ls);
  cover "a narrow float" (is_narrow d);
  match scatter_on b c with
  | None -> ()
  | Some y ->
      cover "reached" reached;
      let got = elements y and xs = elements c.into in
      cover "a wide sum that rounds"
        (not (is_narrow d)
        && Array.exists Fun.id
             (Array.mapi
                (fun t us -> us <> [] && rounds d xs.(t) us got.(t))
                ls));
      let wrong =
        List.filter_map Fun.id
          (Array.to_list
             (Array.mapi
                (fun t us ->
                  if us = [] then None
                  else
                    Option.map (Printf.sprintf "%d: %s" t)
                      (past d xs.(t) us got.(t)))
                ls))
      in
      equal (list string) [] (List.filteri (fun i _ -> i < 5) wrong)

(* Refusals *)

(* A gather's shapes, [x], [idx] and [dst], or a scatter's, [into], [idx],
   [updates] and [dst], along [along], each near a fit. *)
type misfit = { along : int; shapes : int array array }

let pp_misfit what ppf c =
  Format.fprintf ppf "%s along %d of %a" what c.along
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_ints)
    (Array.to_list c.shapes)

let gather_misfits =
  Gen.with_pp (pp_misfit "gather")
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* axis = axis_near x in
     let* n = int_range 0 4 in
     let* idx = near (along x axis n) in
     let+ dst = near idx in
     { along = axis; shapes = [| x; idx; dst |] })

let scatter_misfits =
  Gen.with_pp (pp_misfit "scatter")
    (let open Gen in
     let* into = Nx_array_gen.shape in
     let* axis = axis_near into in
     let* n = int_range 0 4 in
     let u = along into axis n in
     let* idx = near u in
     let* updates = near u in
     let+ dst = near into in
     { along = axis; shapes = [| into; idx; updates; dst |] })

let zeros32 s = A.of_array D.Float32 s (Array.make (total s) 0.)
let zeros64 s = A.of_array D.Int64 s (Array.make (total s) 0L)

(* [got] is Done where [want] is [dst]'s shape, else Shape_mismatch. *)
let check_fit want dst got =
  let fit = want = Ok dst in
  cover "fits" fit;
  cover "misfits" (not fit);
  equal answer (if fit then A.Done else A.Shape_mismatch) got

let law_gather_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let x = c.shapes.(0) and idx = c.shapes.(1) and dst = c.shapes.(2) in
  check_fit
    (Nx_kernel.Shape.gather ~axis:c.along idx x)
    dst
    (K.gather ~axis:c.along ~dst:(zeros32 dst) (zeros64 idx) (zeros32 x))

let law_scatter_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let into = c.shapes.(0) and idx = c.shapes.(1) in
  let u = c.shapes.(2) and dst = c.shapes.(3) in
  check_fit
    (Nx_kernel.Shape.scatter ~axis:c.along into idx u)
    dst
    (K.scatter Set ~unique:false ~axis:c.along ~dst:(zeros32 dst)
       ~into:(zeros32 into) (zeros64 idx) (zeros32 u))

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let idx = A.of_array D.Int64 [| 2; 2 |] [| 0L; 1L; 2L; 0L |] in
  let wrong = A.create Rig.host D.Float32 [| 3; 2 |] in
  equal ~msg:"a gather into another shape" answer A.Shape_mismatch
    (K.gather ~axis:1 ~dst:wrong idx x);
  equal ~msg:"positions off the source's shape" answer A.Shape_mismatch
    (K.gather ~axis:0
       ~dst:(A.create Rig.host D.Float32 [| 2; 2 |])
       idx x);
  let u = A.of_array D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let dst = A.create Rig.host D.Float32 [| 2; 3 |] in
  equal ~msg:"a scatter of positions and updates of two shapes" answer
    A.Shape_mismatch
    (K.scatter Add ~unique:false ~axis:1 ~dst ~into:x
       (A.of_array D.Int64 [| 2; 1 |] [| 0L; 1L |])
       u);
  equal ~msg:"a scatter into another shape" answer A.Shape_mismatch
    (K.scatter Add ~unique:false ~axis:1 ~dst:wrong ~into:x idx u);
  let bools = A.of_array D.Bool [| 2 |] [| true; false |] in
  let bdst = A.create Rig.host D.Bool [| 2 |] in
  equal ~msg:"an Add of booleans" answer A.Wrong_dtype
    (K.scatter Add ~unique:false ~axis:0 ~dst:bdst ~into:bools
       (A.of_array D.Int64 [| 1 |] [| 0L |])
       (A.of_array D.Bool [| 1 |] [| true |]))

(* nx.cpu's own cases *)

(* A float16 target that 2048 updates of one reach: in float32 the sum is exact,
   where float16 steps would stop at 2048. *)
let test_float16_sum (b : Support.backend) () =
  let n = 4096 in
  let into = A.of_array D.Float16 [| 1 |] [| 0. |] in
  let u = A.of_array D.Float16 [| n |] (Array.make n 1.) in
  let idx = A.of_array D.Int64 [| n |] (Array.make n 0L) in
  let dst = A.create Rig.host D.Float16 [| 1 |] in
  let module K = (val b.kernels) in
  match K.scatter Add ~unique:false ~axis:0 ~dst ~into idx u with
  | A.Done -> equal (array float_exact) [| 4096. |] (A.to_array dst)
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A boolean is true where its byte is not zero: Max and Min of bytes 2,
   255 and 1 are Or and And of truths, stored as 0 or 1. *)
let test_bool_extremes (b : Support.backend) () =
  let module K = (val b.kernels) in
  let bools s =
    A.v D.Bool (L.contiguous [| 3 |]) (Rig.Buffer.of_string s)
  in
  let idx = A.of_array D.Int64 [| 3 |] [| 0L; 1L; 2L |] in
  let run c =
    let dst = A.create Rig.host D.Bool [| 3 |] in
    match
      K.scatter c ~unique:false ~axis:0 ~dst
        ~into:(bools "\002\000\255") idx (bools "\001\002\000")
    with
    | A.Done -> elements (A.Any dst)
    | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r
  in
  equal ~msg:"Max" (array string) [| "\001"; "\001"; "\001" |] (run Nx_kernel.Max);
  equal ~msg:"Min" (array string) [| "\001"; "\000"; "\000" |] (run Nx_kernel.Min)

let large_shapes =
  Gen.of_list
    [ [| 300_000 |]; [| 600; 500 |]; [| 3; 100_000 |]; [| 100_000; 3 |] ]

(* Constructed cases, one per covered regime, so that every seed reaches
   each: positions in order from [ps], elements drawn from seed 1. *)

let positions_of ps =
  let k = ref 0 in
  fun () ->
    let p = ps.(!k mod Array.length ps) in
    incr k;
    p

let gather_example d ~xs ~axis ~n ?(broadcast = []) ps =
  let r = Array.length xs in
  let rs = Random.State.make [| 1 |] in
  let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
  let iv = { (plain r) with broadcast } in
  {
    axis;
    idx = int64s is iv (positions_of ps);
    x = operand d xs (plain r) (fun _ -> element d rs);
    gviews = List.map (( ^ ) "idx ") (view_names iv);
  }

let scatter_example d combine ~ts ~axis ~m ?(broadcast = []) ?tv ?uv
    ?(draw = element) ps =
  let r = Array.length ts in
  let rs = Random.State.make [| 1 |] in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  let iv = { (plain r) with broadcast } in
  let tv = Option.value tv ~default:(plain r)
  and uv = Option.value uv ~default:(plain r) in
  {
    combine;
    saxis = axis;
    into = operand d ts tv (fun _ -> draw d rs);
    sidx = int64s us iv (positions_of ps);
    updates = operand d us uv (fun _ -> draw d rs);
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

let gather_examples =
  [
    (* Positions outside the axis, from below and above. *)
    gather_example (D.Any D.Float32) ~xs:[| 3; 4 |] ~axis:1 ~n:5
      [| 0L; -1L; 4L; Int64.max_int; 3L |];
    (* A row take: positions broadcast along the rows. *)
    gather_example (D.Any D.Int16) ~xs:[| 5; 3 |] ~axis:0 ~n:4 ~broadcast:[ 1 ]
      [| 4L; 0L; 2L; 4L |];
  ]

let scatter_examples =
  [
    (* Shared targets and dropped updates. *)
    scatter_example (D.Any D.Float32) Nx_kernel.Add ~ts:[| 4 |] ~axis:0 ~m:8
      [| 0L; 1L; 0L; -1L; 4L; 1L; 1L; Int64.max_int |];
    (* A narrow float's sum. *)
    scatter_example (D.Any D.Float16) Nx_kernel.Add ~ts:[| 3 |] ~axis:0 ~m:6
      [| 0L; 0L; 2L; 5L; 0L; 1L |];
  ]

(* Set: shared targets and positions past both ends, every update on one
   target, no update, and one row. *)
let set_examples =
  let ex d = scatter_example d Nx_kernel.Set in
  [
    ex (D.Any D.Float32) ~ts:[| 4 |] ~axis:0 ~m:6 [| 1L; 3L; 1L; -1L; 4L; 1L |];
    ex (D.Any D.Int8) ~ts:[| 3; 2 |] ~axis:1 ~m:5 [| 0L |];
    ex (D.Any D.Bfloat16) ~ts:[| 3 |] ~axis:0 ~m:0 [| 0L |];
    ex (D.Any D.Complex128) ~ts:[| 5; 4 |] ~axis:0 ~m:1 ~broadcast:[ 1 ] [| 3L |];
  ]

(* Add's bound: many updates on few targets, an embedding gradient's rows,
   every update on one target, no update. *)
let bound_examples =
  let ex d = scatter_example d Nx_kernel.Add ~draw:finite in
  [
    ex (D.Any D.Float32) ~ts:[| 2 |] ~axis:0 ~m:40 [| 0L; 1L |];
    ex (D.Any D.Bfloat16) ~ts:[| 3; 8 |] ~axis:0 ~m:50 ~broadcast:[ 1 ]
      [| 2L; 0L; 2L; 1L |];
    ex (D.Any D.Float16) ~ts:[| 1 |] ~axis:0 ~m:60 [| 0L |];
    ex (D.Any D.Complex64) ~ts:[| 4 |] ~axis:0 ~m:30 [| 3L; 0L; 3L |];
    ex (D.Any D.Float64) ~ts:[| 3 |] ~axis:0 ~m:0 [| 0L |];
  ]

(* Values the contract states: [run c dt into idx u] is the scatter's
   result into a fresh destination. *)
let scatter1 (b : Support.backend) c dt into idx u =
  let module K = (val b.kernels) in
  let n = Array.length into and m = Array.length idx in
  let dst = A.create Rig.host dt [| n |] in
  match
    K.scatter c ~unique:false ~axis:0 ~dst
      ~into:(A.of_array dt [| n |] into)
      (A.of_array D.Int64 [| m |] idx)
      (A.of_array dt [| m |] u)
  with
  | A.Done -> Some (A.to_array dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let stated (b : Support.backend) =
  let f32 = D.Float32 in
  let snan = Int32.float_of_bits 0x7FA00001l in
  let floats name c dt into idx u want =
    ( name,
      fun () ->
        match scatter1 b c dt into idx u with
        | None -> ()
        | Some y -> equal (array float_exact) want y )
  in
  let bits32 a = Array.map Int32.bits_of_float a in
  [
    floats "Set keeps a target's last update" Set f32 [| 10.; 20.; 30.; 40. |]
      [| 1L; 3L; 1L; 1L |] [| 1.; 2.; 3.; 4. |] [| 10.; 4.; 30.; 2. |];
    floats "Set drops positions outside the axis" Set f32 [| 10.; 20.; 30.; 40. |]
      [| -1L; 4L; Int64.min_int; Int64.max_int; 0L |]
      [| 1.; 2.; 3.; 4.; 5. |] [| 5.; 20.; 30.; 40. |];
    floats "Add starts from +0: -0 and -0 updates sum to +0" Add f32 [| -0. |]
      [| 0L; 0L |] [| -0.; -0. |] [| 0. |];
    floats "Add of opposite infinities is a NaN" Add f32 [| infinity; 1. |]
      [| 0L; 1L |] [| neg_infinity; infinity |] [| nan; infinity |];
    floats "Add rounds a bfloat16 sum once, from float32" Add D.Bfloat16
      [| 1. |] (Array.make 256 0L) (Array.make 256 0x1p-8) [| 2. |];
    ( "a target no update reaches keeps its bits, -0 and a signalling NaN's",
      fun () ->
        match
          scatter1 b Add f32 [| -0.; snan |] [| 2L; -1L |] [| 1.; 1. |]
        with
        | None -> ()
        | Some y ->
            equal (array int32) (bits32 [| -0.; snan |]) (bits32 y) );
    ( "Add of int8 wraps",
      fun () ->
        match scatter1 b Add D.Int8 [| 127; -128 |] [| 0L; 1L |] [| 1; -1 |] with
        | None -> ()
        | Some y -> equal (array int) [| -128; 127 |] y );
  ]

(* Positions past the axis on both sides, cycling over 1003 targets. *)
let cycled = Array.init 1009 (fun i -> Int64.of_int ((i * 7 mod 1003) - 1))

(* Updates a sorted scatter sorts at once into an axis of 1000: past them,
   a narrow float's sums carry to the next block. *)
let first_block = 1 lsl 19

(* Positions that miss targets 900 to 999 over the first block, then
   [cycled]'s. *)
let late =
  Array.init 600_000 (fun i ->
      if i < first_block then Int64.of_int (i * 7 mod 900) else cycled.(i mod 1009))

(* Whether a target of [c] is first reached past the first block. *)
let reached_late c =
  let ps = positions c.sidx and d = (shape_of c.into).(c.saxis) in
  let first = Array.make d max_int in
  Array.iteri
    (fun i p ->
      let p = Int64.to_int p in
      if p >= 0 && p < d && first.(p) = max_int then first.(p) <- i)
    ps;
  Array.exists (fun i -> i <> max_int && i >= first_block) first

(* Past a block of a sorted scatter: shared targets, a narrow float's sum;
   targets first reached past the first block, into reversed; sums carried
   through stepped and reversed updates. *)
let blocked_examples =
  let ex d = scatter_example (D.Any d) Nx_kernel.Add ~ts:[| 1000 |] ~axis:0 in
  [
    ex D.Float16 ~m:600_000 cycled;
    ex D.Bfloat16 ~m:600_000 ~tv:{ (plain 1) with reversed = Some 0 } late;
    ex D.Float8_e4m3fn ~m:600_000
      ~uv:{ (plain 1) with stepped = true; reversed = Some 0 }
      cycled;
  ]

(* A narrow Add's sums past the first block, through views. *)
let cover_blocks c =
  let views side = List.exists (String.starts_with ~prefix:side) c.sviews in
  let carried =
    c.combine = Nx_kernel.Add
    && is_narrow (dtype_of c.into)
    && (shape_of c.updates).(c.saxis) > first_block
  in
  cover "a narrow sum first reached past a block, into through a view"
    (carried && views "into" && reached_late c);
  cover "a narrow sum carried through a view of the updates"
    (carried && views "updates")

(* An embedding gradient's rows, sorted: into transposed and the updates
   stepped and reversed; a narrow float's sum past a block. *)
let row_examples =
  [
    scatter_example (D.Any D.Float32) Nx_kernel.Add ~ts:[| 1000; 5 |] ~axis:0
      ~m:70_000 ~broadcast:[ 1 ]
      ~tv:{ (plain 2) with perm = [| 1; 0 |] }
      ~uv:{ (plain 2) with stepped = true; reversed = Some 0 }
      cycled;
    scatter_example (D.Any D.Bfloat16) Nx_kernel.Add ~ts:[| 1000; 2 |] ~axis:0
      ~m:530_000 ~broadcast:[ 1 ] cycled;
  ]

(* A sorted run of 1025 slices, whose last piece is one slice: the updates
   broadcast along the axis, positions cycling over 0, 1 and 2 into an axis
   of 2, so that both targets end as the row. *)
let test_last_piece (b : Support.backend) () =
  let d = D.Any D.Complex128 and m = 65_536 and len = 1025 in
  let rs = Random.State.make [| 2 |] in
  let row = Array.init len (fun _ -> element d rs) in
  let c =
    {
      combine = Nx_kernel.Set;
      saxis = 0;
      into = operand d [| 2; len |] (plain 2) (fun _ -> element d rs);
      sidx =
        int64s [| m; len |]
          { (plain 2) with broadcast = [ 1 ] }
          (positions_of [| 0L; 1L; 2L |]);
      updates =
        operand d [| m; len |] { (plain 2) with broadcast = [ 0 ] } (fun k ->
            row.(k));
      sviews = [];
    }
  in
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y ->
      equal (array string)
        (Array.init (2 * len) (fun t -> row.(t mod len)))
        (elements y)

let gathers = Gen.with_pp pp_gather (gather_gen ())
let large_gathers = Gen.with_pp pp_gather (gather_gen ~shapes:large_shapes ())
let scatters = Gen.with_pp pp_scatter (scatter_gen ())

let sets =
  Gen.with_pp pp_scatter (scatter_gen ~combines:Nx_kernel.[ Set ] ())

let float_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:float_dtypes ~combines:Nx_kernel.[ Add ] ())

let finite_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:float_dtypes ~combines:Nx_kernel.[ Add ]
       ~draw:finite ())

let large_scatters =
  Gen.with_pp pp_scatter (scatter_gen ~shapes:large_shapes ())

(* Many updates along the axis of few slices: threads split the targets. *)
let long_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen
       ~shapes:(Gen.of_list [ [| 1000 |]; [| 3; 7 |]; [| 5 |] ])
       ~ms:(Gen.int_range 70_000 100_000)
       ())

(* More steps along the axis than a partitioned scatter sorts at once. *)
let blocked_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen
       ~shapes:(Gen.of_list [ [| 1000 |]; [| 5 |] ])
       ~ms:(Gen.int_range 524_289 600_000)
       ())

(* Embedding gradients: many updates along axis 0, their positions
   broadcast along the rows, into and the updates through drawn views. *)
let row_scatters =
  let open Gen in
  let* d = of_list ~pp:pp_dtype all_dtypes in
  let* combine = combine_gen d in
  let* ts =
    of_list ~pp:pp_ints
      [ [| 300; 5 |]; [| 50; 7 |]; [| 20; 2; 3 |]; [| 40; 3 |] ]
  in
  let r = Array.length ts in
  let* m = int_range 65_536 70_000 in
  let* tv = view_of r in
  let* uv = view_of r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let us = Array.mapi (fun i e -> if i = 0 then m else e) ts in
  let iv = { (plain r) with broadcast = List.init (r - 1) succ } in
  {
    combine;
    saxis = 0;
    into = operand d ts tv (fun _ -> element d rs);
    sidx = int64s us iv (fun () -> position rs ts.(0));
    updates = operand d us uv (fun _ -> element d rs);
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

let row_scatters = Gen.with_pp pp_scatter row_scatters

(* A scatter of [d] along [axis] into [ts], [m] updates along it, its
   positions broadcast along [broadcast], into and the updates through
   [tv] and [uv], its elements drawn from [rs]. *)
let shaped d combine (ts, axis, m, broadcast) ~tv ~uv rs =
  let r = Array.length ts in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  {
    combine;
    saxis = axis;
    into = operand d ts tv (fun _ -> element d rs);
    sidx =
      int64s us { (plain r) with broadcast } (fun () -> position rs ts.(axis));
    updates = operand d us uv (fun _ -> element d rs);
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

let narrow_dtypes = List.filter is_narrow all_dtypes

(* Positions a parted narrow Add counts at once in a run: past them, its
   count cuts the run into pieces. *)
let count_piece = 1 lsl 18

(* A bfloat16 Add of 3 slices, 300 000 updates along an axis of 1000:
   threads part each slice's targets and count its positions in pieces. *)
let counted = ([| 1000; 3 |], 0, 300_000)

(* A narrow float's Adds, a shape for each way its sums group the
   updates: an axis of more updates than targets, one of fewer, 3 slices
   of positions of their own, and rows of 600 slices cut into pieces of
   512, the positions broadcast along them. *)
let narrow_shapes =
  [
    ("more updates than targets", ([| 1000 |], 0, 70_000, []));
    ("more targets than updates", ([| 200_000 |], 0, 70_000, []));
    ("positions of each slice", ([| 3; 7 |], 1, 70_000, []));
    ("rows cut into pieces", ([| 40; 600 |], 0, 600, [ 1 ]));
  ]

let narrow_examples =
  List.mapi
    (fun i (_, ((ts, _, _, _) as s)) ->
      let d = List.nth narrow_dtypes (i mod List.length narrow_dtypes) in
      let r = Array.length ts in
      shaped d Nx_kernel.Add s ~tv:(plain r) ~uv:(plain r)
        (Random.State.make [| 1 |]))
    narrow_shapes
  @ [
      (* Positions crowding the first of the parts the targets split into,
         and one outside the axis. *)
      scatter_example (D.Any D.Bfloat16) Nx_kernel.Add ~ts:[| 500; 3 |]
        ~axis:0 ~m:70_000
        (Array.init 1009 (fun i -> Int64.of_int ((i * i mod 38) - 1)));
      (let ts, axis, m = counted in
       scatter_example (D.Any D.Bfloat16) Nx_kernel.Add ~ts ~axis ~m cycled);
    ]
  @ blocked_examples

let narrow_adds =
  let open Gen in
  let* d = of_list ~pp:pp_dtype narrow_dtypes in
  let* _, ((ts, _, _, _) as s) = of_list narrow_shapes in
  let r = Array.length ts in
  let* tv = view_of r in
  let* uv = view_of r in
  let+ seed = int in
  shaped d Nx_kernel.Add s ~tv ~uv (Random.State.make [| seed |])

let narrow_adds = Gen.with_pp pp_scatter narrow_adds

(* A scatter's bits are one thread's. *)
let law_threads (b : Support.backend) c =
  match (scatter_on b c, Nx_cpu.one_thread (fun () -> scatter_on b c)) with
  | Some y, Some y' -> equal (array string) (elements y) (elements y')
  | _ -> failf "%s declined a scatter" b.name

let law_narrow_threads (b : Support.backend) c =
  let ts = shape_of c.into and m = (shape_of c.updates).(c.saxis) in
  List.iter
    (fun (name, (ts', axis, m', _)) ->
      cover name (ts = ts' && c.saxis = axis && m = m'))
    narrow_shapes;
  cover "sums carried from block to block" (m > 1 lsl 19);
  (let ts', axis, m' = counted in
   cover "parted positions counted in pieces"
     (ts = ts' && c.saxis = axis && m = m' && m > count_piece));
  cover_blocks c;
  law_threads b c

(* Sub-byte scatters *)

let sub_byte_dtypes = List.filter (fun (D.Any dt) -> D.bits dt < 8) all_dtypes

(* Shapes whose threads store elements of one byte: an axis of 9 split
   into parts, rows of 1001 slices cut into pieces at 512 and starting at
   odd elements, and 20 units of 33 elements. Each is [(into, axis,
   updates along it, broadcast axes of the positions)]. *)
let shared_bytes =
  [
    ([| 9 |], 0, 70_000, []);
    ([| 9; 1001 |], 0, 64, [ 1 ]);
    ([| 20; 11; 3 |], 1, 2000, []);
  ]

let sub_byte_examples =
  List.concat_map
    (fun d ->
      List.concat_map
        (fun (ts, axis, m, broadcast) ->
          let ps = Array.init ts.(axis) Int64.of_int in
          List.map
            (fun combine ->
              scatter_example d combine ~ts ~axis ~m ~broadcast ps)
            (if takes_add d then Nx_kernel.[ Set; Add; Max; Min ]
             else Nx_kernel.[ Set; Max; Min ]))
        shared_bytes)
    sub_byte_dtypes

let sub_byte_scatters =
  let open Gen in
  let* d = of_list ~pp:pp_dtype sub_byte_dtypes in
  let* combine = combine_gen d in
  let* ((ts, _, _, _) as s) = of_list shared_bytes in
  let+ seed = int in
  let r = Array.length ts in
  shaped d combine s ~tv:(plain r) ~uv:(plain r) (Random.State.make [| seed |])

let sub_byte_scatters = Gen.with_pp pp_scatter sub_byte_scatters

(* A sub-byte scatter's bits are one thread's where threads store elements
   of one byte. A lost store needs two threads to meet at the byte, which a
   run may not show; the shapes pin where they would. *)
let law_sub_byte (b : Support.backend) c =
  let d = dtype_of c.into in
  List.iter
    (fun e -> cover (Format.asprintf "%a" pp_dtype e) (e = d))
    sub_byte_dtypes;
  let ts = shape_of c.into in
  cover "an axis split into parts" (Array.length ts = 1);
  cover "a row cut into pieces"
    (Array.length ts = 2 && c.combine = Nx_kernel.Add && is_narrow d);
  cover "several units" (Array.length ts = 3);
  law_threads b c

let int_gathers =
  Gen.with_pp pp_gather (gather_gen ~dtypes:[ D.Any D.Int64 ] ())

(* Guard pages *)

let covers_views ~after views =
  cover "a stepped operand at the page before"
    ((not after) && List.exists (String.ends_with ~suffix:"stepped") views);
  cover "a reversed operand at the page after"
    (after && List.exists (String.ends_with ~suffix:"reversed") views);
  cover "broadcast positions" (List.mem "idx broadcast" views)

(* The same result with the operand and positions, through their views, and
   the destination against an inaccessible page. *)
let law_gather_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let r = Array.length (shape_of c.x) in
  cover "along the innermost axis" (c.axis = r - 1);
  cover "along an outer axis" (c.axis < r - 1);
  covers_views ~after:(side = Support.After) c.gviews;
  let call xs ds =
    let (A.Any x) = xs.(1) in
    let dst = A.expect (A.dtype x) ds.(0) in
    K.gather ~axis:c.axis ~dst (A.expect D.Int64 xs.(0)) x
  in
  equal answer A.Done
    (same_guarded side ~call [| c.idx; c.x |]
       [| (dtype_of c.x, shape_of c.idx) |])

let law_scatter_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let r = Array.length (shape_of c.into) in
  cover (combine_name c.combine) true;
  cover "along the innermost axis" (c.saxis = r - 1);
  cover "along an outer axis" (c.saxis < r - 1);
  covers_views ~after:(side = Support.After) c.sviews;
  let call xs ds =
    let (A.Any into) = xs.(0) in
    let dst = A.expect (A.dtype into) ds.(0) in
    let idx = A.expect D.Int64 xs.(1) in
    let updates = A.expect (A.dtype into) xs.(2) in
    K.scatter c.combine ~unique:false ~axis:c.saxis ~dst ~into idx updates
  in
  equal answer A.Done
    (same_guarded side ~call
       [| c.into; c.sidx; c.updates |]
       [| (dtype_of c.into, shape_of c.into) |])

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a gather and an add-scatter are adjoint" int_gathers
        (run (law_adjoint b));
      prop "a scatter into its own operand gives the fresh result" scatters
        (run (law_in_place b));
      prop ~examples:set_examples
        "a Set scatter keeps each target's last update in C order" sets
        (run (law_set_last b));
      prop "a float Add gives the same bits again and through any views"
        float_adds
        (run (law_add_bits b));
      prop "a float Add's target has the bits of its updates alone" float_adds
        (run (law_add_alone b));
      prop ~examples:bound_examples
        "a float Add is within γ(n + 1) (|x| + Σ|ui|) of the exact sum"
        finite_adds
        (run (law_add_bound b));
      cases ~name:fst "a scatter's stated values" (stated b) (fun (_, f) ->
          b.around f);
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples:gather_examples
        "a gather reads each position, zero outside the axis" gathers
        (run (law_gather_reference b));
      prop ~examples:scatter_examples
        "a scatter combines each target's updates in C order" scatters
        (run (law_scatter_reference b));
      prop "a scatter's Max and Min give the first NaN quieted" nan_scatters
        (run (law_nan_extremes b));
      prop ~count:10 "large gathers on the job's threads" large_gathers
        (run (gather_agrees b));
      prop ~count:10 "large scatters on the job's threads" large_scatters
        (run (scatter_agrees b));
      prop ~count:10 "long scatters split their targets among threads"
        long_scatters
        (run (scatter_agrees b));
      prop ~count:4 ~examples:blocked_examples
        "longer scatters split them block by block" blocked_scatters
        (run (fun c ->
             cover_blocks c;
             law_scatter_reference b c));
      prop ~count:10 ~examples:row_examples
        "an embedding gradient's rows combine in C order, in place too"
        row_scatters
        (run (fun c ->
             law_scatter_reference b c;
             law_in_place b c));
      prop ~count:6 ~examples:narrow_examples
        "a narrow float Add's bits are one thread's" narrow_adds
        (run (law_narrow_threads b));
      prop ~count:10 ~examples:row_examples
        "an embedding gradient's bits are one thread's" row_scatters
        (run (fun c ->
             cover "a narrow float's Add"
               (c.combine = Nx_kernel.Add && is_narrow (dtype_of c.into));
             law_threads b c));
      prop ~count:10 ~examples:sub_byte_examples
        "a sub-byte scatter's bits are one thread's" sub_byte_scatters
        (run (law_sub_byte b));
      test "a sorted run's last piece of one slice" (fun () ->
          b.around (test_last_piece b));
      test "boolean extremes are of truths" (fun () ->
          b.around (test_bool_extremes b));
      test "a float16 sum runs in float32 and rounds once" (fun () ->
          b.around (test_float16_sum b));
      prop "a gather refuses what Nx_kernel.Shape refuses and other shapes"
        gather_misfits
        (run (law_gather_refuses b));
      prop "a scatter refuses what Nx_kernel.Shape refuses and other shapes"
        scatter_misfits
        (run (law_scatter_refuses b));
      test "refuses shapes that do not fit and an Add of booleans" (fun () ->
          b.around (test_refusals b));
      prop ~count:300
        "a gather reads and writes no byte past its operands and dst, \
         whatever pages lie beside them"
        (Gen.pair (gather_gen ~shapes:tail_shapes ()) sides)
        (run (law_gather_guards b));
      prop ~count:300
        "a scatter reads and writes no byte past its operands and dst, \
         whatever pages lie beside them"
        (Gen.pair (scatter_gen ~shapes:tail_shapes ()) sides)
        (run (law_scatter_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.index"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
