(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the suites of Nx_kernel.S's gathers and scatters and of nx.cpu's
   share: the cases, their views and draws, the calls, a scatter's expected
   result and its check, and a scatter into its own operand. *)

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

let scatter_on (b : Support.backend) ?(in_place = false) ?(unique = false) c =
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
  match K.scatter c.combine ~unique ~axis:c.saxis ~dst ~into idx u with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Whether an extreme of a and b keeps a: a NaN, else the greater (lesser), a
   on a tie. *)
let keeps combine ~nan_a ~nan_b cmp =
  if nan_a then true
  else if nan_b then false
  else if combine = Nx_kernel.Max then cmp >= 0
  else cmp <= 0

(* An expected element: its bits, or its value, any NaN for a NaN. *)
type want = Bits of string | Value of string

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
              (fun s u -> round32 (s +. decode d u))
              (decode d x) us
          in
          Value (encode d s)
      | Add when D.is D.Float dt ->
          Value (List.fold_left (add_bits ~w) x us)
      | Add when D.is D.Complex dt ->
          let h = w / 2 in
          let part e k = String.sub e (k * h) h in
          let sum k =
            List.fold_left (fun s u -> add_bits ~w:h s (part u k)) (part x k) us
          in
          Value (sum 0 ^ sum 1)
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
          Value (List.fold_left pick x us))

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
           | Value v when Testable.equal (same d) v got.(t) -> None
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
      | Bits b | Value b ->
          String.concat ""
            (List.map
               (fun ch -> Printf.sprintf "%02x" (Char.code ch))
               (List.of_seq (String.to_seq b))))
  in
  equal (list string) [] (List.map show (List.filteri (fun i _ -> i < 5) wrong))

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

(* Laws for every library *)

(* A scatter into its own operand, donated, gives the fresh result, any NaN
   for a NaN. *)
let law_in_place (b : Support.backend) c =
  match (scatter_on b c, scatter_on b ~in_place:true c) with
  | Some fresh, Some own ->
      equal (array (same (dtype_of c.into))) (elements fresh) (elements own)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

(* Bounds of Add *)

let float_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt)
    all_dtypes

(* Constructed cases, one per covered regime, so that every seed reaches
   each: positions in order from [ps], elements drawn from seed 1. *)

let positions_of ps =
  let k = ref 0 in
  fun () ->
    let p = ps.(!k mod Array.length ps) in
    incr k;
    p

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

(* Draws *)

let scatters = Gen.with_pp pp_scatter (scatter_gen ())

let float_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:float_dtypes ~combines:Nx_kernel.[ Add ] ())
