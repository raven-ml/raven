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

(* Gathers of [dtypes] from [shapes], along [axis] if given, of [ns]
   positions drawn by [position], through drawn views, or plain ones where
   [plain] (broadcast positions aside). *)
let gather_gen ?(dtypes = all_dtypes) ?(shapes = shape_gen) ?axis
    ?(ns = Gen.frequency [ (4, Gen.int_range 0 6); (1, Gen.int_range 7 40) ])
    ?(position = position) ?(plain = false) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* xs = shapes in
  let r = Array.length xs in
  let* axis =
    match axis with Some a -> constant a | None -> int_range 0 (r - 1)
  in
  let* n = ns in
  let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
  let* xv = if plain then constant (Elements.plain r) else view_of r in
  let* bc =
    array ~size:(constant r)
      (frequency [ (3, constant false); (1, constant true) ])
  in
  let broadcast =
    List.filter (fun i -> bc.(i) && i <> axis) (List.init r Fun.id)
  in
  let* iv =
    if plain then constant { (Elements.plain r) with broadcast }
    else view_of ~broadcast r
  in
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

(* The gather by [b]'s kernels, on its device, into a fresh destination;
   [None] where [b] may decline it. *)
let gather_on (b : Support.backend) ~axis idx x =
  let module K = (val b.kernels) in
  let (A.Any x) = x in
  let x = Support.on b x and idx = Support.on b (A.expect D.Int64 idx) in
  let dst = A.create b.device (A.dtype x) (L.shape (A.layout idx)) in
  match K.gather ~axis ~dst idx x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> failf "%s declined a gather" b.name
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

(* The scatter by [b]'s kernels, on its device, into a fresh destination or,
   [in_place], into a copy of [into] it also reads; [None] where [b] may
   decline it. *)
let scatter_on (b : Support.backend) ?(in_place = false) ?(unique = false) c =
  let module K = (val b.kernels) in
  let (A.Any into) = c.into in
  let into = Support.on b into in
  let u = Support.on b (A.expect (A.dtype into) c.updates) in
  let idx = Support.on b (A.expect D.Int64 c.sidx) in
  let dst, into =
    if in_place then
      (* Copied where the host addresses it: sub-byte copies need it. *)
      let d = Support.on b (A.copy (Support.host into)) in
      (d, d)
    else (A.create b.device (A.dtype into) (L.shape (A.layout into)), into)
  in
  match K.scatter c.combine ~unique ~axis:c.saxis ~dst ~into idx u with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> failf "%s declined a scatter" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* [c] with its arrays on [b]'s device, through their views. *)
let scatter_placed (b : Support.backend) c =
  let on (A.Any x) = A.Any (Support.on b x) in
  { c with into = on c.into; updates = on c.updates; sidx = on c.sidx }

(* Whether an extreme of a and b keeps a: a NaN, else the greater (lesser), a
   on a tie. *)
let keeps combine ~nan_a ~nan_b cmp =
  if nan_a then true
  else if nan_b then false
  else if combine = Nx_kernel.Max then cmp >= 0
  else cmp <= 0

(* An expected element: its bits, or its value, any NaN for a NaN. *)
type want = Bits of string | Value of string

(* [u] combined by [combine] into [a], two elements of [d], a narrow float's
   [Add] aside. *)
let step (D.Any dt as d) combine a u =
  let w = String.length a in
  match combine with
  | Nx_kernel.Set -> u
  | Add when D.is D.Float dt -> add_bits ~w a u
  | Add when D.is D.Complex dt ->
      let h = w / 2 in
      let part e k = String.sub e (k * h) h in
      add_bits ~w:h (part a 0) (part u 0) ^ add_bits ~w:h (part a 1) (part u 1)
  | Add -> int_bits d w (Int64.add (int_value d a) (int_value d u))
  | (Max | Min) when D.is D.Boolean dt ->
      (* Or and And of truths, stored as 0 or 1. *)
      let op = if combine = Nx_kernel.Max then ( || ) else ( && ) in
      if op (a <> "\000") (u <> "\000") then "\001" else "\000"
  | Max | Min ->
      let keep =
        if D.is D.Complex dt then
          let h = w / 2 in
          let get = if h = 4 then get32 else get64 in
          let ra = get a 0 and ia = get a h and rb = get u 0 and ib = get u h in
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
          let x = get a and y = get u in
          keeps combine ~nan_a:(Float.is_nan x) ~nan_b:(Float.is_nan y)
            (compare_floats x y)
        else
          let x = int_value d a and y = int_value d u in
          let c =
            if D.is D.Signed dt then Int64.compare x y
            else Int64.unsigned_compare x y
          in
          keeps combine ~nan_a:false ~nan_b:false c
      in
      if keep then a else u

(* The elements of shape [s] at each index along axis [a]: its extents after [a]
   multiplied. *)
let inner s a = total (Array.sub s (a + 1) (Array.length s - a - 1))

(* The target of each of [c]'s updates, in C order: its flat index into [into],
   [-1] for a position outside the axis. Update [k] at index [i] targets [i]
   with the axis replaced by its position. *)
let targets c =
  let ts = shape_of c.into and us = shape_of c.updates and a = c.saxis in
  let per = inner ts a and n = ts.(a) and m = us.(a) in
  Array.mapi
    (fun k p ->
      if p < 0L || p >= Int64.of_int n then -1
      else (((k / (per * m) * n) + Int64.to_int p) * per) + (k mod per))
    (positions c.sidx)

(* The updates that land on each target of [c], in C order, so in increasing
   position along the scattered axis, each with that position. *)
let landings c =
  let us = shape_of c.updates and a = c.saxis in
  let per = inner us a and ues = elements c.updates in
  let lands = Array.make (total (shape_of c.into)) [] in
  let ts = targets c in
  for k = Array.length ts - 1 downto 0 do
    let t = ts.(k) in
    if t >= 0 then lands.(t) <- (k / per mod us.(a), ues.(k)) :: lands.(t)
  done;
  lands

(* The updates that land on each target of [c], in C order. *)
let lands c = Array.map (List.map snd) (landings c)

(* Each target's element of [c]'s result: [into]'s where no update lands, else
   the combine of its updates in C order, folded into it one at a time; a narrow
   float's Add summed in float32 and rounded once. A move's or an integer's are
   bits, the others values, any NaN for a NaN. *)
let scatter_expected c =
  let (D.Any dt as d) = dtype_of c.into in
  let xs = elements c.into and us = elements c.updates and ts = targets c in
  let reached = Array.make (Array.length xs) false in
  Array.iter (fun t -> if t >= 0 then reached.(t) <- true) ts;
  let fold f acc =
    Array.iteri (fun k t -> if t >= 0 then acc.(t) <- f acc.(t) us.(k)) ts;
    acc
  in
  let result want acc =
    Array.mapi (fun t x -> if reached.(t) then want acc.(t) else Bits x) xs
  in
  if c.combine = Nx_kernel.Add && is_narrow d then
    let s =
      fold (fun s u -> round32 (s +. decode d u)) (Array.map (decode d) xs)
    in
    result (fun s -> Value (encode d s)) s
  else
    let moved =
      match c.combine with
      | Nx_kernel.Set -> true
      | Add -> not (D.is D.Float dt || D.is D.Complex dt)
      | Max | Min -> D.is D.Boolean dt
    in
    result
      (fun e -> if moved then Bits e else Value e)
      (fold (step d c.combine) (Array.copy xs))

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
  let ts = targets c in
  let landed = Array.make (total (shape_of c.into)) 0 in
  Array.iter (fun t -> if t >= 0 then landed.(t) <- landed.(t) + 1) ts;
  (Array.exists (fun n -> n > 1) landed, Array.exists (fun t -> t < 0) ts)

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
