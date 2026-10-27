(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Folds through every kernel library the host runs: a fold is the padded load's
   adjoint over integers, and a float fold within its bound of the exact sum
   at each index. nx.cpu's group checks each result element against a
   reference that sums, with no +0 before the first, the elements the load
   reads from it in C order of their taps, +0 where it reads none, bit for
   bit or both NaN, through views of the operand, at every dtype but the
   booleans; and its maps of padded loads, through the
   same geometry: a loaded element is the element the fold adds it into, or
   the fill. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* A fold's case; [load] views the operand of the map whose padded load
   reads the fold's geometry. *)
type case = {
  shape : int array;
  pad : Nx_kernel.pad;
  x : A.any;
  views : string list;
  load : view;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf
    "fold into %a of %a %a: lo %a hi %a interior %a, windows %s (%s), load \
     (%s)"
    pp_ints c.shape D.pp (A.dtype x) pp_ints (shape_of c.x) pp_ints
    (Iarray.to_array c.pad.lo) pp_ints (Iarray.to_array c.pad.hi) pp_ints
    (Iarray.to_array c.pad.interior)
    (String.concat "; "
       (Iarray.to_list
          (Iarray.map
             (fun (w : M.window) ->
               Printf.sprintf "%d:%dx%d+%d" w.axis w.size w.step w.dilation)
             c.pad.windows)))
    (String.concat ", " c.views)
    (String.concat ", " (view_names c.load))

(* The shape a load by [p] gives an array of shape [s], if any. *)
let loaded (p : Nx_kernel.pad) s = Result.to_option (Nx_kernel.Shape.padded p s)

let folded = List.filter (fun (D.Any d) -> not (D.is D.Boolean d)) D.all

(* How a case's first axis leads: as drawn; unwindowed and unpadded, so
   that the fold's units are slabs of the result; or unwindowed and padded,
   so that the fold zeroes the whole result first. The last two window the
   last axis, of a result of two or three axes. *)
type lead = Drawn | Slabs | Padded

(* The complex and sub-byte dtypes, which the laws cover, drawn more often than
   the others, so that every run of a hundred cases reaches them. *)
let weight (D.Any d) =
  if D.is D.Complex d then 10 else if D.bits d < 8 then 6 else 2

(* [last], where given, draws the result's last extent. *)
let case_gen ?(dtypes = folded) ?last () =
  let open Gen in
  let* d = frequency (List.map (fun d -> (weight d, constant d)) dtypes) in
  let* lead =
    frequency
      [ (2, constant Drawn); (1, constant Slabs); (1, constant Padded) ]
  in
  let* shape =
    match lead with
    | Drawn -> array ~size:(int_range 0 3) (int_range 0 5)
    | Slabs | Padded ->
        map
          (fun s -> Array.mapi (fun a v -> if a = 0 then max v 2 else v) s)
          (array ~size:(int_range 2 3) (int_range 1 5))
  in
  let* shape =
    match last with
    | Some n when Array.length shape > 0 ->
        let+ n = n in
        let last = Array.length shape - 1 in
        Array.mapi (fun a v -> if a = last then n else v) shape
    | _ -> constant shape
  in
  let r = Array.length shape in
  let small = int_range (-2) 3 in
  let* lo = array ~size:(constant r) small in
  let* hi = array ~size:(constant r) small in
  let* interior = array ~size:(constant r) (int_range 0 2) in
  let* on = array ~size:(constant r) bool in
  let* sizes = array ~size:(constant r) (int_range 1 3) in
  let* steps = array ~size:(constant r) (int_range 1 3) in
  let* dilations = array ~size:(constant r) (int_range 1 2) in
  let first a v z = if lead <> Drawn && a = 0 then z else v in
  let lo0 = if lead = Padded then 1 else 0 in
  let lo = Array.mapi (fun a v -> first a v lo0) lo in
  let hi = Array.mapi (fun a v -> first a v (max v 0)) hi in
  let interior = Array.mapi (fun a v -> first a v 0) interior in
  let on =
    Array.mapi (fun a v -> (lead <> Drawn && a = r - 1) || first a v false) on
  in
  let windows =
    Array.of_list
      (List.filter_map
         (fun a ->
           if on.(a) then
             Some
               {
                 M.axis = a;
                 size = sizes.(a);
                 step = steps.(a);
                 dilation = dilations.(a);
               }
           else None)
         (List.init r Fun.id))
  in
  let pad =
    let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
    let interior = Iarray.of_array interior in
    { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }
  in
  let* seed = int in
  let xs = loaded pad shape in
  let rx = match xs with Some s -> Array.length s | None -> 0 in
  let* v = view_of rx in
  let+ load = view_of r in
  match xs with
  | None -> None
  | Some xs ->
      let rs = Random.State.make [| seed |] in
      let x = operand d xs v (fun _ -> element d rs) in
      Some { shape; pad; x; views = view_names v; load }

(* Folds into a destination of other elements: the fold writes each. *)
let fold_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) and rs = Random.State.make [| 5 |] in
  let dst =
    A.expect (A.dtype x)
      (operand d c.shape (plain (Array.length c.shape)) (fun _ -> element d rs))
  in
  match K.fold c.pad ~dst x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The result element each operand element lands on, or [None] for one the load
   reads from padding. *)
let target c xi =
  let r = Array.length c.shape in
  let q = Array.sub xi 0 r in
  Iarray.iteri
    (fun k (w : M.window) ->
      q.(w.axis) <- (xi.(w.axis) * w.step) + (xi.(r + k) * w.dilation))
    c.pad.windows;
  let i =
    Array.mapi
      (fun a q ->
        let t = q - Iarray.get c.pad.lo a
        and s = Iarray.get c.pad.interior a + 1 in
        if t < 0 || t mod s <> 0 || t / s >= c.shape.(a) then -1 else t / s)
      q
  in
  if Array.exists (fun v -> v < 0) i then None else Some (flat c.shape i)

(* Prog.Add on two elements of [d]: floats of 32 and 64 bits by
   {!Elements.add_bits}, integers wrapping, and narrow floats and complex
   numbers as nx.cpu's map of one [Add] adds two elements. *)
let add (D.Any dt as d) a b =
  if D.Any D.Float32 = d || D.Any D.Float64 = d then
    add_bits ~w:(String.length a) a b
  else if D.is D.Signed dt || D.is D.Unsigned dt then
    int_bits d (String.length a) (Int64.add (int_value d a) (int_value d b))
  else
    let (A.Any x) = of_elements d [| 1 |] [| a |] in
    let y = A.expect (A.dtype x) (of_elements d [| 1 |] [| b |]) in
    let dst = A.create Rig.host (A.dtype x) [| 1 |] in
    match Support.map2 (module Nx_cpu) (Binary Add) ~dst x y with
    | A.Done -> (elements (A.Any dst)).(0)
    | r -> failf "an add answered %a" Nx_array_support.pp_answer r

(* The number of the operand's elements the load reads from each result
   index. *)
let reached c =
  let xs = shape_of c.x in
  let counts = Array.make (total c.shape) 0 in
  for k = 0 to total xs - 1 do
    match target c (index_of xs k) with
    | Some y -> counts.(y) <- counts.(y) + 1
    | None -> ()
  done;
  counts

let expected c =
  let d = dtype_of c.x in
  let xs = shape_of c.x and es = elements c.x in
  let r = Array.length c.shape and nw = Iarray.length c.pad.windows in
  let acc = Array.make (total c.shape) None in
  let taps = Array.sub xs r nw and outer = Array.sub xs 0 r in
  for t = 0 to total taps - 1 do
    let j = index_of taps t in
    for k = 0 to total outer - 1 do
      let xi = Array.append (index_of outer k) j in
      match target c xi with
      | None -> ()
      | Some y ->
          let e = es.(flat xs xi) in
          acc.(y) <- Some (match acc.(y) with None -> e | Some a -> add d a e)
    done
  done;
  Array.map (function None -> zero d | Some a -> a) acc

(* The axes before the first a window takes. *)
let leading c =
  let r = Array.length c.shape in
  let windowed a =
    Iarray.exists (fun (w : M.window) -> w.axis = a) c.pad.windows
  in
  let rec go a = if a < r && not (windowed a) then go (a + 1) else a in
  go 0

(* Whether every result index along the leading axes receives an operand
   index: the fold then zeroes and adds its result a slab at a time. *)
let slabs c =
  let p = c.pad in
  let covers a =
    Iarray.get p.lo a = 0
    && Iarray.get p.hi a >= 0
    && (Iarray.get p.interior a = 0 || c.shape.(a) <= 1)
  in
  List.for_all covers (List.init (leading c) Fun.id)

let law_reference (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let n = leading c and r = Array.length c.shape in
      let work =
        Iarray.length c.pad.windows > 0 && total c.shape > 0 && total (shape_of c.x) > 0
      in
      let units = total (Array.sub c.shape 0 n) in
      cover "windows" (Iarray.length c.pad.windows > 0);
      cover "interior" (Iarray.exists (fun i -> i > 0) c.pad.interior);
      cover "cropped" (Iarray.exists (fun l -> l < 0) c.pad.lo);
      cover "slabs started per unit" (work && n > 0 && units > 1 && slabs c);
      cover "whole started, a leading axis padded"
        (work && n > 0 && not (slabs c));
      (let (D.Any d) = dtype_of c.x in
       let counts = reached c in
       let float = D.is D.Float d || D.is D.Complex d in
       cover "a float element no tap reaches"
         (work && float && Array.exists (( = ) 0) counts);
       cover "a float element of one tap"
         (work && float && Array.exists (( = ) 1) counts));
      let taps = Array.sub (shape_of c.x) r (Iarray.length c.pad.windows) in
      cover "taps innermost" (work && n < r && total taps > 1);
      (let (D.Any d) = dtype_of c.x in
       cover "a narrow float" (work && is_narrow (D.Any d));
       cover "sub-byte" (work && D.bits d < 8);
       cover "complex" (work && D.is D.Complex d));
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y -> equal (array (same (dtype_of c.x))) (expected c) (elements y))

(* Σ load(a) · x = Σ a · fold(x) over wrapping int64. *)
let law_adjoint (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let d = D.Any D.Int64 in
      let rs = Random.State.make [| 5 |] in
      let ints s =
        operand d s
          (plain (Array.length s))
          (fun _ ->
            int_bits d 8 (Int64.of_int (Random.State.int rs 2001 - 1000)))
      in
      let c = { c with x = ints (shape_of c.x) } in
      match fold_on b c with
      | None -> ()
      | Some f ->
          let a = Array.map (int_value d) (elements (ints c.shape)) in
          let xs = shape_of c.x
          and es = Array.map (int_value d) (elements c.x) in
          let lhs = ref 0L in
          for k = 0 to total xs - 1 do
            match target c (index_of xs k) with
            | None -> ()
            | Some y -> lhs := Int64.add !lhs (Int64.mul a.(y) es.(k))
          done;
          let fs = Array.map (int_value d) (elements f) in
          let rhs = ref 0L in
          Array.iteri (fun i v -> rhs := Int64.add !rhs (Int64.mul a.(i) v)) fs;
          equal int64 !lhs !rhs)

(* A float fold's element is within γ(n - 1) Σ|x| of the exact sum of the
   [n] elements it adds, where they are finite and their magnitudes' sum is
   below half the dtype's largest float, so that no sum overflows. *)
let law_bound (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let f32 = dtype_of c.x = D.Any D.Float32 in
      let get e = if f32 then get32 e 0 else get64 e 0 in
      let u = if f32 then 0x1p-24 else 0x1p-53 in
      let top = if f32 then 0x1.fffffep127 else max_float in
      let xs = shape_of c.x and es = elements c.x in
      let terms = Array.make (total c.shape) [] in
      for k = 0 to total xs - 1 do
        match target c (index_of xs k) with
        | None -> ()
        | Some y -> terms.(y) <- get es.(k) :: terms.(y)
      done;
      let held t =
        List.for_all Float.is_finite t
        && List.fold_left (fun a x -> a +. Float.abs x) 0. t <= top /. 2.
      in
      cover "an element of several finite terms"
        (Array.exists (fun t -> List.length t > 1 && held t) terms);
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y ->
          let got = Array.map get (elements y) in
          let bad = ref [] in
          Array.iteri
            (fun i t ->
              if held t then begin
                (* The exact sum: TwoSum's errors carried in a second part. *)
                let hi = ref 0. and lo = ref 0. in
                List.iter
                  (fun x ->
                    let s = !hi +. x in
                    let z = s -. !hi in
                    lo := !lo +. (!hi -. (s -. z)) +. (x -. z);
                    hi := s)
                  t;
                let exact = !hi +. !lo in
                let mag = List.fold_left (fun a x -> a +. Float.abs x) 0. t in
                let n = Float.of_int (max 0 (List.length t - 1)) in
                let g = n *. u /. (1. -. (n *. u)) in
                let bound = (g *. mag) +. (0x1p-100 *. mag) in
                if not (Float.abs (got.(i) -. exact) <= bound) then
                  bad :=
                    Printf.sprintf "element %d: %h is %h from %h, past %h" i
                      got.(i) (got.(i) -. exact) exact bound
                    :: !bad
              end)
            terms;
          equal (list string) []
            (List.filteri (fun i _ -> i < 8) (List.rev !bad)))

(* A map of the padded load of an array of the case's result shape, seen
   through the case's [load] view, loads at each index the element the fold
   would add that index into, or the fill where the load reads padding. *)
let law_load (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let module K = (val b.kernels) in
      let (A.Any x) = c.x in
      let dt = A.dtype x and d = dtype_of c.x in
      let rs = Random.State.make [| 7 |] in
      let a = A.expect dt (operand d c.shape c.load (fun _ -> element d rs)) in
      let drawn = Rig.Buffer.of_string (element d rs) in
      let fill = A.get (A.v dt (L.contiguous [||]) drawn) [||] in
      let fill_bits = Nx_kernel.Prog.bits dt fill in
      let xs = shape_of c.x in
      let p = Nx_kernel.Prog.v ~ins:[| d |] [| In 0 |] ~outs:[| 0 |] in
      (* The load beside 100 unused constants: a program long enough that
         nx.cpu prices it with the domain lock released. *)
      let long =
        Nx_kernel.Prog.v ~ins:[| d |]
          (Iarray.init 101 (fun i ->
               if i = 0 then Nx_kernel.Prog.In 0
               else Const (D.Any D.Uint8, "\000")))
          ~outs:[| 0 |]
      in
      let some = total xs > 0 in
      cover "padding read" (some && total c.shape < total xs);
      cover "a windowed load" (some && Iarray.length c.pad.windows > 0);
      cover "sub-byte" (some && D.bits dt < 8);
      cover "a stepped operand" (some && c.load.stepped);
      cover "a reversed operand" (some && c.load.reversed <> None);
      let load = Nx_kernel.Padded { x = a; fill; pad = c.pad } in
      let es = elements (A.Any a) in
      let want =
        Array.init (total xs) (fun k ->
            match target c (index_of xs k) with
            | Some y -> es.(y)
            | None -> fill_bits)
      in
      List.iter
        (fun p ->
          let dst = A.create Rig.host dt (Iarray.of_array xs) in
          match K.map p ~dsts:[| A.Any dst |] [| load |] with
          | A.Declined -> failf "%s declined a padded map" b.name
          | A.Done -> equal (array string) want (elements (A.Any dst))
          | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r)
        [ p; long ])

(* A convolution's input gradient, large enough for the job's threads, at
   float32, int4 and bfloat16, the last two from views of the operand. *)
let test_threads (b : Support.backend) () =
  let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
  let pad =
    {
      Nx_kernel.lo = [| 0; 0; 1; 1 |];
      hi = [| 0; 0; 1; 1 |];
      interior = [| 0; 0; 0; 0 |];
      windows = [| w 2; w 3 |];
    }
  in
  let fold d shape v =
    let rs = Random.State.make [| 11 |] in
    let x =
      operand d (Option.get (loaded pad shape)) v (fun _ -> element d rs)
    in
    let c = { shape; pad; x; views = view_names v; load = plain 4 } in
    match fold_on b c with
    | None -> failf "%s declined a fold" b.name
    | Some y -> equal (array (same (dtype_of c.x))) (expected c) (elements y)
  in
  fold (D.Any D.Float32) [| 8; 16; 20; 20 |] (plain 6);
  (* Slabs of an odd number of int4 elements: neighbouring units' results
     share a byte. *)
  fold (D.Any D.Int4) [| 8; 16; 21; 21 |] { (plain 6) with stepped = true };
  fold (D.Any D.Bfloat16) [| 8; 16; 21; 21 |] { (plain 6) with reversed = Some 3 }

(* Rows longer than a slot of float32, of a narrow float and of int4: they add
   through their carrier a piece at a time. *)
let test_long_rows (b : Support.backend) () =
  let pad =
    {
      Nx_kernel.lo = [| 0; 1 |];
      hi = [| 0; 1 |];
      interior = [| 0; 0 |];
      windows = [| { M.axis = 1; size = 3; step = 1; dilation = 1 } |];
    }
  in
  let shape = [| 2; 5000 |] in
  List.iter
    (fun d ->
      let rs = Random.State.make [| 13 |] in
      let x =
        operand d
          (Option.get (loaded pad shape))
          (plain 3)
          (fun _ -> element d rs)
      in
      let c = { shape; pad; x; views = []; load = plain 2 } in
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y -> equal (array (same (dtype_of c.x))) (expected c) (elements y))
    D.[ Any Bfloat16; Any Int4 ]

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p =
    { Nx_kernel.lo = [| 1 |]; hi = [| 1 |]; interior = [| 0 |]; windows = [||] }
  in
  let x = A.of_array D.Float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  equal ~msg:"an operand of another shape" answer A.Shape_mismatch
    (K.fold p
       ~dst:(A.create Rig.host D.Float32 [| 3 |])
       (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]));
  equal ~msg:"a destination of another shape" answer A.Shape_mismatch
    (K.fold p ~dst:(A.create Rig.host D.Float32 [| 4 |]) x);
  let id = Nx_kernel.Prog.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |] in
  let bad = { p with lo = [| 1; 0 |] } in
  equal ~msg:"a map's pad of another rank" answer A.Shape_mismatch
    (K.map id
       ~dsts:[| A.Any (A.create Rig.host D.Float32 [| 7 |]) |]
       [| Padded { x; fill = 0.; pad = bad } |]);
  equal ~msg:"booleans" answer A.Wrong_dtype
    (K.fold p
       ~dst:(A.create Rig.host D.Bool [| 3 |])
       (A.of_array D.Bool [| 5 |] [| true; false; true; false; true |]))

(* The suite *)

let cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ())

let all_cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ~dtypes:D.all ())

let float_cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ~dtypes:D.[ Any Float32; Any Float64 ] ())

let int_cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ~dtypes:[ D.Any D.Int64 ] ())

(* Guard pages *)

let guard_cases =
  Gen.pair
    (Gen.with_pp
       (fun ppf -> function
         | None -> Format.pp_print_string ppf "no load"
         | Some c -> pp_case ppf c)
       (case_gen ~last:tails ()))
    sides

(* The same result with the operand, through its view, and the destination
   against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  match c with
  | None -> reject ()
  | Some c ->
      let after = side = Support.After in
      cover "windows" (Iarray.length c.pad.windows > 0);
      cover "sub-byte" (let (D.Any dt) = dtype_of c.x in D.bits dt < 8);
      cover "a stepped operand at the page before"
        ((not after) && List.mem "stepped" c.views);
      cover "a reversed operand at the page after"
        (after && List.mem "reversed" c.views);
      let call xs ds =
        let (A.Any x) = xs.(0) in
        K.fold c.pad ~dst:(A.expect (A.dtype x) ds.(0)) x
      in
      equal answer A.Done
        (same_guarded ~any_nan:true side ~call [| c.x |]
           [| (dtype_of c.x, c.shape) |])

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a fold is the padded load's adjoint" int_cases (run (law_adjoint b));
      prop "a float fold is within γ(n - 1) Σ|x| of the exact sum" float_cases
        (run (law_bound b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "sums each element's taps in order, +0 where none" cases
        (run (law_reference b));
      prop "a map's padded load reads what the fold adds into" all_cases
        (run (law_load b));
      test "a convolution's gradient on the job's threads" (fun () ->
          b.around (test_threads b));
      test "long rows of narrow dtypes" (fun () -> b.around (test_long_rows b));
      test "refuses shapes and pads that do not fit, and booleans" (fun () ->
          b.around (test_refusals b));
      prop ~count:300
        "reads and writes no byte past the operand and dst, whatever pages lie \
         beside them"
        guard_cases (run (law_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.fold"
       (List.map laws Support.backends @ [ cpu Support.cpu ]))
