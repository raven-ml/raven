(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Folds through every kernel library the host runs: a fold is the padded load's
   adjoint over integers. nx.cpu's group checks each result element against a
   reference that adds, from +0, the elements the load reads from it in C order
   of their taps, bit for bit, through views of the operand, at the dtypes
   nx_cpu.mli lists. *)

open Windtrap
open Elements
module S = Nx_kernel.Spec
module Support = Nx_kernels_support

type case = { shape : int array; pad : S.pad; x : A.any; views : string list }

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf
    "fold into %a of %a %a: lo %a hi %a interior %a, windows %s (%s)" pp_ints
    c.shape D.pp (A.dtype x) pp_ints (shape_of c.x) pp_ints c.pad.lo pp_ints
    c.pad.hi pp_ints c.pad.interior
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (w : M.window) ->
               Printf.sprintf "%d:%dx%d+%d" w.axis w.size w.step w.dilation)
             c.pad.windows)))
    (String.concat ", " c.views)

(* The shape a load by [p] gives an array of shape [s], if any. *)
let loaded (p : S.pad) s =
  match S.fold ~shape:s p with
  | exception Invalid_argument _ -> None
  | _ ->
      let padded =
        Array.mapi
          (fun i d ->
            p.lo.(i) + p.hi.(i) + d
            + if d > 0 then p.interior.(i) * (d - 1) else 0)
          s
      in
      Some
        (if p.windows = [||] then padded else M.shape (Window p.windows) padded)

let folded =
  D.[ Any Float32; Any Float64; Any Int32; Any Int8; Any Uint16; Any Int64 ]

let case_gen ?(dtypes = folded) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* shape = array ~size:(int_range 0 3) (int_range 0 5) in
  let r = Array.length shape in
  let small = int_range (-2) 3 in
  let* lo = array ~size:(constant r) small in
  let* hi = array ~size:(constant r) small in
  let* interior = array ~size:(constant r) (int_range 0 2) in
  let* on = array ~size:(constant r) bool in
  let* sizes = array ~size:(constant r) (int_range 1 3) in
  let* steps = array ~size:(constant r) (int_range 1 3) in
  let* dilations = array ~size:(constant r) (int_range 1 2) in
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
  let pad = { S.lo; hi; interior; windows } in
  let* seed = int in
  let xs = loaded pad shape in
  let rx = match xs with Some s -> Array.length s | None -> 0 in
  let+ v = view_of rx in
  match xs with
  | None -> None
  | Some xs ->
      let rs = Random.State.make [| seed |] in
      let x = operand d xs v (fun _ -> element d rs) in
      Some { shape; pad; x; views = view_names v }

let fold_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let dst = A.create Rig.host (A.dtype x) c.shape in
  match K.fold (S.fold ~shape:c.shape c.pad) ~dst x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The result element each operand element lands on, or [None] for one the load
   reads from padding. *)
let target c xi =
  let r = Array.length c.shape in
  let q = Array.sub xi 0 r in
  Array.iteri
    (fun k (w : M.window) ->
      q.(w.axis) <- (xi.(w.axis) * w.step) + (xi.(r + k) * w.dilation))
    c.pad.windows;
  let i =
    Array.mapi
      (fun a q ->
        let t = q - c.pad.lo.(a) and s = c.pad.interior.(a) + 1 in
        if t < 0 || t mod s <> 0 || t / s >= c.shape.(a) then -1 else t / s)
      q
  in
  if Array.exists (fun v -> v < 0) i then None else Some (flat c.shape i)

(* Prog.Add on two elements of [d]: floats by {!Elements.add_bits}, integers
   wrapping. *)
let add d a b =
  if D.Any D.Float32 = d || D.Any D.Float64 = d then
    add_bits ~w:(String.length a) a b
  else int_bits d (String.length a) (Int64.add (int_value d a) (int_value d b))

let expected c =
  let d = dtype_of c.x in
  let xs = shape_of c.x and es = elements c.x in
  let r = Array.length c.shape and nw = Array.length c.pad.windows in
  let acc = Array.make (total c.shape) (zero d) in
  let taps = Array.sub xs r nw and outer = Array.sub xs 0 r in
  for t = 0 to total taps - 1 do
    let j = index_of taps t in
    for k = 0 to total outer - 1 do
      let xi = Array.append (index_of outer k) j in
      match target c xi with
      | None -> ()
      | Some y -> acc.(y) <- add d acc.(y) es.(flat xs xi)
    done
  done;
  acc

let law_reference (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      cover "windows" (c.pad.windows <> [||]);
      cover "interior" (Array.exists (fun i -> i > 0) c.pad.interior);
      cover "cropped" (Array.exists (fun l -> l < 0) c.pad.lo);
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y -> equal (array string) (expected c) (elements y))

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

(* A convolution's input gradient, large enough for the job's threads. *)
let test_threads (b : Support.backend) () =
  let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
  let pad =
    {
      S.lo = [| 0; 0; 1; 1 |];
      hi = [| 0; 0; 1; 1 |];
      interior = [| 0; 0; 0; 0 |];
      windows = [| w 2; w 3 |];
    }
  in
  let shape = [| 8; 16; 20; 20 |] in
  let d = D.Any D.Float32 and rs = Random.State.make [| 11 |] in
  let x =
    operand d (Option.get (loaded pad shape)) (plain 6) (fun _ -> element d rs)
  in
  let c = { shape; pad; x; views = [] } in
  match fold_on b c with
  | None -> failf "%s declined a fold" b.name
  | Some y -> equal (array string) (expected c) (elements y)

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p =
    { S.lo = [| 1 |]; hi = [| 1 |]; interior = [| 0 |]; windows = [||] }
  in
  let s = S.fold ~shape:[| 3 |] p in
  let x = A.of_array D.Float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  equal ~msg:"an operand of another shape" answer A.Shape_mismatch
    (K.fold s
       ~dst:(A.create Rig.host D.Float32 [| 3 |])
       (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]));
  equal ~msg:"a destination of another shape" answer A.Shape_mismatch
    (K.fold s ~dst:(A.create Rig.host D.Float32 [| 4 |]) x);
  equal ~msg:"booleans" answer A.Wrong_dtype
    (K.fold s
       ~dst:(A.create Rig.host D.Bool [| 3 |])
       (A.of_array D.Bool [| 5 |] [| true; false; true; false; true |]))

(* The suite *)

let cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ())

let int_cases =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ~dtypes:[ D.Any D.Int64 ] ())

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a fold is the padded load's adjoint" int_cases (run (law_adjoint b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "adds each element's taps in order from +0" cases
        (run (law_reference b));
      test "a convolution's gradient on the job's threads" (fun () ->
          b.around (test_threads b));
      test "refuses shapes that do not fit and booleans" (fun () ->
          b.around (test_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.fold"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
