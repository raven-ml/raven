(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Assemblies through every kernel library the host runs: an assembly whose
   first piece is the destination over the whole result gives the fresh result.
   nx.cpu's group checks each element against a reference that copies the pieces
   in order over the fill, at every dtype, over regions that tile the result (a
   concatenation), sit inside it (a pad), step, reverse and overlap, with pieces
   through views. *)

open Windtrap
open Elements
module S = Nx_kernel.Spec
module Support = Nx_kernels_support

type case = {
  dtype : D.any;
  shape : int array;
  fill : string;
  regions : M.range array array;
  pieces : A.any array;
  form : string;
}

let pp_case ppf c =
  Format.fprintf ppf "%s of %a into %a from %d pieces %s" c.form pp_dtype
    c.dtype pp_ints c.shape (Array.length c.pieces)
    (String.concat " "
       (Array.to_list
          (Array.map
             (fun r ->
               "["
               ^ String.concat "; "
                   (Array.to_list
                      (Array.map
                         (fun (x : M.range) ->
                           Printf.sprintf "%d+%d*%d" x.start x.count x.step)
                         r))
               ^ "]")
             c.regions)))

let whole d = { M.start = 0; count = d; step = 1 }

(* Regions: any slices; a concatenation's along an axis; a pad's one piece
   inside the result. *)
let regions_gen shape =
  let open Gen in
  let r = Array.length shape in
  let any_region =
    let rec go i acc =
      if i < 0 then constant (Array.of_list acc)
      else
        let* x = Nx_array_gen.range shape.(i) in
        go (i - 1) (x :: acc)
    in
    go (r - 1) []
  in
  let any =
    let+ rs = array ~size:(int_range 0 7) any_region in
    ("slices", rs)
  in
  let concat =
    if r = 0 then any
    else
      let* axis = int_range 0 (r - 1) in
      let d = shape.(axis) in
      let* cut = int_range 0 d in
      let piece start count =
        Array.mapi
          (fun i e ->
            if i = axis then { M.start; count; step = 1 } else whole e)
          shape
      in
      let+ three = bool in
      if three && d >= 2 then
        let cut2 = max cut 1 in
        ( "concatenation",
          [| piece 0 (cut2 - 1); piece (cut2 - 1) 1; piece cut2 (d - cut2) |] )
      else ("concatenation", [| piece 0 cut; piece cut (d - cut) |])
  in
  let pad =
    let rec go i acc =
      if i < 0 then constant (Array.of_list acc)
      else
        let d = shape.(i) in
        let* start = int_range 0 d in
        let* count = int_range 0 (d - start) in
        go (i - 1) ({ M.start; count; step = 1 } :: acc)
    in
    let+ rs = go (r - 1) [] in
    ("pad", [| rs |])
  in
  frequency [ (2, any); (1, concat); (1, pad) ]

let case_gen ?(dtypes = D.all) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* shape = array ~size:(int_range 0 3) (int_range 0 5) in
  let* form, regions = regions_gen shape in
  let* views =
    array ~size:(constant (Array.length regions)) (view_of (Array.length shape))
  in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let pieces =
    Array.mapi
      (fun j r ->
        operand d (M.shape (Slice r) shape) views.(j) (fun _ -> element d rs))
      regions
  in
  (* A fill is an element's canonical bits: a boolean's 0 or 1. *)
  let fill = element d rs in
  let fill = if d = D.Any D.Bool && fill <> "\000" then "\001" else fill in
  { dtype = d; shape; fill; regions; pieces; form }

let assemble_on (b : Support.backend) ?(in_place = false) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.dtype in
  let pieces = Array.map (A.expect dt) c.pieces in
  let dst, pieces =
    if in_place then
      let p = A.copy pieces.(0) in
      (p, Array.mapi (fun j x -> if j = 0 then p else x) pieces)
    else (A.create Rig.host dt c.shape, pieces)
  in
  match
    K.assemble (S.assemble ~shape:c.shape ~fill:c.fill c.regions) ~dst pieces
  with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let expected c =
  let out = Array.make (total c.shape) c.fill in
  Array.iteri
    (fun j r ->
      let ps = M.shape (Slice r) c.shape and es = elements c.pieces.(j) in
      Array.iteri
        (fun k e ->
          let i = index_of ps k in
          let t = Array.mapi (fun a x -> r.(a).start + (x * r.(a).step)) i in
          out.(flat c.shape t) <- e)
        es)
    c.regions;
  out

let law_reference (b : Support.backend) c =
  cover "concatenation" (c.form = "concatenation");
  cover "pad" (c.form = "pad");
  cover "fill alone" (c.pieces = [||]);
  match assemble_on b c with
  | None -> failf "%s declined an assembly" b.name
  | Some y -> equal (array string) (expected c) (elements y)

(* A first piece that is the destination over the whole result gives the fresh
   result. *)
let law_in_place (b : Support.backend) c =
  let rs = Random.State.make [| 7 |] in
  let base =
    operand c.dtype c.shape
      (plain (Array.length c.shape))
      (fun _ -> element c.dtype rs)
  in
  let c =
    {
      c with
      regions = Array.append [| Array.map whole c.shape |] c.regions;
      pieces = Array.append [| base |] c.pieces;
    }
  in
  match (assemble_on b c, assemble_on b ~in_place:true c) with
  | Some fresh, Some own -> equal (array string) (elements fresh) (elements own)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2 |] [| 1.; 2. |] in
  let s fill =
    S.assemble ~shape:[| 4 |] ~fill
      [| [| { M.start = 1; count = 2; step = 1 } |] |]
  in
  let dst = A.create Rig.host D.Float32 [| 4 |] in
  equal ~msg:"a fill of another width" answer A.Wrong_dtype
    (K.assemble (s "\000\000") ~dst [| x |]);
  equal ~msg:"a piece off its region" answer A.Shape_mismatch
    (K.assemble
       (s (f32 0.))
       ~dst
       [| A.of_array D.Float32 [| 3 |] [| 1.; 2.; 3. |] |]);
  equal ~msg:"a destination of another shape" answer A.Shape_mismatch
    (K.assemble (s (f32 0.)) ~dst:(A.create Rig.host D.Float32 [| 5 |]) [| x |])

(* The suite *)

let cases = Gen.with_pp pp_case (case_gen ())

(* A fill alone, which drawn cases reach rarely. *)
let examples =
  [
    {
      dtype = D.Any D.Float32;
      shape = [| 2; 3 |];
      fill = f32 1.5;
      regions = [||];
      pieces = [||];
      form = "slices";
    };
  ]

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a first piece that is the destination gives the fresh result" cases
        (run (law_in_place b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples
        "the last piece holding an element wins, the fill elsewhere" cases
        (run (law_reference b));
      test "refuses fills and shapes that do not fit" (fun () ->
          b.around (test_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.assemble"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
