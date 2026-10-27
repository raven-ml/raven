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
module Support = Nx_kernels_support

type case = {
  dtype : D.any;
  shape : int array;
  fill : string;
  regions : M.range array array;
  pieces : A.any array;
  form : string;
}

(* A region as its ranges, each [start+count*step]. *)
let region_string r =
  "["
  ^ String.concat "; "
      (Array.to_list
         (Array.map
            (fun (x : M.range) ->
              Printf.sprintf "%d+%d*%d" x.start x.count x.step)
            r))
  ^ "]"

let pp_case ppf c =
  Format.fprintf ppf "%s of %a into %a from %d pieces %s" c.form pp_dtype
    c.dtype pp_ints c.shape (Array.length c.pieces)
    (String.concat " " (Array.to_list (Array.map region_string c.regions)))

(* The value of the element of bits [e], and the bits a kernel stores for it:
   a boolean's 0 or 1, a NaN as its float reads it. *)
let value (type v s) (dt : (v, s) D.t) e : v =
  A.get (A.expect dt (of_elements (D.Any dt) [||] [| e |])) [||]

let canonical (D.Any dt) e = Nx_kernel.Prog.bits dt (value dt e)

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
  (* Up to 7 pieces, or more than a call reads at once,
     Nx_array.max_operands operands with the destination. *)
  let any =
    let pieces =
      frequency
        [
          (3, int_range 0 7);
          (1, int_range A.max_operands (3 * A.max_operands));
        ]
    in
    let+ rs = array ~size:pieces any_region in
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

let small = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5)

(* The shape of the region [r] of an array of shape [s]. *)
let sliced r s =
  Iarray.to_array (M.shape (Slice (Iarray.of_array r)) (Iarray.of_array s))

let case_gen ?(dtypes = D.all) ?(shapes = small) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* shape = shapes in
  let* form, regions = regions_gen shape in
  let* views =
    array ~size:(constant (Array.length regions)) (view_of (Array.length shape))
  in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let pieces =
    Array.mapi
      (fun j r ->
        operand d (sliced r shape) views.(j) (fun _ -> element d rs))
      regions
  in
  let fill = canonical d (element d rs) in
  { dtype = d; shape; fill; regions; pieces; form }

let assemble_on (b : Support.backend) ?(in_place = false) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.dtype in
  let pieces = Array.map (A.expect dt) c.pieces in
  let dst, pieces =
    if in_place then
      let p = A.copy pieces.(0) in
      (p, Array.mapi (fun j x -> if j = 0 then p else x) pieces)
    else (A.create Rig.host dt (Iarray.of_array c.shape), pieces)
  in
  let pieces =
    Array.map2 (fun r x -> (Iarray.of_array r, x)) c.regions pieces
  in
  match K.assemble ~fill:(value dt c.fill) ~dst pieces with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let expected c =
  let out = Array.make (total c.shape) c.fill in
  Array.iteri
    (fun j r ->
      let ps = sliced r c.shape
      and es = elements c.pieces.(j) in
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
  cover "more pieces than operands a call reads at once"
    (Array.length c.pieces + 1 > A.max_operands);
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

(* Fills large enough for the job's threads: an identity's, around its
   diagonal every 513th element of 512 x 512 merged, at a byte-wide and a
   sub-byte dtype; a pad's, around a 400 x 400 piece inside 600 x 600, and
   of a row longer than a unit's part, at float32; a pad's suffix of 1.6 Mi
   int4 elements after a piece at index 0, its fill starting mid-byte. *)
let test_threads (b : Support.backend) () =
  let rs = Random.State.make [| 3 |] in
  let case d form shape region =
    let piece =
      operand d (sliced region shape)
        (plain (Array.length shape))
        (fun _ -> element d rs)
    in
    let regions = [| region |] and pieces = [| piece |] in
    let fill = canonical d (element d rs) in
    { dtype = d; shape; fill; regions; pieces; form }
  in
  let check c =
    match assemble_on b c with
    | None -> failf "%s declined an assembly" b.name
    | Some y -> equal (array string) (expected c) (elements y)
  in
  let range start count = { M.start; count; step = 1 } in
  let diagonal = [| { M.start = 0; count = 512; step = 513 } |] in
  let f32 = D.Any D.Float32 and i4 = D.Any D.Int4 in
  check (case f32 "slices" [| 512 * 512 |] diagonal);
  check (case i4 "slices" [| 512 * 512 |] diagonal);
  check (case f32 "pad" [| 600; 600 |] [| range 100 400; range 100 400 |]);
  check (case f32 "pad" [| 200_001 |] [| range 70_000 100_000 |]);
  check (case i4 "pad" [| 1_600_001 |] [| range 0 1 |])

(* Refusals *)

(* An assembly's destination shape [into] and its pieces, each a region and
   a shape near a fit. *)
type misfit = { into : int array; parts : (M.range array * int array) array }

let pp_misfit ppf c =
  Format.fprintf ppf "assembly into %a of %s" pp_ints c.into
    (String.concat ", "
       (Array.to_list
          (Array.map
             (fun (r, s) -> Format.asprintf "%s %a" (region_string r) pp_ints s)
             c.parts)))

(* A range of an axis of extent [d] mostly, else one Move.Slice refuses:
   past the axis, before it, longer, of step 0, backwards out of it, or
   reaching past every int. *)
let range_near d =
  let r start count step = Gen.constant { M.start; count; step } in
  Gen.frequency
    [
      (12, Nx_array_gen.range d);
      (1, r d 1 1);
      (1, r (-1) 1 1);
      (1, r 0 (d + 1) 1);
      (1, r 0 1 0);
      (1, r 0 2 (-1));
      (1, r 0 max_int max_int);
    ]

let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let* into = Nx_array_gen.shape in
     let part =
       let rec go i acc =
         if i < 0 then constant (Array.of_list acc)
         else
           let* x = range_near into.(i) in
           go (i - 1) (x :: acc)
       in
       let* region = go (Array.length into - 1) [] in
       let r = Array.length region in
       let* region =
         frequency
           [
             (8, constant region);
             (1, constant (Array.append region [| whole 1 |]));
             (1, constant (Array.sub region 0 (max 0 (r - 1))));
           ]
       in
       (* A count past any array's is that of a region refused anyway. *)
       let counts = Array.map (fun (x : M.range) -> min 5 x.count) region in
       let+ s = near counts in
       (region, s)
     in
     let+ parts = array ~size:(int_range 0 3) part in
     { into; parts })

let law_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let zeros s =
    A.of_array D.Float32 (Iarray.of_array s) (Array.make (total s) 0.)
  in
  let want =
    Nx_kernel.Shape.assemble c.into
      (Array.map (fun (r, _) -> Iarray.of_array r) c.parts)
  in
  let fit = want = Ok (Array.map snd c.parts) in
  cover "fits" fit;
  cover "misfits" (not fit);
  equal answer
    (if fit then A.Done else A.Shape_mismatch)
    (K.assemble ~fill:0.
       ~dst:(A.create Rig.host D.Float32 (Iarray.of_array c.into))
       (Array.map (fun (r, s) -> (Iarray.of_array r, zeros s)) c.parts))

(* Whether no claim holds [b]'s memory: a donation of it is exclusive. *)
let unclaimed b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c -> Rig.Claim.exclusive c b)

(* A refusal of a piece read in a later run than the destination's leaves
   no claim on any operand. *)
let test_late_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 3 * A.max_operands in
  let one i : _ iarray = [| { M.start = i; count = 1; step = 1 } |] in
  let refused ~msg want last =
    let dst = A.create Rig.host D.Float32 [| n |] in
    let xs = Array.init (n - 1) (fun _ -> A.create Rig.host D.Float32 [| 1 |]) in
    let pieces =
      Array.append
        (Array.mapi (fun i x -> (one i, x)) xs)
        [| (one (n - 1), last dst) |]
    in
    equal ~msg answer want (K.assemble ~fill:0. ~dst pieces);
    Array.iter
      (fun x ->
        equal ~msg:(msg ^ ", released") bool true (unclaimed (A.buffer x)))
      (Array.append [| dst |] xs)
  in
  refused ~msg:"a last piece off its region" A.Shape_mismatch (fun _ ->
      A.create Rig.host D.Float32 [| 2 |]);
  refused ~msg:"a last piece over the destination's bytes" A.Overlapping
    (fun dst ->
      A.v D.Float32 (L.v ~offset:1 ~strides:[| 1 |] [| 1 |]) (A.buffer dst))

(* Guard pages *)

(* The same result with the pieces, through their views, and the
   destination against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  cover c.form true;
  cover "at the page after" (side = Support.After);
  cover "at the page before" (side = Support.Before);
  let (D.Any dt) = c.dtype in
  let call xs ds =
    let pieces =
      Array.map2 (fun r x -> (Iarray.of_array r, A.expect dt x)) c.regions xs
    in
    K.assemble ~fill:(value dt c.fill) ~dst:(A.expect dt ds.(0)) pieces
  in
  equal answer A.Done
    (same_guarded side ~call c.pieces [| (c.dtype, c.shape) |])

(* The suite *)

let cases = Gen.with_pp pp_case (case_gen ())

(* A fill alone, which drawn cases reach rarely; and pieces of one element
   at each index, as many as the operands a call reads at once and its
   neighbours, and twice as many. *)
let examples =
  let m = A.max_operands in
  let ones n =
    let np = n - 1 in
    let rs = Random.State.make [| n |] in
    {
      dtype = D.Any D.Float32;
      shape = [| np |];
      fill = f32 (-1.);
      regions =
        Array.init np (fun i -> [| { M.start = i; count = 1; step = 1 } |]);
      pieces =
        Array.init np (fun _ ->
            operand (D.Any D.Float32) [| 1 |] (plain 1) (fun _ ->
                element (D.Any D.Float32) rs));
      form = "slices";
    }
  in
  {
    dtype = D.Any D.Float32;
    shape = [| 2; 3 |];
    fill = f32 1.5;
    regions = [||];
    pieces = [||];
    form = "slices";
  }
  :: List.map ones [ m - 1; m; m + 1; (2 * m) - 1; 2 * m; (2 * m) + 1 ]

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
      test "fills on the job's threads" (fun () -> b.around (test_threads b));
      prop "refuses what Nx_kernel.Shape refuses and other shapes" misfits
        (run (law_refuses b));
      test "a refusal in a later run of reads leaves no claim" (fun () ->
          b.around (test_late_refusals b));
      prop ~count:300
        "reads and writes no byte past the pieces and dst, whatever pages lie \
         beside them"
        (Gen.pair (case_gen ~shapes:tail_shapes ()) sides)
        (run (law_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.assemble"
       (List.map laws Support.backends @ [ cpu Support.cpu ]))
