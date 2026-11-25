(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S's assemble on every kernel library the host runs: at each index
   the element of the last piece whose region holds it, or the fill, bit for
   bit, at every dtype, through views, over regions that tile, overlap, step or
   leave gaps; a first piece that is the destination over the whole result; the
   calls that do not fit and the sharing the door refuses; and nx.cpu's threads
   and guard pages. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

type case = {
  dtype : D.any;
  shape : int array;
  fill : string;
  init : string array; (* dst's elements before the call *)
  regions : M.range array array;
  pieces : A.any array;
  form : string;
  views : string list;
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
  Format.fprintf ppf "%s of %a into %a from %d pieces %s (%s)" c.form pp_dtype
    c.dtype pp_ints c.shape (Array.length c.pieces)
    (String.concat " " (Array.to_list (Array.map region_string c.regions)))
    (String.concat ", " c.views)

(* The value of the element of bits [e], and the bits a kernel stores for it:
   a boolean's 0 or 1, a NaN as its float reads it. *)
let value (type v s) (dt : (v, s) D.t) e : v =
  A.get (A.expect dt (of_elements (D.Any dt) [||] [| e |])) [||]

let canonical (D.Any dt) e = Nx_kernel.Prog.bits dt (value dt e)
let whole d = { M.start = 0; count = d; step = 1 }
let range start count step = { M.start; count; step }

(* The shape of the region [r] of an array of shape [s]. *)
let sliced r s =
  Iarray.to_array (M.shape (Slice (Iarray.of_array r)) (Iarray.of_array s))

(* The case of the pieces of [regions] of [shape] at [d], each through its view
   of [views]: their elements, the fill and dst's elements before the call drawn
   from [rs]. *)
let case_of d form shape regions views rs =
  let pieces =
    Array.mapi
      (fun j r -> operand d (sliced r shape) views.(j) (fun _ -> element d rs))
      regions
  in
  let fill = canonical d (element d rs) in
  let init = Array.init (total shape) (fun _ -> element d rs) in
  let views = List.concat_map view_names (Array.to_list views) in
  { dtype = d; shape; fill; init; regions; pieces; form; views }

(* Plain views for the pieces of [regions] of [shape]. *)
let plains shape regions =
  Array.map (fun _ -> plain (Array.length shape)) regions

(* The reference: at each index the element of the last piece whose region holds
   it, the fill where none does. *)
let expected c =
  let out = Array.make (total c.shape) c.fill in
  Array.iteri
    (fun j r ->
      let ps = sliced r c.shape and es = elements c.pieces.(j) in
      Array.iteri
        (fun k e ->
          let i = index_of ps k in
          let t = Array.mapi (fun a x -> r.(a).start + (x * r.(a).step)) i in
          out.(flat c.shape t) <- e)
        es)
    c.regions;
  out

(* [c] assembled on [b]'s device into a destination holding [c.init]. *)
let assemble_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.dtype in
  let dst = Support.on b (A.expect dt (of_elements c.dtype c.shape c.init)) in
  let pieces =
    Array.map2
      (fun r x -> (Iarray.of_array r, Support.on b (A.expect dt x)))
      c.regions c.pieces
  in
  match K.assemble ~fill:(value dt c.fill) ~dst (Iarray.of_array pieces) with
  | A.Done -> A.Any dst
  | r -> failf "%s answered %a" b.name Nx_array_support.pp_answer r

(* Cases *)

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
          (3, int_range 0 7); (1, int_range A.max_operands (3 * A.max_operands));
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
          (fun i e -> if i = axis then range start count 1 else whole e)
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
        go (i - 1) (range start count 1 :: acc)
    in
    let+ rs = go (r - 1) [] in
    ("pad", [| rs |])
  in
  frequency [ (2, any); (1, concat); (1, pad) ]

(* A piece's view of rank [r]: broadcast along some axes one time in four,
   windowed one time in four. *)
let piece_view r =
  let open Gen in
  let* broadcast =
    frequency [ (3, constant []); (1, subsequence (List.init r Fun.id)) ]
  in
  view_of ~broadcast ~windows:true r

let small = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5)

let case_gen shapes =
  let open Gen in
  let* d = of_list ~pp:pp_dtype D.all in
  let* shape = shapes in
  let* form, regions = regions_gen shape in
  let* views =
    array
      ~size:(constant (Array.length regions))
      (piece_view (Array.length shape))
  in
  let+ seed = int in
  case_of d form shape regions views (Random.State.make [| seed |])

(* Pieces of regions near a tiling (Nx_array_gen.near_tiling) at a dtype of each
   width. *)
let tilings =
  let open Gen in
  let* d =
    of_list ~pp:pp_dtype
      D.[ Any Float32; Any Int8; Any Int4; Any Bool; Any Complex64 ]
  in
  let* shape, regions = Nx_array_gen.near_tiling in
  let* views =
    array
      ~size:(constant (Array.length regions))
      (view_of (Array.length shape))
  in
  let+ seed = int in
  case_of d "tiling" shape regions views (Random.State.make [| seed |])

let drawn =
  Gen.with_pp pp_case (Gen.frequency [ (1, case_gen small); (1, tilings) ])

(* Examples *)

(* A pad's one piece of [region], pairs of a start and a count, inside
   [shape]. *)
let pad ?view ?(dtype = D.Any D.Float32) shape region =
  let regions =
    [| Array.map (fun (start, count) -> range start count 1) region |]
  in
  let views =
    match view with Some v -> [| v |] | None -> plains shape regions
  in
  case_of dtype "pad" shape regions views
    (Random.State.make [| Array.length shape |])

(* Pieces of one element at each index of [n - 1], as many as the operands a
   call reads at once for [n] of [Nx_array.max_operands]. *)
let ones n =
  let regions = Array.init (n - 1) (fun i -> [| range i 1 1 |]) in
  let c =
    case_of (D.Any D.Float32) "slices"
      [| n - 1 |]
      regions (plains [| 1 |] regions)
      (Random.State.make [| n |])
  in
  { c with fill = f32 (-1.) }

(* Two pieces of a [2; 3] result that overlap at one index and leave one to the
   fill, through a reversed and a stepped view, at [d]. *)
let overlapping (D.Any dt as d) =
  let regions =
    [| [| range 0 2 1; range 0 2 1 |]; [| range 1 1 1; range 1 2 1 |] |]
  in
  let views =
    [|
      { (plain 2) with reversed = Some 1 }; { (plain 2) with stepped = true };
    |]
  in
  let rs = Random.State.make [| D.bits dt |] in
  let c = case_of d "slices" [| 2; 3 |] regions views rs in
  let rec nonzero () =
    let e = canonical d (element d rs) in
    if e = zero d then nonzero () else e
  in
  { c with fill = nonzero () }

let bits32 x =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 x;
  Bytes.to_string b

let bits64 x =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 x;
  Bytes.to_string b

(* A pad's piece of NaNs of [d], signalling and quiet, of both signs and of
   payloads that mark them: [es], each an element's bits. *)
let nans d es =
  let c = pad ~dtype:d [| 1; 6 |] [| (0, 1); (1, Array.length es) |] in
  let n = Array.length es in
  let piece = operand d [| 1; n |] (plain 2) (fun k -> es.(k)) in
  { c with pieces = [| piece |] }

(* A fill alone, which drawn cases reach rarely; pads whose piece covers inner
   axes whole, which join the rows where the piece's steps compose, through
   plain, reversed, stepped and broadcast views, and an empty piece that starts
   far outside, of a byte dtype and of a sub-byte one; pieces of one element at
   each index, as many as the operands a call reads at once and its neighbours,
   and twice as many; two overlapping pieces at every dtype; NaNs whose payloads
   a copy keeps; the residues of an axis, which tile it; and two intervals that
   overlap as many indices as one leaves. *)
let examples =
  let m = A.max_operands in
  let joined = [| (0, 2); (1, 1); (0, 2) |] in
  let fill_alone =
    case_of (D.Any D.Float32) "slices" [| 2; 3 |] [||] [||]
      (Random.State.make [| 0 |])
  in
  let two shape regions =
    case_of (D.Any D.Int8) "slices" shape regions (plains shape regions)
      (Random.State.make [| 5 |])
  in
  [
    fill_alone;
    pad [| 3; 4; 1; 1 |] [| (1, 1); (0, 4); (0, 1); (0, 1) |];
    pad [| 2; 3; 2 |] joined;
    pad [| 1; 3; 1; 2 |] [| (0, 1); (1, 2); (0, 1); (0, 2) |];
    pad ~view:{ (plain 3) with reversed = Some 2 } [| 2; 3; 2 |] joined;
    pad ~view:{ (plain 3) with stepped = true } [| 2; 3; 2 |] joined;
    pad ~view:{ (plain 3) with broadcast = [ 2 ] } [| 2; 3; 2 |] joined;
    pad [| 2; 3 |] [| (max_int, 0); (0, 3) |];
    pad ~dtype:(D.Any D.Int4) [| 2; 3 |] [| (max_int, 0); (0, 3) |];
    nans (D.Any D.Float32) (Array.map bits32 nans32);
    nans (D.Any D.Float64) (Array.map bits64 nans64);
    nans (D.Any D.Complex64)
      [| bits32 0xFFA00002l ^ f32 1.; f32 (-0.) ^ bits32 0x7F800001l |];
    two [| 5; 2 |] [| [| range 0 3 2; whole 2 |]; [| range 1 2 2; whole 2 |] |];
    two [| 4 |] [| [| range 0 2 1 |]; [| range 1 2 1 |] |];
  ]
  @ List.map ones [ m - 1; m; m + 1; (2 * m) - 1; 2 * m; (2 * m) + 1 ]
  @ List.map overlapping D.all

(* Laws *)

let law_assemble (b : Support.backend) c =
  let want = expected c in
  let held = Nx_array_gen.coverage c.shape c.regions in
  let has f = Array.exists (Array.exists f) c.regions in
  let tile = Array.for_all (( = ) 1) held in
  cover "concatenation" (c.form = "concatenation");
  cover "pad" (c.form = "pad");
  cover "fill alone" (c.pieces = [||]);
  cover "more pieces than operands a call reads at once"
    (Array.length c.pieces + 1 > A.max_operands);
  cover "tile" tile;
  cover "tile by residues"
    (tile && has (fun (x : M.range) -> x.count > 1 && x.step > 1));
  cover "overlap, as many elements as the result"
    (Array.exists (fun n -> n > 1) held
    && Array.fold_left ( + ) 0 held = total c.shape);
  cover "leave a gap" (Array.exists (( = ) 0) held);
  cover "a broadcast piece" (List.mem "broadcast" c.views);
  cover "a windowed piece" (List.mem "windowed" c.views);
  cover "a signalling NaN moved" (Array.exists (signalling c.dtype) want);
  List.iter (fun (D.Any dt as d) -> cover (D.name dt) (c.dtype = d)) D.all;
  equal (array string) want (elements (assemble_on b c))

(* The same call with its first piece over the whole result as dst itself, and
   as a fresh array of the same elements. *)
let law_in_place (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.dtype in
  let first = of_elements c.dtype c.shape c.init in
  let fresh =
    {
      c with
      regions = Array.append [| Array.map whole c.shape |] c.regions;
      pieces = Array.append [| first |] c.pieces;
    }
  in
  let want = elements (assemble_on b fresh) in
  (* Copied where the host addresses it: sub-byte copies need it. *)
  let dst = Support.on b (A.copy (A.expect dt first)) in
  let pieces =
    Array.map2
      (fun r x -> (Iarray.of_array r, Support.on b (A.expect dt x)))
      c.regions c.pieces
  in
  let pieces =
    Array.append [| (Iarray.of_array (Array.map whole c.shape), dst) |] pieces
  in
  equal answer A.Done
    (K.assemble ~fill:(value dt c.fill) ~dst (Iarray.of_array pieces));
  equal (array string) want (elements (A.Any dst))

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
  let r start count step = Gen.constant (range start count step) in
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

let law_misfit (b : Support.backend) c =
  let module K = (val b.kernels) in
  let zeros s =
    Support.on b
      (A.of_array D.Float32 (Iarray.of_array s) (Array.make (total s) 0.))
  in
  let want =
    Nx_kernel.Shape.assemble c.into
      (Array.map (fun (r, _) -> Iarray.of_array r) c.parts)
  in
  let fit = want = Ok (Array.map snd c.parts) in
  cover "fits" fit;
  cover "misfits" (not fit);
  let pieces =
    Iarray.of_array
      (Array.map (fun (r, s) -> (Iarray.of_array r, zeros s)) c.parts)
  in
  Misfit.law ~msg:"an assembly" ~what:"a shape" b ~fit
    [| (D.Any D.Float32, c.into) |]
    (fun g ds ->
      K.assemble ~fill:0. ~dst:(A.expect D.Float32 ds.(0))
        (Iarray.map (fun (r, x) -> (r, g.Misfit.g x)) pieces))

(* Sharing the door refuses, into a dst of four elements: dst reversed as a
   first piece over the whole result, and a piece of dst's second element over
   the region of its first. The door accepts an operand identical to dst, which
   only the first piece over the whole result may be. *)
let overlaps =
  [
    ("dst reversed as the first piece", ([| whole 4 |], `Reversed));
    ("an element of dst at another index", ([| range 0 1 1 |], `Second));
  ]

let test_overlap (b : Support.backend) (_, piece) =
  let module K = (val b.kernels) in
  let dst = Support.on b (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let view l = A.v D.Float32 l (A.buffer dst) in
  let region, piece = piece in
  let piece =
    match piece with
    | `Reversed -> view (L.v ~offset:3 ~strides:[| -1 |] [| 4 |])
    | `Second -> view (L.v ~offset:1 ~strides:[| 1 |] [| 1 |])
  in
  let pieces : _ iarray = [| (Iarray.of_array region, piece) |] in
  let before = elements (A.Any dst) in
  equal answer A.Overlapping (K.assemble ~fill:0. ~dst pieces);
  equal (array string) before (elements (A.Any dst))

(* Calls whose last piece, read in a later run of claims than dst, is refused: a
   piece off its region, which does not fit, and one over dst's bytes. *)
let late_refusals =
  [
    ("a last piece off its region", `Long);
    ("a last piece over dst's bytes", `Dst);
  ]

let test_late_refusal (b : Support.backend) (_, last) =
  let module K = (val b.kernels) in
  let n = 3 * A.max_operands in
  let one i : _ iarray = [| range i 1 1 |] in
  let dst = A.create b.device D.Float32 [| n |] in
  let xs = Array.init (n - 1) (fun _ -> A.create b.device D.Float32 [| 1 |]) in
  let last =
    match last with
    | `Long -> A.create b.device D.Float32 [| 2 |]
    | `Dst ->
        A.v D.Float32 (L.v ~offset:1 ~strides:[| 1 |] [| 1 |]) (A.buffer dst)
  in
  let pieces =
    Array.append
      (Array.mapi (fun i x -> (one i, x)) xs)
      [| (one (n - 1), last) |]
  in
  not_equal answer A.Done (K.assemble ~fill:0. ~dst (Iarray.of_array pieces));
  Array.iteri
    (fun i x ->
      Nx_array_gen.released
        ~msg:(Printf.sprintf "array %d released: a take of it succeeds" i)
        (A.buffer x))
    (Array.append [| dst |] xs)

(* nx.cpu *)

(* Fills large enough for the job's threads: an identity's, around its diagonal
   every 513th element of 512 x 512 merged, at a byte-wide and a sub-byte dtype;
   a pad's, around a 400 x 400 piece inside 600 x 600, and of a row longer than
   a unit's part, at float32; a pad's suffix of 1.6 Mi int4 elements after a
   piece at index 0, its fill starting mid-byte. *)
let threads =
  let f32 = D.Any D.Float32 and i4 = D.Any D.Int4 in
  let diagonal = [| range 0 512 513 |] in
  [
    ("a float32 diagonal", f32, "slices", [| 512 * 512 |], diagonal);
    ("an int4 diagonal", i4, "slices", [| 512 * 512 |], diagonal);
    ( "a float32 square",
      f32,
      "pad",
      [| 600; 600 |],
      [| range 100 400 1; range 100 400 1 |] );
    ("a float32 row", f32, "pad", [| 200_001 |], [| range 70_000 100_000 1 |]);
    ("an int4 suffix", i4, "pad", [| 1_600_001 |], [| range 0 1 1 |]);
  ]

let test_threads (b : Support.backend) (_, d, form, shape, region) =
  let regions = [| region |] in
  let c =
    case_of d form shape regions (plains shape regions)
      (Random.State.make [| 3 |])
  in
  equal (array string) (expected c) (elements (assemble_on b c))

(* The same result with the pieces, through their views, and the
   destination against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  List.iter (fun f -> cover f (c.form = f)) [ "slices"; "concatenation"; "pad" ];
  cover "at the page after" (side = Support.After);
  cover "at the page before" (side = Support.Before);
  let (D.Any dt) = c.dtype in
  let call xs ds =
    let pieces =
      Array.map2 (fun r x -> (Iarray.of_array r, A.expect dt x)) c.regions xs
    in
    K.assemble ~fill:(value dt c.fill) ~dst:(A.expect dt ds.(0))
      (Iarray.of_array pieces)
  in
  equal answer A.Done
    (same_guarded side ~call c.pieces [| (c.dtype, c.shape) |])

(* The suite *)

let laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group d.label
    [
      group "assemble"
        [
          prop ~count:300 ~examples
            "stores at each index the element of the last piece whose region \
             holds it, or fill where none does"
            drawn (run law_assemble);
          prop
            "a first piece identical to dst over the whole result gives what a \
             fresh dst receives"
            drawn (run law_in_place);
          prop
            "a call of shapes Shape.assemble does not give is not Done and \
             reads and writes nothing outside its arrays"
            misfits (run law_misfit);
          cases
            "a piece that shares dst's bytes and is not identical to dst is \
             refused as Overlapping before any write"
            ~id:fst overlaps (run test_overlap);
          cases
            "a refusal after the door's first run of claims leaves every array \
             unclaimed"
            ~id:fst late_refusals (run test_late_refusal);
        ];
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f b x) in
  let id (n, _, _, _, _) = n in
  group ("nx.cpu " ^ b.name)
    [
      cases
        "an assembly over the job's threads stores each piece's elements and \
         the fill"
        ~id threads (run test_threads);
      prop ~count:300
        "reads and writes no byte past the pieces and dst, whatever pages lie \
         beside them"
        (Gen.pair (Gen.with_pp pp_case (case_gen tail_shapes)) sides)
        (run law_guards);
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so an assembly through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree call =
    Elements.agree_bits (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's"
        [
          prop ~examples "of assemblies" drawn (fun c ->
              agree (fun b -> [| assemble_on b c |]));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.assemble"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [
                 Support.declared_cuda;
                 Support.declared_amd;
                 Support.declared_metal;
               ])
       @ [ cpu Support.cpu ]
       @ nv))
