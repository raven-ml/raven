(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's gathers and scatters, element by element against a reference
   built here from the elements' bits: zero reads and dropped writes outside
   the axis, a target's updates in C order, a narrow float's sum in float32
   rounded once, any NaN for a NaN result; at every dtype and through
   strided, reversed, permuted and broadcast views; the same bits on one
   thread as on the job's, and the refusals. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support
open Elements
open Index_cases

(* Gathers *)

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

let scatter_agrees (b : Support.backend) c =
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> check_scatter c y

let law_scatter_reference (b : Support.backend) c =
  let shared, dropped = sharing c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  cover "narrow float add" (c.combine = Nx_kernel.Add && is_narrow (dtype_of c.into));
  scatter_agrees b c

(* The exponent and fraction widths of [d]'s floats, or of its parts, where
   a NaN of them can lack the quiet bit, the top of its fraction:
   float8_e4m3fn's one NaN of each sign has every fraction bit set, and
   float4_e2m1fn has none. *)
let nan_fields (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Some (11, 52)
  | D.Float32 | D.Complex64 -> Some (8, 23)
  | D.Float16 -> Some (5, 10)
  | D.Bfloat16 -> Some (8, 7)
  | D.Float8_e5m2 -> Some (5, 2)
  | _ -> None

(* Whether a part of [e], an element of [d], is a signalling NaN. *)
let signals d e =
  match nan_fields d with
  | None -> false
  | Some (exp, frac) ->
      let w = (1 + exp + frac) / 8 in
      let part p =
        let v = ref 0L in
        for j = w - 1 downto 0 do
          v := Int64.(logor (shift_left !v 8) (of_int (Char.code e.[(p * w) + j])))
        done;
        let ones = Int64.(sub (shift_left 1L exp) 1L) in
        let field = Int64.(logand (shift_right_logical !v frac) ones) in
        let fraction = Int64.(logand !v (sub (shift_left 1L frac) 1L)) in
        let quiet = Int64.(logand !v (shift_left 1L (frac - 1))) in
        field = ones && fraction <> 0L && quiet = 0L
      in
      List.exists part (List.init (String.length e / w) Fun.id)

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
    match nan_fields part with
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

(* A Max or Min scatter is a NaN where a target's element or an update is;
   a target no update reaches keeps its bits, a signalling NaN included. *)
let law_nan_extremes (b : Support.backend) c =
  let d = dtype_of c.into in
  let signals = signals d in
  let xs = elements c.into and ls = lands c in
  let some f = Array.exists Fun.id (Array.mapi f xs) in
  cover "a signalling NaN reaches a target"
    (some (fun t x -> ls.(t) <> [] && List.exists signals (x :: ls.(t))));
  cover "a signalling NaN no update reaches"
    (some (fun t x -> ls.(t) = [] && signals x));
  scatter_agrees b c

(* Laws *)

(* Each operand as a C-contiguous copy of its view gives the same bits, any
   NaN for a NaN. *)
let law_add_views (b : Support.backend) c =
  let plain (A.Any x) = A.Any (A.copy x) in
  let c' =
    { c with into = plain c.into; sidx = plain c.sidx; updates = plain c.updates }
  in
  cover "shared targets" (fst (sharing c));
  cover "through views" (c.sviews <> []);
  match (scatter_on b c, scatter_on b c') with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y) (elements y')
  | None, None -> ()
  | _ -> failf "%s declined a view or its copy" b.name

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

let zeros32 s =
  A.of_array D.Float32 (Iarray.of_array s) (Array.make (total s) 0.)

let zeros64 s =
  A.of_array D.Int64 (Iarray.of_array s) (Array.make (total s) 0L)

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

(* An Add of booleans answers Wrong_dtype. *)
let test_bool_add (b : Support.backend) () =
  let module K = (val b.kernels) in
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

(* Positions past the axis on both sides, cycling over 1003 targets. *)
let cycled = Array.init 1009 (fun i -> Int64.of_int ((i * 7 mod 1003) - 1))

(* Positions that miss targets 900 to 999 over the first half million
   updates, then [cycled]'s. *)
let late =
  Array.init 600_000 (fun i ->
      if i < 1 lsl 19 then Int64.of_int (i * 7 mod 900) else cycled.(i mod 1009))

(* Hundreds of thousands of updates into an axis of a thousand: a narrow
   float's Add over shared targets, targets first reached late, into
   reversed, sums through stepped and reversed updates; and a float32 Set
   and Max over shared targets. *)
let blocked_examples =
  let ex d = scatter_example (D.Any d) Nx_kernel.Add ~ts:[| 1000 |] ~axis:0 in
  let f32 combine =
    scatter_example (D.Any D.Float32) combine ~ts:[| 1000 |] ~axis:0
      ~m:600_000 cycled
  in
  [
    ex D.Float16 ~m:600_000 cycled;
    ex D.Bfloat16 ~m:600_000 ~tv:{ (plain 1) with reversed = Some 0 } late;
    ex D.Float8_e4m3fn ~m:600_000
      ~uv:{ (plain 1) with stepped = true; reversed = Some 0 }
      cycled;
    f32 Nx_kernel.Set;
    f32 Nx_kernel.Max;
  ]

(* An embedding gradient's rows: into transposed and the updates stepped
   and reversed; a narrow float's sum of half a million updates. *)
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

(* A Set of 65 536 rows of 1025 complex128 slices, the updates one row
   broadcast along the axis, positions cycling over 0, 1 and 2 into an axis
   of 2. *)
let wide_rows =
  let d = D.Any D.Complex128 and m = 65_536 and len = 1025 in
  let rs = Random.State.make [| 2 |] in
  let row = Array.init len (fun _ -> element d rs) in
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
    sviews = [ "updates broadcast" ];
  }

(* Parted scratch *)

(* The address space a forked child gives a parted float4_e2m1fn Add past
   its own: the Add's sort takes 32 bytes for each of its 4Mi updates, and
   the pool's threads their stacks. *)
let parted_cap = 512 lsl 20

(* A float4_e2m1fn array of shape [s], each element +0. *)
let f4_zeros s =
  let n = Array.fold_left ( * ) 1 s in
  A.of_array D.Float4_e2m1fn (Iarray.of_array s) (Array.make n 0.)

(* A float4_e2m1fn Add along axis 0 of [2Mi, 2], each update at a position
   of its own, so that threads part each column's targets, its elements
   sharing bytes, fits [parted_cap]: the child that runs it exits 0 where
   it is done, 1 where it raised Out_of_memory. Linux only. *)
let test_parted_memory (b : Support.backend) () =
  if not (Sys.file_exists "/proc/self/statm") then
    skip ~reason:"limits the address space through Linux's /proc" ();
  let module K = (val b.kernels) in
  let rows = 1 lsl 21 in
  let s = [| rows; 2 |] in
  let ps =
    Bigarray.Array1.init Bigarray.int64 Bigarray.c_layout (2 * rows) (fun k ->
        Int64.of_int (k * 7919 mod rows))
  in
  let idx =
    A.of_bigarray D.Int64 (Bigarray.reshape (Bigarray.genarray_of_array1 ps) s)
  in
  let into = f4_zeros s and updates = f4_zeros s in
  let dst = A.create Rig.host D.Float4_e2m1fn (Iarray.of_array s) in
  let run () =
    if not (Support.cap_memory parted_cap) then 3
    else
      match
        K.scatter Nx_kernel.Add ~unique:false ~axis:0 ~dst ~into idx updates
      with
      | A.Done -> 0
      | _ -> 2
      | exception Out_of_memory -> 1
  in
  match Unix.fork () with
  | 0 -> Unix._exit (try run () with _ -> 4)
  | pid -> (
      match snd (Unix.waitpid [] pid) with
      | Unix.WEXITED 3 -> skip ~reason:"no address-space limit" ()
      | Unix.WEXITED code ->
          equal string "done"
            (match code with
            | 0 -> "done"
            | 1 -> "Out_of_memory"
            | 2 -> "not done"
            | _ -> "raised")
      | Unix.WSIGNALED n | Unix.WSTOPPED n ->
          failf "the child ended by signal %d" n)

let gathers = Gen.with_pp pp_gather (gather_gen ())

let large_gathers = Gen.with_pp pp_gather (gather_gen ~shapes:large_shapes ())

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

(* A narrow float's Adds: an axis of more updates than targets, one of
   fewer, 3 slices of positions of their own, and rows of 600 slices, the
   positions broadcast along them. *)
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
      (* Positions crowding the first targets, and one outside the axis,
         at bfloat16 and at float4_e2m1fn, whose elements share bytes. *)
      scatter_example (D.Any D.Bfloat16) Nx_kernel.Add ~ts:[| 500; 3 |]
        ~axis:0 ~m:70_000
        (Array.init 1009 (fun i -> Int64.of_int ((i * i mod 38) - 1)));
      scatter_example (D.Any D.Float4_e2m1fn) Nx_kernel.Add ~ts:[| 500; 3 |]
        ~axis:0 ~m:70_000
        (Array.init 1009 (fun i -> Int64.of_int ((i * i mod 38) - 1)));
      (* 3 slices of 300 000 float4_e2m1fn updates into an axis of 1000. *)
      scatter_example (D.Any D.Float4_e2m1fn) Nx_kernel.Add ~ts:[| 1000; 3 |]
        ~axis:0 ~m:300_000 cycled;
    ]

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

(* A scatter's bits are one thread's, any NaN for a NaN. *)
let law_threads (b : Support.backend) c =
  let d = dtype_of c.into in
  cover "a narrow float's Add" (c.combine = Nx_kernel.Add && is_narrow d);
  cover "a sub-byte dtype" (let (D.Any dt) = d in D.bits dt < 8);
  cover "positions broadcast along the rows"
    (let (A.Any p) = c.sidx in
     Iarray.exists (( = ) 0) (L.strides (A.layout p)));
  cover "through views" (c.sviews <> []);
  cover "more targets than updates"
    ((shape_of c.into).(c.saxis) > (shape_of c.updates).(c.saxis));
  match (scatter_on b c, Nx_cpu.one_thread (fun () -> scatter_on b c)) with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y) (elements y')
  | _ -> failf "%s declined a scatter" b.name

(* Sub-byte scatters *)

let sub_byte_dtypes = List.filter (fun (D.Any dt) -> D.bits dt < 8) all_dtypes

(* Shapes whose neighbouring elements share bytes between threads: a short
   axis of many updates, rows of an odd count of slices, and slices along
   a middle axis. Each is [(into, axis, updates along it, broadcast axes of
   the positions)]. *)
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

(* Scatters of many updates: narrow floats' Adds, embedding gradients'
   rows, sub-byte elements sharing bytes, and long axes of few slices. *)
let threaded_scatters =
  Gen.with_pp pp_scatter
    (Gen.frequency
       [
         (3, narrow_adds); (3, row_scatters); (3, sub_byte_scatters);
         (1, long_scatters);
       ])

let threaded_examples =
  narrow_examples @ row_examples @ sub_byte_examples @ [ wide_rows ]

(* Scatters of many updates against the reference: large shapes, long axes
   of few slices, hundreds of thousands of updates into one axis, and
   embedding gradients' rows. *)
let many_scatters =
  Gen.with_pp pp_scatter
    (Gen.frequency
       [
         (3, large_scatters); (3, long_scatters); (1, blocked_scatters);
         (3, row_scatters);
       ])

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
    (same_guarded ~any_nan:true side ~call
       [| c.into; c.sidx; c.updates |]
       [| (dtype_of c.into, shape_of c.into) |])

(* The suite *)

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "a float Add gives the same result through any views" float_adds
        (run (law_add_views b));
      prop ~examples:gather_examples
        "a gather reads each position, zero outside the axis" gathers
        (run (law_gather_reference b));
      prop ~examples:scatter_examples
        "a scatter combines each target's updates in C order" scatters
        (run (law_scatter_reference b));
      prop "a scatter's Max and Min give a NaN where one reaches a target"
        nan_scatters
        (run (law_nan_extremes b));
      prop ~count:10 "large gathers on the job's threads" large_gathers
        (run (gather_agrees b));
      prop ~count:10 ~examples:blocked_examples
        "scatters of many updates combine each target's updates in C order"
        many_scatters
        (run (fun c ->
             cover "hundreds of thousands of updates into one axis"
               ((shape_of c.updates).(c.saxis) > 500_000);
             law_scatter_reference b c));
      prop ~count:16 ~examples:threaded_examples
        "a scatter's bits are one thread's" threaded_scatters
        (run (law_threads b));
      test "boolean extremes are of truths" (fun () ->
          b.around (test_bool_extremes b));
      test "a float16 sum runs in float32 and rounds once" (fun () ->
          b.around (test_float16_sum b));
      test "a parted float4_e2m1fn Add of 4Mi updates fits 512 MiB" (fun () ->
          b.around (test_parted_memory b));
      prop "a gather refuses what Nx_kernel.Shape refuses and other shapes"
        gather_misfits
        (run (law_gather_refuses b));
      prop "a scatter refuses what Nx_kernel.Shape refuses and other shapes"
        scatter_misfits
        (run (law_scatter_refuses b));
      test "refuses an Add of booleans" (fun () -> b.around (test_bool_add b));
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

let () = exit (Windtrap.run "nx_cpu.index" [ cpu Support.cpu ])
