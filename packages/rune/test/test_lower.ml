(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The core of the lowering: dtypes, captures, movements, placements and reads.
   A traced value is checked by evaluating its graph with its captures bound
   (Traces.value) against the value nx gives eagerly. *)

open Windtrap
open Nx_test
open Traces
open Rune_internals
module Ops = Tolk_next.Ops
module Dtype = Tolk_next.Dtype
module Op = Tolk_next.Op

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let floats = viewed ~pp:pp_float Nx.float32 Gen.any_float
let arange n = Nx.create Nx.float32 [| n |] (Array.init n float_of_int)
let grid r c = Nx.reshape [| r; c |] (arange (r * c))

(* A traced copy of [x], and its value. *)
let copied x =
  let s, y = trace (fun () -> Nx.copy x) in
  (s, value s y)

(* Dtypes *)

let tolk_dtype = option (Testable.make ~pp:Dtype.pp ~equal:Dtype.equal)

let dtypes =
  group "dtypes"
    [
      test "floats, integers and booleans keep their width and signedness"
        (fun () ->
          let is dt expected =
            equal tolk_dtype (Some expected) (Lower.dtype dt)
          in
          is Nx.float16 Float16;
          is Nx.bfloat16 Bfloat16;
          is Nx.float32 Float32;
          is Nx.float64 Float64;
          is Nx.int8 Int8;
          is Nx.uint8 Uint8;
          is Nx.int16 Int16;
          is Nx.uint16 Uint16;
          is Nx.int32 Int32;
          is Nx.uint32 Uint32;
          is Nx.int64 Int64;
          is Nx.uint64 Uint64;
          is Nx.bool Bool);
      test "the 8-bit floats are the OCP formats" (fun () ->
          equal tolk_dtype (Some Dtype.Fp8e4m3) (Lower.dtype Nx.float8_e4m3);
          equal tolk_dtype (Some Dtype.Fp8e5m2) (Lower.dtype Nx.float8_e5m2));
      test "4-bit integers and complex numbers have no counterpart" (fun () ->
          is_none (Lower.dtype Nx.int4);
          is_none (Lower.dtype Nx.uint4);
          is_none (Lower.dtype Nx.complex64);
          is_none (Lower.dtype Nx.complex128));
      test "a trace reads a device's dtypes from its renderer once" (fun () ->
          let calls = ref 0 in
          let renderer d =
            incr calls;
            host d
          in
          let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          ignore (trace ~renderer (fun () -> Nx.mul (Nx.add x x) (Nx.neg x)));
          equal int 1 !calls);
    ]

(* Captures *)

(* A grid of [r] by [c] elements windowed along its rows, then along the columns
   of each window, each by a size and a step. *)
let windowed =
  let open Gen in
  let window n = pair (int_range 1 n) (int_range 1 3) in
  let* r, c = pair (int_range 1 6) (int_range 1 6) in
  let+ (w0, s0), (w1, s1) = pair (window r) (window c) in
  ( (r, c, w0, s0, w1, s1),
    Nx.sliding_window ~axis:1 ~window:w1 ~step:s1
      (Nx.sliding_window ~axis:0 ~window:w0 ~step:s0 (grid r c)) )

let bound s = List.length (Lower.captures s)

(* [n] floats over memory that starts 4 bytes past a 16-byte boundary, as a
   tensor mapped from a file may. *)
let past_boundary n =
  let ba = Bigarray.Array1.create Float32 C_layout (n + 4) in
  let at b =
    Nativeint.to_int (Nativeint.rem (Nx_device.Buffer.address b) 16n)
  in
  let k = (4 - at (Nx_device.Buffer.of_bigarray ba) + 16) mod 16 / 4 in
  let sub = Bigarray.Array1.sub ba k n in
  for i = 0 to n - 1 do
    sub.{i} <- float_of_int i
  done;
  Nx.of_bigarray (Bigarray.genarray_of_array1 sub)

(* The phase of the storage [u] reads, a parameter or a buffer. *)
let phase u =
  let storage v = Ops.op v = Op.Param || Ops.op v = Op.Buffer in
  match List.filter storage (Ops.toposort u) with
  | [ v ] -> (
      match Ops.arg v with Param p -> p.phase | _ -> fail "a parameter")
  | _ -> fail "one storage"

let captures =
  group "captures"
    [
      prop "a captured view computes its elements, whatever its strides" floats
        (fun x -> exact x (snd (copied x)));
      test "overlapping windows are read from their storage" (fun () ->
          let x = Nx.sliding_window ~window:3 ~step:1 (arange 6) in
          exact x (snd (copied x)));
      prop "windows of windows of any size and step are read from their storage"
        (Gen.map snd
           (Gen.with_pp
              (fun ppf ((r, c, w0, s0, w1, s1), _) ->
                Format.fprintf ppf
                  "%dx%d, windows of %d every %d, then of %d every %d" r c w0 s0
                  w1 s1)
              windowed))
        (fun x -> exact x (snd (copied x)));
      test "windows of windows are read from their storage" (fun () ->
          let x =
            Nx.sliding_window ~axis:1 ~window:2
              (Nx.sliding_window ~window:3 (grid 4 5))
          in
          exact x (snd (copied x)));
      test "a value that reaches one element is a constant, bound nowhere"
        (fun () ->
          let s, y =
            copied (Nx.broadcast_to [| 3; 2 |] (Nx.scalar Nx.float32 2.))
          in
          exact (Nx.full Nx.float32 [| 3; 2 |] 2.) y;
          equal int 0 (bound s));
      test "a single element of a larger storage is a constant" (fun () ->
          let s, y = copied (Nx.slice [ I 3 ] (arange 8)) in
          exact (Nx.scalar Nx.float32 3.) y;
          equal int 0 (bound s));
      test "a borrowed single element is storage, which its owner may change"
        (fun () ->
          let ba = Bigarray.Array1.of_array Float32 C_layout [| 7. |] in
          let x = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          let s, y = copied x in
          exact x y;
          equal int 1 (bound s));
      test "a capture met twice is bound once" (fun () ->
          let x = grid 2 3 in
          let s, _ = trace (fun () -> (Nx.copy x, Nx.copy x)) in
          equal int 1 (bound s));
      test "two views of one storage are bound apart" (fun () ->
          let x = grid 2 3 in
          let t = Nx.transpose x in
          let s, _ = trace (fun () -> (Nx.copy x, Nx.copy t)) in
          equal int 2 (bound s));
      test "storage is bound from the 16-byte boundary at or below its view"
        (fun () ->
          let s, _ = copied (Nx.slice [ R (5, 9) ] (arange 16)) in
          match Lower.captures s with
          | [ (u, [ b ]) ] ->
              equal int 5 (Nx_device.Buffer.length b);
              equal (list int) [ 5 ]
                (List.map
                   (function Ops.Int n -> n | Ops.Sym _ -> -1)
                   (Ops.shape u))
          | _ -> fail "one capture on one device");
      test "a capture of storage 4 bytes past a 16-byte boundary has phase 4"
        (fun () ->
          let x = past_boundary 12 in
          let s, y = copied x in
          exact x y;
          match Lower.captures s with
          | [ (u, _) ] -> equal int 4 (phase u)
          | _ -> fail "one capture");
      test "an empty view is bound nowhere" (fun () ->
          let s, y = copied (Nx.slice [ R (2, 2) ] (arange 4)) in
          equal (array int) [| 0 |] (Nx.shape y);
          equal int 0 (bound s));
    ]

(* Devices *)

let runtime name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let d1 = runtime "CPU:1"
let d2 = runtime "CPU:2"
let twin = runtime "CPU:1"
let names s = List.map fst (Lower.devices s)

(* Parameters *)

let parameter x =
  let s = scope () in
  let y = within s (fun () -> Nx.copy (argument s x)) in
  value s y

let parameters =
  group "parameters"
    [
      prop "a parameter views its argument's storage, whatever its strides"
        floats (fun x -> exact x (parameter x));
      test "a parameter of storage 4 bytes past a 16-byte boundary has phase 4"
        (fun () ->
          let x = past_boundary 12 in
          let s = scope () in
          let y = Lower.param s ~slot:0 x in
          equal int 4 (phase (Lower.uop y));
          exact x (parameter x));
      test "a parameter of a slice is where its run starts within 16 bytes"
        (fun () ->
          let x = Nx.slice [ R (5, 12) ] (past_boundary 12) in
          let s = scope () in
          equal int 4 (phase (Lower.uop (Lower.param s ~slot:0 x)));
          exact x (parameter x));
      test
        "a program over an argument 4 bytes past a 16-byte boundary computes \
         nx's values" (fun () ->
          let x = past_boundary 12 in
          let s = scope () in
          let y =
            within s (fun () ->
                let a = argument s x in
                Nx.add a (Nx.mul a a))
          in
          let phases =
            List.filter_map
              (fun u ->
                match (Ops.op u, Ops.arg u) with
                | Op.Param, Param p -> Some p.phase
                | _ -> None)
              (Ops.toposort (Programs.kernels y))
          in
          equal (list int) [ 0; 4 ] (List.sort compare phases);
          exact (Nx.add x (Nx.mul x x)) (Programs.compiled s y));
      test "a buffer on the disk has phase 0 (D54)" (fun () ->
          let path = Filename.temp_file "lower" ".bin" in
          let b = Result.get_ok (Nx_device.Buffer.create_file path 64) in
          Sys.remove path;
          equal int 0 (Lower.phase Dtype.Uint8 b 3));
      test "a parameter binds nothing when traced" (fun () ->
          let s = scope () in
          ignore (Lower.param s ~slot:0 (grid 2 3));
          equal int 0 (bound s));
      test "a parameter is at its argument's placement" (fun () ->
          let x = Nx.place (Nx.Placement.device d1) (grid 2 3) in
          let s = scope () in
          let y = Lower.param s ~slot:0 x in
          is_true (Nx.Placement.equal (Nx.placement x) (Nx.placement y));
          equal (list string) [ "CPU:1" ] (names s));
    ]

(* Movements *)

let movements =
  group "movements"
    [
      prop "a traced movement computes what the eager one gives"
        (Gen.pair layout floats) (fun (steps, x) ->
          let s, y = trace (fun () -> lay_out steps (Nx.copy x)) in
          exact (lay_out steps x) (value s y));
      test "windows along an inner axis take its place, their elements last"
        (fun () ->
          let x = grid 3 5 in
          let s, y =
            trace (fun () -> Nx.sliding_window ~axis:1 ~window:2 ~step:2 x)
          in
          exact (Nx.sliding_window ~axis:1 ~window:2 ~step:2 x) (value s y));
      test "a movement of a capture moves its storage, bound once" (fun () ->
          let x = grid 3 4 in
          let s, y = trace (fun () -> Nx.flip (Nx.transpose x)) in
          exact (Nx.flip (Nx.transpose x)) (value s y);
          equal int 1 (bound s));
    ]

(* Placements *)

let placements =
  group "placements"
    [
      test "a traced value placed on a device is copied there" (fun () ->
          let x = grid 2 3 in
          let s, y =
            trace (fun () -> Nx.place (Nx.Placement.device d1) (Nx.transpose x))
          in
          exact (Nx.transpose x) (value s y);
          is_true (Nx.Placement.equal (Nx.Placement.device d1) (Nx.placement y));
          equal (list string) [ "CPU"; "CPU:1" ] (names s));
      test "a traced value placed on devices is a copy on each" (fun () ->
          let p = Nx.Placement.replicated [ d1; d2 ] in
          let s, y = trace (fun () -> Nx.place p (Nx.flip (arange 4))) in
          exact (Nx.flip (arange 4)) (value s y);
          equal (list string) [ "CPU"; "CPU:1"; "CPU:2" ] (names s));
      test "a capture on another placement is placed once, then bound there"
        (fun () ->
          let p = Nx.Placement.device d1 and x = grid 2 2 in
          let s, y = trace (fun () -> Nx.place p x) in
          exact (grid 2 2) (value s y);
          match Lower.captures s with
          | [ (_, [ b ]) ] ->
              equal string "CPU:1" (Nx_device.name (Nx_device.Buffer.device b))
          | _ -> fail "one capture on one device");
      test "a placed capture is bound where it lies, without a copy" (fun () ->
          let x = Nx.place (Nx.Placement.device d1) (grid 2 3) in
          let s, y = copied x in
          exact (grid 2 3) y;
          equal (list string) [ "CPU:1" ] (names s));
      test "a traced value split over devices holds its slices" (fun () ->
          let p = Nx.Placement.sharded ~axis:0 [ d1; d2 ] and x = grid 4 3 in
          let s, y = trace (fun () -> Nx.place p (Nx.copy x)) in
          exact x (value s y);
          is_true (Nx.Placement.equal p (Nx.placement y)));
      test "a split capture is its shards, reassembled" (fun () ->
          let p = Nx.Placement.sharded ~axis:1 [ d1; d2 ] in
          let x = Nx.place p (grid 3 4) in
          let s, y = copied x in
          exact (grid 3 4) y;
          equal int 1 (bound s));
      test "two devices of one name cannot meet in one trace" (fun () ->
          let a = Nx.place (Nx.Placement.device d1) (grid 2 2) in
          let b = Nx.place (Nx.Placement.device twin) (grid 2 2) in
          raises_match (Exn.invalid_arg ~substring:"CPU:1") (fun () ->
              trace (fun () -> (Nx.copy a, Nx.copy b))));
    ]

(* Reads *)

let reads =
  group "reads"
    [
      test "a capture's elements are read eagerly" (fun () ->
          let x = grid 2 2 in
          let _, v = trace (fun () -> Nx.to_array x) in
          equal (array float_exact) [| 0.; 1.; 2.; 3. |] v);
      test "a traced value's elements cannot be read" (fun () ->
          raises
            (Lower.Jit_error
               "Nx.to_array: cannot read the value of a traced tensor inside \
                jit; return it from the compiled function instead") (fun () ->
              trace (fun () -> Nx.to_array (Nx.copy (grid 2 2)))));
      cases ~name:fst "a refused read names the function the program called"
        [
          ("Nx.item", fun x -> ignore (Nx.item [ 0; 0 ] x));
          ( "Nx.compress",
            fun x ->
              ignore
                (Nx.compress ~axis:0
                   ~condition:(Nx.create Nx.bool [| 2 |] [| true; false |])
                   x) );
          ( "Nx.positions",
            fun x -> ignore (Nx.positions (Nx.flatten (Nx.less_s x 1.))) );
          ("Nx.unique", fun x -> ignore (Nx.unique x));
          ("Nx.print", Nx.print);
        ]
        (fun (name, f) ->
          raises_match
            (function
              | Lower.Jit_error m -> String.starts_with ~prefix:(name ^ ": ") m
              | _ -> false)
            (fun () -> trace (fun () -> f (Nx.copy (grid 2 2)))));
    ]

(* Staged scans

   A step's part that varies with no trip is computed once, before the loop;
   what it reads through a movement, the loop reads in place. *)

(* The sizes of the buffers each kernel of the schedule storing [y] reads or
   writes, a list per kernel, the staged loop's kernels included. *)
let kernel_buffers y =
  let n = Nx.numel y in
  let out =
    Ops.new_buffer (Single "CPU") n (Option.get (Lower.dtype (Nx.dtype y)))
  in
  let view = Ops.reshape out [ Ops.Int n ] in
  let linear, _ =
    Tolk_next.Schedule.create_linear_with_vars
      (Ops.sink
         [
           Ops.after view
             [ Ops.store view (Ops.reshape (Lower.uop y) [ Ops.Int n ]) ];
         ])
  in
  let rec bodies u =
    if Ops.op u = Op.Call then [ Ops.body u ]
    else List.concat_map bodies (Ops.src u)
  in
  List.map
    (fun k ->
      List.filter_map
        (fun u -> if Ops.op u = Op.Param then Some (Ops.max_numel u) else None)
        (Ops.toposort k))
    (List.concat_map bodies (Ops.src linear))

let staged_scans =
  group "staged scans"
    [
      test
        "a step reads its weight in place and its bias as a vector, through \
         the product's transpose and the bias's broadcast" (fun () ->
          let s = scope () in
          let w = argument s (grid 32 32)
          and bias = argument s (arange 32)
          and xs = argument s (Nx.reshape [| 5; 8; 32 |] (arange 1280)) in
          let c, _ =
            Staged.install s (fun () ->
                Rune.scan'
                  ~f:(fun c x ->
                    let c = Nx.add (Nx.add (Nx.matmul (Nx.tanh c) w) bias) x in
                    (c, c))
                  ~init:(Nx.zeros Nx.float32 [| 8; 32 |])
                  xs)
          in
          let reading m = List.filter (List.mem m) (kernel_buffers c) in
          equal ~msg:"kernels reading the weight's 1024 elements" int 1
            (List.length (reading 1024));
          is_true ~msg:"the step reads the bias's 32 elements"
            (List.exists (List.mem 32) (reading 1024)));
    ]

(* Refusals *)

let metal _ =
  match Tolk_next.Device.renderer "METAL" with
  | Ok r -> r
  | Error e -> failwith e

let refusals =
  group "refusals"
    [
      test "a dtype the device's renderer lacks is refused, naming the device"
        (fun () ->
          raises
            (Lower.Jit_error
               "cannot compile contiguous: float64 is not supported on CPU")
            (fun () ->
              trace ~renderer:metal (fun () ->
                  Nx.copy (Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]))));
      test "an 8-bit float the host's renderer lacks is traced, tolk emulating it"
        (fun () ->
          let x =
            Nx.create Nx.float8_e4m3 [| 2; 2 |] [| 1.; 2.; 3.; -1. |]
          in
          let s, y = trace (fun () -> Nx.matmul x x) in
          Traces.exact (Nx.matmul x x) (Traces.value s y));
      test "a dtype with no counterpart is refused" (fun () ->
          raises
            (Lower.Jit_error "cannot compile contiguous: int4 is not supported")
            (fun () ->
              trace (fun () -> Nx.copy (Nx.create Nx.int4 [| 2 |] [| 1; 2 |]))));
      test "a Fourier transform is refused" (fun () ->
          let z = Nx.cast Nx.complex64 (arange 4) in
          raises (Lower.Jit_error "cannot compile fft") (fun () ->
              trace (fun () -> Nx.fft z)));
    ]

let () =
  exit
  @@ run "lower"
       [
         dtypes;
         captures;
         parameters;
         movements;
         placements;
         reads;
         staged_scans;
         refusals;
       ]
