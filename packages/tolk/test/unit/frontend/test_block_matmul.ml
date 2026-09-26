(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The block kernel: its values on the default device against a host
   reference, and its generated code on every renderer, where the loop over
   the contraction reads the block's id on a GPU and is constant on the CPU. *)

open Windtrap
open Tolk
module T = Tolk_frontend.Tensor
module Op = Tolk_frontend.Op
module Dt = Tolk_frontend.Dtype_ops
module Run = Tolk_frontend.Run
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module Ops = Tolk_uop.Ops

let rng = Random.State.make [| 11 |]
let floats n = Array.init n (fun _ -> Random.State.float rng 2.0 -. 1.0)

(* Values *)

(* [expected ~transpose ~m ~n ~k xs ws ids] is each block times its matrix at
   float64, and whether the block selects one. *)
let expected ~transpose ~m ~n ~k ~e xs ws ids =
  let nb = Array.length ids in
  Array.init (nb * m * n) (fun o ->
      let b = o / (m * n) and i = o / n mod m and j = o mod n in
      let id = ids.(b) in
      if id < 0 || id >= e then None
      else
        let acc = ref 0.0 and mag = ref 0.0 in
        for c = 0 to k - 1 do
          let w =
            if transpose then ws.((((id * n) + j) * k) + c)
            else ws.((((id * k) + c) * n) + j)
          in
          let p = xs.((((b * m) + i) * k) + c) *. w in
          acc := !acc +. p;
          mag := !mag +. Float.abs p
        done;
        Some (!acc, !mag))

(* [v] rounded to [dtype] on the host, to nearest with ties to even, to
   [bits] significant bits and at least the quantum [2^least]. Inputs rounded
   here are exact in [dtype], so the device's cast to it keeps them and the
   reference multiplies the values the kernel reads. *)
let rounded dtype v =
  let bits, least =
    if D.equal dtype D.float16 then (11, -24)
    else if D.equal dtype D.bfloat16 then (8, -133)
    else (24, -149)
  in
  let round x =
    if x = 0.0 then 0.0
    else
      let _, e = Float.frexp x in
      let q = Float.ldexp 1.0 (max (e - bits) least) in
      let r = x /. q in
      let f = Float.floor r in
      let up = r -. f > 0.5 || (r -. f = 0.5 && Float.rem f 2.0 <> 0.0) in
      (if up then f +. 1.0 else f) *. q
  in
  Array.map round v

(* The product at [dtype], whose unit roundoff is [u]: products are exact, the
   float32 sum rounds at each term, and the result rounds once to [dtype],
   which at float16 may land on a subnormal and lose up to half its quantum. A
   block that selects no matrix is exactly +0, whatever its rows hold. *)
let check ?(dtype = D.float32) ?(u = 0.0) ?(transpose = false) ?poison ~m ~n
    ~k ~e ids =
  let nb = Array.length ids in
  let round = rounded dtype in
  let tiny =
    if D.equal dtype D.float16 then Float.ldexp 1.0 (-25) else 0.0
  in
  let xs = round (floats (nb * m * k)) and ws = round (floats (e * n * k)) in
  let xs =
    match poison with
    | None -> xs
    | Some b ->
        Array.mapi
          (fun i v ->
            if i / (m * k) <> b then v
            else if i mod 2 = 0 then Float.nan
            else Float.infinity)
          xs
  in
  let x = Dt.cast (Run.of_float_array ~shape:[ nb; m; k ] xs) dtype in
  let w =
    Dt.cast
      (Run.of_float_array
         ~shape:(if transpose then [ e; n; k ] else [ e; k; n ])
         ws)
      dtype
  in
  let got =
    Run.to_float_array
      (Dt.cast
         (Op.block_matmul ~transpose x w
            ~ids:(Run.of_int_array ~shape:[ nb ] ids))
         D.float32)
  in
  let eps = Float.ldexp 1.0 (-24) in
  Array.iteri
    (fun o expect ->
      let a = got.(o) in
      match expect with
      | None ->
          if Int64.bits_of_float a <> 0L then
            failf "element %d: a block with no matrix gave %h" o a
      | Some (v, mag) ->
          let tol =
            (2.0 *. float_of_int k *. eps *. mag) +. (u *. Float.abs v) +. tiny
          in
          if not (Float.abs (a -. v) <= tol) then
            failf "element %d: expected %h, got %h" o v a)
    (expected ~transpose ~m ~n ~k ~e xs ws ids)

(* Bit for bit against float32 products summed at float32, on operands whose
   products need more bits than bfloat16 and float16 hold and whose sums
   float32 holds exactly: multiples of 2^-8 below 1, so a product is a multiple
   of 2^-16 below 1, and at most 256 of them sum to a multiple of 2^-16 below
   2^8. The result is then the exact sum rounded once, whatever order the
   device adds in, and a product rounded to [dtype] shows. *)
let exact ?(transpose = false) ~dtype ~m ~n ~k ~e ids =
  assert (k <= 256);
  let nb = Array.length ids in
  let draw count =
    Array.init count (fun _ ->
        float_of_int (Random.State.int rng 511 - 255) /. 256.0)
  in
  let xs = draw (nb * m * k) and ws = draw (e * n * k) in
  let x = Dt.cast (Run.of_float_array ~shape:[ nb; m; k ] xs) dtype in
  let w =
    Dt.cast
      (Run.of_float_array
         ~shape:(if transpose then [ e; n; k ] else [ e; k; n ])
         ws)
      dtype
  in
  let got =
    Run.to_float_array
      (Dt.cast
         (Op.block_matmul ~transpose x w
            ~ids:(Run.of_int_array ~shape:[ nb ] ids))
         D.float32)
  in
  Array.iteri
    (fun o expect ->
      let v = match expect with None -> 0.0 | Some (v, _) -> v in
      let want = (rounded dtype [| v |]).(0) in
      if Int64.bits_of_float got.(o) <> Int64.bits_of_float want then
        failf "%s, element %d: expected %h, got %h" (D.to_string dtype) o want
          got.(o))
    (expected ~transpose ~m ~n ~k ~e xs ws ids)

let bf16 = Float.ldexp 1.0 (-8)
let f16 = Float.ldexp 1.0 (-11)

let value_tests =
  group "values"
    [
      test "each block by its matrix" (fun () ->
          check ~m:3 ~n:5 ~k:12 ~e:3 [| 2; 0; 1; 2 |]);
      test "transposed" (fun () ->
          check ~transpose:true ~m:3 ~n:5 ~k:12 ~e:3 [| 1; 1; 0 |]);
      test "ids outside the matrices" (fun () ->
          check ~m:2 ~n:4 ~k:16 ~e:3 [| -1; 2; 3; 0; -7; 100 |]);
      test "a block with no matrix ignores its rows" (fun () ->
          check ~poison:1 ~m:2 ~n:4 ~k:16 ~e:3 [| 0; -1; 2 |];
          check ~poison:0 ~transpose:true ~m:8 ~n:16 ~k:32 ~e:2 [| 5; 1 |];
          check ~poison:1 ~m:2 ~n:64 ~k:8 ~e:3 [| 0; -1; 2 |];
          check ~poison:2 ~m:1 ~n:4 ~k:1 ~e:3 [| 0; 1; 3 |]);
      test "one block" (fun () ->
          check ~m:4 ~n:3 ~k:8 ~e:2 [| 1 |];
          (* The fewest tiles a bounded loop runs: two of 8. *)
          check ~m:8 ~n:8 ~k:16 ~e:2 [| 1 |];
          check ~poison:0 ~m:8 ~n:8 ~k:16 ~e:2 [| -1 |];
          check ~poison:0 ~transpose:true ~m:1 ~n:5 ~k:16 ~e:2 [| 2 |]);
      test "tensor-core tiles" (fun () ->
          let ids = [| 0; -1; 2; 2; 1; 3 |] in
          check ~m:16 ~n:24 ~k:32 ~e:3 ids;
          check ~transpose:true ~m:64 ~n:48 ~k:64 ~e:3 ids;
          check ~dtype:D.bfloat16 ~u:bf16 ~m:16 ~n:24 ~k:32 ~e:3 ids;
          check ~dtype:D.float16 ~u:f16 ~transpose:true ~m:8 ~n:48 ~k:32 ~e:3
            ids);
      test "narrow dtypes over few inputs" (fun () ->
          let ids = [| 1; -1; 0; 2 |] in
          List.iter
            (fun k ->
              check ~dtype:D.float16 ~u:f16 ~m:3 ~n:5 ~k ~e:3 ids;
              check ~dtype:D.bfloat16 ~u:bf16 ~transpose:true ~m:2 ~n:4 ~k
                ~e:3 ids)
            [ 1; 2; 7; 8; 9 ]);
      test "exact products" (fun () ->
          let ids = [| 0; -1; 2; 2; 1; 3 |] in
          List.iter
            (fun dtype ->
              exact ~dtype ~m:16 ~n:24 ~k:32 ~e:3 ids;
              exact ~dtype ~transpose:true ~m:64 ~n:48 ~k:256 ~e:3 ids;
              exact ~dtype ~m:3 ~n:5 ~k:12 ~e:3 ids;
              exact ~dtype ~transpose:true ~m:1 ~n:40 ~k:64 ~e:3 ids;
              exact ~dtype ~m:2 ~n:4 ~k:1 ~e:3 ids)
            [ D.bfloat16; D.float16 ]);
      test "shapes and dtypes" (fun () ->
          let raises msg f = raises (Invalid_argument msg) f in
          let x = Run.of_float_array ~shape:[ 2; 3; 4 ] (floats 24) in
          let w = Run.of_float_array ~shape:[ 2; 4; 5 ] (floats 40) in
          let ids = Run.of_int_array ~shape:[ 2 ] [| 0; 1 |] in
          raises "Op.block_matmul: w does not match x's inputs" (fun () ->
              ignore (Op.block_matmul ~transpose:true x w ~ids));
          raises "Op.block_matmul: ids must be [blocks]" (fun () ->
              ignore
                (Op.block_matmul x w
                   ~ids:(Run.of_int_array ~shape:[ 3 ] [| 0; 1; 0 |])));
          raises "Op.block_matmul: integer ids required" (fun () ->
              ignore
                (Op.block_matmul x w
                   ~ids:(Run.of_float_array ~shape:[ 2 ] [| 0.; 1. |])));
          List.iter
            (fun dtype ->
              raises "Op.block_matmul: x must be a float of at most 32 bits"
                (fun () ->
                  ignore
                    (Op.block_matmul (Dt.cast x dtype) (Dt.cast w dtype) ~ids)))
            [ D.float64; D.int32 ]);
    ]

(* Code *)

(* A device that renders for [ren] and runs nothing. *)
let render_only prefix ren =
  let allocator = Device.Allocator.Pack
      (Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  Device.register prefix (fun name ->
      Device.make ~name ~allocator
        ~renderer_set:
          (Device.Renderer_set.make ~device:prefix [ (prefix, Fun.const ren) ])
        ~runtime:(fun _ -> failwith "render only")
        ~synchronize:ignore ())

let renderers =
  [
    ("RCLANG", Cstyle.clang_no_abi Gpu_target.X86_64);
    ("RCUDA", Cstyle.cuda Gpu_target.SM80);
    ("RMETAL", Cstyle.metal (Gpu_target.Apple 7));
    ("ROPENCL", Cstyle.opencl "");
    ("RAMD", Cstyle.amd Gpu_target.RDNA3);
  ]

let () = List.iter (fun (prefix, ren) -> render_only prefix ren) renderers

let param ~device ~slot ~dtype shape =
  T.of_uop
    (U.param ~slot ~dtype
       ~shape:(U.stack (List.map U.const_int shape))
       ~device:(Single device) ())

(* The block kernel of [block_matmul] on [device], lowered for [ren]: its
   program and source. *)
let lowered ~device ren ~dtype ~nb ~m ~n ~k ~e =
  let x = param ~device ~slot:0 ~dtype [ nb; m; k ] in
  let w = param ~device ~slot:1 ~dtype [ e; n; k ] in
  let ids = param ~device ~slot:2 ~dtype:D.int32 [ nb ] in
  let y = Op.block_matmul ~transpose:true x w ~ids in
  let graph =
    Rangeify.get_kernel_graph (U.sink [ U.contiguous ~src:(T.uop y) () ])
  in
  let kernel =
    List.find_map
      (fun u ->
        match U.as_call u with
        | Some { body; _ } -> (
            match U.as_kernel_info body with
            | Some ki when String.starts_with ~prefix:"block_matmul" ki.name ->
                Some body
            | _ -> None)
        | None -> None)
      (U.toposort graph)
    |> Option.get
  in
  let sink = Codegen.full_rewrite_to_sink ren kernel in
  let program = Linearizer.linearize sink in
  let opts =
    match U.as_kernel_info sink with Some ki -> ki.applied_opts | None -> []
  in
  (program, Renderer.render ren ~name:"block" program, opts)

(* The sizes of the program's loops that are not constant, and whether each
   reads memory. *)
let loaded_bounds program =
  List.filter_map
    (fun u ->
      match U.as_range u with
      | Some { size; _ } when U.const_int_value size = None ->
          Some
            (List.exists (fun v -> U.op v = Ops.Load) (U.backward_slice size))
      | _ -> None)
    program

let contains s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

(* The index expressions of a load's address. *)
let rec address u =
  match U.op u with
  | Ops.Index -> List.tl (Array.to_list (U.src u))
  | Ops.Cast | Ops.Bitcast -> address (U.src u).(0)
  | _ -> [ u ]

let code_tests =
  let shapes =
    [
      ("tensor-core tiles", 16, 64, 48, 64);
      ("a row per block", 12, 1, 40, 24);
      ("one block", 1, 16, 32, 16);
      ("a contraction of 8", 6, 4, 16, 8);
      ("one input", 6, 4, 16, 1);
    ]
  in
  group "code"
    (List.concat_map
       (fun (prefix, ren) ->
         let gpu = Renderer.has_local ren in
         List.map
           (fun (shape, nb, m, n, k) ->
             test
               (Printf.sprintf "%s, %s" (Renderer.device ren) shape)
               (fun () ->
                 let program, _, _ =
                   lowered ~device:prefix ren ~dtype:D.float32 ~nb ~m ~n ~k
                     ~e:4
                 in
                 match (gpu && k > 1, loaded_bounds program) with
                 | true, [ true ] | false, [] -> ()
                 | _, bounds ->
                     failf "%d loop bounds that are not constant, %d loaded"
                       (List.length bounds)
                       (List.length (List.filter Fun.id bounds))))
           shapes)
       renderers
    @ [
        test "a lone block's loads at its id are gated" (fun () ->
            (* Where the kernel gates its loads rather than bounding its
               contraction by the id, which it reads at a fixed address. *)
            List.iter
              (fun (prefix, ren) ->
                List.iter
                  (fun k ->
                    if not (Renderer.has_local ren && k > 1) then
                      let program, _, _ =
                        lowered ~device:prefix ren ~dtype:D.float32 ~nb:1 ~m:4
                          ~n:16 ~k ~e:4
                      in
                      let ungated =
                        List.filter
                          (fun u ->
                            match U.as_load u with
                            | Some { src; gate = None; _ } ->
                                List.exists
                                  (fun i ->
                                    List.exists
                                      (fun v -> U.op v = Ops.Load)
                                      (U.backward_slice i))
                                  (address src)
                            | _ -> false)
                          program
                      in
                      equal
                        ~msg:(Printf.sprintf "%s, k = %d" prefix k)
                        int 0 (List.length ungated))
                  [ 1; 8 ])
              renderers);
        test "Metal takes its tensor cores" (fun () ->
            let ren = List.assoc "RMETAL" renderers in
            List.iter
              (fun dtype ->
                let _, src, _ =
                  lowered ~device:"RMETAL" ren ~dtype ~nb:65 ~m:64 ~n:2880
                    ~k:2880 ~e:32
                in
                is_true
                  ~msg:(D.to_string dtype)
                  (contains src "simdgroup_multiply_accumulate"))
              [ D.float32; D.float16; D.bfloat16 ]);
        test "the CPU tiles rows, columns and the contraction" (fun () ->
            let ren = List.assoc "RCLANG" renderers in
            List.iter
              (fun (dtype, m, expect) ->
                let _, _, opts =
                  lowered ~device:"RCLANG" ren ~dtype ~nb:4 ~m ~n:5760 ~k:2880
                    ~e:32
                in
                equal (list string)
                  ~msg:(Printf.sprintf "%s, %d rows" (D.to_string dtype) m)
                  expect
                  (List.map U.Opt.to_string opts))
              [
                ( D.float32,
                  8,
                  [
                    "SPLIT:1:8:upcast:false";
                    "SPLIT:1:8:upcast:false";
                    "SPLIT:4:4:unroll:false";
                  ] );
                ( D.bfloat16,
                  1,
                  [ "SPLIT:1:16:upcast:false"; "SPLIT:3:4:unroll:false" ] );
              ]);
      ])

let () = exit (run "block_matmul" [ value_tests; code_tests ])
