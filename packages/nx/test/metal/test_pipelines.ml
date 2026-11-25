(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's pipelines and kept plans, skipping without a Metal GPU: a call's
   first launch on a device waits for the pipelines of its own launches, and of
   a kernel it weighs against them, and for no other's; a device keeps the plans
   of maps of distinct shapes. Each test opens GPU 0 afresh, since a device
   keeps a pipeline and a plan once made, and the process opens it for nothing
   else. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module K = Nx_metal.Kernels
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

(* [f d] on GPU 0 opened afresh as [d], closed after. *)
let fresh f =
  if Sys.getenv_opt "NX_KERNELS" = Some "cpu" then
    skip ~reason:"NX_KERNELS=cpu" ();
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error _ -> skip ~reason:"no Metal GPU" ()
  | Ok d ->
      Fun.protect
        ~finally:(fun () -> Rig.close d)
        (fun () ->
          if not (Nx_metal.computes_on d) then
            skip ~reason:"no Metal GPU nx.metal computes on" ();
          f d)

let names ks = List.sort_uniq compare (List.map K.name ks)
let made d = names (List.filter (Nx_metal.resolved d) (Array.to_list K.all))

(* The kernels that [call d] launched on a fresh device [d], after [setup d] has
   placed its operands, and those it waited for. *)
let first setup call =
  fresh @@ fun d ->
  let x = setup d in
  let (), traced = Nx_metal.Plan.traced (fun () -> call d x) in
  (names (List.map (fun (t : Nx_metal.Plan.traced) -> t.kernel) traced), made d)

(* A float32 array of [shape] on [d], its elements drawn in [-1, 1]. *)
let drawn d shape =
  let n = List.fold_left ( * ) 1 shape in
  let st = Random.State.make [| n |] in
  A.to_device d
    (A.of_array D.Float32 (Iarray.of_list shape)
       (Array.init n (fun _ -> Random.State.float st 2. -. 1.)))

let transposed x = Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) x)

(* [p] of float32 [loads] into a float32 destination of [shape]. *)
let mapped p shape loads d =
  let z = A.create d D.Float32 (Iarray.of_list shape) in
  equal answer A.Done
    (Nx_metal.map p ~dsts:[| A.Any z |]
       (Iarray.of_list (List.map (fun x -> Nx_kernel.Plain x) loads)))

let add =
  P.v
    ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
    [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
    ~outs:[| 2 |]

(* A cast of a float32 square of 1024 into float16, as it lies or transposed. *)
let cast ~transposed:t () =
  first
    (fun d ->
      let x = drawn d [ 1024; 1024 ] in
      if t then transposed x else x)
    (fun d x ->
      let z = A.create d D.Float16 (Iarray.of_list [ 1024; 1024 ]) in
      equal answer A.Done (Support.map1 (module Nx_metal) P.Cast ~dst:z x))

let test_cast () =
  let launched, waited = cast ~transposed:false () in
  expect (String.concat " " launched) @@ __POS_OF__ {|loop_cast_f32_16|};
  equal (list string) launched waited

(* A transposed cast weighs its cast's kernel against a map's tiles, and the
   map's tiles against its flat kernel. *)
let test_cast_transposed () =
  let launched, waited = cast ~transposed:true () in
  expect (String.concat " " launched) @@ __POS_OF__ {|loop_map_tiled|};
  equal (list string)
    (names [ K.Loop_cast_f32_16; Loop_map_v4; Loop_map_tiled ])
    waited

(* An add of 1024 float32 elements runs its direct kernel flat. *)
let test_direct_flat () =
  let launched, waited =
    first
      (fun d -> (drawn d [ 1024 ], drawn d [ 1024 ]))
      (fun d (x, y) -> mapped add [ 1024 ] [ x; y ] d)
  in
  equal (list string) launched waited

(* An add of a transposed float32 square of 1024 weighs its direct kernel's
   tiles of 32-bit lanes against its flat kernel, which reads the transposed
   load an element a thread. *)
let test_direct_strided () =
  let _, waited =
    first
      (fun d -> (transposed (drawn d [ 1024; 1024 ]), drawn d [ 1024; 1024 ]))
      (fun d (x, y) -> mapped add [ 1024; 1024 ] [ x; y ] d)
  in
  equal (list string)
    (names [ K.Loop_map_direct; Loop_map_direct32_tiled ])
    waited

(* (x + y) x over float32 squares of 1024, which no direct kernel takes. *)
let test_map () =
  let p =
    P.v
      ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
      [| P.In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 0) |]
      ~outs:[| 3 |]
  in
  let launched, waited =
    first
      (fun d -> (drawn d [ 1024; 1024 ], drawn d [ 1024; 1024 ]))
      (fun d (x, y) -> mapped p [ 1024; 1024 ] [ x; y ] d)
  in
  expect (String.concat " " launched) @@ __POS_OF__ {|loop_map_v4|};
  equal (list string) launched waited

(* A sum along the rows of a float32 square of 1024. *)
let test_row_sum () =
  let id = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let launched, waited =
    first
      (fun d -> drawn d [ 1024; 1024 ])
      (fun d x ->
        let z = A.create d D.Float32 (Iarray.of_list [ 1024 ]) in
        equal answer A.Done
          (Nx_metal.reduce id
             ~shape:(Iarray.of_list [ 1024; 1024 ])
             ~axes:(Iarray.of_list [ 1 ])
             [| (Monoid Sum, 0) |]
             ~dsts:[| A.Any z |] [| Nx_kernel.Plain x |]))
  in
  expect (String.concat " " launched) @@ __POS_OF__ {|loop_row_sum_f32|};
  equal (list string) launched waited

let tiles =
  List.filter
    (fun k ->
      k <> K.Contract_finish
      && String.starts_with ~prefix:"contract_" (K.name k))
    (Array.to_list K.all)

(* [Nx_metal.contract] on [d] of float32 a m x k by b k x n, and the tile
   kernels it launched. *)
let contracted d m n k =
  let operand shape =
    let x = A.create d D.Float32 (Iarray.of_list shape) in
    { Nx_kernel.array = A.Any x; scale = None }
  in
  let dst = A.create d D.Float32 (Iarray.of_list [ m; n ]) in
  let got, launches =
    Nx_metal.Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list [])
      ~contracting:(Iarray.of_list [ (1, 0) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
      (operand [ m; k ])
      (operand [ k; n ])
  in
  equal answer A.Done got;
  List.filter_map
    (fun (t : Nx_metal.Plan.traced) ->
      if List.mem t.kernel tiles then Some (K.name t.kernel) else None)
    launches

let made_tiles d = List.map K.name (List.filter (Nx_metal.resolved d) tiles)

(* A contraction without terms makes no tile's pipeline. *)
let test_contract () =
  fresh @@ fun d ->
  equal (list string) [] (contracted d 64 64 0);
  equal (list string) [] (made_tiles d);
  let launched = contracted d 64 64 64 in
  equal (list string) launched (made_tiles d)

(* nx_metal.mli: a device keeps maps' plans up to 2 MiB, in groups by a hash of
   every fact of their operands. Copies of one program over 1000 uint8 shapes of
   1 to 25 rows of 1 to 40 elements, destination and load of one shape: about
   0.5 MiB of plans, which 256 groups hold whole as long as their shapes spread
   over the groups, so each copy keeps its plan. *)
let kept_rows = 25
let kept_cols = 40

let test_kept_shapes () =
  fresh @@ fun d ->
  let copy = P.v ~ins:[| D.Any D.Uint8 |] [| P.In 0 |] ~outs:[| 0 |] in
  let bytes = kept_rows * kept_cols in
  let x = Rig.Buffer.create d bytes and y = Rig.Buffer.create d bytes in
  let at b r c =
    let l = Nx_array.Layout.contiguous (Iarray.of_list [ r; c ]) in
    A.v D.Uint8 l (Rig.Buffer.view b ~first:0 ~length:(r * c))
  in
  let before = Nx_metal.kept_plans d in
  for r = 1 to kept_rows do
    for c = 1 to kept_cols do
      equal answer A.Done
        (Nx_metal.map copy
           ~dsts:[| A.Any (at y r c) |]
           [| Nx_kernel.Plain (at x r c) |])
    done
  done;
  equal int (kept_rows * kept_cols) (Nx_metal.kept_plans d - before)

let () =
  exit
    (run "nx_metal.pipelines"
       [
         test "a contraction makes its own launches' pipelines alone"
           test_contract;
         test "a cast waits for its own kernel alone" test_cast;
         test "a transposed cast waits for the kernels it weighs"
           test_cast_transposed;
         test "a flat direct map waits for its own kernel alone"
           test_direct_flat;
         test "a strided direct map waits for the kernels it weighs"
           test_direct_strided;
         test "a map waits for its own kernel alone" test_map;
         test "a sum along rows waits for its own kernel alone" test_row_sum;
         test "a fresh device keeps a copy's plan for each of 1000 shapes"
           test_kept_shapes;
       ])
