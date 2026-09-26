(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [Op.quant_matmul] against a host reference on the default device ([DEV]: run
   it with [DEV=CPU] and [DEV=METAL]), and the code each renderer emits for it:
   on a GPU, a position whose id selects no matrix runs no multiply. *)

open Windtrap
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module Ops = Tolk_uop.Ops
module T = Tolk_frontend.Tensor
module Op = Tolk_frontend.Op
module Run = Tolk_frontend.Run
module Creation = Tolk_frontend.Creation
module Dtype_ops = Tolk_frontend.Dtype_ops

let rng = Random.State.make [| 11 |]

(* Values *)

let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |]

let code_value c =
  let v = e2m1.(c land 7) in
  if c land 8 = 0 then v else -.v

let bf16_bits v =
  Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float v) 16)
  land 0xffff

(* Inputs are values every dtype holds exactly: multiples of 2^-6 in [-1, 1]. *)
let input () = float_of_int (Random.State.int rng 129 - 64) /. 64.0

let f16_bits v =
  if v = 0.0 then 0
  else
    let s = if v < 0.0 then 0x8000 else 0 in
    let fr, ex = Float.frexp (Float.abs v) in
    s lor ((ex + 14) lsl 10) lor int_of_float (((fr *. 2.0) -. 1.0) *. 1024.0)

let encode dtype xs =
  let item = D.itemsize dtype in
  let b = Bytes.create (item * Array.length xs) in
  Array.iteri
    (fun i v ->
      match dtype with
      | D.Float32 -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v)
      | D.Bfloat16 -> Bytes.set_uint16_le b (2 * i) (bf16_bits v)
      | _ -> Bytes.set_uint16_le b (2 * i) (f16_bits v))
    xs;
  b

(* A case *)

type case = {
  ix : int;
  m : int;
  n : int;
  k : int;
  e : int;
  ids : int array option;
  codes : Bytes.t;
  scales : Bytes.t;
  x : float array;
}

let make ?ids ?(scale = fun () -> 100 + Random.State.int rng 51) ?(poison = [])
    ?(input = input) ~ix ~m ~n ~k ~e () =
  let x = Array.init (ix * m * k) (fun _ -> input ()) in
  List.iter
    (fun t ->
      Array.fill x (t * m * k) (m * k) Float.nan;
      x.(t * m * k) <- Float.infinity)
    poison;
  {
    ix;
    m;
    n;
    k;
    e;
    ids;
    codes =
      Bytes.init (e * n * k / 2) (fun _ -> Char.chr (Random.State.int rng 256));
    scales = Bytes.init (e * n * k / 32) (fun _ -> Char.chr (scale ()));
    x;
  }

let instances c = match c.ids with Some ids -> Array.length ids | None -> c.e

(* The product at float64, and the sum of its terms' magnitudes; [None] where no
   matrix is selected. *)
let reference c t r col =
  let matrix = match c.ids with Some ids -> ids.(t) | None -> t in
  if matrix < 0 || matrix >= c.e then None
  else begin
    let groups = c.k / 32 and rep = instances c / c.ix in
    let line = (matrix * c.n) + col in
    let sum = ref 0.0 and mag = ref 0.0 in
    for j = 0 to c.k - 1 do
      let byte = Char.code (Bytes.get c.codes ((line * c.k / 2) + (j / 2))) in
      let s = Char.code (Bytes.get c.scales ((line * groups) + (j / 32))) in
      let w =
        code_value (if j land 1 = 0 then byte land 15 else byte lsr 4)
        *. if s = 255 then Float.nan else Float.ldexp 1.0 (s - 127)
      in
      let p = c.x.((((t / rep * c.m) + r) * c.k) + j) *. w in
      sum := !sum +. p;
      mag := !mag +. Float.abs p
    done;
    Some (!sum, !mag)
  end

let product dtype c =
  let x = Run.of_bytes ~dtype ~shape:[ c.ix; c.m; c.k ] (encode dtype c.x) in
  let codes =
    Run.of_bytes ~dtype:D.uint8 ~shape:[ c.e; c.n; c.k / 2 ] c.codes
  in
  let scales =
    Run.of_bytes ~dtype:D.uint8 ~shape:[ c.e; c.n; c.k / 32 ] c.scales
  in
  let ids =
    Option.map
      (fun ids -> Run.of_int_array ~shape:[ Array.length ids ] ids)
      c.ids
  in
  Run.to_float_array
    (Dtype_ops.cast (Op.quant_matmul ?ids x ~codes ~scales) D.float32)

(* Within a float32 sum of [k] terms rounded once to the dtype of unit roundoff
   [u] and least half-ulp [tiny], plus what a device that flushes subnormal
   float32 loses; NaN exactly where the reference is; exact positive zeros where
   no matrix is selected. *)
let check ~device (name, dtype, u, tiny) c =
  let got = product dtype c in
  let flush = device = "METAL" in
  let largest =
    Array.fold_left
      (fun a v -> if Float.is_finite v then Float.max a (Float.abs v) else a)
      0.0 c.x
  in
  let k = float_of_int c.k in
  for t = 0 to instances c - 1 do
    for r = 0 to c.m - 1 do
      for col = 0 to c.n - 1 do
        let a = got.((((t * c.m) + r) * c.n) + col) in
        let wrong expected =
          fail
            (Printf.sprintf
               "%s, %s: instance %d row %d column %d: expected %s, got %h"
               device name t r col expected a)
        in
        match reference c t r col with
        | None -> if Int64.bits_of_float a <> 0L then wrong "+0"
        | Some (e, mag) ->
            if Float.is_nan e || Float.is_nan a then
              begin if not (Float.is_nan e && Float.is_nan a) then
                wrong (Printf.sprintf "%h" e)
              end
            else
              let tol =
                (2.0 *. k *. Float.ldexp 1.0 (-24) *. mag)
                +. (u *. Float.abs e)
                +. tiny
                +.
                if flush then
                  k *. Float.ldexp 1.0 (-126) *. ((3.0 *. largest) +. 2.0)
                else 0.0
              in
              if Float.abs (a -. e) > tol then wrong (Printf.sprintf "%h" e)
      done
    done
  done

let devices = [ Tolk.Device.canonicalize (Run.device_name ()) ]

let dtypes =
  [
    ("float32", D.float32, 0.0, 0.0);
    ("bfloat16", D.bfloat16, Float.ldexp 1.0 (-8), 0.0);
    ("float16", D.float16, Float.ldexp 1.0 (-11), Float.ldexp 1.0 (-25));
  ]

let on_every c () =
  List.iter
    (fun device -> List.iter (fun dt -> check ~device dt c) dtypes)
    devices

(* Scales that keep float16 results in range. *)
let small () = 110 + Random.State.int rng 21

let product_tests =
  group "product"
    [
      slow "one row"
        (on_every (make ~scale:small ~ix:1 ~m:1 ~n:48 ~k:64 ~e:1 ()));
      slow "rows no tile divides"
        (on_every (make ~scale:small ~ix:1 ~m:6 ~n:40 ~k:96 ~e:1 ()));
      slow "sixteen rows"
        (on_every (make ~scale:small ~ix:1 ~m:16 ~n:64 ~k:288 ~e:1 ()));
      slow "a matrix per instance"
        (on_every (make ~scale:small ~ix:3 ~m:2 ~n:40 ~k:64 ~e:3 ()));
      slow "x shared by the instances"
        (on_every (make ~scale:small ~ix:1 ~m:2 ~n:40 ~k:64 ~e:3 ()));
      slow "ids"
        (on_every
           (make ~scale:small ~ids:[| 2; -1; 5; 2; 0; 7 |] ~poison:[ 1; 5 ]
              ~ix:6 ~m:1 ~n:32 ~k:128 ~e:5 ()));
      slow "ids, x per group of instances"
        (on_every
           (make ~scale:small
              ~ids:[| 2; -1; 5; 2; 0; 7; 1; 1 |]
              ~ix:2 ~m:1 ~n:32 ~k:128 ~e:5 ()));
      slow "ids, one group"
        (on_every
           (make ~scale:small ~ids:[| 1; -2; 0 |] ~poison:[ 1 ] ~ix:3 ~m:2 ~n:48
              ~k:32 ~e:2 ()));
      slow "no id selects a matrix"
        (on_every
           (make ~ids:[| -1; 3 |] ~poison:[ 0; 1 ] ~ix:2 ~m:3 ~n:32 ~k:64 ~e:3
              ()));
      slow "a stack against one shared row"
        (on_every (make ~scale:small ~ix:1 ~m:1 ~n:8 ~k:32 ~e:2 ()));
      slow "a stack against one shared row, long rows"
        (on_every (make ~scale:small ~ix:1 ~m:1 ~n:16 ~k:2880 ~e:5 ()));
      slow "a stack against a row per pair of matrices"
        (on_every (make ~scale:small ~ix:2 ~m:1 ~n:4 ~k:2880 ~e:4 ()));
      slow "scale bytes 0, 1 and 255"
        (on_every
           (make
              ~scale:(fun () ->
                match Random.State.int rng 4 with
                | 0 -> 0
                | 1 -> 1
                | 2 -> 255
                | _ -> small ())
              ~ids:[| 1; 0 |] ~ix:2 ~m:1 ~n:48 ~k:64 ~e:2 ()));
    ]

(* [v] rounded to [dtype], to nearest with ties to even. *)
let nearest dtype v =
  let bits, least =
    match dtype with
    | D.Float16 -> (11, -24)
    | D.Bfloat16 -> (8, -133)
    | _ -> (24, -149)
  in
  if v = 0.0 || not (Float.is_finite v) then v
  else
    let _, e = Float.frexp v in
    let q = Float.ldexp 1.0 (max (e - bits) least) in
    let r = v /. q in
    let f = Float.floor r in
    let up = r -. f > 0.5 || (r -. f = 0.5 && Float.rem f 2.0 <> 0.0) in
    (if up then f +. 1.0 else f) *. q

(* Bit for bit against float32 products summed at float32, on inputs whose
   products with a code need more bits than bfloat16 holds and whose sums
   float32 holds exactly: multiples of 2^-8 below 1 and scales 2^-1 to 2^1, so
   a term is a multiple of 2^-10 below 12, and at most 1024 of them sum to a
   multiple of 2^-10 below 2^14. The result is then the exact sum rounded once,
   whatever order the device adds in. *)
let exact_products c () =
  List.iter
    (fun (name, dtype, _, _) ->
      let got = product dtype c in
      for t = 0 to instances c - 1 do
        for r = 0 to c.m - 1 do
          for col = 0 to c.n - 1 do
            let o = (((t * c.m) + r) * c.n) + col in
            let want =
              match reference c t r col with
              | None -> 0.0
              | Some (v, _) -> nearest dtype v
            in
            if Int64.bits_of_float got.(o) <> Int64.bits_of_float want then
              fail
                (Printf.sprintf
                   "%s: instance %d row %d column %d: expected %h, got %h" name
                   t r col want got.(o))
          done
        done
      done)
    dtypes

let exact ?ids ~ix ~m ~n ~k ~e () =
  assert (k <= 1024);
  exact_products
    (make ?ids
       ~scale:(fun () -> 126 + Random.State.int rng 3)
       ~input:(fun () -> float_of_int (Random.State.int rng 511 - 255) /. 256.0)
       ~ix ~m ~n ~k ~e ())

let exact_tests =
  group "exact products"
    [
      slow "one row" (exact ~ix:1 ~m:1 ~n:48 ~k:1024 ~e:1 ());
      slow "a tile of rows" (exact ~ix:1 ~m:8 ~n:40 ~k:96 ~e:1 ());
      slow "ids" (exact ~ids:[| 2; -1; 0; 2 |] ~ix:4 ~m:1 ~n:32 ~k:128 ~e:3 ());
      slow "ids, one group"
        (exact ~ids:[| 1; 0 |] ~ix:2 ~m:3 ~n:48 ~k:32 ~e:2 ());
    ]

(* The largest scales, 253 and 254 included, on float32 inputs small enough that
   no sum overflows. *)
let large_scales () =
  let c =
    make
      ~scale:(fun () -> 240 + Random.State.int rng 15)
      ~ix:2 ~m:1 ~n:16 ~k:64 ~e:2 ()
  in
  let c = { c with x = Array.map (fun v -> v *. Float.ldexp 1.0 (-40)) c.x } in
  List.iter
    (fun device -> check ~device ("float32", D.float32, 0.0, 0.0) c)
    devices

(* Random shapes, ids and dtypes against the reference: rows up to 9, one to
   ninety groups, ids with duplicates, -1 and ids past the matrices, and x
   shared by any divisor of the instances. *)
let random_products () =
  let pick l = List.nth l (Random.State.int rng (List.length l)) in
  for _ = 1 to 12 do
    let k = pick [ 32; 64; 96; 128; 2880 ] and m = 1 + Random.State.int rng 9 in
    let n = pick [ 4; 8; 16; 40; 48 ] and e = 1 + Random.State.int rng 5 in
    let ids =
      if Random.State.bool rng then None
      else
        Some
          (Array.init
             (1 + Random.State.int rng 6)
             (fun _ -> Random.State.int rng (e + 3) - 1))
    in
    let i = match ids with Some a -> Array.length a | None -> e in
    let ix = pick (List.filter (fun d -> i mod d = 0) (List.init i succ)) in
    on_every (make ~scale:small ?ids ~ix ~m ~n ~k ~e ()) ()
  done

(* No inputs: a sum of no terms. Only the nonempty output needs storage. *)
let no_inputs () =
  List.iter
    (fun device ->
      check ~device ("float32", D.float32, 0.0, 0.0)
        (make ~ix:2 ~m:3 ~n:8 ~k:0 ~e:2 ()))
    devices

(* Code per renderer *)

let renderer_device name ren =
  let allocator = Tolk.Device.Allocator.Pack
      (Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  Tolk.Device.register name (fun canonical ->
      Tolk.Device.make ~name:canonical ~allocator
        ~renderer_set:
          (Tolk.Device.Renderer_set.make ~device:name [ (name, Fun.const ren) ])
        ~runtime:(fun _ ->
          {
            Tolk.Device.call =
              (fun _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ -> None);
            handle = 0n;
            free = (fun () -> ());
          })
        ~synchronize:(fun timeout -> ignore timeout; ())
        ())

let renderers =
  [
    ("metal", Tolk.Cstyle.metal (Tolk.Gpu_target.Apple 7), true);
    ("cuda", Tolk.Cstyle.cuda Tolk.Gpu_target.SM80, true);
    ("amd", Tolk.Cstyle.amd Tolk.Gpu_target.RDNA3, true);
    ("opencl", Tolk.Cstyle.opencl "", true);
    ("clang", Tolk.Cstyle.clang Tolk.Gpu_target.Arm64, false);
  ]

let () =
  List.iter
    (fun (name, ren, _) ->
      renderer_device ("QMM" ^ String.uppercase_ascii name) ren)
    renderers

(* The linearized kernel of a product with [instances] ids (default 4) over 32
   matrices of 64 columns, [m] rows of [k] inputs per instance, on [ren]. *)
let program ?(instances = 4) name ren ~m ~k =
  let device = U.Single ("QMM" ^ String.uppercase_ascii name) in
  let empty dtype shape = Creation.empty ~dtype ~device shape in
  let y =
    Op.quant_matmul
      ~ids:(empty D.int32 [ instances ])
      (empty D.bfloat16 [ instances; m; k ])
      ~codes:(empty D.uint8 [ 32; 64; k / 2 ])
      ~scales:(empty D.uint8 [ 32; 64; k / 32 ])
  in
  let sink = U.sink [ U.contiguous ~src:(T.uop y) () ] in
  let kernel =
    List.find_map
      (fun node ->
        match (U.op node, U.as_call node) with
        | Ops.Call, Some { body; _ } when U.as_kernel_info body <> None ->
            Some body
        | _ -> None)
      (U.toposort (Tolk.Rangeify.get_kernel_graph sink))
  in
  Tolk.Linearizer.linearize
    (Tolk.Codegen.full_rewrite_to_sink ren (Option.get kernel))

let reads_memory u =
  List.exists (fun n -> U.op n = Ops.Load) (U.backward_slice u)

(* On a GPU, one loop's bound reads the position's id, and every float multiply
   lies inside it: at gpt-oss's rows of 2880 inputs, one row or a tile of 8.
   With a single group per row (32 inputs) the kernel clamps the id instead,
   and no loop's bound reads memory, as on the CPU: this row fails when the
   kernel's two-iteration cap is removed, which is right once a reduce over a
   possibly empty loop stops being rewritten to its body times the loop's
   size. *)
let codegen ~m ~k (name, ren, gpu) =
  test (Printf.sprintf "%s, %d rows of %d" name m k) (fun () ->
      let prog = program name ren ~m ~k in
      let gpu = gpu && k >= 64 in
      let bounded =
        List.filter
          (fun u ->
            match U.as_range u with
            | Some r -> reads_memory r.size
            | None -> false)
          prog
      in
      if not gpu then
        equal ~msg:"loops bounded by memory" int 0 (List.length bounded)
      else begin
        equal ~msg:"loops bounded by the id" int 1 (List.length bounded);
        let loop = List.hd bounded in
        let inside = ref false and outside = ref 0 in
        List.iter
          (fun u ->
            (if u == loop then inside := true
             else
               match U.as_end u with
               | Some e when List.memq loop e.ranges -> inside := false
               | _ -> ());
            if
              (not !inside)
              && D.is_float (U.dtype u)
              && (U.op u = Ops.Mul || U.op u = Ops.Mulacc)
            then incr outside)
          prog;
        equal ~msg:"float multiplies outside the loop" int 0 !outside
      end)

(* Where no loop's bound reads the id, an id outside the matrices reads matrix
   0: every load at an address read from memory, the instance's id, is ungated
   and its address selects on the id, a lone instance included, whose id is
   read at a fixed address outside every loop. The values tests hold such an
   instance's result at +0, and CHECK_OOB=1 its loads in bounds. *)
let rec address u =
  match U.op u with
  | Ops.Index -> List.tl (Array.to_list (U.src u))
  | Ops.Cast | Ops.Bitcast -> address (U.src u).(0)
  | _ -> [ u ]

let clamped ~k (name, ren, gpu) =
  test (Printf.sprintf "%s, one instance of %d inputs clamps its id" name k)
    (fun () ->
      if not (gpu && k >= 64) then begin
        let at_id =
          List.filter_map
            (fun u ->
              match U.as_load u with
              | Some { src; gate; _ }
                when List.exists reads_memory (address src) ->
                  Some (gate, address src)
              | _ -> None)
            (program ~instances:1 name ren ~m:1 ~k)
        in
        let selects a =
          List.exists (fun n -> U.op n = Ops.Where) (U.backward_slice a)
        in
        is_true ~msg:"loads at the id" (at_id <> []);
        List.iter
          (fun (gate, address) ->
            is_true ~msg:"no gate" (Option.is_none gate);
            is_true ~msg:"the address selects on the id"
              (List.exists selects address))
          at_id
      end)

let refusals =
  let empty dtype shape =
    Creation.empty ~dtype ~device:(U.Single "CPU") shape
  in
  let raises_invalid f =
    match f () with
    | _ -> fail "expected Invalid_argument"
    | exception Invalid_argument _ -> ()
  in
  test "refuses shapes that disagree" (fun () ->
      raises_invalid (fun () ->
          Op.quant_matmul
            (empty D.float32 [ 1; 1; 48 ])
            ~codes:(empty D.uint8 [ 1; 8; 24 ])
            ~scales:(empty D.uint8 [ 1; 8; 1 ]));
      raises_invalid (fun () ->
          Op.quant_matmul
            (empty D.float32 [ 1; 1; 64 ])
            ~codes:(empty D.uint8 [ 1; 8; 32 ])
            ~scales:(empty D.uint8 [ 1; 8; 3 ]));
      raises_invalid (fun () ->
          Op.quant_matmul ~ids:(empty D.int32 [ 3 ])
            (empty D.float32 [ 2; 1; 64 ])
            ~codes:(empty D.uint8 [ 2; 8; 32 ])
            ~scales:(empty D.uint8 [ 2; 8; 2 ]));
      raises_invalid (fun () ->
          Op.quant_matmul
            (empty D.float64 [ 1; 1; 64 ])
            ~codes:(empty D.uint8 [ 1; 8; 32 ])
            ~scales:(empty D.uint8 [ 1; 8; 2 ])))

let () =
  exit (run "Tolk_frontend_quant_matmul"
    [
      product_tests;
      exact_tests;
      test "the largest scales" large_scales;
      slow "random products" random_products;
      test "no inputs" no_inputs;
      refusals;
      group "code per renderer"
        (List.concat_map
           (fun r ->
             [
               codegen ~m:1 ~k:2880 r;
               codegen ~m:8 ~k:2880 r;
               codegen ~m:1 ~k:32 r;
               clamped ~k:32 r;
               clamped ~k:64 r;
             ])
           renderers);
    ])
