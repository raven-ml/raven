(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows: each prepares its operands on the GPU, checks its result once,
   and gives the run to time, with the floor kernel that moves its bytes in its
   directions where it is memory-bound. *)

module S = Nx_amd_support
module D = Nx_array.Dtype

let strf = Printf.sprintf

type work = Bytes of int | Flops of float
type prepared = { run : S.run; work : work; floor : S.run option }
type row = { name : string; make : S.gpu -> prepared }

(* [n] as a row names a size: 64K, 16M. *)
let size n =
  if n >= 1 lsl 20 then strf "%dM" (n lsr 20) else strf "%dK" (n lsr 10)

(* Floors *)

(* A row of [work] whose run [run] makes, with no floor. *)
let row name work run =
  { name; make = (fun g -> { run = run g; work; floor = None }) }

let harness g launches = S.record (S.harness g) launches

let copy n =
  row
    (strf "floor/copy-%sB" (size n))
    (Bytes (2 * n))
    (fun g ->
      harness g [ S.floor_copy ~ins:[ S.buffer g n ] ~out:(S.buffer g n) ])

(* The driver's copy, which a copy floor should not trail. *)
let copy_driver n =
  row
    (strf "floor/copy-driver-%sB" (size n))
    (Bytes (2 * n))
    (fun g -> S.driver_copy ~src:(S.buffer g n) ~dst:(S.buffer g n))

let read n =
  row
    (strf "floor/read-%sB" (size n))
    (Bytes n)
    (fun g -> harness g [ S.floor_read g (S.buffer g n) ])

let launch =
  row "floor/launch" (Bytes 0) (fun g ->
      harness g [ S.launch "empty" ~groups:(1, 1, 1) ~threads:1 [] ])

(* A peak row: 2 workgroups of 8 waves per compute unit, each wave [rounds]
   rounds of 16 instructions of [flops] each, per lane for an fma. *)
let peak name ~rounds ~flops =
  let kernel = String.map (fun c -> if c = '-' then '_' else c) name in
  let make g =
    let blocks = 2 * S.cus g in
    let o = S.buffer g (8 * blocks * 8) in
    let total = float (blocks * 8 * rounds * 16) *. flops in
    let launch =
      S.launch kernel ~groups:(blocks, 1, 1) ~threads:S.threads
        [ A o; W rounds ]
    in
    { run = harness g [ launch ]; work = Flops total; floor = None }
  in
  { name = "floor/" ^ name ^ "-peak"; make }

let peaks =
  [
    peak "wmma-bf16" ~rounds:2000 ~flops:8192.;
    peak "fma-f32" ~rounds:40000 ~flops:64.;
  ]

(* Copies and reads in the R9700's 8 MB L2, its 64 MB Infinity Cache and DRAM; a
   copy's working set is twice its size. *)
let floors =
  let copies = [ 1 lsl 18; 1 lsl 21; 1 lsl 24; 1 lsl 28 ] in
  [ launch ]
  @ List.concat_map (fun n -> [ copy n; copy_driver n ]) copies
  @ List.map read [ 1 lsl 22; 1 lsl 25; 1 lsl 28 ]
  @ peaks

(* Contract *)

(* The operands of y[z, i, j] = Σ_k a[z, i, k] · b[z, j, k], a [m; k] and b [n;
   k] per batch element of [d], in the layouts [la] and [lb] ([`K] or [`Free]
   axis contiguous), y [m; n] of [o], and the plan's run. *)
let plan g ?(batch = 1) ?(la = `K) ?(lb = `K) (D.Any d) (D.Any c) (D.Any o) m n
    k =
  let operand (type v s) (d : (v, s) D.t) rows cols layout : S.operand =
    let buffer = S.buffer g (D.bits d / 8 * batch * rows * cols) in
    let draw = if D.is D.Float d then S.Uniform else S.Small in
    S.generate g buffer d draw ~seed:rows;
    let s = if layout = `K then [| cols; 1 |] else [| 1; rows |] in
    let strides = Array.append [| rows * cols |] s in
    let shape = [| batch; rows; cols |] in
    { buffer; dtype = D.code d; shape; strides; first = 0 }
  in
  let a = operand d m k la and b = operand d n k lb and y = operand o m n `K in
  let contracting = [ (2, 2) ] and acc = D.code c in
  match S.contract g ~a ~b ~y ~batch:[ (0, 0) ] ~contracting ~acc () with
  | Some run -> (a, b, y, run)
  | None -> failwith "the plan declines"

(* The contraction, checked once at 512 outputs against the error bound. A
   product with m <= 16 is memory-bound: its floor reads its operands. *)
let contract ?(batch = 1) ?(la = `K) ?(lb = `K) (D.Any d as dt) (D.Any c as acc)
    out m n k =
  let l = function `K -> "k" | `Free -> "f" in
  let name =
    strf "contract/%s-%s%dx%dx%d%s" (D.name d)
      (if batch > 1 then strf "%dx" batch else "")
      m n k
      (if la = `K && lb = `K then "" else "-" ^ l la ^ l lb)
  in
  let make g =
    let a, b, y, run = plan g ~batch ~la ~lb dt acc out m n k in
    S.run g run;
    let view (x : S.operand) =
      {
        Nx_gpu_ref.bytes = S.read x.buffer;
        dtype = x.dtype;
        strides = x.strides;
      }
    in
    let r =
      Nx_gpu_ref.contract ~a:(view a) ~b:(view b) ~y:(view y) ~batch ~m ~n ~k
        ~acc:(D.code c) ~samples:512 ()
    in
    if r.wrong > 0 || not (r.worst <= 1.) then
      failwith (strf "%s: %g of the bound, %d wrong" name r.worst r.wrong);
    let bytes = D.bits d / 8 * batch * (m + n) * k in
    if m > 16 then
      {
        run;
        work = Flops (2. *. float (batch * m * n) *. float k);
        floor = None;
      }
    else
      let operands = S.buffer g (16 * ((bytes + 15) / 16)) in
      let floor = harness g [ S.floor_read g operands ] in
      { run; work = Bytes bytes; floor = Some floor }
  in
  { name; make }

let contracts =
  let open D in
  let bf16 = Any Bfloat16 and f16 = Any Float16 and f32 = Any Float32 in
  let i8 = Any Int8 and i32 = Any Int32 in
  let squares dt =
    List.map
      (fun s -> contract dt f32 dt s s s)
      [ 256; 512; 1024; 2048; 4096; 8192 ]
  in
  let layouts dt =
    List.map
      (fun (la, lb) -> contract ~la ~lb dt f32 dt 4096 4096 4096)
      [ (`K, `Free); (`Free, `K); (`Free, `Free) ]
  in
  List.concat_map squares [ bf16; f16; f32 ]
  @ List.concat_map layouts [ bf16; f16; f32 ]
  @ [
      contract i8 i32 i32 4096 4096 4096;
      contract bf16 f32 bf16 512 5120 2880;
      contract bf16 f32 bf16 512 2880 4096;
      contract bf16 f32 bf16 512 201088 2880;
      contract bf16 f32 bf16 1 5120 2880;
      contract bf16 f32 bf16 1 201088 2880;
      (* Products of m <= 16 whose grids the plan splits 4, 2 and 8 ways. *)
      contract bf16 f32 bf16 1 2880 4096;
      contract bf16 f32 bf16 16 4096 4096;
      contract bf16 f32 bf16 1 1024 16384;
      contract bf16 f32 bf16 4096 14336 4096;
      contract ~batch:64 bf16 f32 bf16 512 512 512;
      contract f32 f32 f32 1 5120 2880;
      contract ~lb:`Free f32 f32 f32 1 5120 2880;
    ]

let all = floors @ contracts
