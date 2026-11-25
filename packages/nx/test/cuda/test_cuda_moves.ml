(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cuda's moves on each rig driver that reaches the machine's NVIDIA GPU 0: a
   scatter Add as nx_cuda.mli states it, at each target x + S, S the sum of the
   position tree over the updates' positions along the axis, bit for bit, for
   axes past one tile, for updates whose positions step 0 along the last axis
   and into its own operand, of no update, target or column, NaNs and infinities
   among the values, operands through strided, permuted and reversed views;
   every move of complex64 arrays aligned to their parts' 4 bytes alone, as
   nx.cpu's; and the workspace that moves in a loop hold, within nx_cuda.mli's
   bytes. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* Cases *)

(* Updates [m; cols] along axis 0, or [cols; m] along axis 1, into [d]
   targets along the axis; with [rows], idx is a column broadcast along the
   last axis; with [in_place], dst is [into]; into, idx and the updates
   through the views [vi], [vx] and [vu]; NaNs and infinities among the
   values where [specials]. *)
type case = {
  dt : D.any;
  m : int;
  d : int;
  cols : int;
  axis : int;
  rows : bool;
  in_place : bool;
  vi : view;
  vx : view;
  vu : view;
  specials : bool;
  seed : int;
}

let viewed c = List.exists (fun v -> v <> plain 2) [ c.vi; c.vx; c.vu ]

let pp_case ppf c =
  let names v =
    match view_names v with [] -> "plain" | ns -> String.concat " " ns
  in
  Format.fprintf ppf
    "%a: %d updates by %d along %d into %d%s%s%s, into %s, idx %s, updates %s \
     (seed %d)"
    pp_dtype c.dt c.m c.cols c.axis c.d
    (if c.rows then ", idx broadcast along the last axis" else "")
    (if c.in_place then ", in place" else "")
    (if c.specials then ", NaNs and infinities" else "")
    (names c.vi) (names c.vx) (names c.vu) c.seed

let dtypes =
  D.
    [
      Any Float32;
      Any Float64;
      Any Bfloat16;
      Any Float16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Complex64;
      Any Complex128;
    ]

let gen =
  let open Gen in
  let* dt = of_list ~pp:pp_dtype dtypes in
  let* m =
    frequency [ (1, constant 0); (8, int_range 1 40); (8, int_range 1025 3000) ]
  in
  let* d =
    frequency [ (1, constant 0); (16, int_range 1 6); (8, int_range 50 2000) ]
  in
  let* cols =
    frequency [ (1, constant 0); (16, int_range 1 5); (8, int_range 30 40) ]
  in
  let* axis = int_range 0 1 in
  let* rows = bool in
  let* in_place = bool in
  let* views = bool in
  let view = if views then view_of 2 else constant (plain 2) in
  let* vi = view and+ vx = view and+ vu = view in
  let* specials = bool in
  let+ seed = int in
  {
    dt;
    m;
    d;
    cols;
    axis;
    rows = rows && axis = 0 && cols > 1;
    in_place;
    vi;
    vx;
    vu;
    specials;
    seed;
  }

let cases = Gen.with_pp pp_case gen

(* Draws reach row tiles past one tile, and no update, target or column, too
   rarely for every run (p 0.098 and 0.133 a case, so 100 cases miss them 3e-5
   and 7e-7 of runs): an example each. *)
let examples =
  let c =
    {
      dt = D.Any D.Float32;
      m = 0;
      d = 5;
      cols = 3;
      axis = 0;
      rows = false;
      in_place = false;
      vi = plain 2;
      vx = plain 2;
      vu = plain 2;
      specials = false;
      seed = 1;
    }
  in
  [
    { c with dt = D.Any D.Bfloat16; m = 1500; d = 300; rows = true };
    c;
    { c with m = 40; d = 0 };
    { c with m = 40; cols = 0 };
  ]

(* Values *)

let narrow (D.Any dt) = D.is D.Float dt && D.bits dt < 32
let complex (D.Any dt) = D.is D.Complex dt

(* A part's value: an integer of up to 12 bits times a power of two, -0 now
   and then, a NaN or an infinity now and then where [specials]; a narrow
   float's within its range over 3000 terms. *)
let value ?(specials = false) dt rs =
  let r = Random.State.int rs 64 in
  if r < 4 then -0.
  else if specials && r = 4 then Float.nan
  else if specials && r = 5 then Float.infinity
  else if specials && r = 6 then Float.neg_infinity
  else
    let e = if narrow dt then 3 else 12 in
    let k = Float.of_int (Random.State.int rs 4095 - 2047) in
    Float.ldexp k (Random.State.int rs ((2 * e) + 1) - e - 11)

(* Arithmetic in the format a sum computes in: float64, or float32. *)
let wide (D.Any dt) = D.bits dt = if D.is D.Complex dt then 128 else 64
let add dt a b = if wide dt then a +. b else round32 (a +. b)

(* An element's parts and back. *)
let parts dt e =
  if narrow dt then [| decode dt e |]
  else if complex dt then
    let h = String.length e / 2 in
    if h = 4 then [| get32 e 0; get32 e 4 |] else [| get64 e 0; get64 e 8 |]
  else if String.length e = 4 then [| get32 e 0 |]
  else [| get64 e 0 |]

(* The codes of the narrow float [dt] nearest each of [xs], by one nx.cpu cast:
   one call, where a cast an element would cost the suite seconds. *)
let encoded (D.Any dt) xs =
  let n = Array.length xs in
  let src = A.of_array D.Float32 [| n |] xs in
  let dst = A.create Rig.host dt [| n |] in
  match Support.map1 Support.cpu.kernels Cast ~dst src with
  | A.Done -> elements (A.Any dst)
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* A wide element from its parts. *)
let of_parts dt ps =
  String.concat "" (Array.to_list (Array.map (if wide dt then f64 else f32) ps))

(* [n] elements drawn by [value] from [rs], in order. *)
let drawn ?specials dt rs n =
  let part () = round32 (value ?specials dt rs) in
  if narrow dt then encoded dt (Array.init n (fun _ -> part ()))
  else
    let k = if complex dt then 2 else 1 in
    Array.init n (fun _ -> of_parts dt (Array.init k (fun _ -> part ())))

(* The reference *)

(* The position tree's sum of [vs.(lo)] to [vs.(hi - 1)], at the increasing
   positions [js], all below [2^l] and sharing the bits above: each half of the
   block summed apart, the left one first; [None] for none. *)
let rec tree dt l js vs lo hi =
  if lo >= hi then None
  else if hi - lo = 1 then Some vs.(lo)
  else
    let mid = ref lo in
    while !mid < hi && (js.(!mid) lsr (l - 1)) land 1 = 0 do
      incr mid
    done;
    match (tree dt (l - 1) js vs lo !mid, tree dt (l - 1) js vs !mid hi) with
    | Some a, Some b -> Some (add dt a b)
    | (Some _ as a), None | None, (Some _ as a) -> a
    | None, None -> None

let rec ceil_log2 ?(b = 0) n =
  if 1 lsl b >= n then b else ceil_log2 ~b:(b + 1) n

(* The scatter's result as nx_cuda.mli states it, as elements in C order: at
   each target, its updates' positions in increasing order and each part's
   values, summed by the tree and added to the target's part. *)
let expected c into idx upd =
  let tgt = Array.map (fun _ -> []) into in
  for j = c.m - 1 downto 0 do
    for k = 0 to c.cols - 1 do
      let p = idx j k in
      if p >= 0 && p < c.d then begin
        let t = if c.axis = 0 then (p * c.cols) + k else (k * c.d) + p in
        tgt.(t) <- (j, parts c.dt (upd j k)) :: tgt.(t)
      end
    done
  done;
  let l = ceil_log2 c.m in
  let sum us x =
    let us = Array.of_list us in
    let js = Array.map fst us and n = Array.length us in
    let part i xi =
      let vs = Array.map (fun (_, u) -> u.(i)) us in
      add c.dt xi (Option.get (tree c.dt l js vs 0 n))
    in
    Array.mapi part (parts c.dt x)
  in
  let sums =
    Array.mapi
      (fun t x -> match tgt.(t) with [] -> None | us -> Some (sum us x))
      into
  in
  let codes =
    if narrow c.dt then
      encoded c.dt (Array.map (function Some s -> s.(0) | None -> 0.) sums)
    else [||]
  in
  Array.mapi
    (fun t x ->
      match sums.(t) with
      | None -> x
      | Some _ when narrow c.dt -> codes.(t)
      | Some s -> of_parts c.dt s)
    into

(* The law *)

let law (b : Support.backend) c =
  let module K = (val b.kernels) in
  cover "an axis past one tile" (c.m > 1024);
  cover "row tiles" c.rows;
  cover "row tiles past one tile" (c.rows && c.m > 256);
  cover "targets of several updates" (c.m > c.d);
  cover "in place" c.in_place;
  cover "a narrow float" (narrow c.dt);
  cover "no update, target or column" (c.m = 0 || c.d = 0 || c.cols = 0);
  cover "NaNs and infinities" c.specials;
  let rs = Random.State.make [| c.seed |] in
  let ts = if c.axis = 0 then [| c.d; c.cols |] else [| c.cols; c.d |] in
  let us = if c.axis = 0 then [| c.m; c.cols |] else [| c.cols; c.m |] in
  (* The elements of the array a view is made from, in order. *)
  let drawn s v = drawn ~specials:c.specials c.dt rs (total (made v s)) in
  let into_es = drawn ts c.vi in
  let (A.Any into_a) = operand c.dt ts c.vi (fun i -> into_es.(i)) in
  let u_es = drawn us c.vu in
  let u = operand c.dt us c.vu (fun i -> u_es.(i)) in
  let into = elements (A.Any into_a) and ups = elements u in
  let pos () = Int64.of_int (Random.State.int rs (c.d + 2) - 1) in
  let idx_a =
    if c.rows then
      Option.get
        (A.move
           (M.Broadcast (Iarray.of_array us))
           (A.of_array D.Int64 [| c.m; 1 |] (Array.init c.m (fun _ -> pos ()))))
    else
      let (A.Any p) = int64s us c.vx pos in
      A.expect D.Int64 (A.Any p)
  in
  let ids = Array.map Int64.to_int (positions (A.Any idx_a)) in
  let at j k = if c.axis = 0 then (j * c.cols) + k else (k * c.m) + j in
  let idx j k = ids.(at j k) in
  let upd j k = ups.(at j k) in
  let want = expected c into idx upd in
  let into_a = Support.on b into_a in
  let u = Support.on b (A.expect (A.dtype into_a) u) in
  let idx_a = Support.on b idx_a in
  let dst, into_a =
    if c.in_place then
      let d = Support.on b (A.copy (Support.host into_a)) in
      (d, d)
    else (A.create b.device (A.dtype into_a) (Iarray.of_array ts), into_a)
  in
  let layouts = [ A.layout into_a; A.layout u; A.layout idx_a ] in
  cover "views"
    (viewed c && List.exists (fun l -> not (L.is_contiguous l)) layouts);
  cover "a view with gaps" (List.exists Support.gapped layouts);
  match K.scatter Add ~unique:false ~axis:c.axis ~dst ~into:into_a idx_a u with
  | A.Done -> equal (array (same c.dt)) want (elements (A.Any dst))
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Complex64 aligned to 4 bytes *)

(* A complex64 array of shape [s] holding [es], on [b]'s device with its
   first element at 4 mod 8 bytes, or on the host for nx.cpu. *)
let c64 (b : Support.backend) s es =
  let h = A.expect D.Complex64 (of_elements (D.Any D.Complex64) s es) in
  if Rig.equal b.device Rig.host then h
  else
    let n = total s in
    let buf = Rig.Buffer.create b.device ((8 * n) + 8) in
    let view = Rig.Buffer.view buf ~first:4 ~length:(8 * n) in
    let a = A.v D.Complex64 (L.contiguous (Iarray.of_array s)) view in
    A.transfer ~src:h ~dst:a;
    a

let ints (b : Support.backend) s xs =
  Support.on b
    (A.of_array D.Int64 (Iarray.of_array s) (Array.map Int64.of_int xs))

let c64s rs n =
  Array.init n (fun _ ->
      let part () =
        if Random.State.int rs 8 = 0 then -0. else value (D.Any D.Float32) rs
      in
      f32 (round32 (part ())) ^ f32 (round32 (part ())))

let done_ r =
  if r <> A.Done then
    failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Each move of complex64 arrays at 4 mod 8 bytes, by [b], as the elements
   of its result. *)
let misaligned_moves (b : Support.backend) =
  let module K = (val b.kernels) in
  let rs () = Random.State.make [| 7 |] in
  let fresh s = c64 b s (Array.make (total s) (String.make 8 '\000')) in
  let scatter c ~unique ?(in_place = false) ids =
    let r = rs () in
    let into = c64 b [| 5; 2 |] (c64s r 10) in
    let u = c64 b [| 4; 2 |] (c64s r 8) in
    let dst = if in_place then into else fresh [| 5; 2 |] in
    done_ (K.scatter c ~unique ~axis:0 ~dst ~into (ints b [| 4; 2 |] ids) u);
    dst
  in
  let same_ids = [| 1; 0; 1; 4; 1; 4; 3; -1 |] in
  let distinct = [| 0; 4; 2; 1; 4; 0; 1; 3 |] in
  [
    ( "gather",
      fun () ->
        let x = c64 b [| 6; 3 |] (c64s (rs ()) 18) and dst = fresh [| 5; 3 |] in
        let idx =
          ints b [| 5; 3 |] (Array.init 15 (fun i -> (i * 7 mod 8) - 1))
        in
        done_ (K.gather ~axis:0 ~dst idx x);
        dst );
    ( "gather_ranges",
      fun () ->
        let x = c64 b [| 8; 2 |] (c64s (rs ()) 16) and dst = fresh [| 6; 2 |] in
        done_
          (K.gather_ranges ~dst
             ~starts:(ints b [| 3 |] [| 5; 0; 2 |])
             ~ends:(ints b [| 3 |] [| 2; 3; 7 |])
             x);
        dst );
    ("scatter Set", fun () -> scatter Set ~unique:false same_ids);
    ("scatter Set ~unique", fun () -> scatter Set ~unique:true distinct);
    ("scatter Max", fun () -> scatter Max ~unique:false same_ids);
    ( "scatter Min, in place",
      fun () -> scatter Min ~unique:false ~in_place:true same_ids );
    ("scatter Add", fun () -> scatter Add ~unique:false same_ids);
    ("scatter Add ~unique", fun () -> scatter Add ~unique:true distinct);
    ( "assemble",
      fun () ->
        let r = rs () in
        let p0 = c64 b [| 2; 3 |] (c64s r 6)
        and p1 = c64 b [| 3; 2 |] (c64s r 6) in
        let dst = fresh [| 4; 4 |] in
        let g start count = { M.start; count; step = 1 } in
        done_
          (K.assemble
             ~fill:{ Complex.re = 1.5; im = -0. }
             ~dst
             [|
               (Iarray.of_array [| g 0 2; g 0 3 |], p0);
               (Iarray.of_array [| g 1 3; g 2 2 |], p1);
             |]);
        dst );
    ( "fold",
      fun () ->
        let pad =
          {
            Nx_kernel.lo = Iarray.of_array [| 1; 0 |];
            hi = Iarray.of_array [| 0; 1 |];
            interior = Iarray.of_array [| 0; 1 |];
            windows =
              Iarray.of_array
                [| { M.axis = 1; size = 2; step = 1; dilation = 2 } |];
          }
        in
        let xs = Result.get_ok (Nx_kernel.Shape.padded pad [| 3; 3 |]) in
        let x = c64 b xs (c64s (rs ()) (total xs)) and dst = fresh [| 3; 3 |] in
        done_ (K.fold pad ~dst x);
        dst );
  ]

let test_misaligned (b : Support.backend) () =
  let want =
    List.map
      (fun (n, f) -> (n, elements (A.Any (f ()))))
      (misaligned_moves Support.cpu)
  in
  List.iter2
    (fun (n, w) (_, f) ->
      equal ~msg:n
        (array (same (D.Any D.Complex64)))
        w
        (elements (A.Any (f ()))))
    want (misaligned_moves b)

(* nx_cuda.mli: a [Set] without [~unique] takes 4 bytes a target while it
   runs, here past the 64 MiB the device keeps: its workspace, a buffer of
   the call's own, which counts in the driver's units ({!Rig.allocation}). *)
let test_loop_set (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 1 lsl 25 and d = b.device in
  let into = A.create d D.Uint8 [| n |] and dst = A.create d D.Uint8 [| n |] in
  let idx = A.create d D.Int64 [| 4 |] and u = A.create d D.Uint8 [| 4 |] in
  Kernels_cases.loop_held b ~bound:(Rig.allocation d (4 * n)) (fun () ->
      done_ (K.scatter Set ~unique:false ~axis:0 ~dst ~into idx u);
      Rig.Buffer.wait (A.buffer dst) Read_write)

(* nx_cuda.mli: a [Max] scatter of complex numbers takes 4 bytes a target while
   it runs. These tests read a call's own buffers, past the 64 MiB of moves'
   workspace nx.cuda keeps (nx_cuda.ml), within the 192 MiB nx_cuda.mli allows:
   a device that kept more would show none held. *)
let test_loop_max (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 1 lsl 25 and d = b.device in
  let into = A.create d D.Complex64 [| n |] in
  let dst = A.create d D.Complex64 [| n |] in
  let idx = ints b [| 4 |] [| 0; 7; 7; n - 1 |] in
  let u = A.create d D.Complex64 [| 4 |] in
  Kernels_cases.loop_held b
    ~bound:(Rig.allocation d (4 * n))
    (fun () ->
      done_ (K.scatter Max ~unique:false ~axis:0 ~dst ~into idx u);
      Rig.Buffer.wait (A.buffer dst) Read_write)

(* nx_cuda.mli: a tiled add takes 4 bytes a target of [into], at most [83 + 4 w]
   bytes an update, [w] 4 at float32, and 5 KiB: 2^16 updates, so that their 6.5
   MiB pass the driver's 2 MiB granules (the 5 KiB stay within one). *)
let test_loop_tiled (b : Support.backend) () =
  let module K = (val b.kernels) in
  let t = 1 lsl 25 and m = 1 lsl 16 and d = b.device in
  let into = A.create d D.Float32 [| t |]
  and dst = A.create d D.Float32 [| t |] in
  let idx = ints b [| m |] (Array.init m (fun j -> j * 7 mod t)) in
  let u = A.create d D.Float32 [| m |] in
  let bound = (4 * t) + ((83 + 16) * m) + 5120 in
  Kernels_cases.loop_held b ~bound:(Rig.allocation d bound) (fun () ->
      done_ (K.scatter Add ~unique:false ~axis:0 ~dst ~into idx u);
      Rig.Buffer.wait (A.buffer dst) Read_write)

(* nx_cuda.mli: a range gather takes 24 bytes a run and 8 for each 4096 runs of
   a row: here one row of 2^22 runs of one element, whose 8 KiB for each 4096
   runs fall within a driver's granule. *)
let test_loop_ranges (b : Support.backend) () =
  let module K = (val b.kernels) in
  let k = 1 lsl 22 and d = b.device in
  let x = A.create d D.Float32 [| k |] and dst = A.create d D.Float32 [| k |] in
  let starts = ints b [| k |] (Array.init k Fun.id) in
  let ends = ints b [| k |] (Array.init k (fun i -> i + 1)) in
  let bound = (24 * k) + (8 * ((k + 4095) / 4096)) in
  Kernels_cases.loop_held b ~bound:(Rig.allocation d bound) (fun () ->
      done_ (K.gather_ranges ~dst ~starts ~ends x);
      Rig.Buffer.wait (A.buffer dst) Read_write)

(* GPU 0 through CUDA and through rig.nv, opened at a test's first call;
   a test skips without its GPU. *)
let cuda () =
  match Support.gpus () with
  | b :: _ -> b
  | [] -> skip ~reason:"CUDA opens no GPU" ()

let nv () =
  match Support.nv () with
  | Some b -> b
  | None -> skip ~reason:"rig.nv opens no GPU" ()

let () =
  exit
    (Windtrap.run "nx.cuda moves"
       (List.map
          (fun (name, b) ->
            let on f x = f (b ()) x in
            group name
              [
                prop ~examples
                  "a float Add is x + S, S the position tree's sum of the \
                   updates"
                  cases (on law);
                test "moves complex64 aligned to 4 bytes as nx.cpu does"
                  (on test_misaligned);
                test
                  "range gathers in a loop hold 24 bytes a run and 8 each 4096 \
                   runs"
                  (on test_loop_ranges);
                test "scatters in a loop hold one scatter's workspace"
                  (on test_loop_set);
                test "complex Max scatters in a loop hold 4 bytes a target"
                  (on test_loop_max);
                test
                  "tiled adds in a loop hold their targets', updates' and 5 \
                   KiB's workspace"
                  (on test_loop_tiled);
              ])
          [ ("cuda", cuda); ("cuda/nv", nv) ]))
