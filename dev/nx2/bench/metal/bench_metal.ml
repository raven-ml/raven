(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx2's Metal floors and kernels on the Mac's GPU, each kernel row beside the
   floor that bounds it. A floor row runs [n] launches in one submission, and a
   contraction row [n] eager calls of nx.metal, enough for 10 ms of GPU time, so
   that the host's wait is a small part of a run. A call row ([.../call]) runs
   one call.

   [bench_metal.exe] runs the rows under thumper, which times runs on the
   host's clock. [bench_metal.exe gate [PAT]] prints each row's wall time per
   launch or call, the median of 30 runs, and its distance to its floor; for a
   call row, the wall time of a call. [bench_metal.exe probe] prints what the
   compiler and the GPU do to float arithmetic. *)

module S = Nx_metal_support
module Dt = Nx_array.Dtype

let strf = Printf.sprintf

external loadavg : unit -> float = "nx_metal_bench_loadavg"

let k = 1024
let m = k * k

(* Rows *)

(* A row: the floor that bounds it (none for a floor), the bytes or flops a
   launch moves or computes, whether a call is one launch submitted and waited
   for alone, as an eager call runs, and its setup, which opens the device and
   is a launch. *)
type row = {
  name : string;
  floor : string option;
  work : [ `Bytes of int | `Flops of int | `Launch ];
  call : bool;
  setup : S.t -> S.run;
}

let dev () =
  S.hold_gpu ();
  match S.open_ () with Some t -> t | None -> failwith "no Metal GPU"

let operands t n count =
  List.init count (fun i ->
      let o = S.operand t (4 * n) in
      S.generate t o Dt.Float32 n ~seed:(i + 1);
      o)

let size_name n = if n >= m then strf "%dM" (n / m) else strf "%dK" (n / k)

(* Reads [ins] arrays of [n] float32 and writes one: move. read reads one. *)
let stream_row kernel ~ins ~out n =
  let name = if kernel = "read" then "read" else strf "move%d" ins in
  {
    name = strf "floor/%s-%s" name (size_name n);
    floor = None;
    call = false;
    work = `Bytes (4 * n * (ins + if out then 1 else 0));
    setup =
      (fun t ->
        let vecs = n / 4 in
        let ((gx, _, _) as groups) = S.groups vecs in
        let out = S.operand t (if out then 4 * n else 16 * gx) in
        let ins = operands t n ins in
        (* The kernel reads its first [ins] inputs: the rest repeat the first. *)
        let addrs =
          out :: (ins @ List.init (3 - List.length ins) (fun _ -> List.hd ins))
        in
        S.launch kernel ~groups ~addrs ~words:[ List.length ins; vecs ]);
  }

let launch_row =
  {
    name = "floor/launch";
    floor = None;
    call = false;
    work = `Launch;
    setup = (fun _ -> S.launch "empty" ~addrs:[] ~words:[]);
  }

(* The GPU's peaks: 2 flops per fma, and 1,024 per 8x8x8 product, per chain, per
   round. *)
let peak_row name kernel (Dt.Any dt) ~flops_per_round =
  let threads = m / 2 and iters = 4096 in
  {
    name;
    floor = None;
    call = false;
    work = `Flops (threads * flops_per_round * iters);
    setup =
      (fun t ->
        (* A thread's element, or a simdgroup's 64. *)
        let out = S.operand t (Dt.bytes dt (2 * threads)) in
        S.generate t out dt (2 * threads) ~seed:7;
        S.launch kernel ~groups:(S.groups threads)
          ~addrs:[ out ]
          ~words:[ iters; 0 ]);
  }

(* Contractions of a [batch][m][k] and b [batch][k][n], stored transposed where
   [trans] says ("nt": b stored [n][k]), against the simdgroup-matrix peak of
   the type their fragments hold. A skinny product (m at most 16) streams b: its
   floor reads as many bytes. Each row's setup runs it once and checks 512 of
   its outputs, against the error bound for floats and exactly for integers: a
   row whose results are wrong fails instead of timing them. *)
let samples = 512

let contract_row ?(batch = 1) ?name ?acc ?out (Dt.Any dt) ~m ~k ~n trans =
  let a_t = trans.[0] = 't' and b_t = trans.[1] = 't' in
  let short = function
    | "float32" -> "f32"
    | "float16" -> "f16"
    | "bfloat16" -> "bf16"
    | "int8" -> "i8"
    | d -> d
  in
  let base = strf "contract-%s" (short (Dt.name dt)) in
  let name =
    match name with
    | Some x -> base ^ "-" ^ x
    | None -> strf "%s-%dx%dx%d-%s" base m k n trans
  in
  let peak =
    match Dt.Any dt with
    | Dt.Any Dt.Float16 | Dt.Any Dt.Int8 -> "floor/simdgroup-matrix-f16-peak"
    | _ -> "floor/simdgroup-matrix-f32-peak"
  in
  let b_bytes = Dt.bytes dt (batch * k * n) in
  let floor, work =
    if m <= 16 then
      (strf "floor/read-%dMB" (b_bytes / 1_000_000), `Bytes b_bytes)
    else (peak, `Flops (2 * batch * m * n * k))
  in
  {
    name;
    floor = Some floor;
    work;
    call = false;
    setup =
      (fun t ->
        let matrix ~trans ~rows ~cols ~seed =
          let o = S.operand t (Dt.bytes dt (batch * rows * cols)) in
          S.generate t o dt (batch * rows * cols) ~seed;
          S.arg o dt
            (if trans then (rows * cols, 1, rows) else (rows * cols, cols, 1))
        in
        let a = matrix ~trans:a_t ~rows:m ~cols:k ~seed:1 in
        let b = matrix ~trans:b_t ~rows:k ~cols:n ~seed:2 in
        let (Dt.Any out_dt) = Option.value out ~default:(Dt.Any dt) in
        let out =
          S.arg
            (S.operand t (Dt.bytes out_dt (batch * m * n)))
            out_dt
            (m * n, n, 1)
        in
        let dims = (batch, m, n, k) in
        let run = Option.get (S.plan_contract ?acc t dims ~a ~b ~out) in
        ignore (S.run t run);
        (match acc with
        | Some acc ->
            let wrong, first = S.contract_wrong ~samples ~acc dims ~a ~b ~out in
            if wrong > 0 then
              failwith
                (strf "%s: %d outputs wrong, the first %d" name wrong first)
        | None ->
            let worst, at = S.contract_error ~samples dims ~a ~b ~out in
            if not (worst <= 1.) then
              failwith (strf "%s: output %d at %g of the bound" name at worst));
        run);
  }

(* Squares, the four orders, gpt-oss's prefill, few-row and decode products (k
   2880), a decode product with b stored [k][n], Llama 3 8B's MLP up projection,
   and a batch of small products. *)
let contract_rows =
  let floats = [ Dt.Any Dt.Bfloat16; Dt.Any Dt.Float16; Dt.Any Dt.Float32 ] in
  List.concat_map
    (fun dt ->
      List.map
        (fun s -> contract_row ~name:(string_of_int s) dt ~m:s ~k:s ~n:s "nn")
        [ 256; 512; 1024; 2048; 4096 ]
      @ List.map
          (fun tr ->
            contract_row ~name:("4096-" ^ tr) dt ~m:4096 ~k:4096 ~n:4096 tr)
          [ "nt"; "tn"; "tt" ]
      @ List.map
          (fun (m, k, n) -> contract_row dt ~m ~k ~n "nt")
          [
            (512, 2880, 5120);
            (512, 4096, 2880);
            (512, 2880, 201088);
            (1, 2880, 5120);
            (3, 2880, 5120);
            (8, 2880, 5120);
            (16, 2880, 5120);
            (1, 2880, 201088);
            (4096, 4096, 14336);
          ]
      @ [
          contract_row dt ~m:1 ~k:2880 ~n:5120 "nn";
          contract_row dt ~m:8 ~k:2880 ~n:5120 "nn";
        ]
      @ [ contract_row ~batch:64 ~name:"64x512" dt ~m:512 ~k:512 ~n:512 "nn" ])
    floats
  @ [
      contract_row ~name:"4096" ~acc:(Dt.Any Dt.Int32) ~out:(Dt.Any Dt.Int32)
        (Dt.Any Dt.Int8) ~m:4096 ~k:4096 ~n:4096 "nn";
    ]
  @ List.concat_map
      (fun dt ->
        List.map
          (fun (m, k, n) -> contract_row dt ~m ~k ~n "nt")
          [ (32, 2880, 201088); (48, 5120, 2880); (1000, 1000, 1000) ]
        @ List.map
            (fun (m, k, n) -> contract_row dt ~m ~k ~n "nn")
            [ (32, 2880, 201088); (48, 5120, 2880); (1000, 1000, 1000) ])
      [ Dt.Any Dt.Bfloat16; Dt.Any Dt.Float16; Dt.Any Dt.Float32 ]

(* Decode's rows as one call at a time, each waited for, b resident in the
   GPU's cache from the call before. *)
let call_rows =
  List.filter_map
    (fun r ->
      let decode =
        List.exists
          (fun s -> String.ends_with ~suffix:s r.name)
          [ "-1x2880x5120-nt"; "-1x2880x5120-nn"; "-1x2880x201088-nt" ]
      in
      if decode then Some { r with name = r.name ^ "/call"; call = true }
      else None)
    contract_rows

(* The floors of the skinny rows, a read of the bytes of b they stream. *)
let skinny_floors =
  List.sort_uniq compare
    (List.filter_map
       (fun r ->
         match (r.work, r.floor) with
         | `Bytes b, Some f -> Some (f, b)
         | _ -> None)
       contract_rows)
  |> List.map (fun (name, b) ->
      { (stream_row "read" ~ins:1 ~out:false (b / 4)) with name })

let rows =
  [ launch_row ]
  @ List.concat_map
      (fun n ->
        List.map (fun ins -> stream_row "move" ~ins ~out:true n) [ 1; 2; 3 ]
        @ [ stream_row "read" ~ins:1 ~out:false n ])
      [ 64 * k; m; 16 * m ]
  @ [
      peak_row "floor/fma-f32-peak" "fma_f32" (Dt.Any Dt.Float32)
        ~flops_per_round:64;
      peak_row "floor/fma-f16-peak" "fma_f16" (Dt.Any Dt.Float16)
        ~flops_per_round:64;
      peak_row "floor/simdgroup-matrix-f32-peak" "mma_f32" (Dt.Any Dt.Float32)
        ~flops_per_round:(8 * 1024 / 32);
      peak_row "floor/simdgroup-matrix-f16-peak" "mma_f16" (Dt.Any Dt.Float16)
        ~flops_per_round:(8 * 1024 / 32);
    ]
  @ skinny_floors @ contract_rows @ call_rows

(* Timing *)

(* A row's launches per run: the same count on every run and build, about
   [target_ns] of GPU time at an M1 Max's nominal rates for the row's work; one
   for a call row. A count measured per run would time different work from one
   run to the next. *)
let target_ns = 10_000_000.
let flops_per_ns = 8_000. (* 8 TFLOP/s *)
let bytes_per_ns = 400. (* 400 GB/s *)
let launch_ns = 3_000.

let launches r =
  if r.call then 1
  else
    let ns =
      match r.work with
      | `Flops f -> float f /. flops_per_ns
      | `Bytes b -> float b /. bytes_per_ns
      | `Launch -> launch_ns
    in
    max 1 (min 8192 (int_of_float (Float.ceil (target_ns /. ns))))

(* A row's run: its [n] launches or calls, after three runs of one that make
   the pipelines and warm the caches and the GPU's clock. *)
let sized t r =
  let launch = r.setup t in
  let one = S.prepare t launch in
  for _ = 1 to 3 do
    ignore (one ())
  done;
  let n = launches r in
  if n = 1 then (1, one)
  else (n, S.prepare t (S.seq (List.init n (fun _ -> launch))))

let median l =
  let a = Array.of_list l in
  Array.sort compare a;
  a.(Array.length a / 2)

(* The median wall time per launch or call, in nanoseconds, over 30 runs. *)
let per_launch t r =
  let n, go = sized t r in
  for _ = 1 to 3 do
    ignore (go ())
  done;
  float (median (List.init 30 (fun _ -> go ()))) /. float n

(* Gate *)

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let gate pat =
  let t = dev () in
  let selected r = contains ~sub:pat r.name in
  let needed r =
    selected r
    || List.exists (fun r' -> selected r' && r'.floor = Some r.name) rows
  in
  let floors = Hashtbl.create 16 in
  Printf.printf "load %.2f\n%-40s %12s %12s %8s\n" (loadavg ()) "row"
    "us/launch" "rate" "/floor";
  (* A call row: the median of 30 samples of the wall time of 50 calls on the
     host's clock. *)
  let call_time r =
    let _, go = sized t r in
    let sample () =
      let t0 = Unix.gettimeofday () in
      for _ = 1 to 50 do
        ignore (go ())
      done;
      (Unix.gettimeofday () -. t0) /. 50.
    in
    let wall = median (List.init 30 (fun _ -> sample ())) in
    Gc.full_major ();
    Printf.printf "%-40s %12.2f\n%!" r.name (wall *. 1e6)
  in
  let time r =
    if r.call then call_time r
    else
      let ns = per_launch t r in
      Gc.full_major ();
      if r.floor = None then Hashtbl.replace floors r.name (ns, r.work);
      if selected r then
        let rate =
          match r.work with
          | `Bytes b -> strf "%.1f GB/s" (float b /. ns)
          | `Flops f -> strf "%.0f GF/s" (float f /. ns)
          | `Launch -> ""
        in
        (* A memory-bound row's time over its floor's; a compute-bound row's
           fraction of its peak. *)
        let ratio =
          match (r.work, Option.bind r.floor (Hashtbl.find_opt floors)) with
          | `Flops f, Some (fns, `Flops ff) ->
              strf "%.0f%%" (100. *. (float f /. ns) /. (float ff /. fns))
          | _, Some (fns, _) -> strf "%.2f" (ns /. fns)
          | _, None -> ""
        in
        Printf.printf "%-40s %12.2f %12s %8s\n%!" r.name (ns /. 1000.) rate
          ratio
  in
  List.iter (fun r -> if needed r then time r) rows;
  Printf.printf "load %.2f\n" (loadavg ())

(* Probes *)

let n = 1 lsl 24
let floats o = S.view Bigarray.float32 o
let words o = S.view Bigarray.int32 o
let tiny = Int32.float_of_bits 1l (* 2^-149 *)
let least_normal = Int32.float_of_bits 0x00800000l (* 2^-126 *)

(* Operand tuples: drawn ones, then [edges] at the end. *)
let tuples t ~arity ~spread edges =
  let o = S.operand t (4 * arity * n) in
  S.generate ~spread t o Dt.Float32 (arity * n) ~seed:spread;
  let v = floats o in
  List.iteri
    (fun i tuple ->
      List.iteri (fun j x -> v.{(arity * (n - 1 - i)) + j} <- x) tuple)
    edges;
  o

let probe () =
  let t = dev () in
  Printf.printf "load %.2f, GPU %s; results with float32 subnormals flushed\n%!"
    (loadavg ())
    (Rig.arch (S.rig t));
  let edges =
    [
      [ least_normal; 0.5; 0. ]; [ -.tiny; 1.; -0. ]; [ least_normal; 1.; tiny ];
    ]
  in
  List.iter
    (fun spread ->
      let abc = tuples t ~arity:3 ~spread edges in
      let run which = floats (S.probe t "probe_contract" abc ~which n) in
      let apart, fused, wrong =
        S.probe_contract ~show:4 (floats abc) (run 0) (run 1)
      in
      Printf.printf
        "contract, spread %d, %d triples: a*b + c differs from two roundings \
         in %d, from one in %d; fma differs from one rounding in %d\n\
         %!"
        spread n apart fused wrong)
    [ 0; 20 ];
  let xy =
    tuples t ~arity:2 ~spread:126 [ [ 1e-40; 7. ]; [ 1.; tiny ]; [ -1.; 2. ] ]
  in
  let run which = floats (S.probe t "probe_div_sqrt" xy ~which n) in
  let d, s, sub = S.probe_div_sqrt ~show:4 (floats xy) (run 0) (run 1) in
  Printf.printf
    "div_sqrt, %d pairs (%d subnormal quotients): x / y differs from the \
     correctly rounded quotient in %d, sqrt x from the root in %d\n\
     %!"
    n sub d s;
  let xs = S.operand t (4 * n) in
  S.generate t xs Dt.Uint32 n ~seed:3;
  let run which = words (S.probe t "probe_half" xs ~which n) in
  let to_, from, sum = S.probe_half (words xs) (run 0) (run 1) (run 2) in
  Printf.printf
    "half, %d words: half(x) differs from nx_float_to_f16 in %d, float(h) from \
     nx_f16_to_float in %d, h + h from the exact sum in %d\n\
     %!"
    n to_ from sum

(* Thumper *)

let case r =
  Thumper.bench_with_setup r.name
    ~setup:(fun () ->
      let t = dev () in
      snd (sized t r))
    (fun go -> go ())

let () =
  match Array.to_list Sys.argv with
  | _ :: "gate" :: rest -> gate (String.concat "" rest)
  | [ _; "probe" ] -> probe ()
  | _ ->
      S.hold_gpu ();
      exit (Thumper.run "nx_metal" (List.map case rows))
