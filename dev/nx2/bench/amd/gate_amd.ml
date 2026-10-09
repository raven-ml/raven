(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The gate's table: each row's device time per run under two protocols, its
   rate, and its distance to the floor kernel that moves its bytes, timed in the
   same run, with the compute unit clock each protocol ended at.

   gate_amd.exe [PATTERN...]

   A row runs if its name starts with a PATTERN, or is a PATTERN that ends with
   $ but for it; every row runs without one. First the GPU's clock settles on
   the row: it runs 100 ms at a time, at most ten times, until two clocks read
   after it agree within 1%. Then:

   - idle: the median of 30 measurements, each of enough runs to span 10 ms of
   the GPU's time, after the GPU idled 20 ms, as the reference's ref.py times
   PyTorch, so that both run in the same power state; - sustained: the median of
   10 such measurements back to back, after 1 s of the row without pause, as a
   training loop runs it: a GPU busy without pause reaches its power cap and
   lowers its clock. *)

module S = Nx_amd_support

let strf = Printf.sprintf
let span = 10e-3
let warm = 0.1
let idle = 20e-3

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

(* Seconds per run of [r] when idle and sustained, and the clock each ended
   at. *)
let time g r =
  let t = S.device_time g r ~count:3 in
  let runs s = Int.max 3 (int_of_float (Float.ceil (s /. t))) in
  let rec settle last tries =
    ignore (S.device_time g r ~count:(runs warm));
    let c = S.cu_clock g in
    if tries > 1 && Float.abs (c -. last) > 0.01 *. c then settle c (tries - 1)
  in
  settle 0. 10;
  let measure () = S.device_time g r ~count:(runs span) in
  let idle =
    median
      (List.init 30 (fun _ ->
           Unix.sleepf idle;
           measure ()))
  in
  let idle_clock = S.cu_clock g in
  ignore (S.device_time g r ~count:(runs 1.));
  let sustained = median (List.init 10 (fun _ -> measure ())) in
  ((idle, idle_clock), (sustained, S.cu_clock g))

let rate (p : Rows.prepared) t =
  match p.work with
  | Bytes 0 -> ""
  | Bytes n -> strf "%7.1f GB/s" (float n /. t *. 1e-9)
  | Flops f -> strf "%7.1f TF/s" (f /. t *. 1e-12)

let row g (r : Rows.row) =
  let p = r.make g in
  let (t, c), (ts, cs) = time g p.run in
  let floor =
    match p.floor with
    | None -> ""
    | Some f ->
        let (tf, _), _ = time g f in
        strf "floor %9.2f us  x%.2f" (tf *. 1e6) (t /. tf)
  in
  Printf.printf
    "%-34s idle %10.2f us %s [%.0f MHz]  sustained %10.2f us %s [%.0f MHz]  %s\n\
     %!"
    r.name (t *. 1e6) (rate p t) c (ts *. 1e6) (rate p ts) cs floor

let () =
  let patterns = List.tl (Array.to_list Sys.argv) in
  let selected (r : Rows.row) =
    patterns = []
    || List.exists
         (fun p ->
           match String.ends_with ~suffix:"$" p with
           | true -> String.sub p 0 (String.length p - 1) = r.name
           | false -> String.starts_with ~prefix:p r.name)
         patterns
  in
  if Rig_amd_amdgpu.count () = 0 then begin
    print_endline "nx.amd gate: no AMD GPU";
    exit 0
  end;
  S.hold_gpu ();
  let g = S.gpu () in
  let kernels, bytes = S.library_size g in
  Printf.printf
    "# %s, %d compute units; nx.amd: %d kernels, contract's, %d bytes\n%!"
    (S.arch g) (S.cus g) kernels bytes;
  List.iter (row g) (List.filter selected Rows.all)
