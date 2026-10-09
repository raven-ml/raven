(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The gate's table: each row's device time per run, its rate, and its distance
   to the floor kernel that moves its bytes, timed in the same run, with the
   compute unit's clock the row ended at.

   gate_amd.exe [PATTERN...]

   A row runs if its name contains a PATTERN, every row without one. A time is
   the median of five measurements, each of enough runs to span 10 ms of the
   GPU's time, once the GPU's clock has settled on the row: it runs 100 ms at a
   time, at most ten times, until two clocks read after it agree within 1%.
   After an idle spell the clock climbs over some 100 ms of work. *)

module S = Nx_amd_support

let strf = Printf.sprintf
let span = 10e-3
let warm = 0.1
let measurements = 5

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

(* Seconds per run of [r]. *)
let time g r =
  let t = S.device_time g r ~count:3 in
  let runs s = Int.max 3 (int_of_float (Float.ceil (s /. t))) in
  let rec settle last tries =
    ignore (S.device_time g r ~count:(runs warm));
    let c = S.cu_clock g in
    if tries > 1 && Float.abs (c -. last) > 0.01 *. c then settle c (tries - 1)
  in
  settle 0. 10;
  median
    (List.init measurements (fun _ -> S.device_time g r ~count:(runs span)))

let rate (p : Rows.prepared) t =
  match p.work with
  | Bytes 0 -> ""
  | Bytes n -> strf "%7.1f GB/s" (float n /. t *. 1e-9)
  | Flops f -> strf "%7.1f TF/s" (f /. t *. 1e-12)

let row g (r : Rows.row) =
  let p = r.make g in
  let t = time g p.run in
  let clock = S.cu_clock g in
  let floor =
    match p.floor with
    | None -> ""
    | Some f ->
        let tf = time g f in
        strf "floor %9.2f us  x%.2f" (tf *. 1e6) (t /. tf)
  in
  Printf.printf "%-28s %10.2f us  %s  %s  [%.0f MHz]\n%!" r.name (t *. 1e6)
    (rate p t) floor clock

let () =
  let patterns = List.tl (Array.to_list Sys.argv) in
  let selected (r : Rows.row) =
    patterns = []
    || List.exists
         (fun p ->
           let n = String.length p and m = String.length r.name in
           let rec at i =
             i + n <= m && (String.sub r.name i n = p || at (i + 1))
           in
           at 0)
         patterns
  in
  if Rig_amd_amdgpu.count () = 0 then begin
    print_endline "nx.amd gate: no AMD GPU";
    exit 0
  end;
  S.hold_gpu ();
  let g = S.gpu () in
  Printf.printf "# %s, %d compute units\n%!" (S.arch g) (S.cus g);
  List.iter (row g) (List.filter selected Rows.all)
