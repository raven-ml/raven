(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A probe, run by hand on an AMD GPU, of what splits
   copy/amd/within-device-4K into two modes, and of boxes: [check] copies
   boxes with 1 MiB of slack past their span and compares every byte;
   [shapes] times box shapes; [series], [chunks] and [gaps] time 4 KiB
   copies four ways (Buffer.copy; a prepared submission waited by spinning
   on the word, by Point.wait, by Point.reached), per call, per chunk of
   1024 with the GPU's clock levels after each, and after a host idle gap.
   It is no bench row: it prints raw numbers. *)

module A = Rig_amd
module B = Rig.Buffer

external now : unit -> int = "probe_now" [@@noalloc]
external cpu : unit -> int = "probe_cpu" [@@noalloc]
external spin : int -> unit = "probe_spin" [@@noalloc]

let pr = Printf.printf
let n4k = 4096
let fast_below = 11_500

let read_file p =
  try In_channel.with_open_text p In_channel.input_all with _ -> "?"

(* The starred level of a pp_dpm_* file. *)
let star p =
  match
    List.find_opt
      (fun l -> String.contains l '*')
      (String.split_on_char '\n' (read_file p))
  with
  | Some l -> String.trim l
  | None -> "?"

let card =
  let d = "/sys/class/drm" in
  try
    Array.to_list (Sys.readdir d)
    |> List.filter (fun c ->
           String.length c = 5
           && String.sub c 0 4 = "card"
           && Sys.file_exists (Filename.concat d (c ^ "/device/pp_dpm_sclk")))
    |> List.map (fun c -> Filename.concat d (c ^ "/device/"))
    |> List.hd
  with _ -> "/nonexistent/"

let clocks () =
  String.concat " "
    (List.map
       (fun f -> f ^ "=" ^ star (card ^ "pp_dpm_" ^ f))
       [ "sclk"; "mclk"; "fclk"; "socclk" ])

let freq c =
  String.trim
    (read_file
       (Printf.sprintf "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq"
          c))

let pct a p =
  let b = Array.copy a in
  Array.sort compare b;
  b.(min (Array.length b - 1) (Array.length b * p / 100))

let pair d =
  let src = B.create d n4k and dst = B.create d n4k in
  B.copy ~src:(B.of_string (String.make n4k 'w')) ~dst:src;
  (src, dst)

(* Chunks of [k] calls of [f], each followed by a reading of the clocks:
   per chunk, its median and the clocks after it. *)
let chunks name ~k ~n f =
  let t = Array.make k 0 in
  for _ = 1 to 2000 do
    f ()
  done;
  let slow_chunks = ref 0 in
  for c = 1 to n do
    let c0 = cpu () in
    for i = 0 to k - 1 do
      let a = now () in
      f ();
      t.(i) <- now () - a
    done;
    let c1 = cpu () in
    let m = pct t 50 in
    let slow = Array.fold_left (fun s x -> if x >= fast_below then s + 1 else s) 0 t in
    if m >= fast_below then incr slow_chunks;
    pr "%s chunk %d p10 %d p50 %d p90 %d slow %d/%d cpu %d->%d %skHz %s\n%!"
      name c (pct t 10) m (pct t 90) slow k c0 c1 (freq c1) (clocks ())
  done;
  pr "%s slow chunks %d/%d\n%!" name !slow_chunks n

(* One series of [n] calls: per call its time; then the runs of each mode. *)
let series name ~n f =
  let t = Array.make n 0 and cs = Array.make n 0 in
  for _ = 1 to 2000 do
    f ()
  done;
  let start = now () in
  for i = 0 to n - 1 do
    let a = now () in
    f ();
    t.(i) <- now () - a;
    cs.(i) <- cpu ()
  done;
  let total = now () - start in
  let runs = ref [] and len = ref 1 in
  for i = 1 to n - 1 do
    if (t.(i) >= fast_below) = (t.(i - 1) >= fast_below) then incr len
    else begin
      runs := ((t.(i - 1) >= fast_below), !len) :: !runs;
      len := 1
    end
  done;
  runs := ((t.(n - 1) >= fast_below), !len) :: !runs;
  let runs = List.rev !runs in
  let hist = Array.make 41 0 in
  Array.iter (fun x -> let b = min 40 (x / 500) in hist.(b) <- hist.(b) + 1) t;
  pr "%s n %d total %.1f ms p10 %d p50 %d p90 %d p99 %d\n" name n
    (float total /. 1e6) (pct t 10) (pct t 50) (pct t 90) (pct t 99);
  Array.iteri
    (fun b c -> if c > 0 then pr "%s hist %5.1f-%5.1f us %d\n" name
        (float b *. 0.5) (float (b + 1) *. 0.5) c) hist;
  let long = List.filter (fun (_, l) -> l >= 100) runs in
  pr "%s runs %d, of 100 calls or more %d\n" name (List.length runs)
    (List.length long);
  List.iteri (fun i (slow, l) -> if i < 60 then pr "%s run %s %d\n" name
      (if slow then "slow" else "fast") l) long;
  let migrations = ref 0 in
  for i = 1 to n - 1 do if cs.(i) <> cs.(i - 1) then incr migrations done;
  pr "%s cpu migrations %d\n%!" name !migrations

let gaps = [ 0; 500; 1000; 2000; 3000; 4000; 5000; 6000; 8000; 10000; 15000;
             20000; 30000; 50000; 100000; 300000; 1000000 ]

(* Each gap: [n] calls of [f] each followed by a spin of the gap. *)
let gapped name ~n f =
  List.iter
    (fun g ->
      let t = Array.make n 0 in
      for _ = 1 to 200 do f (); spin g done;
      for i = 0 to n - 1 do
        let a = now () in
        f ();
        t.(i) <- now () - a;
        spin g
      done;
      let slow = Array.fold_left (fun s x -> if x >= fast_below then s + 1 else s) 0 t in
      pr "%s gap %d ns p10 %d p50 %d p90 %d slow %d/%d\n%!" name g (pct t 10)
        (pct t 50) (pct t 90) slow n)
    gaps

(* Boxes: one box copied between two buffers of [slack] bytes past the
   box's span, each byte checked against the host's model. *)
let pattern n seed = String.init n (fun i -> Char.chr ((seed + (i * 7)) land 255))

let check_box d ~so ~d_o ~width ~rows ~sp ~dp =
  let span off pitch = off + ((rows - 1) * pitch) + width in
  let slack = 1 lsl 20 in
  let ns = span so sp + slack and nd = span d_o dp + slack in
  let from = pattern ns 1 and into = pattern nd 2 in
  let src = B.create d ns and dst = B.create d nd in
  B.copy ~src:(B.of_string from) ~dst:src;
  B.copy ~src:(B.of_string into) ~dst;
  let box = { B.src_offset = so; dst_offset = d_o; width; rows;
              src_pitch = sp; dst_pitch = dp } in
  Rig.Point.wait (B.submit_copy ~src ~dst [| box |]);
  let model = Bytes.of_string into in
  for r = 0 to rows - 1 do
    Bytes.blit_string from (so + (r * sp)) model (d_o + (r * dp)) width
  done;
  let h = B.create Rig.host nd in
  B.copy ~src:dst ~dst:h;
  let got = Bytes.create nd in
  B.blit_to_bytes h 0 got 0 nd;
  let bad = ref 0 and first = ref (-1) in
  Bytes.iteri (fun i c -> if c <> Bytes.get model i then begin
      incr bad; if !first < 0 then first := i end) got;
  pr "box %d rows of %d at %d pitch %d into %d pitch %d: %s\n%!" rows width so
    sp d_o dp
    (if !bad = 0 then "exact" else Printf.sprintf "%d bytes wrong, first at %d" !bad !first);
  !bad = 0

let boxes_checked d =
  List.for_all
    (fun (so, d_o, width, rows, sp, dp) -> check_box d ~so ~d_o ~width ~rows ~sp ~dp)
    [
      (1024, 2048, 16, 4, 64, 64);
      (1025, 2050, 7, 4, 64, 68);
      (0, 0, 4, 3, 8, 12);
      (3, 1, 13, 5, 100, 4000);
      (0, 0, 4, 16383, 8, 8);
      (0, 0, 4, 16384, 8, 8);
      (0, 0, 4, 16385, 8, 8);
      (2, 2, 6, 65537, 8, 12);
      (0, 0, 16 * 65536, 2, 16 * 65536, 16 * 65536 + 16);
    ]

(* Box shapes timed: submit_copy then wait, the median of 21. *)
let shapes =
  [ (2, 1 lsl 20, (1 lsl 20) + 16); (64, 64 lsl 10, 128 lsl 10);
    (1024, 64 lsl 10, 128 lsl 10); (4096, 256, 512); (65536, 4, 64);
    (16384, 1, 4); (16384, 16, 32); (100_000, 4, 64);
    (64, 4096, 2 lsl 20) ]

let time_boxes d =
  List.iter
    (fun (rows, width, pitch) ->
      let n = rows * pitch in
      let src = B.create d n and dst = B.create d n in
      let box = { B.src_offset = 0; dst_offset = 0; width; rows;
                  src_pitch = pitch; dst_pitch = pitch } in
      let once () = Rig.Point.wait (B.submit_copy ~src ~dst [| box |]) in
      for _ = 1 to 5 do once () done;
      let t = Array.init 21 (fun _ -> let a = now () in once (); now () - a) in
      pr "shape %dx%d pitch %d: p50 %d ns min %d ns, %.2f GB/s\n%!" rows width
        pitch (pct t 50) (pct t 0)
        (float (rows * width) /. float (pct t 50)))
    shapes

(* The split of a call's time: [submit] host time up to its return, after
   the doorbell, then the wait; per mode of the whole call (fast below
   [fast_below]), the mean of each, and the minor words per call. *)
let split name ~n (submit : unit -> unit -> unit) =
  let ts = Array.make n 0 and tw = Array.make n 0 in
  for _ = 1 to 2000 do
    submit () ()
  done;
  let w0 = Gc.minor_words () in
  for i = 0 to n - 1 do
    let a = now () in
    let wait = submit () in
    let b = now () in
    wait ();
    let c = now () in
    ts.(i) <- b - a;
    tw.(i) <- c - b
  done;
  let words = (Gc.minor_words () -. w0) /. float n in
  let mean sel a =
    let s = ref 0 and k = ref 0 in
    Array.iteri (fun i x -> if sel i then begin s := !s + x; incr k end) a;
    if !k = 0 then (0, 0) else (!s / !k, !k)
  in
  let slow i = ts.(i) + tw.(i) >= fast_below in
  let mid i = ts.(i) + tw.(i) >= 7000 && ts.(i) + tw.(i) < 14000 in
  List.iter
    (fun (mode, sel) ->
      let s, k = mean sel ts and w, _ = mean sel tw in
      pr "%s split %s calls %d submit %d ns wait %d ns\n" name mode k s w)
    [ ("all", fun _ -> true); ("fast", fun i -> not (slow i) && not (mid i));
      ("7-14us", mid); ("slow", slow) ];
  pr "%s submit p10 %d p50 %d p90 %d, wait p50 %d p90 %d, %.1f minor words\n%!"
    name (pct ts 10) (pct ts 50) (pct ts 90) (pct tw 50) (pct tw 90) words

(* Per call of [f] in a profile: from the call to its work's start on the
   GPU, the work, and from its end to the call's return, by mode. *)
let gpu_split name ~n f =
  for _ = 1 to 2000 do
    f ()
  done;
  let t0 = Array.make n 0 and t1 = Array.make n 0 in
  let (), events =
    Rig.Profile.take (fun () ->
        for i = 0 to n - 1 do
          t0.(i) <- now ();
          f ();
          t1.(i) <- now ()
        done)
  in
  let works =
    List.filter_map
      (function Rig.Profile.Work { start; stop; _ } -> Some (start, stop) | _ -> None)
      events
    |> Array.of_list
  in
  pr "%s gpu events %d for %d calls\n" name (Array.length works) n;
  if Array.length works = n then begin
    let buckets = [ ("fast", 0, 7000); ("7-14us", 7000, 14000); ("slow", 14000, max_int) ] in
    List.iter
      (fun (mode, lo, hi) ->
        let k = ref 0 and pre = ref 0 and ex = ref 0 and post = ref 0 in
        Array.iteri
          (fun i (a, b) ->
            let total = t1.(i) - t0.(i) in
            if total >= lo && total < hi then begin
              incr k;
              pre := !pre + (a - t0.(i));
              ex := !ex + (b - a);
              post := !post + (t1.(i) - b)
            end)
          works;
        if !k > 0 then
          pr "%s gpu %s calls %d call->start %d ns work %d ns end->return %d ns\n%!"
            name mode !k (!pre / !k) (!ex / !k) (!post / !k))
      buckets
  end

(* Fine host gaps after each call of [f], and each mode's share. *)
let fine_gaps name ~n f =
  List.iter
    (fun g ->
      let t = Array.make n 0 in
      for _ = 1 to 500 do f (); spin g done;
      for i = 0 to n - 1 do
        let a = now () in
        f ();
        t.(i) <- now () - a;
        spin g
      done;
      let count lo hi =
        Array.fold_left (fun k x -> if x >= lo && x < hi then k + 1 else k) 0 t
      in
      pr "%s fine gap %d ns p50 %d: fast %d, 7-14us %d, slow %d of %d\n%!" name
        g (pct t 50) (count 0 7000) (count 7000 14000) (count 14000 max_int) n)
    [ 0; 100; 200; 300; 400; 500; 600; 800; 1000; 1500; 2000 ]

(* Spins until [reached ()], reading the clock once every [check] spins so
   that the spin's latency stays a spin's; past [patience_ns], [late ()]
   waits instead, which checks the hang bound. *)
let check = 4096
let patience_ns = 1_000_000_000

let spin_until reached late =
  let start = now () in
  let rec go i =
    if reached () then ()
    else if i mod check = 0 && now () - start > patience_ns then late ()
    else begin
      Domain.cpu_relax ();
      go (i + 1)
    end
  in
  go 1

let probe what { Rig_amd_support.d; g } =
  let src, dst = pair d in
  let copy () = B.copy ~src ~dst in
  (* A prepared submission of one copy, waited by spinning on the word. *)
  let run =
    Rig.Submission.(
      Run.make (make d [| copy ~src:(Buffer src) ~dst:(Buffer dst) |]))
  in
  let raw_spin () =
    let p = Rig.submit run ~buffers:[||] in
    let v = Rig.Point.value p in
    spin_until (fun () -> A.signaled g >= v) (fun () -> Rig.Point.wait p)
  in
  let raw_wait () = Rig.Point.wait (Rig.submit run ~buffers:[||]) in
  let poll_reached () =
    let p = Rig.submit run ~buffers:[||] in
    spin_until (fun () -> Rig.Point.reached p) (fun () -> Rig.Point.wait p)
  in
  (* The run Buffer.copy submits: one copy from slot 0 into slot 1. *)
  let slots =
    Rig.Submission.(
      Run.make (make d [| copy ~src:(Slot 0) ~dst:(Slot 1) |]))
  in
  let spin_point p =
    spin_until (fun () -> Rig.Point.reached p) (fun () -> Rig.Point.wait p)
  in
  let submit_slots () =
    let p = Rig.submit slots ~buffers:[| src; dst |] in
    fun () -> spin_point p
  in
  let submit_fixed () =
    let p = Rig.submit run ~buffers:[||] in
    fun () -> spin_point p
  in
  let host_copy =
    let a = B.create Rig.host n4k and b = B.create Rig.host n4k in
    fun () -> B.copy ~src:a ~dst:b
  in
  let fine () =
    split "host-copy" ~n:50_000 (fun () -> host_copy (); ignore);
    fine_gaps "slots" ~n:5000 (fun () -> submit_slots () ());
    fine_gaps "buffer-copy" ~n:5000 copy
  in
  let split_all () =
    split "buffer-copy" ~n:50_000 (fun () -> copy (); ignore);
    split "slots" ~n:50_000 submit_slots;
    split "fixed" ~n:50_000 submit_fixed;
    gpu_split "buffer-copy" ~n:20_000 copy;
    gpu_split "slots" ~n:20_000 (fun () -> submit_slots () ())
  in
  let fs = [ ("buffer-copy", copy); ("raw-spin", raw_spin);
             ("raw-wait", raw_wait); ("raw-reached", poll_reached) ] in
  pr "load %s" (read_file "/proc/loadavg");
  pr "perf_level %s" (read_file (card ^ "power_dpm_force_performance_level"));
  pr "clocks %s\n%!" (clocks ());
  match what with
  | "check" -> if boxes_checked d then 0 else 1
  | "shapes" -> time_boxes d; 0
  | "split" -> split_all (); 0
  | "fine" -> fine (); 0
  | "series" -> List.iter (fun (n, f) -> series n ~n:200_000 f) fs; 0
  | "chunks" -> List.iter (fun (n, f) -> chunks n ~k:1024 ~n:100 f) fs; 0
  | "gaps" -> List.iter (fun (n, f) -> gapped n ~n:3000 f) fs; 0
  | s -> (
      match List.assoc_opt s fs with
      | Some f ->
          series s ~n:200_000 f;
          chunks s ~k:1024 ~n:100 f;
          gapped s ~n:3000 f;
          0
      | None ->
          prerr_endline "check | shapes | series | chunks | gaps | <name>";
          2)

(* The device closes before the exit, an exception's included. *)
let () =
  let what = if Array.length Sys.argv > 1 then Sys.argv.(1) else "all" in
  exit (Rig_amd_support.with_ (probe what))
