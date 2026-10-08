(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let strf = Printf.sprintf

external read : int -> int -> string = "rig_amd_test_read"
external write : int -> string -> unit = "rig_amd_test_write"
external pages : int -> int = "rig_amd_test_pages"
external now_ns : unit -> int = "rig_amd_test_now"
external free_pages : int -> int -> unit = "rig_amd_test_free_pages"

type arg =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external fill_arg :
  nativeint -> nativeint -> int array -> int -> int -> int -> arg
  = "rig_amd_test_fill_arg_byte" "rig_amd_test_fill_arg"

external fill_entry : unit -> nativeint = "rig_amd_test_fill_entry"
external fill_address : arg -> int = "rig_amd_test_fill_address"
external data : arg -> int = "rig_amd_test_data"

(* The machine's GPU lock *)

external lock : string -> string -> int = "rig_amd_test_lock"

let gpu_lock = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let hold_gpu () = if Rig_amd_amdgpu.count () > 0 then take 0

(* The device gpu opened and rig's device over it, until a test stops it or rig
   loses it: one a failed test left open is stopped by the next gpu. Each open
   has a name of its own, since rig keeps a name's device after the driver's
   stop. *)
let opened = ref None
let opens = ref 0

let stop g =
  (match !opened with Some (o, _) when o == g -> opened := None | _ -> ());
  Rig_amd.stop g

let gpu () =
  if Rig_amd_amdgpu.count () = 0 then
    skip ~reason:"the machine has no AMD GPU" ();
  hold_gpu ();
  Option.iter (fun (o, _) -> stop o) !opened;
  incr opens;
  let g = ref None in
  let make () =
    Result.map
      (fun x ->
        g := Some x;
        x)
      (Rig_amd_amdgpu.open_ 0)
  in
  match Rig.open_ (module Rig_amd) ~name:(strf "AMD:test-%d" !opens) make with
  | Error why -> failwith why
  | Ok c ->
      let g = Option.get !g in
      opened := Some (g, c);
      g

let core g =
  match !opened with
  | Some (o, c) when o == g -> c
  | _ -> invalid_arg "Rig_amd_support.core: the device is not open"

let submit g parts =
  let s = Rig.Submission.make ~reads:0 ~writes:0 ~waits:0 (core g) parts in
  match Rig.submit s with
  | p -> Rig.Point.value p
  | exception (Rig.Lost _ as e) ->
      opened := None;
      raise e

let with_gpu f =
  let g = gpu () in
  let stop_left () =
    match !opened with Some (o, _) when o == g -> stop g | _ -> ()
  in
  Fun.protect ~finally:stop_left (fun () -> f g)

let wait g v =
  let rec loop () =
    let seen = Rig_amd.signaled g in
    if seen < v then begin
      Rig_amd.sleep g ~seen ~still_ms:200;
      loop ()
    end
  in
  loop ()

let still ?msg w x f ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal ?msg w x (f ())
  done

(* Fills *)

type fill = { entry : nativeint; arg : arg }

let fill ?(code = 0) ?(split = 0) (c : Rig_amd.capability) ws ~bytes =
  {
    entry = fill_entry ();
    arg = fill_arg c.place c.segment ws split bytes code;
  }

let fill_address f = fill_address f.arg

let fill_part ~queue ?(after = [||]) f ~units ~bytes =
  {
    Rig.Submission.queue;
    after;
    work =
      Fill
        {
          fill = f.entry;
          arg = Rig.Buffer.of_bigarray f.arg;
          ring_units = units;
          segment_bytes = bytes;
        };
  }

let words_part ~queue ?(after = [||]) ws =
  let b = Bigarray.(Array1.create int32 c_layout (Array.length ws)) in
  Array.iteri (fun i w -> b.{i} <- Int32.of_int w) ws;
  { Rig.Submission.queue; after; work = Words (Rig.Buffer.of_bigarray b) }

(* The C entries *)

module Edge = struct
  (* The ints the C side reads: queue, fill, argument, ring units, segment
     bytes, copy destination, source and bytes, the counts of [after] indices
     and of words, the indices, the words. *)
  type part = { ints : int array; keep : arg option }

  let index = function
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | q -> invalid_arg ("Rig_amd_support.Edge: queue " ^ q)

  let make ?keep ~queue ~after ?(fill = 0n) ?(arg = 0) ?(units = 0) ?(bytes = 0)
      ?(dst = 0) ?(src = 0) ?(copy = 0) words =
    let head =
      [|
        queue;
        Nativeint.to_int fill;
        arg;
        units;
        bytes;
        dst;
        src;
        copy;
        Array.length after;
        Array.length words;
      |]
    in
    {
      ints =
        Array.concat
          [ head; after; Array.map (fun w -> w land 0xffff_ffff) words ];
      keep;
    }

  let words ~queue ?(after = [||]) ws = make ~queue:(index queue) ~after ws

  let fill ~queue ?(after = [||]) f ~units ~bytes =
    make ~keep:f.arg ~queue:(index queue) ~after ~fill:f.entry ~arg:(data f.arg)
      ~units ~bytes [||]

  let copy ?(after = [||]) ~dst ~src n =
    make ~queue:1 ~after ~dst ~src ~copy:n [||]

  let raw ~queue ?(words = 0) ?(fill = false) ?(copy = 0) ?(after = [||]) () =
    let fill = if fill then fill_entry () else 0n in
    make ~queue ~after ~fill ~copy (Array.make words 0)

  external room_c : nativeint -> nativeint -> int array array -> int
    = "rig_amd_test_room"

  external submit_c :
    nativeint ->
    nativeint ->
    int ->
    int array ->
    int array array ->
    string option = "rig_amd_test_submit"

  let room g ps =
    match
      room_c Rig_amd.room_entry (Rig_amd.self g)
        (Array.map (fun p -> p.ints) ps)
    with
    | 0 -> `Fits
    | 1 -> `Later
    | _ -> `Never

  let submit g ~v ?(waits = [||]) ps =
    let w =
      Array.concat (Array.to_list (Array.map (fun (a, x) -> [| a; x |]) waits))
    in
    let r =
      submit_c Rig_amd.submit_entry (Rig_amd.self g) v w
        (Array.map (fun p -> p.ints) ps)
    in
    (* The fills' arguments lived through the call. *)
    Array.iter (fun p -> ignore (Sys.opaque_identity p.keep)) ps;
    match r with None -> `Ok | Some why -> `Failed why
end
