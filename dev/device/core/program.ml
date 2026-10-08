(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

external blit_string : string -> int -> unit = "caml_device_core_blit_string"

type t = program

(* Places [code] at the start of the region [r] of [d]: by a copy on [d]'s copy
   queue from pinned memory, or by the host where [d] runs no copy, after [d]'s
   queued work either way. *)
let place d r code =
  let n = String.length code in
  let address, handle, host = Memory.region_info r in
  let stamps = Memory.stamps_new () in
  let entry = Memory.entry ~region:r d Memory.kept_kind n stamps in
  let dst =
    Buffer.of_memory
      (Memory.make ~host ~address ~handle d n entry)
      Scalar.UInt8 n
  in
  if Dev.copies d then begin
    let src = Buffer.create ~memory:Buffer.Pinned d Scalar.UInt8 n in
    blit_string code src.mem.host;
    let queue =
      Array.to_list d.queues |> List.find (String.starts_with ~prefix:"COPY:")
    in
    let part =
      { Submission.queue; after = [||]; work = Submission.Copy { src; dst } }
    in
    let p =
      Submission.submit
        (Submission.make ~reads:0 ~writes:0 ~waits:0 d [| part |])
    in
    Dev.wait d (Point.value p);
    Memory.stamps_unref stamps
  end
  else begin
    if host < 0 then
      invalid_argf
        "Device_core.Program.load: %s's code region has no host address" d.name;
    Dev.wait d (Dev.submitted d);
    blit_string code host;
    Memory.stamps_unref stamps
  end

let rec load_image d binary round =
  match d.kind with
  | Driver { m; h } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.image h binary) with
      | Ok (i, upload) ->
          let bytes =
            match upload with
            | None -> 0
            | Some (r, code) ->
                place d (Region { m; h; r }) code;
                String.length code
          in
          Ok (Image { m; h; i }, bytes)
      | Error (`Refused why) -> Error (strf "%s: %s" d.name why)
      | Error (`No_memory n) ->
          if round >= Memory.rounds then raise (Dev.Out_of_memory (d, n));
          Memory.reclaim d round;
          load_image d binary (round + 1))
  | _ -> invalid_argf "Device_core.Program.load: %s loads no code" d.name

let load d binary =
  if Dev.is_lost d then Dev.raise_lost d;
  Memory.drain d;
  match load_image d binary 1 with
  | Error _ as e -> e
  | Ok (image, bytes) ->
      Mutex.protect d.lock (fun () -> d.used <- d.used + bytes);
      Memory.note d;
      let ptoken =
        Memory.token d.release
          (Program (image, bytes))
          bytes (Memory.room d) (-1)
      in
      let p = { pdev = d; image; code_bytes = bytes; ptoken } in
      if Prof.enabled () then
        Prof.record (Load { program = p; binary; time = Prof.now () });
      Ok p

let device p = p.pdev

let entry p f =
  let (Image { m; i; _ }) = p.image in
  let module D = (val m) in
  Dev.counted p.pdev (fun () -> D.entry i f)
