(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = image

(* Places [code] at the start of [d]'s code memory [e]: by a copy on [d]'s copy
   queue from pinned memory, or by the host where [d] runs no copy, after [d]'s
   queued work either way, and waits for it. *)
let place d (e : entry) code =
  let n = String.length code in
  let address, handle, host =
    match e.region with Some r -> Memory.region_info r | None -> (-1, 0n, -1)
  in
  match d.copy_queue with
  | Some queue ->
      let dst =
        Buffer.of_memory (Memory.make ~host ~address ~handle d e.bytes e) n
      in
      let src = Buffer.create ~memory:Buffer.Pinned d n in
      Buffer.blit_string code 0 src.mem.host n;
      Copy.queued d queue ~src ~dst
  | None ->
      if host < 0 then
        invalid_argf "Rig.Image.load: %s's code memory has no host address"
          d.name;
      Dev.wait d (Dev.submitted d);
      Buffer.blit_string code 0 host n

(* The driver's image of [binary] on [d], and the memory its code lies in where
   [d]'s memory holds it. *)
let loaded d binary =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.image h binary) with
      | Error why -> Error (strf "%s: %s" d.name why)
      | Ok (`Loaded i) -> Ok (Loaded { m; h; i }, None)
      | Ok (`Place (n, lay)) -> (
          let e = Memory.alloc_entry d Device n in
          (* [d]'s regions are of [d]'s region type. *)
          match e.region with
          | Some (Region { r; rid = rid'; _ }) -> (
              match Type.Id.provably_equal rid rid' with
              | Some Type.Equal -> (
                  (* A failure gives the code memory back once [d]'s work that
                     may use it is done, or, lost, once [d] counts as stopped,
                     and raises its own exception. *)
                  match
                    let i, code = lay r in
                    (try place d e code
                     with x ->
                       (try Memory.unload d (Loaded { m; h; i })
                        with Dev.Lost _ -> ());
                       raise x);
                    i
                  with
                  | i -> Ok (Loaded { m; h; i }, Some e)
                  | exception x ->
                      Memory.retire d e;
                      raise x)
              | None -> assert false)
          | None -> assert false))
  | _ -> invalid_argf "Rig.Image.load: %s loads no code" d.name

let load d binary =
  if Dev.is_lost d then Dev.raise_lost d;
  match loaded d binary with
  | Error _ as e -> e
  | Ok (loaded, code) ->
      let bytes = match code with Some e -> e.bytes | None -> 0 in
      let itoken =
        Memory.token d.release (Image (loaded, code)) bytes (Memory.room d) (-1)
      in
      let image = { idev = d; loaded; itoken } in
      if Prof.enabled () then
        Prof.record (Load { image; binary; time = Prof.now () });
      Ok image

let device image = image.idev

let entry image f =
  let (Loaded { m; i; _ }) = image.loaded in
  let module D = (val m) in
  Dev.counted image.idev (fun () -> D.entry i f)
