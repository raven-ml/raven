(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = image

(* Places [code] at the start of [d]'s code memory [e], whose region is
   [region]: by a copy on [d]'s copy engine from pinned memory, or by the host
   where [d] runs no copy, after [d]'s queued work either way, and waits for
   it. *)
let place d (e : entry) region code =
  let n = String.length code in
  let l = Memory.locate region in
  let m =
    Memory.make ?host:l.host ?address:l.address ~handle:l.handle d e
  in
  let dst = Buffer.of_memory m n in
  match d.live.copy_engine with
  | Some _ ->
      let src = Buffer.create ~memory:Buffer.Pinned d n in
      Buffer.blit_string code 0 src 0 n;
      Copy.queued d ~src ~dst
  | None ->
      if l.host = None then
        invalid_argf "Rig.Image.load: %s's code memory has no host address"
          d.name;
      Dev.wait d (Dev.submitted d);
      Buffer.blit_string code 0 dst 0 n

(* The driver's image of [binary] on [d], or the host's linked code, and the
   memory its code lies in where [d]'s memory holds it. *)
let loaded d binary =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      match Dev.counted d Dev.Any (fun () -> D.image h binary) with
      | Error why -> Error (strf "%s: %s" d.name why)
      | Ok (Rig_edge.Loaded i) -> Ok (Loaded { m; h; i }, None)
      | Ok (Place (n, lay)) -> (
          let e = Memory.alloc_entry d Rig_edge.Device n in
          (* [d]'s regions are of [d]'s region type. *)
          match e.backing with
          | Driver_memory { region = Region { r; rid = rid'; _ } as region; _ }
            -> (
              match Type.Id.provably_equal rid rid' with
              | Some Type.Equal -> (
                  (* A failure gives the code memory back once [d]'s work that
                     may use it is done, or, lost, once [d] counts as stopped,
                     and raises its own exception. *)
                  match
                    let i, code = Dev.counted d Dev.Any (fun () -> lay r) in
                    (try place d e region code
                     with x ->
                       Memory.unload d (Loaded { m; h; i }) None;
                       raise x);
                    i
                  with
                  | i -> Ok (Loaded { m; h; i }, Some e)
                  | exception x ->
                      Memory.retire d e;
                      raise x)
              | None -> assert false)
          | Io_memory _ | Kept _ -> assert false))
  | Host -> (
      match Host_code.link binary with
      | Ok c -> Ok (Linked c, None)
      | Error why -> Error (strf "%s: %s" d.name why))
  | Io _ -> invalid_argf "Rig.Image.load: %s loads no code" d.name

let load d binary =
  if Dev.is_lost d then Dev.raise_lost d;
  match loaded d binary with
  | Error _ as e -> e
  | Ok (loaded, code) ->
      let bytes = match code with Some e -> e.bytes | None -> 0 in
      let itoken =
        Memory.token d.live.release (Image (loaded, code)) bytes
          (Memory.room d) (-1)
      in
      if itoken == Memory.no_token then begin
        Memory.unload d loaded code;
        Dev.raise_lost d
      end;
      let image = { idev = d; loaded; itoken } in
      if Profile.enabled () then
        Profile.record (Load { image; binary; time = Profile.now () });
      Ok image

let device image = image.idev

let mem image f = Option.is_some (Memory.kernel_entry image f)

let entry image f =
  Option.map (fun (e : Rig_edge.entry) -> e.code) (Memory.kernel_entry image f)

let found fn image f =
  match Memory.kernel_entry image f with
  | Some e -> e
  | None -> Printf.ksprintf invalid_arg "Rig.Image.%s: no function %S" fn f

let params image f = (found "params" image f).params
let threads image f = (found "threads" image f).threads
