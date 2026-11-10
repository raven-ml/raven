(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = image
type func = Def.func

(* Places [code] at the start of [d]'s code memory [e], whose region is
   [region], after [d]'s queued work. A copy on [d]'s copy engine from memory
   both address, [d]'s Pinned memory or, where the host addresses [d]'s own,
   that, is queued and not waited for: work that runs the image is later work
   of [d], which starts once the copy completed (device order, rig_edge.mli),
   so a load behind a step's queued work returns at once. Where [d] runs no
   copy, the host writes the code once that work is done. *)
let place d (e : entry) region code =
  let n = String.length code in
  let l = Memory.locate region in
  let m =
    Memory.make ?host:l.host ?address:l.address ~handle:l.handle d e
  in
  let dst = Buffer.of_memory m n in
  match d.live.copy_engine with
  | Some _ ->
      let memory = if Buffer.host_addresses d then Buffer.Device else Pinned in
      let src = Buffer.create ~memory d n in
      Buffer.blit_string code 0 src 0 n;
      ignore (Submission.run_copy d ~waits:[||] ~src ~dst : int)
  | None ->
      if l.host = None then
        invalid_argf "Rig.Image.load: %s's code memory has no host address"
          d.name;
      Dev.wait d (Dev.submitted d);
      Buffer.blit_string code 0 dst 0 n

(* The token that unloads [l] once collected, which paces the collector by
   the [bytes] of its code memory. Raises [Lost] if [d]'s list is closed, as
   it is once [d] has stopped. *)
let image_token d l bytes =
  let t =
    Memory.token d.live.release (Image l) bytes (Memory.room d) (-1)
  in
  if t == Memory.no_token then Dev.raise_lost d;
  t

(* [t]'s image, which no caller will hold, goes on its list now, as [t]'s
   collection would put it, so the next drain unloads what it holds; then
   [x] is raised again. *)
let unheld t x =
  let bt = Printexc.get_raw_backtrace () in
  Memory.token_link t;
  Printexc.raise_with_backtrace x bt

(* The driver's image of [binary] on [d], or the host's linked code, with its
   token. A driver's image is recorded, with its token, before each call that
   may make it: [D.image], whose record serves an image the driver loads
   itself, and for an image the driver lays into [d]'s memory, [lay], whose
   record also holds the code memory, kept there before its driver makes it.
   Each call stores its answer in its record inside its count, so an
   exception once the call returned leaves the image to its token, which
   unloads it, then gives back its code memory once [d]'s work that may use
   it is done, or, lost, once [d] counts as stopped. [D.image]'s answer is
   stored as it is, allocating nothing. [lay]'s store wraps the image in its
   option: an exception at that allocation is one inside the driver's call,
   which loses [d] with what the call made, as rig.mli states. A record that
   serves no image goes on its list at once. *)
let loaded d binary =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      let l =
        { m; h; answer = Error ""; laid = None; code = ref Memory.no_entry }
      in
      let t = image_token d (Loaded l) 0 in
      let image () = l.answer <- D.image h binary in
      match
        Dev.counted d Dev.Any image ();
        l.answer
      with
      | exception x -> unheld t x
      | Error why ->
          Memory.token_link t;
          Error (strf "%s: %s" d.name why)
      | Ok (Rig_edge.Loaded _) -> Ok (Loaded l, t)
      | Ok (Place (n, lay)) -> (
          Memory.token_link t;
          let l =
            { m; h; answer = Error ""; laid = None; code = ref Memory.no_entry }
          in
          let t = image_token d (Loaded l) n in
          let laid () =
            let e = Memory.code_entry d n l in
            (* [d]'s regions are of [d]'s region type. *)
            match e.backing with
            | Driver_memory
                { region = Region { r = Some r; rid = rid'; _ } as region; _ }
              -> (
                match Type.Id.provably_equal rid rid' with
                | Some Type.Equal ->
                    let lay () =
                      let i, code = lay r in
                      l.laid <- Some i;
                      code
                    in
                    place d e region (Dev.counted d Dev.Any lay ())
                | None -> assert false)
            | Driver_memory _ | Io_memory _ | Kept _ -> assert false
          in
          match laid () with
          | () -> Ok (Loaded l, t)
          | exception x -> unheld t x))
  | Host -> (
      match Host_code.link binary with
      | Ok c -> Ok (Linked c, image_token d (Linked c) 0)
      | Error why -> Error (strf "%s: %s" d.name why))
  | Io -> invalid_argf "Rig.Image.load: %s loads no code" d.name

let load d binary =
  Dev.check_held "Image.load";
  if Dev.is_lost d then Dev.raise_lost d;
  match loaded d binary with
  | Error _ as e -> e
  | Ok (loaded, itoken) ->
      let image = { idev = d; loaded; itoken } in
      if Profile.enabled () then
        Profile.record (Load { image; binary; time = Profile.now () });
      Ok image

let device image = image.idev

let resolve image f =
  Dev.check_held "Image.resolve";
  Option.map
    (fun e -> { fimage = image; fname = f; fentry = e })
    (Memory.kernel_entry image f)

let prepare image f =
  Dev.check_held "Image.prepare";
  Memory.kernel_prepare image f

let name fn = fn.fname
let image fn = fn.fimage
let params fn = fn.fentry.params
let threads fn = fn.fentry.threads
let entry fn = fn.fentry.code
