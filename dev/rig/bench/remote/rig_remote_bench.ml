(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

external tune : Unix.file_descr -> unit = "rig_remote_bench_tune"

external ask : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_ask"

external echo : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_echo"

external stream_open : Unix.file_descr -> Unix.file_descr -> int -> nativeint
  = "rig_remote_bench_stream_open"

external stream_run : nativeint -> int = "rig_remote_bench_stream_run"
external stream_close : nativeint -> unit = "rig_remote_bench_stream_close"

external rail_run : Rig_remote_abi.area -> Rig_remote_abi.area -> int -> int
  = "rig_remote_bench_rail_run"

external rail_answer : Rig_remote_abi.area -> Rig_remote_abi.area -> int -> int
  = "rig_remote_bench_rail_answer"

(* Requests *)

let alloc = Wire.Alloc { id = 1; device = 0; memory = `Device; bytes = 4096 }

(* A header of 9 bytes, the kind, id, device, memory and bytes; the answer's:
   the header, 0 and the [bool]. *)
let request_bytes = 9 + 1 + 8 + 8 + 1 + 8
let answer_bytes = 9 + 1 + 1

(* The agent *)

let rail l ~id ~send ~receive =
  let e = Link.rail l ~id ~send ~receive in
  let rec answer c =
    if rail_answer e.counts e.counts c = 0 then answer (c + 1)
  in
  ignore (Thread.create answer 1)

let reply : type r.
    Link.t ->
    (int, Rig_remote_abi.area) Hashtbl.t ->
    r Wire.request ->
    (r, string) result =
 fun l memory -> function
  | Wire.Alloc { id; bytes; _ } ->
      Hashtbl.replace memory id (Array1.create char c_layout bytes);
      Ok true
  | Wire.Rail { id; send; receive; _ } ->
      rail l ~id ~send ~receive;
      Ok ()
  | _ -> Error "the bench's agent serves only allocations and rails"

let region memory id offset bytes =
  Array1.sub (Hashtbl.find memory id) offset bytes

(* Runs a hand-over's copies in order; [local] holds the bytes of those from the
   controller's memory. *)
let run l memory (h : Wire.handover) local =
  let k = ref 0 in
  let part = function
    | Wire.Copy { src = Wire.Local; dst = Wire.Region { id; offset }; bytes } ->
        Array1.blit local.(!k) (region memory id offset bytes);
        incr k
    | Wire.Copy { src = Wire.Region { id; offset }; dst = Wire.Local; bytes } ->
        Link.bytes l ~device:h.device ~value:h.value
          (region memory id offset bytes)
    | Wire.Copy
        {
          src = Wire.Region { id = s; offset = so };
          dst = Wire.Region { id = d; offset = doff };
          bytes;
        } ->
        Array1.blit (region memory s so bytes) (region memory d doff bytes)
    | Wire.Copy { src = Wire.Local; dst = Wire.Local; _ } ->
        invalid_arg "a copy within the controller's memory"
    | Wire.Words _ -> failwith "the bench's agent runs no code"
  in
  Array.iter part h.parts;
  Link.word l ~device:h.device h.value

let agent l =
  let memory = Hashtbl.create 8 in
  let rec serve () =
    match Link.next l with
    | Ok (Wire.Request r) ->
        Link.answer l r (reply l memory r);
        serve ()
    | Ok (Wire.Handover (h, local)) ->
        run l memory h local;
        serve ()
    | Ok (Wire.Drop id) ->
        Hashtbl.remove memory id;
        serve ()
    | Ok Wire.Close | Error _ -> ()
  in
  serve ()
