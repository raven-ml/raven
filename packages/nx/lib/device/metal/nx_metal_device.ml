(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external create_device : unit -> nativeint = "caml_nx_metal_create_device"
external arch : nativeint -> string = "caml_nx_metal_arch"
external working_set : nativeint -> int = "caml_nx_metal_working_set"
external unified : nativeint -> bool = "caml_nx_metal_unified"
external new_queue : nativeint -> nativeint = "caml_nx_metal_new_queue"
external new_event : nativeint -> nativeint = "caml_nx_metal_new_event"
external new_fence : nativeint -> nativeint = "caml_nx_metal_new_fence"

external new_residency_set : nativeint -> nativeint -> nativeint
  = "caml_nx_metal_new_residency_set"

external residency : nativeint -> nativeint -> bool -> unit
  = "caml_nx_metal_residency"

module Driver = Nx_device.Driver
module Region = Driver.Region

(* A buffer's host address, GPU address, [MTLBuffer] and size. *)
type buffer = (nativeint * nativeint * nativeint * int) option

external alloc : nativeint -> int -> buffer = "caml_nx_metal_alloc"
external wrap : nativeint -> nativeint -> int -> buffer = "caml_nx_metal_wrap"
external release : nativeint -> unit = "caml_nx_metal_release"

external pipeline : nativeint -> string -> string -> nativeint
  = "caml_nx_metal_pipeline"

external signaled : nativeint -> int = "caml_nx_metal_signaled"
external wait : nativeint -> int -> int -> bool = "caml_nx_metal_wait"
external cycle_pool : unit -> unit = "caml_nx_metal_cycle_pool"
external resolve : nativeint -> unit = "caml_nx_metal_resolve"

type handles = {
  device : nativeint;
  queue : nativeint;
  event : nativeint;
  fence : nativeint;
  residency_set : nativeint option;
}

type metal = {
  dev : Nx_device.t;
  handles : handles;
  resources : (nativeint, unit) Hashtbl.t;
      (* the buffers to declare resident, without a residency set *)
}

let opened = Atomic.make None
let lock = Mutex.create ()

let count () =
  match Atomic.get opened with
  | Some _ -> 1
  | None ->
      let d = create_device () in
      if d = 0n then 0
      else begin
        release d;
        1
      end

let name i = if i = 0 then "METAL" else Printf.sprintf "METAL:%d" i

let open_metal mtl =
  let queue = new_queue mtl in
  let residency_set =
    match new_residency_set mtl queue with 0n -> None | set -> Some set
  in
  let handles =
    {
      device = mtl;
      queue;
      event = new_event mtl;
      fence = new_fence mtl;
      residency_set;
    }
  in
  let resources = Hashtbl.create 64 in
  let resident buffer add =
    match residency_set with
    | Some set -> residency set buffer add
    | None ->
        if add then Hashtbl.replace resources buffer ()
        else Hashtbl.remove resources buffer
  in
  let region =
    Option.map (fun (host, address, buffer, n) ->
        resident buffer true;
        Region.v ~host ~handle:buffer address n)
  in
  let free r =
    resident (Region.handle r) false;
    release (Region.handle r)
  in
  let mapping =
    if unified mtl then
      Some
        {
          Driver.map =
            (fun a n ->
              match region (wrap mtl a n) with
              | Some r -> Ok r
              | None -> Error "Metal cannot wrap it in a buffer");
          unmap = free;
        }
    else None
  in
  let signal =
    {
      Driver.signaled = (fun () -> signaled handles.event);
      wait = (fun v ~timeout_ms -> wait handles.event v timeout_ms);
    }
  in
  let dev =
    Driver.device ~name:(name 0) ~arch:(arch mtl) ~budget:(working_set mtl)
      ~completion:(Signal (fun ~timeline:_ -> signal))
      ~load:(fun ~binary ~entry ->
        match pipeline mtl binary entry with
        | p -> Ok p
        | exception Failure why -> Error why)
      ~synchronized:cycle_pool ~resolve
      (Host_visible
         { memory = { alloc = (fun n -> region (alloc mtl n)); free }; mapping })
  in
  { dev; handles; resources }

let get i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_metal_device.get: %d < 0" i);
  let refuse why = Error (name i ^ ": " ^ why) in
  Mutex.protect lock @@ fun () ->
  match Atomic.get opened with
  | Some m when i = 0 -> Ok m.dev
  | Some _ -> refuse "no such device; there is one Metal device"
  | None -> (
      match create_device () with
      | 0n -> refuse "no GPU of this machine supports Metal"
      | mtl when i > 0 ->
          release mtl;
          refuse "no such device; there is one Metal device"
      | mtl when arch mtl = "" ->
          release mtl;
          refuse "the GPU belongs to no supported GPU family"
      | mtl -> (
          match open_metal mtl with
          | m ->
              Atomic.set opened (Some m);
              Ok m.dev
          | exception Failure why ->
              release mtl;
              refuse why))

let v i = match get i with Ok d -> d | Error msg -> failwith msg

let metal fn d =
  match Atomic.get opened with
  | Some m when Nx_device.equal m.dev d -> m
  | _ ->
      invalid_arg
        (Printf.sprintf "Nx_metal_device.%s: %s is not a Metal device" fn
           (Nx_device.name d))

let handles d = (metal "handles" d).handles

let resources d =
  let m = metal "resources" d in
  Array.of_seq (Hashtbl.to_seq_keys m.resources)
