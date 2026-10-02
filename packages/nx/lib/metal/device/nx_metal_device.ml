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
external msg_send : unit -> nativeint = "caml_nx_metal_msg_send"
external selector : string -> nativeint = "caml_nx_metal_selector"
external max_threads : nativeint -> int = "caml_nx_metal_max_threads"

(* The indirect command buffer, then each command, of the dispatches of the
   pipelines on the buffer, each with its offset and its six launch sizes. *)
external new_icb :
  nativeint -> nativeint -> nativeint array -> int array -> nativeint array
  = "caml_nx_metal_new_icb"

type t = {
  dev : Nx_device.t;
  mtl : nativeint;
  queue : nativeint;
  event : nativeint;
  fence : nativeint;
  residency_set : nativeint option;
  resources : (nativeint, unit) Hashtbl.t;
      (* the buffers to declare resident, without a residency set *)
}

(* The opens of the GPU, the live one first: an open of a lost one is a new
   device, over the same [MTLDevice], with a queue and an event of its own. *)
let opened = Atomic.make []
let lock = Mutex.create ()

let count () =
  match Atomic.get opened with
  | _ :: _ -> 1
  | [] ->
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
  let event = new_event mtl and fence = new_fence mtl in
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
        (Driver.Pages
           {
             map =
               (fun a n ->
                 match region (wrap mtl a n) with
                 | Some r -> Ok r
                 | None -> Error "Metal cannot wrap it in a buffer");
             unmap = free;
           })
    else None
  in
  let signal =
    {
      Driver.signaled = (fun () -> signaled event);
      wait = (fun v ~timeout_ms -> wait event v timeout_ms);
    }
  in
  let dev =
    Driver.device ~name:(name 0) ~arch:(arch mtl) ~budget:(working_set mtl)
      ~completion:(Signal (fun ~timeline:_ -> signal))
      ~load:(fun ~binary ->
        (* A function's pipeline, released with the image. *)
        let made = ref [] in
        let entry name =
          match pipeline mtl binary name with
          | p ->
              made := p :: !made;
              Ok p
          | exception Failure why -> Error why
        in
        Ok
          {
            Driver.code = None;
            entry;
            unload = (fun () -> List.iter release !made);
          })
      ~synchronized:cycle_pool ~resolve
      (Host_visible
         { memory = { alloc = (fun n -> region (alloc mtl n)); free }; mapping })
  in
  { dev; mtl; queue; event; fence; residency_set; resources }

let get i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_metal_device.get: %d < 0" i);
  let refuse why = Error (name i ^ ": " ^ why) in
  let reopen m =
    match open_metal m.mtl with
    | m' ->
        Atomic.set opened (m' :: Atomic.get opened);
        Ok m'.dev
    | exception Failure why -> refuse why
  in
  Mutex.protect lock @@ fun () ->
  match Atomic.get opened with
  | _ :: _ when i > 0 -> refuse "no such device; there is one Metal device"
  | m :: _ when Nx_device.lost m.dev = None -> Ok m.dev
  | m :: _ -> reopen m
  | [] -> (
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
              Atomic.set opened [ m ];
              Ok m.dev
          | exception Failure why ->
              release mtl;
              refuse why))

let of_device d =
  List.find_opt (fun m -> Nx_device.equal m.dev d) (Atomic.get opened)

let queue m = m.queue
let event m = m.event
let fence m = m.fence
let residency_set m = m.residency_set

let resource m b =
  if Nx_device.equal (Nx_device.Buffer.device b) m.dev then
    Some (Region.handle (Region.of_buffer b))
  else None

let resources m = Array.of_seq (Hashtbl.to_seq_keys m.resources)
let msg_send = msg_send ()

type command = {
  program : Nx_device.Program.t;
  offset : int;
  global : int * int * int;
  local : int * int * int;
}

let indirect_commands m args cmds =
  let invalid fmt =
    Printf.ksprintf invalid_arg ("Nx_metal_device.indirect_commands: " ^^ fmt)
  in
  let buffer =
    match resource m args with
    | Some b -> b
    | None -> invalid "the arguments are not on %s" (Nx_device.name m.dev)
  in
  let pipeline c =
    if not (Nx_device.equal (Nx_device.Program.device c.program) m.dev) then
      invalid "%s is not loaded on %s"
        (Nx_device.Program.name c.program)
        (Nx_device.name m.dev);
    Nx_device.Program.handle c.program
  in
  let pipelines = List.map pipeline cmds in
  let too_large c p =
    let x, y, z = c.local in
    if x * y * z > max_threads p then
      Some
        (Printf.sprintf "%s: local size (%d, %d, %d) bigger than %d"
           (Nx_device.Program.name c.program)
           x y z (max_threads p))
    else None
  in
  match List.find_map Fun.id (List.map2 too_large cmds pipelines) with
  | Some why -> Error why
  | None -> (
      let launch c =
        let gx, gy, gz = c.global and lx, ly, lz = c.local in
        [| Nx_device.Buffer.offset args + c.offset; gx; gy; gz; lx; ly; lz |]
      in
      match
        new_icb m.mtl buffer (Array.of_list pipelines)
          (Array.concat (List.map launch cmds))
      with
      | exception Failure why -> Error why
      | objects ->
          let objects = Array.to_list objects in
          let programs = List.map (fun c -> c.program) cmds in
          Nx_device.Driver.depends args (fun () ->
              ignore (Sys.opaque_identity programs);
              List.iter release objects);
          Ok (List.hd objects, List.tl objects))
