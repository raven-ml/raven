(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_next
module B = Nx_device.Buffer
module M = Nx_metal_device

let claims d = Option.is_some (M.of_device d)

let metal d =
  match M.of_device d with
  | Some m -> m
  | None -> invalid_arg (Nx_device.name d ^ " is no Metal device")

(* The words of a device's ["mtl_sel"] placeholder, handles then selectors, and
   the table of the buffers an encoder declares resident without a residency
   set. *)
type sels = {
  buffer : B.t;
  words : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  mutable table :
    (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
}

let int64_words n = Bigarray.(Array1.create int64 c_layout n)
let address a = Int64.of_nativeint (B.address (B.of_bigarray a))
let lock = Mutex.create ()
let opened = ref []

let sels d =
  Mutex.protect lock @@ fun () ->
  match List.assq_opt d !opened with
  | Some s -> s
  | None ->
      let m = metal d in
      let n = List.length Ops_metal.handles + List.length Ops_metal.selectors in
      let buffer = B.create Nx_device.host Nx_dtype.Scalar.Int64 n in
      let words = B.bigarray Bigarray.int64 buffer in
      let table = int64_words 1 in
      List.iteri
        (fun i v -> words.{i} <- v)
        ([
           Int64.of_nativeint (M.queue m);
           Int64.of_nativeint (M.event m);
           Int64.of_nativeint (M.fence m);
           address table;
           0L;
         ]
        @ List.map
            (fun s -> Int64.of_nativeint (M.selector s))
            Ops_metal.selectors);
      let s = { buffer; words; table } in
      opened := (d, s) :: !opened;
      s

(* Without a residency set, the table holds the device's buffers as they are
   when a batch runs. *)
let resident m s () =
  let resources = M.resources m in
  let n = Array.length resources in
  if n > Bigarray.Array1.dim s.table then s.table <- int64_words n;
  Array.iteri (fun i r -> s.table.{i} <- Int64.of_nativeint r) resources;
  s.words.{3} <- address s.table;
  s.words.{4} <- Int64.of_int n

let host_words b =
  match B.borrow Nx_device.host b with
  | Ok h -> B.bigarray Bigarray.int64 h
  | Error why -> invalid_arg why

(* The indirect command buffer of [cmds], in a buffer of [d] that holds its
   commands' arguments, with the words from byte [header]: the indirect command
   buffer, each command, and each pipeline they use. *)
let new_icb d (cmds : Ops_metal.command list) header =
  let pipes =
    List.fold_left
      (fun ps (c : Ops_metal.command) ->
        if List.mem (c.lib, c.name) ps then ps else ps @ [ (c.lib, c.name) ])
      [] cmds
  in
  let n = List.length cmds in
  let buf =
    B.create d Nx_dtype.Scalar.UInt8 (header + (8 * (1 + n + List.length pipes)))
  in
  let program (binary, name) =
    match Nx_device.Program.load d ~binary ~name with
    | Ok p -> p
    | Error why -> failwith why
  in
  let triple = function
    | [ x; y; z ] -> (x, y, z)
    | _ -> invalid_arg "a Metal launch size has three axes"
  in
  let command (c : Ops_metal.command) =
    {
      M.program = program (c.lib, c.name);
      offset = c.offset;
      global = triple c.global;
      local = triple c.local;
    }
  in
  match M.indirect_commands (metal d) buf (List.map command cmds) with
  | Error why -> failwith why
  | Ok (icb, commands) ->
      let words = host_words buf in
      List.iteri
        (fun i v -> words.{(header / 8) + i} <- Int64.of_nativeint v)
        ((icb :: commands)
        @ List.map (fun p -> Nx_device.Program.handle (program p)) pipes);
      buf

(* A batch's slots start zeroed: its host program reads the stamps it wrote. *)
let new_slots d n =
  let buf = B.create ~pinned:true d Nx_dtype.Scalar.UInt8 n in
  Bigarray.Array1.fill (host_words buf) 0L;
  buf

(* The word that holds the address of objc_msgSend, which host programs call:
   the engine's, as nx.device binds the Objective-C runtime. *)
let msg_send =
  lazy
    (let word = B.create Nx_device.host Nx_dtype.Scalar.Int64 1 in
     (B.bigarray Bigarray.int64 word).{0} <- Int64.of_nativeint M.msg_send;
     word)

(* The storage of the placeholders Metal's commands name: those of the device
   [name], and objc_msgSend's word on the host. *)
let placeholder name d u =
  let on_device =
    match Ops.device u with
    | Some (Single n) | Some (Multi [ n ]) -> n = name
    | _ -> false
  in
  match (Ops.tag u, Ops_metal.icb u) with
  | Some (Tuple [ String "cfunc"; String "metal"; String "objc_msgSend" ]), _ ->
      Some (Mutex.protect lock (fun () -> Lazy.force msg_send))
  | _ when not on_device -> None
  | Some (String "mtl_sel"), _ -> Some (sels d).buffer
  | Some (String "slots"), _ ->
      Some (new_slots d (Ops.max_numel u * Dtype.itemsize (Ops.dtype u)))
  | _, Some (cmds, header) -> Some (new_icb d cmds header)
  | _ -> None

let queues devices name d =
  match M.of_device d with
  | None -> None
  | Some m ->
      (* Any name of the host names it: the first one serves. *)
      let host =
        match
          List.find_opt (fun (_, h) -> h == Nx_device.host_of d) devices
        with
        | Some (host, _) -> host
        | None ->
            invalid_arg
              (Printf.sprintf "Tolk_next_engine.device: no device is %s's host"
                 name)
      in
      let residency_set = Option.is_some (M.residency_set m) in
      let queues =
        Ops_metal.queues ~host ~arch:(Nx_device.arch d) ~residency_set
      in
      let submitting = if residency_set then ignore else resident m (sels d) in
      Some (queues, placeholder name d, submitting)
