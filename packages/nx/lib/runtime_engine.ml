(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
module B = Nx_device.Buffer

(* One buffer per device of the cell's placement, in its order. *)
type Nx_effect.storage += Runtime of B.t list

(* Devices *)

let lock = Mutex.create ()
let opened : (Nx_device.t * Nx_effect.device) list ref = ref []

let runtime_of d =
  Mutex.protect lock (fun () ->
      match List.find_opt (fun (_, d') -> d' == d) !opened with
      | Some (rd, _) -> rd
      | None ->
          invalid_arg
            ("Nx: " ^ Nx_effect.Device.name d ^ " is not a runtime device"))

let create d s n =
  try B.create (runtime_of d) s n
  with Nx_device.Out_of_memory (_, bytes) ->
    raise (Nx_effect.Device.Out_of_memory (d, bytes))

(* Reading *)

(* The lowest and one past the highest storage element [v] reaches. *)
let extent v =
  let lo = ref (View.offset v) and hi = ref (View.offset v) in
  Array.iteri
    (fun a n ->
      let s = (View.strides v).(a) * (n - 1) in
      if s < 0 then lo := !lo + s else hi := !hi + s)
    (View.shape v);
  (!lo, !hi + 1)

(* The elements of view [v] of [b]. Int4 storage is read whole: its elements may
   not start on a byte. *)
let read_view b v =
  let s = B.dtype b in
  let n = View.numel v in
  if n = 0 then B.create Nx_device.host s 0
  else
    let lo, hi =
      match s with
      | Nx_dtype.Scalar.Int4 | UInt4 -> (0, B.length b)
      | _ -> extent v
    in
    let span = B.create Nx_device.host s (hi - lo) in
    B.copy
      ~src:(B.view b ~offset:(lo * Nx_dtype.Scalar.bitsize s / 8) s (hi - lo))
      ~dst:span;
    if View.is_c_contiguous v && hi - lo = n && View.offset v = lo then span
    else
      Elements.gather span
        (View.create
           ~offset:(View.offset v - lo)
           ~strides:(View.strides v) (View.shape v))

let read : type a b. (a, b) Nx_effect.resident -> Nx_device.Buffer.t =
 fun r ->
  match Nx_effect.Cell.state r.r_cell with
  | Live (Runtime bufs) ->
      let holders = Nx_effect.Placement.devices r.r_cell.placement in
      let buffer_on d =
        List.nth bufs (Option.get (List.find_index (( == ) d) holders))
      in
      let shape = Nx_effect.global r.r_placement (View.shape r.r_view) in
      Nx_effect.assemble r
        (Array.map (fun n -> (0, n)) shape)
        (fun d v -> read_view (buffer_on d) v)
  | _ -> assert false (* nx reads held and consumed values itself *)

(* Placing *)

(* The elements of [t] in C order, as a host buffer of exactly them. *)
let elements t =
  let buf = Nx_backend.to_host t and v = Nx_backend.view t in
  if View.is_c_contiguous v && View.offset v = 0 then
    B.view buf ~offset:0 (B.dtype buf) (View.numel v)
  else Elements.gather buf v

let place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  let h = Nx_effect.host_of x in
  let dt = Nx_backend.dtype h and shape = View.shape (Nx_backend.view h) in
  let s = Nx_dtype.Scalar.of_dtype dt in
  let ds = Nx_effect.Placement.devices p in
  let windows = List.map (fun d -> Nx_effect.Placement.window p shape d) ds in
  let local = Nx_effect.extents (List.hd windows) in
  let n = Array.fold_left ( * ) 1 local in
  let bufs =
    List.map2
      (fun d w ->
        let b = create d s n in
        if n > 0 then B.copy ~src:(elements (Nx_backend.shrink h w)) ~dst:b;
        b)
      ds windows
  in
  Nx_effect.placed p dt (View.create local)
    (Nx_effect.cell ~placement:p ~length:n (Runtime bufs))

let engine = { Nx_effect.read; place }

let device rd =
  if Nx_device.equal rd Nx_device.host then Nx_effect.Device.host
  else
    Mutex.protect lock (fun () ->
        match
          List.find_opt (fun (rd', _) -> Nx_device.equal rd rd') !opened
        with
        | Some (_, d) -> d
        | None ->
            let d = Nx_effect.Device.make (Nx_device.name rd) engine in
            opened := (rd, d) :: !opened;
            d)
