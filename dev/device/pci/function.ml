(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type addressing = Machine.addressing = Physical | Iommu

(* What the process holds of a function, so that misuse is refused before its
   machine is asked: its BAR windows, DMA memory and pins. The tables are used
   from any domain under [lock]. *)
type t = {
  machine : Machine.t;
  bus : string;
  fn : Machine.fn;
  mutable released : bool;
  lock : Mutex.t;
  maps : (int, Window.t) Hashtbl.t; (* by address *)
  dmas : (int, Window.t) Hashtbl.t; (* by address *)
  pins : (int * int, int) Hashtbl.t; (* counts, by (address, bytes) *)
}

let take machine ~lock bus =
  Machine.take machine ~lock bus
  |> Result.map (fun fn ->
      {
        machine;
        bus;
        fn;
        released = false;
        lock = Mutex.create ();
        maps = Hashtbl.create 16;
        dmas = Hashtbl.create 16;
        pins = Hashtbl.create 16;
      })

let machine f = f.machine
let bus f = f.bus
let addressing f = f.fn.addressing
let released f = f.released

let release f =
  if not f.released then begin
    f.released <- true;
    let maps =
      Mutex.protect f.lock (fun () ->
          Hashtbl.to_seq_values f.maps |> List.of_seq)
    in
    List.iter f.fn.unmap maps;
    Mutex.protect f.lock (fun () -> Hashtbl.reset f.maps);
    f.fn.release ()
  end

(* Configuration space *)

let width fn n =
  if n <> 1 && n <> 2 && n <> 4 then
    invalid_arg (Printf.sprintf "Function.%s: %d bytes" fn n)

let config f off n =
  width "config" n;
  f.fn.config off n

let set_config f off n x =
  width "set_config" n;
  f.fn.set_config off n x

(* BARs *)

let bar f i = f.fn.bar i

let map ?(off = 0) ?length f i =
  let size =
    match f.fn.bar i with
    | Some (_, size) -> size
    | None ->
        invalid_arg (Printf.sprintf "Function.map: %s has no BAR %d" f.bus i)
  in
  let length = Option.value length ~default:(size - off) in
  if off < 0 || length <= 0 || off > size - length then
    invalid_arg
      (Printf.sprintf "Function.map: %d bytes at %d outside BAR %d of %d bytes"
         length off i size);
  let w = f.fn.map i off length in
  Mutex.protect f.lock (fun () -> Hashtbl.replace f.maps (Window.address w) w);
  w

(* Removes the window at [w]'s address from [table], or refuses [w]. *)
let forget f table fn w =
  Mutex.protect f.lock @@ fun () ->
  let a = Window.address w in
  match Hashtbl.find_opt table a with
  | Some w' when Window.length w' = Window.length w -> Hashtbl.remove table a
  | _ ->
      invalid_arg (Printf.sprintf "Function.%s: no such window of %s" fn f.bus)

let unmap f w =
  forget f f.maps "unmap" w;
  f.fn.unmap w

(* Interrupts and reset *)

let interrupt f ms = f.fn.interrupt ms
let reset f = f.fn.reset ()

(* System memory *)

let alloc_dma ?(contiguous = false) ?va f n =
  let ((w, _) as dma) = f.fn.alloc_dma ~contiguous ~va n in
  Mutex.protect f.lock (fun () -> Hashtbl.replace f.dmas (Window.address w) w);
  dma

let free_dma f w =
  forget f f.dmas "free_dma" w;
  f.fn.free_dma w

let pin f a n =
  let runs = f.fn.pin a n in
  Mutex.protect f.lock (fun () ->
      let k = Option.value ~default:0 (Hashtbl.find_opt f.pins (a, n)) in
      Hashtbl.replace f.pins (a, n) (k + 1));
  runs

let unpin f a n =
  Mutex.protect f.lock (fun () ->
      match Hashtbl.find_opt f.pins (a, n) with
      | None ->
          invalid_arg
            (Printf.sprintf "Function.unpin: 0x%x is not pinned for %s" a f.bus)
      | Some 1 -> Hashtbl.remove f.pins (a, n)
      | Some k -> Hashtbl.replace f.pins (a, n) (k - 1));
  f.fn.unpin a n
