(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune_next
open Tolk_next

let host =
  let clang =
    lazy
      (match Device.renderer ~arch:Host.target.arch "CPU" with
      | Ok r -> r
      | Error e -> failwith e)
  in
  fun _ -> Lazy.force clang

let scope ?(renderer = host) () = Lower.scope ~renderer
let within s f = Nx.Op.intercept { run = (fun o -> Lower.op s o) } f

let trace ?renderer f =
  let s = scope ?renderer () in
  (s, within s f)

(* Elements as values *)

type dtype = Dtype : ('a, 'b) Nx_dtype.t -> dtype

let nx_dtype : Dtype.t -> dtype = function
  | Float16 -> Dtype Float16
  | Float32 -> Dtype Float32
  | Float64 -> Dtype Float64
  | Bfloat16 -> Dtype BFloat16
  | Fp8e4m3 -> Dtype Float8_e4m3
  | Fp8e5m2 -> Dtype Float8_e5m2
  | Int8 -> Dtype Int8
  | Uint8 -> Dtype UInt8
  | Int16 -> Dtype Int16
  | Uint16 -> Dtype UInt16
  | Int32 -> Dtype Int32
  | Uint32 -> Dtype UInt32
  | Int64 -> Dtype Int64
  | Uint64 -> Dtype UInt64
  | Bool -> Dtype Bool
  | dt -> Format.kasprintf invalid_arg "no nx dtype for %a" Dtype.pp dt

let to_value : type a b. (a, b) Nx_dtype.t -> a -> Dtype.value =
 fun dt v ->
  match dt with
  | Float16 -> `Float v
  | Float32 -> `Float v
  | Float64 -> `Float v
  | BFloat16 -> `Float v
  | Float8_e4m3 -> `Float v
  | Float8_e5m2 -> `Float v
  | Int4 -> `Int (Z.of_int v)
  | UInt4 -> `Int (Z.of_int v)
  | Int8 -> `Int (Z.of_int v)
  | UInt8 -> `Int (Z.of_int v)
  | Int16 -> `Int (Z.of_int v)
  | UInt16 -> `Int (Z.of_int v)
  | Int32 -> `Int (Z.of_int32 v)
  | UInt32 -> `Int (Z.of_int32_unsigned v)
  | Int64 -> `Int (Z.of_int64 v)
  | UInt64 -> `Int (Z.of_int64_unsigned v)
  | Bool -> `Bool v
  | Complex64 | Complex128 -> invalid_arg "a complex element"

let of_const : type a b. (a, b) Nx_dtype.t -> Dtype.const -> a =
 fun dt c ->
  let int = function `Int z -> z | _ -> invalid_arg "not an integer" in
  let float = function `Float f -> f | _ -> invalid_arg "not a float" in
  match dt with
  | Float16 -> float c
  | Float32 -> float c
  | Float64 -> float c
  | BFloat16 -> float c
  | Float8_e4m3 -> float c
  | Float8_e5m2 -> float c
  | Int4 -> Z.to_int (int c)
  | UInt4 -> Z.to_int (int c)
  | Int8 -> Z.to_int (int c)
  | UInt8 -> Z.to_int (int c)
  | Int16 -> Z.to_int (int c)
  | UInt16 -> Z.to_int (int c)
  | Int32 -> Z.to_int32 (int c)
  | UInt32 -> Z.to_int32_unsigned (int c)
  | Int64 -> Z.to_int64 (int c)
  | UInt64 -> Z.to_int64_unsigned (int c)
  | Bool -> ( match c with `Bool b -> b | _ -> invalid_arg "not a boolean")
  | Complex64 | Complex128 -> invalid_arg "a complex element"

(* The elements of [b], a buffer of [dt]'s elements on any device. *)
let elements (Dtype dt) b =
  let n = Nx_device.Buffer.length b in
  let h =
    if Nx_device.equal (Nx_device.Buffer.device b) Nx_device.host then b
    else
      let h = Nx_array.Elements.create dt n in
      Nx_device.Buffer.copy ~src:b ~dst:h;
      h
  in
  let get = Nx_array.Elements.get dt h in
  Array.init n (fun i -> to_value dt (get i))

(* The run of [x]'s host storage that a parameter binds: from the element at or
   below the first one its view reaches whose offset is a multiple of 16 bytes,
   through the last one it reaches. *)
let run (Nx.P x) =
  match Nx.Repr.v x with
  | Nx.Repr.Host a when Nx_array.View.numel a.view = 0 -> [||]
  | Nx.Repr.Host a ->
      let dt = Nx.dtype x in
      let lo, hi = Nx_array.View.extent a.view in
      let per = Int.max 1 (16 / Nx_dtype.itemsize dt) in
      let start = lo - (lo mod per) in
      let get = Nx_array.Elements.get dt a.buffer in
      Array.init (hi - start) (fun i -> to_value dt (get (start + i)))
  | Nx.Repr.Placed _ | Nx.Repr.Traced _ ->
      invalid_arg "an argument off the host"

(* The arguments of each scope, by slot. Slots come from the counter of storage
   slots, so that no parameter shares its slot with a capture. *)
let arguments : (Lower.scope * (int * Nx.packed)) list ref = ref []
let arguments_lock = Mutex.create ()

let argument s x =
  let slot = Ops.unique_num () in
  Mutex.protect arguments_lock (fun () ->
      arguments := (s, (slot, Nx.P x)) :: !arguments);
  Lower.param s ~slot x

let contents s =
  let args =
    Mutex.protect arguments_lock (fun () ->
        List.filter_map
          (fun (s', a) -> if s' == s then Some a else None)
          !arguments)
  in
  List.map
    (fun (u, bufs) ->
      let dt = nx_dtype (Ops.dtype u) in
      let slot =
        match Ops.arg u with
        | Param { slot; _ } -> slot
        | _ -> invalid_arg "a capture that is not storage"
      in
      (slot, Array.concat (List.map (elements dt) bufs)))
    (Lower.captures s)
  @ List.map (fun (slot, x) -> (slot, run x)) args

let value s y =
  let buffers = contents s in
  let dt = Nx.dtype y in
  match Tensors.eval ~buffers (Lower.uop y) with
  | first :: _ -> Nx.create dt (Nx.shape y) (Array.map (of_const dt) first)
  | [] -> invalid_arg "a value on no device"

(* Bits *)

(* An element's bits, every NaN the same. *)
let element_bits : type a b. (a, b) Nx_dtype.t -> a -> Dtype.value =
 fun dt v ->
  match to_value dt v with
  | `Float f when Float.is_nan f -> `Bool false
  | `Float f -> `Int (Z.of_int64 (Int64.bits_of_float f))
  | v -> v

let exact ?__POS__ expected actual =
  let bits =
    Windtrap.Testable.make ~pp:Nx.pp ~equal:(fun x y ->
        let dt = Nx.dtype x in
        Nx_dtype.equal dt (Nx.dtype y)
        && Nx.shape x = Nx.shape y
        &&
        match Nx_dtype.equal_witness dt (Nx.dtype y) with
        | Some Type.Equal ->
            Array.for_all2
              (fun a b ->
                Dtype.equal_const (element_bits dt a) (element_bits dt b))
              (Nx.to_array x) (Nx.to_array y)
        | None -> false)
  in
  Windtrap.equal ?__POS__ bits expected actual
