(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk
open Tolk_uop
module U = Uop

let uint n = U.const (Const.int Dtype.uint64 n)
let index p i = U.index ~ptr:p ~idxs:[U.const_int i] ()
let load p i = U.load ~src:(index p i) ()
let context name = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
    ~device:(U.Single name) ~allocation:("cuda_context", "") ()
let call name ?after fn dtype args = Hcq2.ccall ~host:name ?after ~name:fn ~dtype args

let lower name u = match U.as_load u with
  | Some {src; _} ->
      (match U.as_index src with
       | Some {ptr; idxs = [i]} when U.const_int_value i = Some 0
           && U.node_tag (U.buf_uop ptr) = Some "timeline" ->
           let deps = if U.op ptr = Ops.After then List.tl (U.children ptr) else [] in
           Some (call name ~after:deps "tolk_cuda_hcq_poll" Dtype.uint64
             [load (context name) 0; index (U.without_after ptr) 0])
       | _ -> None)
  | _ -> None

let encode name u = match U.op u, U.arg u, U.children u with
  | Ops.Custom_function, U.Arg.String ("submit_cuda_compute_0" | "submit_cuda_copy_0" as kind),
      [linear; dependency] ->
      let stream = uint (if kind = "submit_cuda_copy_0" then 1 else 0) in
      let ctx = load (context name) 0 in
      let previous = ref (call name ~after:[dependency] "tolk_cuda_hcq_begin" Dtype.void [ctx]) in
      let emit fn args = previous := call name ~after:[!previous] fn Dtype.void (ctx :: args) in
      List.iter (fun node -> match U.as_call node, U.arg node with
          | Some {body; args}, _ when U.op body = Ops.Program ->
              let info = Option.get (U.as_program_info body) in
              let buffers = List.filter (fun a -> not (U.is_bound_var a)) args in
              let bound = List.filter_map (fun a -> match U.as_bind a with
                  | Some {var; value} -> Option.map (fun n -> n, value) (U.program_var_name var)
                  | None -> None) args in
              let variables = List.map (fun v -> match U.program_var_name v with
                  | Some n -> Option.value (List.assoc_opt n bound) ~default:v | None -> v) info.vars in
              let actuals = List.map (fun i -> U.getaddr ~device:name ~src:(List.nth buffers i) ()) info.globals @ variables in
              let object_ = U.to_elf body in
              let fields = Tiny_elf.layout object_.signature in
              let size = List.fold_left (fun size (field : Tiny_elf.field) ->
                  max size (field.offset + field.size)) 0 fields in
              let rows = List.map (fun (field : Tiny_elf.field) ->
                  let value = List.nth actuals field.argument.slot in
                  let dtype = if field.argument.addrspace = Dtype.Alu then field.argument.dtype else Dtype.uint64 in
                  field.offset, U.cast ~src:value ~dtype) fields in
              let arena_size = max 8 ((size + 7) / 8 * 8) in
              let arena = U.placeholder ~shape:[arena_size] ~dtype:Dtype.uint8
                  ~slot:(U.fresh_buffer_slot ()) ~device:(U.Single name) () |> U.with_tag "kernargs" in
              let patched = Hcq2.patch ~after:[dependency] arena rows in
              let function_ = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
                  ~device:(U.Single name) ~allocation:("cuda_function", Marshal.to_string object_ []) () in
              let dims ds = List.init 3 (fun i -> if i >= List.length ds then uint 1 else
                  match List.nth ds i with
                  | U.Launch_int n -> uint n
                  | U.Launch_float f -> uint (int_of_float f)
                  | U.Launch_sym v -> U.cast ~src:v ~dtype:Dtype.uint64) in
              emit "tolk_cuda_hcq_launch" ([load function_ 0] @ dims info.global_size @ dims info.local_size
                  @ [index patched 0; uint size])
          | Some {body; args = [dst; src]}, _ when U.op body = Ops.Store ->
              let bytes = Bound.(to_int (mul (int (U.max_numel dst)) (int (Dtype.itemsize (U.dtype dst))))) in
              emit "tolk_cuda_hcq_copy" [U.getaddr ~device:name ~src:dst ();
                U.getaddr ~device:name ~src (); uint bytes]
          | _, U.Arg.Typed (("wait" | "store" as kind), _) ->
              let args = U.src node in
              emit (if kind = "wait" then "tolk_cuda_hcq_wait" else "tolk_cuda_hcq_signal")
                [stream; U.getaddr ~device:name ~src:args.(0) (); args.(1)]
          | _, U.Arg.Typed ("timestamp", _) ->
              let timestamp = U.shrink ~src:(U.src node).(0) ~offset:(U.const_int 1)
                  ~size:(U.const_int 1) in
              emit "tolk_cuda_hcq_timestamp" [stream; U.getaddr ~device:"CPU" ~src:timestamp ()]
          | _, U.Arg.Typed ("barrier", _) -> ()
          | _ -> invalid_arg "CUDA queue: unsupported instruction") (U.children linear);
      Some !previous
  | _ -> None
