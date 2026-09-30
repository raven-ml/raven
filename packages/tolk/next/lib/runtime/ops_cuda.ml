(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let u64 n = int ~dtype:Dtype.Uint64 n

(* A C int, as a call passes a number. *)
let cint n = int ~dtype:Dtype.Int32 n
let cu_stream_wait_value_geq = 0
let cu_stream_write_value_default = 0

(* Queue *)

let queue ~host q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q in
  let dev = Multi devs and on = List.hd devs in
  let cuda f args = Hcq2.ccall ~host ~lib:"cuda" ~ret:Dtype.Uint32 f args in
  (* [context, compute stream, copy stream, status] *)
  let rt_vars =
    placeholder ~slot:0 ~device:dev ~tag:(Tag.String "cuda") [ 4 ] Dtype.Uint64
  in
  let kernargs = placeholder ~device:dev [ 8 ] Dtype.Uint8 in
  let h = ref (cuda "cuCtxSetCurrent" [ load (index rt_vars [ int 0 ]) [] ]) in
  (* Read after the last call. *)
  let stream () =
    let copy = String.starts_with ~prefix:"COPY" (Hcq2.Queue.name q) in
    load (index (after rt_vars [ !h ]) [ int (if copy then 2 else 1) ]) []
  in
  (* A function's address, from a word of the host (DIVERGENCES D36). *)
  let extern tag =
    load
      (index
         (placeholder ~slot:0 ~device:(Single host) ~tag [ 1 ] Dtype.Uint64)
         [ int 0 ])
      []
  in
  let launch func global local args =
    let rows = Hcq2.layout_args ~offset:8 args in
    let size =
      List.fold_left
        (fun m (o, w) -> max m (o + Dtype.itemsize (dtype w)))
        8 rows
      - 8
    in
    let addr =
      getaddr ~device:on
        (v Op.Linear
           ~src:(Hcq2.pack_args ((0, u64 size) :: rows) (8 + size))
           ~arg:(String "kernargs"))
    in
    (* The launch's extra words, stacked on the queue. *)
    let extra =
      Hcq2.Queue.q q [ u64 1; add addr (int 8); u64 2; addr; u64 0 ] - 40
    in
    let size = function Int n -> cint n | Sym s -> s in
    h :=
      cuda "cuLaunchKernel"
        ((func :: List.map size global)
        @ List.map size local
        @ [ cint 0; stream (); u64 0; index kernargs [ int extra ] ])
  in
  let exec call prg =
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "a CUDA command runs a compiled program"
    in
    let obj = Device.Tiny_elf.of_program prg in
    let bufs = Realize.get_call_arg_uops call in
    let vals = Realize.get_call_var_uops call prg in
    launch
      (extern (Tag.Tuple [ String "function"; Bytes obj.lib; String obj.name ]))
      info.global_size info.local_size
      (List.map (fun g -> getaddr ~device:on (List.nth bufs g)) info.globals
      @ List.map2 (fun v var -> ccast v (dtype var)) vals info.vars)
  in
  let copy dst src n =
    h :=
      cuda "cuMemcpyAsync"
        [ getaddr ~device:on dst; getaddr ~device:on src; u64 n; stream () ]
  in
  let wait signal value =
    h :=
      cuda "cuStreamWaitValue64_v2"
        [
          stream ();
          getaddr ~device:on signal;
          value;
          cint cu_stream_wait_value_geq;
        ]
  in
  let signal word value =
    h :=
      cuda "cuStreamWriteValue64_v2"
        [
          stream ();
          getaddr ~device:on word;
          value;
          cint cu_stream_write_value_default;
        ]
  in
  (* A slot is [signal][timestamp]. *)
  let timestamp slot =
    h :=
      cuda "cuLaunchHostFunc"
        [
          stream ();
          extern (Tag.String "stamp");
          getaddr ~device:on (shrink slot [ Some (Int 1, Int 2) ]);
        ]
  in
  let submit ka =
    substitute
      (store (index (after rt_vars [ !h ]) [ int 3 ]) (cast !h Dtype.Uint64))
      [ (kernargs, ka) ]
  in
  {
    exec;
    copy;
    wait;
    signal;
    timestamp;
    memory_barrier = (fun () -> ());
    submit;
  }

let queues ~host ~reaches =
  { Hcq2.commands = queue ~host; copy_queue = true; host; reaches }
