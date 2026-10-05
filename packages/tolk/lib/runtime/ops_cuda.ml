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
  (* The status of the last call, into [rt_vars]. *)
  let status () =
    store (index (after rt_vars [ !h ]) [ int 3 ]) (cast !h Dtype.Uint64)
  in
  (* The loops being encoded, innermost first. *)
  let ranges = ref [] in
  (* Read after the last call, and in each trip of the loops: every command
     passes the stream, so none leaves its loop. *)
  let stream () =
    let copy = String.starts_with ~prefix:"COPY" (Hcq2.Queue.name q) in
    load
      (index (after rt_vars (!h :: !ranges)) [ int (if copy then 2 else 1) ])
      []
  in
  (* A function's address, from a word of the device's that the host reads: a
     function is loaded on each device. *)
  let extern tag =
    load
      (index
         (placeholder ~slot:0 ~device:dev ~tag [ 1 ] Dtype.Uint64)
         [ int 0 ])
      []
  in
  (* The launches of the loops being encoded, by where they read their extra
     words: a launch in a loop reads its trip's copy. *)
  let launches = ref [] in
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
           ~arg:(Region { name = "kernargs"; align = 128 }))
    in
    (* The launch's extra words, stacked on the queue. *)
    let extra =
      Hcq2.Queue.q q [ u64 1; add addr (int 8); u64 2; addr; u64 0 ] - 40
    in
    let size = function Int n -> cint n | Sym s -> s in
    let at = index kernargs [ int extra ] in
    launches := at :: !launches;
    h :=
      cuda "cuLaunchKernel"
        ((func :: List.map size global)
        @ List.map size local
        @ [ cint 0; stream (); u64 0; at ])
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
  (* A loop is a loop of the host program around its trip's calls, each trip's
     launches reading their extra words from the trip's copy of them. *)
  let loop r body =
    let outer = !launches in
    launches := [];
    ranges := r :: !ranges;
    let start = Hcq2.Queue.size q in
    Hcq2.Queue.loop q r body;
    let trip =
      (Hcq2.Queue.size q - start) / (Dtype.Value.to_int (vmax r) + 1)
    in
    let moved =
      List.map
        (fun at -> (at, index kernargs [ add (nth at 1) (mul r (int trip)) ]))
        !launches
    in
    h := substitute ~calls:Skip ~pass:Fixed_point !h moved;
    (* A trip ends with its status: a loop ends an effect. *)
    h := end_ (status ()) [ r ];
    ranges := List.tl !ranges;
    launches := List.map snd moved @ outer
  in
  let submit () =
    let ka = Hcq2.bufferize_cmdbuf q "cmdbuf" in
    substitute ~calls:Skip ~pass:Fixed_point
      (if op !h = Op.End then !h else status ())
      [ (kernargs, ka) ]
  in
  {
    exec;
    copy;
    wait;
    signal;
    timestamp;
    memory_barrier = (fun () -> ());
    loop;
    submit;
  }

let queues ~host ~reaches =
  {
    Hcq2.commands = queue ~host;
    copy_queue = true;
    submission = Streamed;
    host;
    reaches;
  }
