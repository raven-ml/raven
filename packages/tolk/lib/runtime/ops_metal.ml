(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

(* Queue *)

let handles = [ "queue"; "fence"; "resources"; "count"; "signaler" ]

(* [retain] and [release] keep a profiled command buffer alive in its stamps
   until the device reads its times; [signal:value:] hands a command buffer to
   the device's signaler. *)
let selectors =
  [
    "commandBuffer";
    "computeCommandEncoder";
    "waitForFence:";
    "updateFence:";
    "encodeSignalEvent:value:";
    "endEncoding";
    "commit";
    "useResources:count:usage:";
    "executeCommandsInBuffer:withRange:";
    "concurrentDispatchThreadgroups:threadsPerThreadgroup:";
    "setComputePipelineState:";
    "dispatchThreadgroups:threadsPerThreadgroup:";
    "signaledValue";
    "retain";
    "release";
    "signal:value:";
  ]

let u64 n = int ~dtype:Dtype.Uint64 n
let part u a b = shrink u [ Some (Int a, Int b) ]

let mtl_sel dev name =
  let words = handles @ selectors in
  let rec position i = function
    | [] -> invalid_arg ("no Metal handle or selector " ^ name)
    | w :: ws -> if w = name then i else position (i + 1) ws
  in
  index
    (placeholder ~slot:0 ~device:dev ~tag:(Tag.String "mtl_sel")
       [ List.length words ]
       Dtype.Uint64)
    [ int (position 0 words) ]

(* The command buffer, and its encoder. *)
let mtl_word tag dev =
  index
    (placeholder ~slot:0 ~device:dev ~volatile:true ~tag:(Tag.String tag) [ 1 ]
       Dtype.Uint64)
    [ int 0 ]

let mtl_cb = mtl_word "mtl_cb"
let mtl_enc = mtl_word "mtl_enc"

(* The message [sel] to [obj], through objc_msgSend. *)
let send ~host ?ret dev obj sel args =
  Hcq2.ccall ~host ~lib:"metal" ?ret "objc_msgSend"
    (obj :: load (mtl_sel dev sel) [] :: args)

(* A message to the object that [target] holds, after [h]. *)
let mtl_msg ~host ?ret dev h target sel args =
  let obj = load (index (after (nth target 0) [ h ]) [ nth target 1 ]) [] in
  send ~host ?ret dev obj sel args

type command = {
  lib : string;
  name : string;
  global : int list;
  local : int list;
  offset : int;
}

let icb_tag cmds header =
  let command c =
    Tag.Tuple
      [
        Bytes c.lib;
        String c.name;
        Tuple (List.map (fun n -> Tag.Int n) (c.global @ c.local));
        Int c.offset;
      ]
  in
  Tag.Tuple [ String "mtl_icb"; Tuple (List.map command cmds); Int header ]

let icb u =
  let command = function
    | Tag.Tuple [ Bytes lib; String name; Tuple dims; Int offset ] ->
        let dims =
          List.map (function Tag.Int n -> n | _ -> raise Exit) dims
        in
        let global = List.filteri (fun i _ -> i < 3) dims
        and local = List.filteri (fun i _ -> i >= 3) dims in
        { lib; name; global; local; offset }
    | _ -> raise Exit
  in
  match tag u with
  | Some (Tuple [ String "mtl_icb"; Tuple cmds; Int header ]) -> (
      try Some (List.map command cmds, header) with Exit -> None)
  | _ -> None

let queue ~host ~arch ~residency_set q : Hcq2.commands =
  let apple9 =
    String.starts_with ~prefix:"Apple" arch
    &&
    match int_of_string_opt (String.sub arch 5 (String.length arch - 5)) with
    | Some family -> family >= 9
    | None -> invalid_arg (Printf.sprintf "%S is not an Apple GPU family" arch)
  in
  let devs = Hcq2.Queue.devices q in
  let dev = Multi devs in
  let msg ?ret h target sel args = mtl_msg ~host ?ret dev h target sel args in
  let sel name = load (mtl_sel dev name) [] in
  let rows = ref [] and cmds = ref [] and sizes = ref [] and stamps = ref [] in
  let nbytes = ref 0 and value = ref None in
  let exec call prg =
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "a Metal command runs a compiled program"
    in
    let bufs = Realize.get_call_arg_uops call in
    let vals = Realize.get_call_var_uops call prg in
    let obj = Device.Tiny_elf.of_program prg in
    let args =
      List.map
        (fun g -> getaddr ~device:(List.hd devs) (List.nth bufs g))
        info.globals
      @ List.map2 (fun v var -> ccast v (dtype var)) vals info.vars
    in
    let off = Helpers.round_up !nbytes 256 in
    let r = Hcq2.layout_args ~offset:off args in
    rows := !rows @ List.map (fun (o, w) -> (int o, w)) r;
    nbytes :=
      List.fold_left
        (fun m (o, w) -> max m (o + Dtype.itemsize (dtype w)))
        (if List.is_empty r then off + 8 else 0)
        r;
    (* Symbolic sizes, set on the command at run time. *)
    let dims = info.global_size @ info.local_size in
    if List.exists (function Sym _ -> true | Int _ -> false) dims then begin
      let at = Helpers.round_up !nbytes 8 in
      sizes := !sizes @ [ (List.length !cmds, at) ];
      let words =
        List.map (function Int d -> u64 d | Sym d -> cast d Dtype.Uint64) dims
      in
      rows :=
        !rows
        @ List.map
            (fun (o, w) -> (int o, w))
            (Hcq2.layout_args ~offset:at words);
      nbytes := at + 48
    end;
    let dims = List.map (function Int d -> d | Sym _ -> 1) dims in
    let global = List.filteri (fun i _ -> i < 3) dims
    and local = List.filteri (fun i _ -> i >= 3) dims in
    cmds :=
      !cmds
      @ [ { lib = obj.lib; name = obj.name; global; local; offset = off } ]
  in
  (* A loop's commands are in the indirect command buffer once per trip, each
     trip's arguments after the trip before's, which the host program writes in
     a loop over the range. *)
  let loop r body =
    let start = Helpers.round_up !nbytes 256 in
    nbytes := start;
    let first_row = List.length !rows
    and first_cmd = List.length !cmds
    and first_size = List.length !sizes in
    body ();
    let trip = Helpers.round_up !nbytes 256 - start
    and n = Dtype.Value.to_int (vmax r) + 1 in
    let split l k =
      (List.filteri (fun i _ -> i < k) l, List.filteri (fun i _ -> i >= k) l)
    in
    let rows_before, trip_rows = split !rows first_row
    and cmds_before, trip_cmds = split !cmds first_cmd
    and sizes_before, trip_sizes = split !sizes first_size in
    let k = List.length trip_cmds in
    let trips f = List.concat (List.init n f) in
    rows :=
      rows_before
      @ List.map (fun (o, w) -> (add o (mul r (int trip)), w)) trip_rows;
    cmds :=
      cmds_before
      @ trips (fun t ->
          List.map
            (fun c -> { c with offset = c.offset + (t * trip) })
            trip_cmds);
    sizes :=
      sizes_before
      @ trips (fun t ->
          List.map (fun (ci, at) -> (ci + (t * k), at + (t * trip))) trip_sizes);
    nbytes := start + (n * trip)
  in
  (* The commands are in the indirect command buffer: there is no command
     stream. *)
  let submit () =
    let cmds = !cmds in
    let n = List.length cmds and zero = Helpers.round_up !nbytes 8 in
    let pipes =
      List.fold_left
        (fun ps c ->
          if List.mem (c.lib, c.name) ps then ps else ps @ [ (c.lib, c.name) ])
        [] cmds
    in
    let size = zero + 24 + (8 * (1 + n + List.length pipes)) in
    let buf =
      placeholder ~device:dev ~volatile:true
        ~tag:(icb_tag cmds (zero + 24))
        [ size ] Dtype.Uint8
    in
    let args =
      Hcq2.patch buf
        (!rows @ List.init 3 (fun i -> (int (zero + (8 * i)), u64 0)))
    in
    let header = part (bitcast args Dtype.Uint64) ((zero / 8) + 3) (size / 8) in
    let cb = mtl_cb dev and enc = mtl_enc dev in
    let value =
      match !value with
      | Some v -> v
      | None -> invalid_arg "a Metal queue signals a value"
    in
    (* Symbolic sizes *)
    let h =
      List.fold_left
        (fun h (ci, off) ->
          msg h
            (index header [ int (1 + ci) ])
            "concurrentDispatchThreadgroups:threadsPerThreadgroup:"
            [ index args [ int off ]; index args [ int (off + 24) ] ])
        args !sizes
    in
    (* A command buffer for the commands [first, first + count). *)
    let run h first count last =
      let first_arg = match first with `Int i -> u64 i | `Range r -> r in
      let cbuf =
        msg ~ret:Dtype.Uint64 h (mtl_sel dev "queue") "commandBuffer" []
      in
      let h = store cb cbuf in
      let h =
        store enc (msg ~ret:Dtype.Uint64 h cb "computeCommandEncoder" [])
      in
      let h = msg h enc "waitForFence:" [ sel "fence" ] in
      (* Without a residency set, the encoder declares the buffers. *)
      let h =
        if residency_set then h
        else
          msg h enc "useResources:count:usage:"
            [ sel "resources"; sel "count"; u64 3 ]
      in
      (* Before Apple9, the encoder must use the pipelines. *)
      let h =
        if apple9 then h
        else
          let r =
            range ~dtype:Dtype.Uint64
              (Int (List.length pipes))
              [ unique_num () ]
          in
          let h =
            msg h enc "setComputePipelineState:"
              [ load (index header [ add (int (1 + n)) r ]) [] ]
          in
          end_
            (msg h enc "dispatchThreadgroups:threadsPerThreadgroup:"
               [ index args [ int zero ]; index args [ int zero ] ])
            [ r ]
      in
      let h =
        msg h enc "executeCommandsInBuffer:withRange:"
          [
            load (index (after header [ h ]) [ int 0 ]) []; first_arg; u64 count;
          ]
      in
      let h = msg h enc "updateFence:" [ sel "fence" ] in
      let h = msg h enc "endEncoding" [] in
      (* The command buffer waits in the command's stamps, retained, until the
         device reads its times. *)
      let h =
        match !stamps with
        | [] -> h
        | s :: _ ->
            (* The first stamp, in a loop its first trip's. *)
            let slots, _, off = Hcq2.lane_offset s in
            let off =
              match off with Int o -> o | Sym o -> Dtype.Value.to_int (vmin o)
            in
            let word k =
              let k = (off / 8) + k in
              match first with
              | `Int i -> int (k + (4 * i))
              | `Range r -> add (int k) (mul (int 4) r)
            in
            let at h k = load (index (after slots [ h ]) [ word k ]) [] in
            let unread = where (eq (at h 3) (u64 0)) (at h 1) (u64 0) in
            let h = send ~host dev unread "release" [] in
            let h = store (index (after slots [ h ]) [ word 3 ]) (u64 0) in
            store
              (index (after slots [ h ]) [ word 1 ])
              (send ~host ~ret:Dtype.Uint64 dev cbuf "retain" [])
      in
      (* The signaler signals the last command buffer's value once it completed,
         and watches the others for a failure. *)
      let h =
        msg h (mtl_sel dev "signaler") "signal:value:"
          [ cbuf; (if last then value else u64 0) ]
      in
      msg h cb "commit" []
    in
    (* Timestamps come from command buffers' metrics: one command buffer
       each. *)
    match !stamps with
    | [] -> run h (`Int 0) n true
    | _ ->
        let h =
          if n > 1 then
            let r = range ~dtype:Dtype.Uint64 (Int (n - 1)) [ unique_num () ] in
            end_ (run (after h [ r ]) (`Range r) 1 false) [ r ]
          else h
        in
        run h (`Int (n - 1)) 1 true
  in
  {
    exec;
    copy =
      (fun _ _ _ -> invalid_arg "Metal copies on the host, not on its queues");
    (* The fence orders the queue. *)
    wait = (fun _ _ -> ());
    signal = (fun _ v -> value := Some v);
    timestamp = (fun dst -> stamps := !stamps @ [ dst ]);
    memory_barrier = (fun () -> ());
    loop;
    submit;
  }

let queues ~host ~arch ~residency_set =
  {
    Hcq2.commands = queue ~host ~arch ~residency_set;
    copy_queue = false;
    submission = Buffered;
    host;
    reaches = (fun _ -> false);
  }
