(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The program test/remote/two.sh runs under rig run, on a machine with an AMD
   or NV GPU and one other machine with a GPU. A program loaded on the other
   machine doubles a frame there and sends it over a rail; this machine's GPU
   waits for its arrival and copies it. Three frames reuse the loaded programs
   and both landing copies, and every word is checked. Kernels and host code
   are the suites' fixtures, read from the directory it runs in. *)

module B = Rig.Buffer
module G = Rig_program
module Sub = Rig.Submission
module A = Rig_amd_abi
module N = Rig_nv_abi

let get = function Ok v -> v | Error why -> failwith why
let words = 64
let bytes = 4 * words

(* The other machine's GPU and the conformance kernels it runs. *)
let far h =
  let binaries =
    [
      ("AMD", Rig_amd_support.launch_binary);
      ("NV", Rig_nv_support.launch_binary);
      ("CUDA", Rig_cuda_support.launch_binary);
    ]
  in
  let found (kind, binary) =
    match Rig_remote.devices h kind with
    | Ok (d :: _) -> Some (d, Option.get (binary ()))
    | Ok [] | Error _ -> None
  in
  match List.find_map found binaries with
  | Some f -> f
  | None -> failwith ("no GPU on " ^ Rig.name h)

(* The ring words and commands that hold a queue until the 64 bits at
   [arrived] are at least [c]. An AMD ring holds the PM4 packets themselves; an
   NV ring holds a GPFIFO entry of the methods at [commands]. *)
let amd d ~arrived ~commands:_ c =
  let g = (Option.get (Rig.capability d A.Capability.key)).gpu in
  ( A.Packet.encode Fun.id
      (A.Pm4.wait_64 g arrived Greater_equal c () @ A.Pm4.acquire_mem g System),
    "" )

let nv _ ~arrived ~commands c =
  let methods = N.Packet.encode Fun.id (N.Method.acquire arrived c) in
  let words = String.length methods / 4 in
  (N.Packet.encode Fun.id (N.Gpfifo.entry commands ~offset:0 ~words), methods)

let near () =
  if Rig_amd_amdgpu.count () > 0 then
    (get (Rig.open_ (Rig_amd_amdgpu.gpu 0)), amd)
  else if Rig_nv_nvidia.count () > 0 then
    (get (Rig.open_ (Rig_nv_nvidia.gpu 0)), nv)
  else failwith "no AMD or NV GPU on this machine"

let view memory offset length = G.Memory { memory; offset; length }
let hole at x = { G.at; width = W64; terms = [| (x, 1) |]; add = 0; shift = 0 }

(* Doubles input 0 into input 1 and adds 1, places the result in landing copy
   [k] of the rail's outbound area, and readies the run's count, its int 0,
   through the fixture [ready]. *)
let sender d binary rail k =
  let params = Bytes.make 20 '\000' in
  Bytes.set_int32_le params 16 1l;
  {
    G.devices = [| Rig.arch d; Rig.arch (Rig.host_of d) |];
    memory =
      [|
        G.Rail { rail; area = Counts };
        G.Rail { rail; area = Outbound };
        G.Alloc
          {
            device = 1;
            kind = Pinned;
            bytes = 8;
            init = { G.bytes = ""; holes = [||] };
            copies = One;
          };
      |];
    images =
      [|
        { G.device = 0; binary = { G.bytes = binary; holes = [||] } };
        {
          G.device = 1;
          binary =
            {
              G.bytes = Rig_host_support.fixture ~dir:"../fixtures" "ready";
              holes = [||];
            };
        };
      |];
    inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes });
    ints = 1;
    steps =
      [|
        G.Submit
          {
            device = 0;
            slots = [| G.Input 0; G.Input 1 |];
            fixed = [||];
            parts =
              [|
                (G.Launch
                      {
                        image = 0;
                        name = "twice";
                        params =
                          { G.bytes = Bytes.to_string params; holes = [||] };
                        refs =
                          [|
                            { G.slot = 1; at = 0; use = Write };
                            { G.slot = 0; at = 8; use = Read };
                          |];
                        groups = (G.Fixed words, G.Fixed 1, G.Fixed 1);
                        threads = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
                        shared = G.Fixed 0;
                      });
              |];
          };
        G.Move { src = G.Input 1; dst = view 1 (k * bytes) bytes };
        (* test/fixtures' [ready] stores [value] at [out], then calls [ready]
           with [counts] and [count]; its last parameter names the outbound
           area it follows. *)
        G.Submit
          {
            device = 1;
            slots = [| view 2 0 8; view 0 0 384; view 1 (k * bytes) bytes |];
            fixed = [||];
            parts =
              [|
                G.Launch
                  {
                    image = 1;
                    name = "ready";
                    params =
                      {
                        G.bytes = String.make 48 '\000';
                        holes =
                          [|
                            hole 16 (G.Leaf (Ready rail)); hole 32 (G.Int 0);
                          |];
                      };
                    refs =
                      [|
                        { G.slot = 0; at = 0; use = Write };
                        { G.slot = 1; at = 8; use = Write };
                        { G.slot = 2; at = 40; use = Read };
                      |];
                    groups = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
                    threads = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
                    shared = G.Fixed 0;
                  };
              |];
          };
      |];
  }

let write b s =
  B.write b Bigarray.char (fun a ->
      String.iteri (fun i c -> a.{i} <- c) s)

let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes h 0 s 0 (B.length b);
  Bytes.to_string s

let le32s f =
  let b = Bytes.create bytes in
  for i = 0 to words - 1 do
    Bytes.set_int32_le b (4 * i) (Int32.of_int (f i))
  done;
  Bytes.to_string b

let () =
  let job =
    match Rig_remote.launched () with
    | Some j -> get j
    | None -> failwith "two.exe runs under rig run (test/remote/two.sh)"
  in
  let h = List.hd (Rig_remote.hosts job) in
  let fd, binary = far h in
  let d, held = near () in
  let r =
    match Rig.capability h Rig_remote_abi.key with
    | Some (Rig_remote_abi.Host r) -> r
    | _ -> failwith "the other machine's host has no rail"
  in
  let t = { Rig_remote_abi.src = 0; dst = 0; length = bytes } in
  let rail = get (r.rail None ~send:[| t |] ~receive:[||]) in
  let e = Option.get rail.local in
  let senders =
    Array.init 2 (fun k ->
        get (G.load (sender fd binary rail.id k) [| fd; h |]))
  in
  let borrow b = Option.get (B.borrow d b) in
  let counts = borrow (B.of_bigarray e.counts) in
  let inbound = borrow (B.of_bigarray e.inbound) in
  let commands = B.view (B.create Rig.host 65536) ~first:0 ~length:4096 in
  let cmds = borrow commands in
  let ring_bytes =
    String.length (fst (held d ~arrived:0L ~commands:0L 1L))
  in
  let out = B.create d bytes in
  (* A receiver per landing copy: the queue's wait, then the copy. Its hold
     keeps the rail, whose end this machine's work reads. The wait makes the
     landed bytes the copy reads, a write of [inbound] that no buffer of the
     words shows, so [inbound] is fixed for writing: the copy follows it. *)
  let receivers =
    Array.init 2 (fun k ->
        let ring = B.create Rig.host ring_bytes in
        let src = B.view inbound ~first:(k * bytes) ~length:bytes in
        let s =
          Sub.make ~keep:rail
            ~fixed:[ (counts, Rig.Read); (inbound, Read_write); (cmds, Read) ]
            d
            Sub.[| words ring; copy ~src:(Buffer src) ~dst:(Buffer out) |]
        in
        (s, ring))
  in
  let x = B.create fd bytes and y = B.create fd bytes in
  for c = 1 to 3 do
    let k = (c - 1) mod 2 in
    let s, ring = receivers.(k) in
    let at b = Int64.of_int (Sub.address s b) in
    let ring_words, methods =
      held d
        ~arrived:(Int64.add (at counts) 256L)
        ~commands:(at cmds) (Int64.of_int c)
    in
    write ring ring_words;
    write commands methods;
    let staged = B.create Rig.host bytes in
    write staged (le32s (fun i -> i + (1000 * c)));
    B.copy ~src:staged ~dst:x;
    let p = Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[||] in
    Array.iter Rig.Point.wait
      (G.run senders.(k) { inputs = [| x; y |]; ints = [| c |] });
    Rig.Point.wait p;
    if read out <> le32s (fun i -> (2 * (i + (1000 * c))) + 1) then
      failwith (Printf.sprintf "frame %d differs" c);
    Printf.printf "frame %d: %d words checked\n%!" c words
  done;
  Printf.printf "%s -> %s: passed\n%!" (Rig.name fd) (Rig.name d);
  Rig_remote.close job
