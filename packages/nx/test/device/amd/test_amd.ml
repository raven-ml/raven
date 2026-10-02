(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD runtime on a machine without an AMD GPU: opening refuses with a
   message, fixes no interface, and the queries refuse devices of other
   vendors. *)

open Windtrap

let no_gpu () =
  if
    Nx_amd_device.count ~interface:Kernel () > 0
    || Nx_amd_device.count ~interface:Pci () > 0
  then skip ~reason:"this machine has an AMD GPU" ()

(* Another machine without GPUs: this process serves it on the loopback, then
   stops serving it. *)
let test_other_machine () =
  let key = "a key of the test, long enough" in
  let s =
    Nx_remote_device.listen ~key (Unix.ADDR_INET (Unix.inet_addr_loopback, 0))
  in
  let port =
    match Nx_device_support.Remote_server.address s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  match Nx_remote_device.connect ~port ~key "127.0.0.1" with
  | Error why -> fail why
  | Ok host ->
      if Nx_amd_device.count ~host () > 0 then
        skip ~reason:"the machine has a GPU" ();
      let named = Printf.sprintf "AMD@127.0.0.1:%d: " port in
      (match Nx_amd_device.get ~host 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg ->
          is_true ~msg:"the GPU's name starts it"
            (String.starts_with ~prefix:named msg));
      (match Nx_amd_device.get ~host ~interface:Kernel 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg -> contains ~msg:"over PCI only" ~sub:"over PCI" msg);
      Nx_device_support.Remote_server.stop s;
      let lost = function Nx_device.Lost (d, _) -> d == host | _ -> false in
      raises_match ~msg:"count, the machine gone" lost (fun () ->
          Nx_amd_device.count ~host ());
      raises_match ~msg:"get, the machine gone" lost (fun () ->
          Nx_amd_device.get ~host 1)

(* Thread traces *)

type traced = {
  file : string;
  markers : (int * int) list;
  waves : Nx_amd_device.Thread_trace.wave list;
}

(* The traces of golden/thread_trace.golden, as tinygrad's decoder reads them
   (gen/thread_trace.py). *)
let traces =
  lazy
    (let lines =
       String.split_on_char '\n'
         (In_channel.with_open_bin "golden/thread_trace.golden"
            In_channel.input_all)
     in
     let ints = List.map int_of_string in
     List.fold_left
       (fun acc line ->
         match (String.split_on_char ' ' line, acc) with
         | [ "trace"; file ], _ -> { file; markers = []; waves = [] } :: acc
         | "markers" :: ms, t :: rest ->
             let marker m =
               match ints (String.split_on_char ':' m) with
               | [ s; r ] -> (s, r)
               | _ -> failwith m
             in
             { t with markers = List.map marker (List.filter (( <> ) "") ms) }
             :: rest
         | [ "wave"; cu; simd; slot; start; stop ], t :: rest ->
             let w =
               match ints [ cu; simd; slot; start; stop ] with
               | [ cu; simd; slot; start; stop ] ->
                   { Nx_amd_device.Thread_trace.cu; simd; slot; start; stop }
               | _ -> assert false
             in
             { t with waves = w :: t.waves } :: rest
         | [ "" ], _ -> acc
         | _ -> failwith ("golden line " ^ line))
       [] lines
     |> List.rev_map (fun t -> { t with waves = List.rev t.waves }))

let trace t = In_channel.with_open_bin ("golden/" ^ t.file) In_channel.input_all

let wave =
  Testable.make
    ~pp:(fun ppf (w : Nx_amd_device.Thread_trace.wave) ->
      Format.fprintf ppf "cu %d simd %d slot %d: %d-%d" w.cu w.simd w.slot
        w.start w.stop)
    ~equal:( = )

let test_waves () =
  List.iter
    (fun t ->
      equal ~msg:t.file (list wave) t.waves
        (Nx_amd_device.Thread_trace.waves (trace t)))
    (Lazy.force traces);
  is_true ~msg:"some traces have waves"
    (List.exists (fun t -> t.waves <> []) (Lazy.force traces))

let test_clock () =
  List.iter
    (fun t ->
      let distinct =
        List.filter
          (fun (s, _) ->
            List.length (List.filter (fun (s', _) -> s = s') t.markers) = 1)
          t.markers
      in
      match (Nx_amd_device.Thread_trace.clock (trace t), distinct) with
      | None, ([] | [ _ ]) -> ()
      | None, _ -> fail (t.file ^ ": no clock through its markers")
      | Some _, ([] | [ _ ]) -> fail (t.file ^ ": a clock without markers")
      | Some clock, markers ->
          List.iter
            (fun (s, r) ->
              equal ~msg:(Printf.sprintf "%s at %d" t.file s) int r (clock s))
            markers)
    (Lazy.force traces);
  is_true ~msg:"some traces are timed"
    (List.exists
       (fun t -> Option.is_some (Nx_amd_device.Thread_trace.clock (trace t)))
       (Lazy.force traces));
  is_true ~msg:"CDNA traces are not"
    (List.for_all
       (fun t ->
         (not (String.starts_with ~prefix:"gfx950" t.file))
         || Option.is_none (Nx_amd_device.Thread_trace.clock (trace t)))
       (Lazy.force traces))

let () =
  exit
    (run "nx.amd.device"
       [
         test "without a GPU, an open fails with a message and fixes nothing"
           (fun () ->
             no_gpu ();
             List.iter
               (fun interface ->
                 match Nx_amd_device.get ~interface 0 with
                 | Ok _ -> fail "a GPU opened"
                 | Error msg ->
                     let name =
                       match interface with
                       | Nx_amd_device.Kernel -> "AMD: "
                       | Pci -> "AMD-PCI: "
                     in
                     is_true ~msg:"the failure names the GPU and its interface"
                       (String.starts_with ~prefix:name msg))
               [ Nx_amd_device.Kernel; Pci; Kernel ];
             is_error (Nx_amd_device.get 0));
         test "another machine's GPUs are opened over PCI" test_other_machine;
         test "a negative index is refused" (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_amd_device.get (-1)));
         test "a thread trace's waves are those its starts and ends pair"
           test_waves;
         test "a thread trace's clock passes through its realtime markers"
           test_clock;
         test "a device of another vendor has no AMD queues" (fun () ->
             is_true (Option.is_none (Nx_amd_device.of_device Nx_device.host)));
       ])
