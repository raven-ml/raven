(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Each path, in the preference order the .mli states: its name, the names
   its opener gives its GPUs, and its listing, as bus addresses or a count
   where its GPUs have none. *)
let names gpu n = List.init n (fun i -> (gpu i : _ Rig_edge.opener).name)
let by_bus gpu =
  Result.map (fun b -> (names gpu (List.length b), List.map Option.some b))

let preferred () =
  let metal =
    Result.map
      (fun n -> (names Rig_metal.gpu n, List.init n (fun _ -> None)))
      (Rig_metal.count ())
  in
  [
    ("METAL", metal);
    ("CUDA", by_bus Rig_cuda.gpu (Rig_cuda.buses ()));
    ("NV", by_bus Rig_nv_nvidia.gpu (Rig_nv_nvidia.buses ()));
    ("AMD", by_bus Rig_amd_amdgpu.gpu (Rig_amd_amdgpu.buses ()));
    ("NV-PCI", by_bus Rig_nv_pci.gpu (Rig_nv_pci.buses ()));
    ("AMD-PCI", by_bus Rig_amd_pci.gpu (Rig_amd_pci.buses ()));
  ]

let seen =
  List.map (fun (p : Rig_gpus.path) ->
      ( p.kind,
        Result.map
          (fun gpus ->
            ( List.map (fun (g : Rig_gpus.gpu) -> g.name) gpus,
              List.map (fun (g : Rig_gpus.gpu) -> g.bus) gpus ))
          (p.gpus ()) ))

let listing =
  list
    (pair string
       (result (pair (list string) (list (option string))) string))

let order () = equal listing (preferred ()) (seen (Rig_gpus.paths ()))

(* The name of the first GPU of the paths that opens, closed again so that
   [first] can open it: several paths open one device of a GPU at a time. *)
let first_name () =
  let opens (g : Rig_gpus.gpu) =
    match g.open_ () with
    | Ok d ->
        Rig.close d;
        Some g.name
    | Error _ -> None
  in
  List.find_map
    (fun (p : Rig_gpus.path) ->
      match p.gpus () with
      | Ok gpus -> List.find_map opens gpus
      | Error _ -> None)
    (Rig_gpus.paths ())

let opens () =
  let want = first_name () in
  match (want, Rig_gpus.first ()) with
  | None, Ok d ->
      Rig.close d;
      failf "first opened %s where no GPU of paths opens" (Rig.name d)
  | None, Error why ->
      List.iter
        (fun (p : Rig_gpus.path) ->
          match p.gpus () with
          | Error _ -> contains ~msg:"names the refusing path" ~sub:p.kind why
          | Ok gpus ->
              List.iter
                (fun (g : Rig_gpus.gpu) ->
                  contains ~msg:"names the refusing GPU" ~sub:g.name why)
                gpus)
        (Rig_gpus.paths ())
  | Some name, first ->
      let d = require_ok ~pp:Format.pp_print_string first in
      Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
      equal string ~msg:"the device of the first GPU that opens" name
        (Rig.name d)

(* One launch, the same bytes *)

let contents f = In_channel.with_open_bin f In_channel.input_all
let add f = contents ("../../examples/01-add/kernels/" ^ f)

(* The binary of examples/01-add's [add] for a GPU of [path] and [arch]. Every
   binary takes the addresses of [a], [b] and [out] at bytes 0, 8 and 16 of
   its 32 bytes of parameters, then [n]. *)
let binary path arch =
  match (path, arch) with
  | "METAL", _ -> Some (add "add.metallib")
  | "CUDA", _ -> Some (add "add.ptx")
  | ("AMD" | "AMD-PCI"), "gfx1201" -> Some (add "add_gfx1201.hsaco")
  | ("NV" | "NV-PCI"), "sm_89" -> Some (add "add_sm_89.cubin")
  | _ -> None

(* The GPUs here whose [add] examples/01-add has, each with its binary,
   opened once. *)
let gpus =
  lazy
    (List.concat_map
       (fun (p : Rig_gpus.path) ->
         match p.gpus () with
         | Error _ -> []
         | Ok gpus ->
             List.filter_map
               (fun (g : Rig_gpus.gpu) ->
                 match g.open_ () with
                 | Error _ -> None
                 | Ok d -> (
                     match binary p.kind (Rig.arch d) with
                     | Some b -> Some (d, b)
                     | None ->
                         Rig.close d;
                         None))
               gpus)
       (Rig_gpus.paths ()))

let host_add = lazy (add ("add_" ^ Rig.arch Rig.host ^ ".o"))

(* The bytes [out] holds after [binary]'s [add] over the [n] floats of [a]
   and [b] on [d], in groups of 64 threads. *)
let run_add d binary ~n a b =
  let image = require_ok ~pp:Format.pp_print_string (Rig.Image.load d binary) in
  let on s =
    let x = Rig.Buffer.create d (String.length s) in
    Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:x;
    x
  in
  let a = on a and b = on b and out = Rig.Buffer.create d (4 * n) in
  let module Sub = Rig.Submission in
  let refs = Sub.[| read 0 ~at:0; read 1 ~at:8; write 2 ~at:16 |] in
  let fn = require_some (Rig.Image.resolve image "add") in
  let s = Sub.make d [| Sub.launch ~refs fn ~params:32 |] in
  let run = Sub.Run.make s and blk = Sub.block s 0 in
  Sub.Run.groups run blk ((n + 63) / 64) 1 1;
  Sub.Run.threads run blk 64 1 1;
  Sub.Run.int64 run blk 24 (Int64.of_int n);
  ignore (Rig.submit run ~buffers:[| a; b; out |]);
  let bytes = Bytes.create (4 * n) in
  Rig.Buffer.blit_to_bytes out 0 bytes 0 (4 * n);
  Bytes.to_string bytes

let same_bytes (n, seed) =
  match Lazy.force gpus with
  | [] -> skip ~reason:"no GPU here whose add examples/01-add has" ()
  | gpus ->
      let st = Random.State.make [| seed |] in
      let floats () =
        let b = Bytes.create (4 * n) in
        for i = 0 to n - 1 do
          let v = Random.State.float st 2e6 -. 1e6 in
          Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v)
        done;
        Bytes.to_string b
      in
      let a = floats () and b = floats () in
      let host = run_add Rig.host (Lazy.force host_add) ~n a b in
      List.iter
        (fun (d, binary) ->
          equal ~msg:(Rig.name d) string host (run_add d binary ~n a b))
        gpus

let launch_case =
  Gen.with_pp
    (fun ppf (n, seed) -> Format.fprintf ppf "%d floats, seed %d" n seed)
    Gen.(pair (frequency [ (1, constant 1); (3, int_range 1 4096) ]) nat)

let () =
  exit
  @@ run "rig_gpus"
       [
         (* Before the launches, which hold every GPU they open. *)
         group ~timeout:60. "paths"
           [
             test
               "each path in preference order, with its GPUs' names and buses \
                or its refusal"
               order;
             test "first opens the first GPU that opens, or names each refusal"
               opens;
           ];
         group ~timeout:120. "launches"
           [
             prop ~count:20
               "one launch leaves the same bytes on the host and on each GPU"
               launch_case same_bytes;
           ];
       ]
