(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Each path's GPUs, in the preference order the .mli states. *)
let preferred () =
  let path name n = List.init n (fun _ -> name) in
  List.concat
    [
      path "METAL" (Rig_metal.count ());
      path "CUDA" (Rig_cuda.count ());
      path "NV" (Rig_nv_nvidia.count ());
      path "AMD" (Rig_amd_amdgpu.count ());
      path "NV-PCI" (Rig_nv_pci.count ());
      path "AMD-PCI" (Rig_amd_pci.count ());
    ]

let order () =
  equal (list string) (preferred ()) (List.map fst (Rig_gpus.openers ()))

let opens () =
  match Rig_gpus.openers () with
  | [] -> skip ~reason:"no driver sees a GPU" ()
  | ("METAL", open_) :: _ when Rig_metal_support.virtual_mac -> (
      match open_ () with
      | Error "Metal made no residency set" ->
          skip ~reason:"a virtual Mac's GPU makes no residency set" ()
      | Error why -> fail why
      | Ok d ->
          Rig.close d;
          fail "a virtual Mac's GPU opened a device")
  | (path, open_) :: _ -> (
      match open_ () with
      | Error why -> fail why
      | Ok d ->
          Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
          equal string ~msg:"the device of the path's GPU 0" path (Rig.name d))

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
    (List.filter_map
       (fun (path, open_) ->
         match open_ () with
         | Error _ -> None
         | Ok d -> (
             match binary path (Rig.arch d) with
             | Some b -> Some (d, b)
             | None ->
                 Rig.close d;
                 None))
       (Rig_gpus.openers ()))

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
         group ~timeout:120. "launches"
           [
             prop ~count:20
               "one launch leaves the same bytes on the host and on each GPU"
               launch_case same_bytes;
           ];
         group ~timeout:60. "openers"
           [
             test "one per GPU a path sees, paths in preference order" order;
             test "the first opens its path's GPU 0" opens;
           ];
       ]
