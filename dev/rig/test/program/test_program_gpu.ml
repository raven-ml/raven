(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A description on each GPU driver the machine has that runs launches: it
   leaves the bytes the same submission leaves when made by hand. Kernels come
   from the driver's conformance fixtures
   (Rig_gpu_support.Conformance.launch_binary). *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module G = Rig_program

module type Gpu = Rig_gpu_support.Conformance

let words = 256

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let le32 v =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int v);
  Bytes.to_string b

let get b =
  let n = B.length b in
  let h = B.create Rig.host n in
  B.copy ~src:b ~dst:h;
  let s = Bytes.create n in
  B.blit_to_bytes h 0 s 0 n;
  Bytes.to_string s

(* [ids] into slot 0 over 4 groups of 64 threads, [a = 5], [b = 3], [f = 2.5]:
   word [k] is [5 + 3k + 2]. Then [twice] from slot 0 into slot 1, [c = 1]. *)
let ids_params =
  le64 0 ^ le64 5 ^ le32 3 ^ le32 (Int32.to_int (Int32.bits_of_float 2.5))

let twice_params = le64 0 ^ le64 0 ^ le32 1

let by_hand d image queue x y =
  let launch kernel params refs =
    {
      Sub.queue;
      after = [||];
      work = Launch { image; kernel; params = String.length params; refs };
    }
  in
  let s =
    Sub.make ~reads:0 ~writes:2 d
      [|
        launch "ids" ids_params [| { Sub.at = 0; slot = 0 } |];
        launch "twice" twice_params
          [| { Sub.at = 0; slot = 1 }; { Sub.at = 8; slot = 0 } |];
      |]
  in
  let run = Sub.Run.make () in
  let store i params ~groups ~threads =
    let b = Sub.block s i in
    Sub.Run.groups run b groups 1 1;
    Sub.Run.threads run b threads 1 1;
    Sub.Run.shared run b 0;
    for w = 0 to (String.length params / 4) - 1 do
      Sub.Run.int32 run b (4 * w)
        (Int32.to_int (String.get_int32_le params (4 * w)))
    done
  in
  store 0 ids_params ~groups:4 ~threads:64;
  store 1 twice_params ~groups:words ~threads:1;
  ignore (Rig.submit s ~run ~reads:[||] ~writes:[| x; y |] ~waits:[||])

let described d binary queue =
  let launch kernel params refs ~groups ~threads =
    {
      G.queue;
      after = [||];
      work =
        G.Launch
          {
            image = 0;
            kernel;
            params = { G.bytes = params; holes = [||] };
            refs;
            groups = (G.Fixed groups, G.Fixed 1, G.Fixed 1);
            threads = (G.Fixed threads, G.Fixed 1, G.Fixed 1);
            shared = G.Fixed 0;
          };
    }
  in
  {
    G.devices = [| Rig.arch d |];
    code = [||];
    ints = 0;
    memory = [||];
    images = [| { G.device = 0; binary = { G.bytes = binary; holes = [||] } } |];
    inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 4 * words });
    steps =
      [|
        G.Submit
          {
            device = 0;
            reads = [||];
            writes = [| G.Input 0; G.Input 1 |];
            fixed = [||];
            parts =
              [|
                launch "ids" ids_params
                  [| { Sub.at = 0; slot = 0 } |]
                  ~groups:4 ~threads:64;
                launch "twice" twice_params
                  [| { Sub.at = 0; slot = 1 }; { Sub.at = 8; slot = 0 } |]
                  ~groups:words ~threads:1;
              |];
          };
      |];
  }

let same_bytes (module Gpu : Gpu) () =
  Gpu.with_ @@ fun t ->
  let d = t.d in
  let binary =
    match Gpu.launch_binary () with
    | Some b -> b
    | None -> skip ~reason:"the driver runs no launch" ()
  in
  let queue =
    match
      List.find_opt
        (fun (q : Rig.queue) -> List.mem Rig.Launch q.runs)
        (Rig.queues d)
    with
    | Some q -> q.name
    | None -> skip ~reason:"the device runs no launch" ()
  in
  let image = Result.get_ok (Rig.Image.load d binary) in
  let x = B.create d (4 * words) and y = B.create d (4 * words) in
  by_hand d image queue x y;
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load (described d binary queue) [| d |])
  in
  let x' = B.create d (4 * words) and y' = B.create d (4 * words) in
  ignore (G.run p { inputs = [| x'; y' |]; ints = [||] });
  let le32s f = String.concat "" (List.init words (fun k -> le32 (f k))) in
  equal string ~msg:"ids, by hand" (le32s (fun k -> 7 + (3 * k))) (get x);
  equal string ~msg:"twice, by hand" (le32s (fun k -> 15 + (6 * k))) (get y);
  equal string ~msg:"ids, described" (get x) (get x');
  equal string ~msg:"twice, described" (get y) (get y')

let gpus : (module Gpu) list =
  [
    (module Rig_metal_support);
    (module Rig_cuda_support);
    (module Rig_nv_support);
    (module Rig_amd_support);
  ]

let () =
  if List.exists (fun (module Gpu : Gpu) -> Gpu.present ()) gpus then
    Rig_gpu_lock.hold ();
  exit
    (run "rig.program.gpu"
       (List.map
          (fun (module Gpu : Gpu) ->
            test ~timeout:120.
              (Gpu.class_
             ^ ": a description leaves the bytes of its submission made by hand"
              )
              (same_bytes (module Gpu)))
          gpus))
