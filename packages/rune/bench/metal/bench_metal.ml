(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune's compiled calls on the Mac's GPU, each row a call a user makes, which
   waits for what it computes. A case makes its values and compiles its call in
   its own worker, so that no process forks after Metal started in it.

   The rows are products whose layout the target's thread counts decide: a
   tensor core's splits keep [core_threads] threads, and a product of few
   outputs streams its operand with [stream_threads]. On an M1 Max, in GPU
   time, where the rows time a call and its wait: the float32 product runs
   1.18x slower at 16384 core threads and the key projection 1.8x slower at
   65536; at 16384 stream threads the vector times 16384 by 8192 runs 1.68x
   faster, and the one times 2880 by 512, a decode step's key and value
   projection, 1.50x slower. The suite runs in about 3 s warm. *)

let f32 = Nx.float32
let f16 = Nx.float16

(* [m] by [k] times [k] by [n], normal values cast to [dt], on the GPU. *)
let product (type b) (dt : (float, b) Nx.dtype) m k n () =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      let draw i s =
        Nx.place D.on (Nx.cast dt (Nx.Rng.normal ~key:(Nx.Rng.key i) f32 s))
      in
      let a = draw 1 [| m; k |] and b = draw 2 [| k; n |] in
      let f =
        Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) Nx.matmul
      in
      fun () ->
        let y = f a b in
        Nx.sync y

let cases =
  [
    ("matmul-f32-1024", product f32 1024 1024 1024);
    ("key-projection-f16-512x2880x512", product f16 512 2880 512);
    ("vecmat-f16-4096x4096", product f16 1 4096 4096);
    ("vecmat-f16-16384x8192", product f16 1 16384 8192);
    ("vecmat-f16-2880x512", product f16 1 2880 512);
  ]

let row (name, make) =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let call = make () in
      call ();
      call)
    (fun call -> call ())

(* Whether the Mac's GPU opens, asked of a fresh process. *)
let opens () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--opens" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let config =
  Thumper.Config.(
    default |> deadline 60. |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--opens" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_metal.gpu 0)) then 0 else 1)
  | [ _; "--warm" ] ->
      if opens () then List.iter (fun (_, make) -> make () ()) cases
  | _ ->
      if not (opens ()) then print_endline "rune_metal: skipped, no Metal"
      else exit (Thumper.run "rune_metal" ~config (List.map row cases))
