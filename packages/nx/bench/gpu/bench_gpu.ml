(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's eager kernels on GPUs, each beside its host twin: a cast from bfloat16
   to float32 at 4K, 1M and 16M elements, timed to the work's completion, and
   the first use of a kernel in a fresh process, which opens the GPU and loads
   the kernel's code objects. AMD loads code objects with no compiler, so the
   first use has no cold and warm cases. Rows exist for the GPUs the machine
   has: AMD GPU 0 under the kernel driver. The GPU is opened in each measuring
   worker, never in the parent that forks them; the host twins run on every
   machine. *)

let sizes = [ ("4K", 4096); ("1M", 1 lsl 20); ("16M", 16 lsl 20) ]

(* The first use: what the child process runs. *)
let first_use_child = "--first-use"

let first_use () =
  let d = Nx_amd.device 0 in
  let x = Nx.place (Nx.Placement.on d) (Nx.ones Nx.bfloat16 [| 4096 |]) in
  ignore (Nx.cast Nx.float32 x);
  Nx_device.synchronize (Nx.Device.memory d)

let cast_bf16_f32 ~gpu (label, n) =
  let input () = Nx.cast Nx.bfloat16 (Nx.rand Nx.float32 [| n |]) in
  let name = "cast-bf16-f32-" ^ label in
  let host =
    Thumper.bench_with_setup ~setup:input (name ^ "-host") (Nx.cast Nx.float32)
  in
  if not gpu then [ host ]
  else
    [
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let d = Nx_amd.device 0 in
          (Nx.Device.memory d, Nx.place (Nx.Placement.on d) (input ())))
        name
        (fun (m, x) ->
          let y = Nx.cast Nx.float32 x in
          Nx_device.synchronize m;
          y);
      host;
    ]

let first_use_case () =
  let exe = Sys.executable_name in
  Thumper.bench "first-use" (fun () ->
      let pid =
        Unix.create_process exe [| exe; first_use_child |] Unix.stdin
          Unix.stdout Unix.stderr
      in
      match Unix.waitpid [] pid with
      | _, WEXITED 0 -> ()
      | _ -> failwith "the first use failed")

let () =
  if Array.length Sys.argv = 2 && Sys.argv.(1) = first_use_child then (
    first_use ();
    exit 0);
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  let gpu = Nx_amd_device.count () > 0 in
  Thumper.run "nx_gpu"
    ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
    [
      Thumper.group "gpu"
        [
          Thumper.group "amd"
            (List.concat_map (cast_bf16_f32 ~gpu) sizes
            @ if gpu then [ first_use_case () ] else []);
        ];
    ]
  |> exit
