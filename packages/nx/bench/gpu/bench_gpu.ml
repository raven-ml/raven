(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's eager kernels on GPUs, each beside its host twin: a cast from bfloat16
   to float32, the exponential of a float32 value and the sum of two, at 4K, 1M
   and 16M elements, timed to the work's completion, and the first use of a
   kernel in a fresh process, which opens the GPU and loads the kernel's code
   objects. AMD loads code objects with no compiler, so the first use has no
   cold and warm cases. Rows exist for the GPUs the machine has: AMD GPU 0 under
   the kernel driver. The GPU is opened in each measuring worker, never in the
   parent that forks them; the host twins run on every machine. *)

let sizes = [ ("4K", 4096); ("1M", 1 lsl 20); ("16M", 16 lsl 20) ]

(* The first use: what the child process runs. *)
let first_use_child = "--first-use"

let first_use () =
  let d = Nx_amd.device 0 in
  let x = Nx.place (Nx.Placement.on d) (Nx.ones Nx.bfloat16 [| 4096 |]) in
  ignore (Nx.cast Nx.float32 x);
  Nx_device.synchronize (Nx.Device.memory d)

(* The device's cache holds memory for [f]'s results. A result's memory returns
   to the cache once the result is collected, and a result that finds none there
   allocates fresh memory from the driver: 1.5 ms for 64 MiB on an R9700 under
   amdgpu, against 2 us from the cache. Results computed and collected before
   the timing leave memory in the cache, so the rows time the kernels and the
   cache, not when the collector last ran. *)
let warmups = 8

let warm m f =
  for _ = 1 to warmups do
    ignore (Sys.opaque_identity (f ()))
  done;
  Nx_device.synchronize m;
  Gc.full_major ()

(* The rows of [op] over [input ()], of [n] elements, named [name] and [name ^
   "-host"]. *)
let rows ~gpu name (label, n) ~input ~op =
  let name = name ^ "-" ^ label in
  let host =
    Thumper.bench_with_setup ~setup:(fun () -> input n) (name ^ "-host") op
  in
  if not gpu then [ host ]
  else
    [
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let d = Nx_amd.device 0 in
          let m = Nx.Device.memory d
          and x = Nx.place (Nx.Placement.on d) (input n) in
          warm m (fun () -> op x);
          (m, x))
        name
        (fun (m, x) ->
          let y = op x in
          Nx_device.synchronize m;
          y);
      host;
    ]

let floats n = Nx.rand Nx.float32 [| n |]

let cases ~gpu size =
  rows ~gpu "cast-bf16-f32" size
    ~input:(fun n -> Nx.cast Nx.bfloat16 (floats n))
    ~op:(Nx.cast Nx.float32)
  @ rows ~gpu "unary-exp" size ~input:floats ~op:Nx.exp
  @ rows ~gpu "binary-add" size ~input:floats ~op:(fun x -> Nx.add x x)

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
            (List.concat_map (cases ~gpu) sizes
            @ if gpu then [ first_use_case () ] else []);
        ];
    ]
  |> exit
