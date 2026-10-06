(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's eager kernels on GPUs, each beside its host twin: a cast from bfloat16
   to float32, the exponential of a float32 value, the sum of two, the sum of
   one's elements and their running sum, a uniform draw, a sort with its
   positions (at 4K and 1M: the host's sort of 16M overruns a case's deadline),
   a gather of every element at drawn positions, a sum of as many updates at
   drawn positions, the 3 by 3 windows of an image and their sum back (at 4K and
   1M: nine times 16M elements overrun the host's case), the concatenation of
   two halves and a square padded by one, at 4K, 1M and 16M elements, and the
   product of a float32 square matrix by itself, of 128, 1,024 and 4,096 rows,
   twenty dependent sums of 4K elements, the many small operations of eager
   code, which pay the launch latency each, timed to the work's completion, and
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

(* The rows of [op] over [input n], named [name] and [name ^ "-host"]; [put]
   places an input on a device. *)
let rows ~put ~gpu name (label, n) ~input ~op =
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
          and x = put (Nx.Placement.on d) (input n) in
          warm m (fun () -> op x);
          (m, x))
        name
        (fun (m, x) ->
          let y = op x in
          Nx_device.synchronize m;
          y);
      host;
    ]

(* Placing one value, and two. *)
let one p x = Nx.place p x
let two p (a, b) = (Nx.place p a, Nx.place p b)
let floats n = Nx.rand Nx.float32 [| n |]

(* [n] positions in [0, n), drawn. *)
let positions n =
  Nx.cast Nx.int64 (Nx.mul_s (Nx.rand Nx.float32 [| n |]) (Float.of_int n))

(* A 3 by 3 window of step 1 padded to keep the extent, over an image of about
   [n] elements. *)
let three = [| 3; 3 |]
let one_step = [| 1; 1 |]
let same = [| (1, 1); (1, 1) |]

let image n =
  let side = Float.to_int (Float.sqrt (Float.of_int n)) in
  Nx.rand Nx.float32 [| 1; 1; side; side |]

(* A square of about [n] elements. *)
let square n =
  let side = Float.to_int (Float.sqrt (Float.of_int n)) in
  Nx.rand Nx.float32 [| side; side |]

let squares = [ ("128", 128); ("1024", 1024); ("4096", 4096) ]

let matrix n = Nx.rand Nx.float32 [| n; n |]

(* Twenty sums, each of the one before and [x]. *)
let chain = 20

let chained x =
  let y = ref x in
  for _ = 1 to chain do
    y := Nx.add !y x
  done;
  !y

let cases ~gpu size =
  rows ~put:one ~gpu "cast-bf16-f32" size
    ~input:(fun n -> Nx.cast Nx.bfloat16 (floats n))
    ~op:(Nx.cast Nx.float32)
  @ rows ~put:one ~gpu "unary-exp" size ~input:floats ~op:Nx.exp
  @ rows ~put:one ~gpu "binary-add" size ~input:floats ~op:(fun x -> Nx.add x x)
  @ rows ~put:one ~gpu "reduce-sum" size ~input:floats ~op:(fun x -> Nx.sum x)
  @ rows ~put:one ~gpu "scan-cumsum" size ~input:floats ~op:(fun x ->
      Nx.cumsum x)
  @ rows
      ~put:(fun p (k, n) -> (Nx.place p k, n))
      ~gpu "rng-uniform" size
      ~input:(fun n -> ((Nx.Rng.key 7 :> Nx.int32_t), n))
      ~op:(fun (k, n) -> Nx.Rng.uniform (Nx.Rng.of_tensor k) Nx.float32 [| n |])
  @ (if snd size > 1 lsl 20 then []
     else rows ~put:one ~gpu "sort" size ~input:floats ~op:(fun x -> Nx.sort x))
  @ rows ~put:two ~gpu "gather" size
      ~input:(fun n -> (floats n, positions n))
      ~op:(fun (x, indices) -> Nx.take ~indices x)
  @ rows
      ~put:(fun p (x, i, v) -> (Nx.place p x, Nx.place p i, Nx.place p v))
      ~gpu "scatter-add" size
      ~input:(fun n -> (Nx.zeros Nx.float32 [| n |], positions n, floats n))
      ~op:(fun (x, indices, values) ->
        Nx.scatter ~mode:`Add ~axis:0 ~indices ~values x)
  @ rows ~put:two ~gpu "cat" size
      ~input:(fun n -> (floats (n / 2), floats (n / 2)))
      ~op:(fun (a, b) -> Nx.concatenate ~axis:0 [ a; b ])
  @ (if snd size > 1 lsl 20 then []
     else
       rows ~put:one ~gpu "unfold" size ~input:image ~op:(fun x ->
           Nx.extract_patches ~kernel_size:three ~stride:one_step
             ~dilation:one_step ~padding:same x)
       @ rows ~put:one ~gpu "fold" size
           ~input:(fun n ->
             Nx.extract_patches ~kernel_size:three ~stride:one_step
               ~dilation:one_step ~padding:same (image n))
           ~op:(fun p ->
             let side =
               Nx.dim (-1) p |> Float.of_int |> Float.sqrt |> Float.to_int
             in
             Nx.combine_patches ~output_size:[| side; side |] ~kernel_size:three
               ~stride:one_step ~dilation:one_step ~padding:same p))
  @ rows ~put:one ~gpu "pad" size ~input:square
      ~op:(Nx.pad [| (1, 1); (1, 1) |] 0.)

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
            @ List.concat_map
                (fun size ->
                  rows ~put:one ~gpu "matmul" size ~input:matrix ~op:(fun x ->
                      Nx.matmul x x))
                squares
            @ rows ~put:one ~gpu "chain20-add" (List.hd sizes) ~input:floats
                ~op:chained
            @ if gpu then [ first_use_case () ] else []);
        ];
    ]
  |> exit
