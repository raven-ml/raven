(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Adding two arrays on a GPU.

   A kernel is a function that a device runs once for each thread of a grid.
   This program opens the first GPU here for which it has a binary of [add],
   loads the binary as an image, copies two arrays into the GPU's memory,
   launches [add] over a million threads and copies the sum back. Without
   such a GPU, the host runs the same launch on its cores, from an ELF
   object. Each binary takes the same parameters, the addresses of the three
   arrays then their length, so the program runs one launch on any device.

   With an argument, the name of a path of [Rig_gpus.openers] such as CUDA
   or NV, it tries only that path's GPUs. The device that ran [add] goes to
   stderr, and the sum to stdout, which is the same on every machine. *)

open Rig

let n = 1 lsl 20
let group = 256
let contents f = In_channel.with_open_bin f In_channel.input_all

(* The file of [add] that [d] loads, by its arch. Metal and CUDA compile a
   portable form for the GPU they load it on. AMD's and NVIDIA's kernel
   drivers and the host take one binary per architecture. *)
let binary d =
  match arch d with
  | ("arm64" | "x86_64") as a -> Some ("add_" ^ a ^ ".o")
  | "gfx1201" -> Some "add_gfx1201.hsaco"
  | "sm_89" -> Some "add_sm_89.cubin"
  | a when String.starts_with ~prefix:"sm_" a -> Some "add.ptx"
  | a when String.starts_with ~prefix:"Apple" a || a = "Mac2" ->
      Some "add.metallib"
  | _ -> None

(* A GPU this program opened but cannot use is closed, so the program keeps
   only the device it runs on. The host is never closed. *)
let give_up d why =
  prerr_endline why;
  if not (equal d host) then close d;
  None

(* [add] loaded on the device [open_device] opens, if it opens and has a
   binary of [add] that loads. *)
let load (open_device : unit -> (t, string) result) =
  match open_device () with
  | Error why ->
      prerr_endline why;
      None
  | Ok d -> (
      match binary d with
      | None -> give_up d (Printf.sprintf "no add for %s (%s)" (name d) (arch d))
      | Some file -> (
          match Image.load d (contents ("kernels/" ^ file)) with
          | Ok image -> Some image
          | Error why -> give_up d why))

(* The GPUs to try, in rig's order of preference, then the host. *)
let devices () =
  let gpus = Rig_gpus.openers () in
  let gpus =
    match Sys.argv with
    | [| _ |] -> gpus
    | [| _; path |] -> List.filter (fun (p, _) -> p = path) gpus
    | _ -> invalid_arg "usage: main.exe [PATH]"
  in
  List.map snd gpus @ [ (fun () -> Ok host) ]

let floats f = Bigarray.(Array1.init float32 c_layout n f)

let () =
  let image = Option.get (List.find_map load (devices ())) in
  let d = Image.device image in

  (* Two arrays in [d]'s memory, copied from the host's, and one for the
     sum. *)
  let x = floats float_of_int and y = floats (fun _ -> 0.5) in
  let on_d v =
    let b = Buffer.create d (4 * n) in
    Buffer.copy ~src:(Buffer.of_bigarray v) ~dst:b;
    b
  in
  let a = on_d x and b = on_d y and out = Buffer.create d (4 * n) in

  (* One launch of [add]. Its 32 bytes of parameters hold refs to the
     submit's buffers 0 and 1, which it reads, and 2, which it writes, then
     [n] at byte 24. The run holds the grid, one thread per element in
     groups of 256. *)
  let add = Option.get (Image.resolve image "add") in
  let refs = Submission.[| read 0 ~at:0; read 1 ~at:8; write 2 ~at:16 |] in
  let s = Submission.(make d [| launch ~refs add ~params:32 |]) in
  let run = Submission.Run.make s and k = Submission.block s 0 in
  Submission.Run.(
    groups run k (n / group) 1 1;
    threads run k group 1 1;
    int64 run k 24 (Int64.of_int n));
  let p = submit run ~buffers:[| a; b; out |] in
  Format.eprintf "%s (%s) ran add at %a@." (name d) (arch d) Point.pp p;

  (* The copy back reads [out], so it follows the launch, which writes it. *)
  let z = floats (fun _ -> 0.) in
  Buffer.copy ~src:out ~dst:(Buffer.of_bigarray z);
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if z.{i} <> x.{i} +. y.{i} then incr wrong
  done;
  Printf.printf "%d floats added: z[%d] = %.1f, %d wrong\n" n (n - 1)
    z.{n - 1} !wrong
