(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a program that differentiates eagerly pays for what it links.

   [start] runs this executable again as [--grad]: a fresh process's start,
   which initialises every module it links, then one eager gradient.
   [full-major] is a major collection after one eager gradient in this process:
   every major cycle of a program marks the heap its linked modules keep
   alive. *)

let x () = Nx.init Nx.float32 [| 16 |] (fun i -> float_of_int i.(0))
let grad () = Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) (x ())

let start () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--grad" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | _ -> failwith "bench_eager: the eager gradient failed"

let config = Thumper.Config.(default |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--grad" ] -> ignore (Sys.opaque_identity (grad ()))
  | _ ->
      let g = grad () in
      Thumper.run "eager" ~config
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        [
          Thumper.bench "start" start;
          Thumper.bench "full-major" (fun () ->
              Gc.full_major ();
              Sys.opaque_identity g);
        ]
      |> exit
