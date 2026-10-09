(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support
module E = S.Edge
module H = Rig_gpu_support.Host

let host r = Option.get (A.locate r).host
let address r = Option.get (A.locate r).address

(* A child forked after the parent opened GPU 0 opens the GPU on its own and
   runs a copy. It exits 0 once its copy holds the bytes, 1 if the open failed,
   2 if the bytes differ, 3 if its value is not reached within 10 s. With no
   kernel driver one process holds a GPU: the child's open is refused, and it
   exits 0 then, 4 if the open succeeds. *)
let forked () =
  S.with_ @@ fun _ ->
  match Unix.fork () with
  | 0 ->
      let code =
        match S.open_gpu () with
        | Error _ when S.driverless () -> 0
        | Ok c when S.driverless () ->
            A.stop c ~fault:None;
            4
        | Error why ->
            prerr_endline why;
            1
        | Ok c ->
            let src = Option.get (A.alloc c Rig_edge.Pinned 64) in
            let dst = Option.get (A.alloc c Rig_edge.Pinned 64) in
            H.write (host src) (String.make 64 'f');
            H.write (host dst) (String.make 64 '\000');
            ignore
              (E.submit c ~v:1
                 [| E.copy ~dst:(address dst) ~src:(address src) 64 |]);
            let rec reached n =
              A.signaled c >= 1
              || n > 0
                 && begin
                   A.sleep c ~seen:(A.signaled c) ~still_ms:500;
                   reached (n - 1)
                 end
            in
            if not (reached 20) then 3
            else if H.read (host dst) 64 = String.make 64 'f' then 0
            else 2
      in
      Unix._exit code
  | pid -> (
      match Unix.waitpid [] pid with
      | _, WEXITED n -> equal int ~msg:"the child's exit" 0 n
      | _, (WSIGNALED n | WSTOPPED n) ->
          failf "the child stopped on signal %d" n)

let () =
  S.hold ();
  exit
    (run "rig_amd fork"
       [
         group ~timeout:60. "fork"
           [
             test
               "a child forked after an open opens the GPU and runs work, \
                where the path lets two processes hold it"
               forked;
           ];
       ])
