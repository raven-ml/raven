(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support
module E = S.Edge

let host r = Option.get (A.host r)
let address r = Option.get (A.address r)

(* A child forked after the parent opened GPU 0 opens the GPU on its own and
   runs a copy. It exits 0 once its copy holds the bytes, 1 if the open failed,
   2 if the bytes differ, 3 if its value is not reached within 10 s. *)
let forked () =
  S.with_gpu @@ fun _ ->
  match Unix.fork () with
  | 0 ->
      let code =
        match Rig_amd_amdgpu.open_ 0 with
        | Error why ->
            prerr_endline why;
            1
        | Ok c ->
            let src = Option.get (A.alloc c `Pinned 64) in
            let dst = Option.get (A.alloc c `Pinned 64) in
            S.write (host src) (String.make 64 'f');
            S.write (host dst) (String.make 64 '\000');
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
            else if S.read (host dst) 64 = String.make 64 'f' then 0
            else 2
      in
      Unix._exit code
  | pid -> (
      match Unix.waitpid [] pid with
      | _, WEXITED n -> equal int ~msg:"the child's exit" 0 n
      | _, (WSIGNALED n | WSTOPPED n) ->
          failf "the child stopped on signal %d" n)

let () =
  S.hold_gpu ();
  exit
    (run "rig_amd fork"
       [
         group ~timeout:60. "fork"
           [
             test "a child forked after an open opens the GPU and runs work"
               forked;
           ];
       ])
