(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Rig.Profile

let record n () =
  P.take (fun () ->
      for _ = 1 to n do
        P.span "work" ignore
      done)

(* [n] submits of a copy on a memory device while a profile is taken: each
   leaves the profile an entry, which [take] reads before it returns. The
   device opens in the process that measures, since a fork loses it. *)
let submits n =
  let copy =
    lazy
      (let d = Rig.memory_device "bench:profile" in
       let src = Rig.Buffer.create d 64 and dst = Rig.Buffer.create d 64 in
       Rig.Submission.(make d [| copy ~src:(Buffer src) ~dst:(Buffer dst) |]))
  and run = Rig.Submission.Run.make () in
  fun () ->
    let s = Lazy.force copy in
    P.take (fun () ->
        for _ = 1 to n do
          ignore (Rig.submit s ~run ~buffers:[||])
        done)

let events n =
  List.init n (fun i ->
      P.Span
        {
          device = Rig.host;
          lane = "host";
          name = "work";
          start = i mod 100 * 1000;
          stop = ((i mod 100) + 1) * 1000;
        })

let () =
  Out_channel.with_open_bin Filename.null (fun oc ->
      exit
        (Thumper.run "rig_profile"
           [
             Thumper.group "record"
               [
                 Thumper.bench "100" (record 100);
                 Thumper.bench "10000" (record 10_000);
               ];
             Thumper.group "take" [ Thumper.bench "submits-100" (submits 100) ];
             Thumper.group "export"
               (List.map
                  (fun n ->
                    let input = Thumper.black_box (events n) in
                    Thumper.bench (string_of_int n) (fun () ->
                        P.output_chrome_trace oc input))
                  [ 100; 10_000 ]);
           ]))
