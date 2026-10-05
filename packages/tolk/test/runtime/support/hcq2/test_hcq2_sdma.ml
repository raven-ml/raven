open Windtrap
open Tolk

(* The suite runs with HCQ_NUM_SDMA=1, which settings read once per process. *)

let target =
  {
    Helpers.Target.device = "AMD";
    renderer = "";
    arch = "gfx1100";
    interface = "";
    indices = "";
  }

let devices =
  let events = Null_queue.events () in
  fun _ ->
    let queues =
      {
        Hcq2.commands = Null_queue.commands events;
        copy_queue = true;
        submission = Buffered;
        host = "CPU";
        reaches = (fun _ -> true);
      }
    in
    { Hcq2.target; queues = Some queues }

let storage d = Ops.new_buffer (Single d) 4 Float32

let copy_queues =
  test
    "AMD copies between peers take the HCQ_NUM_SDMA queues it sets, whatever \
     ALL2ALL says" (fun () ->
      let src = storage "AMD:0" in
      let calls =
        List.map (fun d -> Ops.store_call (storage d) src) [ "AMD:1"; "AMD:2" ]
      in
      let batched =
        Setting.context
          [ B (Setting.all2all, 1) ]
          (fun () ->
            Hcq2.sched_batches ~devices ~profile:false (Ops.v Linear ~src:calls))
      in
      let names =
        List.map
          (fun s ->
            match Ops.arg (Ops.nth (Ops.without_after s) 0) with
            | Queue { queue; _ } -> queue
            | _ -> "")
          (Ops.src (Ops.nth (List.hd (Ops.src batched)) 0))
      in
      let copies = List.filter (String.starts_with ~prefix:"COPY") names in
      equal (list string) [ "COPY:0" ] (List.sort_uniq String.compare copies))

let keyed =
  test "the compiled batches key on HCQ_NUM_SDMA" (fun () ->
      equal (option string) (Some "1")
        (List.assoc_opt "HCQ_NUM_SDMA" (Setting.shaping ())))

let () = exit (run "Hcq2 with HCQ_NUM_SDMA=1" [ copy_queues; keyed ])
