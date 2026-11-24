(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Sub = Rig.Submission

let device () =
  match Nx_kernels_support.metal () with
  | Some b -> b.device
  | None -> Windtrap.skip ~reason:"no Metal GPU nx.metal computes on" ()

let harness =
  lazy
    (match Rig.Image.load (device ()) Harness.metallib with
    | Ok image -> image
    | Error why -> failwith ("the harness does not load: " ^ why))

type word = A of Rig.Buffer.t | W of int

(* One launch's submission: a write ref for each address, its slot the address's
   rank among them. *)
let launch kernel ~groups ~threads ~shared words =
  let d = device () in
  let f =
    match Rig.Image.resolve (Lazy.force harness) kernel with
    | Some f -> f
    | None -> invalid_arg ("Nx_metal_support.launch: no kernel " ^ kernel)
  in
  let ws = Array.of_list words in
  let buffers =
    Array.of_list
      (List.filter_map (function A b -> Some b | W _ -> None) words)
  in
  let slot = ref 0 in
  let refs =
    Array.of_list
      (List.concat
         (List.mapi
            (fun i -> function
              | A _ ->
                  incr slot;
                  [ Sub.write (!slot - 1) ~at:(8 * i) ]
              | W _ -> [])
            words))
  in
  let s = Sub.make d [| Sub.launch ~refs f ~params:(8 * Array.length ws) |] in
  let params = Bytes.make (8 * Array.length ws) '\000' in
  Array.iteri
    (fun i w ->
      let v = match w with A _ -> 0 | W v -> v in
      Bytes.set_int64_le params (8 * i) (Int64.of_int v))
    ws;
  let args =
    { Sub.groups = (groups, 1, 1); threads = (threads, 1, 1); shared; params }
  in
  Rig.Point.wait (Rig.submit s ~buffers ~args:[| args |])

(* Parts

   The bounds are global, so concurrent [parted]s count themselves and the last
   to end restores them. *)

let parts_lock = Mutex.create ()
let parting = ref 0
let budgeted = ref false

let parted ?budget ?(launches = Nx_metal.Plan.part_launches)
    ?(seconds = Nx_metal.Plan.part_budget) f =
  let module Plan = Nx_metal.Plan in
  Mutex.protect parts_lock (fun () ->
      if !parting = 0 then begin
        Plan.bound_parts ~launches ~seconds;
        Option.iter Plan.bound_budget budget;
        budgeted := Option.is_some budget
      end;
      incr parting);
  let restore () =
    Mutex.protect parts_lock (fun () ->
        decr parting;
        if !parting = 0 then begin
          Plan.bound_parts ~launches:Plan.part_launches
            ~seconds:Plan.part_budget;
          if !budgeted then Plan.bound_budget Plan.budget
        end)
  in
  Fun.protect ~finally:restore f
