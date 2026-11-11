(* Tests of Tolk_engine on another machine's host: the other machine is rig's
   test agent, a real process on the loopback (Remote_job). *)

open Windtrap
open Tolk
module Engine = Tolk_engine

(* The host of a job over one agent, which the suite shares. *)
let remote =
  fixture
    ~teardown:(fun (j, a) ->
      Rig_remote.close j;
      Remote_job.kill a)
    (fun () ->
      let a = Remote_job.with_key_file Remote_job.start in
      (Remote_job.connect [ a ], a))

let host () = List.hd (Rig_remote.hosts (fst (remote ())))

(* out[i] = a[i] * n + b[i] for [size] elements, where out, a and b are the
   parameters 0, 1 and 2 and n a variable. *)
let axpy size =
  let open Ops.O in
  let buffer slot = Shape.placeholder ~slot [ size ] Float32 in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let i = Ops.range ~axis_type:Weak (Int size) [ 0 ] in
  let at b = Ops.index (buffer b) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name:"axpy" ())
    [ Ops.end_ (Ops.store (at 0) ((at 1 * Ops.cast n Float32) + at 2)) [ i ] ]

(* A time of a schedule on another machine's host is its kernel's time there,
   which lies within the host's times around the call, and runs the schedule
   once there: one submission of its run's words. *)
let timed () =
  let h = host () in
  let size = 1024 in
  let prg = Codegen.to_program (axpy size) (Engine.renderer h) in
  let s =
    Engine.link_call
      ~devices:(fun _ -> h)
      (Ops.call prg
         (List.init 3 (fun slot ->
              Shape.param ~shape:[ Ops.Int size ] ~device:(Rig.name h) slot
                Float32)))
  in
  let slots = Engine.slots s in
  let before = Rig.submitted h in
  let t0 = Rig.Profile.now () in
  let t = Engine.time ~vars:[ ("n", 3) ] s slots in
  let window = Float.of_int (Rig.Profile.now () - t0) *. 1e-9 in
  greater ~msg:"the time" float_exact ~than:0. t;
  at_most ~msg:"the time" float_exact ~than:window t;
  equal ~msg:"its submits" int (before + 1) (Rig.submitted h)

let () =
  Watchdog.start ();
  exit
    (run "Tolk_engine on another machine's host"
       [ test "a time is its kernel's time there, within the call" timed ])
