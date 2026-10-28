(* Tests of Tolk.Jit's beam width: JITBEAM's, or BEAM's without it. The
   environment is read once per process, so each setting of JITBEAM is a run of
   its own, which runs the group named after it. *)

open Windtrap
open Tolk

let devices name = List.assoc name (Run.devices ())

(* The capture of a kernel adding [c] to four floats on the host. A kernel is
   searched once per process, so each test lowers a kernel of its own. *)
let captured c =
  let i = Ops.range (Int 4) [ 0 ] in
  let at slot = Ops.index (Call.placeholder ~slot [ 4 ] Float32) [ i ] in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"inc" ())
      [
        Ops.end_
          (Ops.store (at 0) (Ops.add (at 1) (Ops.float ~dtype:Float32 c)))
          [ i ];
      ]
  in
  let buffer () = Ops.new_buffer (Single "CPU") 4 Float32 in
  let out = buffer () and x = buffer () in
  (Ops.v Op.Linear ~src:[ Ops.call kernel [ out; x ] ], out, x)

let kernels = ref 0

(* The beam widths the lowering searches a new kernel with, with BEAM at
   [width]. *)
let searched ?beam width =
  incr kernels;
  let widths = ref [] in
  let search w s =
    widths := w :: !widths;
    s
  in
  let linear, out, x = captured (float_of_int !kernels) in
  Setting.context
    [ B (Setting.beam, width) ]
    (fun () ->
      ignore
        (Jit.jit_lower ?beam ~search
           ~targets:(fun n -> Tolk_engine.target (devices n))
           ~held_bufs:[ out ] ~inputs:[ x ] linear));
  !widths

(* A test that assumes JITBEAM holds [value]. *)
let under value name f =
  test name (fun () ->
      if Sys.getenv_opt "JITBEAM" <> value then
        skip ~reason:"runs under its own JITBEAM" ()
      else f ())

(* Two lowerings, each [(beam, width)] at an explicit width or none under BEAM
   at [width]: their settings are equal and they search at one width. *)
let alike (beam, width) (beam', width') =
  let at beam width =
    let shaping =
      Setting.context [ B (Setting.beam, width) ] (fun () ->
          Jit.shaping ?beam ())
    in
    (shaping, searched ?beam width)
  in
  let s, w = at beam width and s', w' = at beam' width' in
  equal (list (pair string string)) ~msg:"settings" s s';
  equal (list int) ~msg:"widths" w w'

let unset =
  group "JITBEAM unset"
    [
      under None "a width given and BEAM's alike compile alike" (fun () ->
          alike (Some 2, 5) (None, 2));
      under None "BEAM at 0 searches no kernel" (fun () ->
          equal (list int) [] (searched 0));
      under None "a kernel is searched with BEAM's width" (fun () ->
          equal (list int) [ 2 ] (searched 2));
    ]

let three =
  group "JITBEAM=3"
    [
      under (Some "3") "a width given and JITBEAM's alike compile alike"
        (fun () -> alike (Some 3, 1) (None, 0));
      under (Some "3") "a kernel is searched with JITBEAM's width, not BEAM's"
        (fun () -> equal (list int) [ 3 ] (searched 2));
      under (Some "3") "a kernel is searched with BEAM at 0" (fun () ->
          equal (list int) [ 3 ] (searched 0));
      under (Some "3") "BEAM keeps its value once the lowering returns"
        (fun () ->
          equal int 2
            (Setting.context
               [ B (Setting.beam, 2) ]
               (fun () ->
                 ignore (searched 2);
                 Setting.value Setting.beam)));
    ]

let zero =
  group "JITBEAM=0"
    [
      under (Some "0") "no kernel is searched whatever BEAM is" (fun () ->
          equal (list int) [] (searched 2));
    ]

let () = exit (run "Jit JITBEAM" [ unset; three; zero ])
