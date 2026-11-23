(* Tests of Tolk_engine.compile over the disk cache: a later process that
   compiles an equal sink reads its schedule back whole, runs no compiler and
   computes what the sink computes; nothing is kept where a hit would skip what
   the call asks for; and an entry that does not read back is made anew. Each
   process is a part of this executable, played in a child. *)

open Windtrap
open Tolk
module Engine = Tolk_engine

let cpu = "CPU"

let devices = function
  | "CPU" -> Rig.host
  | name -> invalid_arg ("no device is named " ^ name)

(* Cases

   A case is a sink over float32 storage on the host: the storage each run
   binds, in slot order, and the storage the link binds, each with its contents
   before the run. *)

type case = {
  sink : Ops.t;
  inputs : (Ops.t * float array) list;
  held : (Ops.t * float array) list;
}

let buffer xs = Ops.new_buffer cpu (Array.length xs) Float32
let storage xs = (buffer xs, xs)
let stored out v = Ops.after out [ Ops.store out v ]

(* [(x + c) * x], [c] held: a constant the function captures. *)
let arithmetic () =
  let ((x, _) as xi) = storage [| 3.; 5. |] in
  let ((c, _) as ci) = storage [| 1.; 2. |] in
  let ((out, _) as oi) = storage [| 0.; 0. |] in
  {
    sink = Ops.sink [ stored out Ops.O.((x + c) * x) ];
    inputs = [ xi; oi ];
    held = [ ci ];
  }

let reduction () =
  let ((x, _) as xi) = storage [| 1.; 3.; 5.; 7. |] in
  let ((out, _) as oi) = storage [| 0. |] in
  {
    sink =
      Ops.sink [ stored out (Shape.reshape (Shape.rop x Add [ 0 ]) [ Int 1 ]) ];
    inputs = [ xi; oi ];
    held = [];
  }

(* A loop of three trips around one call: the call adds its row of [xs] to the
   carry [c] and stores twice the carry into its row of [ys]. *)
let loop ?(axis = 100) () =
  let k = 4 and n = 3 in
  let p slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let body =
    Ops.sink
      [
        Ops.store (p 0) Ops.O.(p 0 + p 1);
        Ops.store (p 2) Ops.O.(p 0 * float 2.);
      ]
  in
  let ((c, _) as ci) = storage [| 1.; 0.; -1.; 2. |] in
  let ((xs, _) as xsi) = storage (Array.init (n * k) Float.of_int) in
  let ((ys, _) as ysi) = storage (Array.make (n * k) 0.) in
  let r = Ops.range ~axis_type:Loop (Int n) [ axis ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let e =
    Ops.end_ (Ops.call ~precompile:true body [ c; row xs; row ys ]) [ r ]
  in
  {
    sink = Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ];
    inputs = [ ci; xsi; ysi ];
    held = [];
  }

(* [loop] in a process whose counters of storage have moved, its range numbered
   apart: the same sink up to the numbers of its ranges and storage. *)
let renumbered () =
  ignore (List.init 3 (fun _ -> buffer [| 0. |]));
  loop ~axis:7 ()

(* [a - b], its inputs taken in the order [a b], or [b a]. *)
let subtraction ~swapped () =
  let ((a, _) as ai) = storage [| 10. |] in
  let ((b, _) as bi) = storage [| 3. |] in
  let ((out, _) as oi) = storage [| 0. |] in
  {
    sink = Ops.sink [ stored out Ops.O.(a - b) ];
    inputs = (if swapped then [ bi; ai; oi ] else [ ai; bi; oi ]);
    held = [];
  }

(* [x * c], [c] held at [k]: one sink whatever [k]. *)
let scaled k () =
  let ((x, _) as xi) = storage [| 4.; 5. |] in
  let ((c, _) as ci) = storage [| k; k |] in
  let ((out, _) as oi) = storage [| 0.; 0. |] in
  {
    sink = Ops.sink [ stored out Ops.O.(x * c) ];
    inputs = [ xi; oi ];
    held = [ ci ];
  }

let cases =
  [
    ("arithmetic", arithmetic);
    ("reduction", reduction);
    ("loop", fun () -> loop ());
    ("renumbered", renumbered);
    ("sub_ab", subtraction ~swapped:false);
    ("sub_ba", subtraction ~swapped:true);
    ("times_two", scaled 2.);
    ("times_three", scaled 3.);
  ]

(* What each input holds after a run of the case, as Tensors computes the
   sink. *)
let expected name =
  let c = (List.assoc name cases) () in
  let slot u =
    match Ops.arg u with
    | Param p -> p.slot
    | _ -> invalid_arg "storage has a parameter argument"
  in
  let buffers =
    List.map
      (fun (u, xs) -> (slot u, Array.map (fun x -> `Float x) xs))
      (c.inputs @ c.held)
  in
  let writes = Tensors.writes ~buffers c.sink in
  List.concat_map
    (fun (u, xs) ->
      let after = Array.copy xs in
      List.iter
        (fun (s, i, v) ->
          if s = slot u then after.(i) <- Dtype.Value.to_float v)
        writes;
      Array.to_list after)
    c.inputs

(* Parts

   A part compiles its case, links and runs it, and writes on a line what each
   input holds after the run, then on another how many programs the process made
   or read ({!Codegen.programs}). Settings such as DEBUG write lines of their
   own around them. *)

let run_case case =
  let c = case () in
  let linear, vars =
    Engine.compile
      ~search:(fun _ k -> k)
      ~devices ~held_bufs:(List.map fst c.held) ~inputs:(List.map fst c.inputs)
      c.sink
  in
  let filled (u, xs) =
    (u, Run.buffer Rig.host Float32 (Array.map (fun x -> `Float x) xs))
  in
  let held = List.map filled c.held and inputs = List.map filled c.inputs in
  let s = Engine.link ~devices ~bound:held linear in
  Engine.run ~vars s (Array.of_list (List.map (fun (_, b) -> Some b) inputs));
  let values (_, b) =
    List.map
      (fun v -> Printf.sprintf "%h" (Dtype.Value.to_float v))
      (Array.to_list (Run.values Float32 b))
  in
  print_endline ("values " ^ String.concat " " (List.concat_map values inputs));
  Printf.printf "programs %d\n" (Codegen.programs ())

(* [arithmetic], while the kernels the process compiles are collected. *)
let collected () = ignore (Realize.kernels (fun () -> run_case arithmetic))

let parts =
  ("collected", collected)
  :: List.map (fun (name, case) -> (name, fun () -> run_case case)) cases

(* A part's report *)

type report = { values : float list; programs : int }

let output ?env part db =
  (* A part compiles for the host: skip where rig links no host code. *)
  ignore (Run.target Rig.host);
  match Disk_cache.child ?env ~cachedb:db part with
  | Error err -> failf "%s failed: %s" part err
  | Ok out -> (
      let lines = String.split_on_char '\n' out in
      let line prefix =
        List.find_map
          (fun l ->
            if String.starts_with ~prefix:(prefix ^ " ") l then
              Some
                (String.sub l
                   (String.length prefix + 1)
                   (String.length l - String.length prefix - 1))
            else None)
          lines
      in
      let count prefix = Option.map int_of_string (line prefix) in
      match (line "values", count "programs") with
      | Some values, Some programs ->
          {
            values = List.map float_of_string (String.split_on_char ' ' values);
            programs;
          }
      | _ -> failf "%s wrote %S" part out)

let values = list float_exact
let compiled = "compiled"
let kept db = Disk_cache.entries ~table:compiled db

(* A later process *)

(* A first process makes the schedule; a second reads it back, making and
   reading no program, and computes what the sink computes. *)
let reads_back part () =
  let db = Disk_cache.fresh () in
  let first = output part db in
  greater ~msg:"programs made first" ~than:0 int first.programs;
  let later = output part db in
  equal ~msg:"programs made later" int 0 later.programs;
  equal values (expected part) later.values

(* Two processes compile one sink, its inputs in other orders: each run computes
   [a - b] from its own slots. *)
let positions () =
  let db = Disk_cache.fresh () in
  ignore (output "sub_ab" db);
  equal values (expected "sub_ba") (output "sub_ba" db).values

(* A held buffer's contents are the link's: the entry one process made from one
   constant serves another's. *)
let held_buffers () =
  let db = Disk_cache.fresh () in
  ignore (output "times_two" db);
  let three = output "times_three" db in
  equal ~msg:"programs made" int 0 three.programs;
  equal values (expected "times_three") three.values

(* A sink renumbered reads the entry the first numbering made. *)
let renumbering () =
  let db = Disk_cache.fresh () in
  ignore (output "loop" db);
  let later = output "renumbered" db in
  equal ~msg:"programs made" int 0 later.programs;
  equal values (expected "renumbered") later.values

(* Damage *)

(* Replaces every [Ops.] by [Opz.]: a whole entry, but not a graph. *)
let not_a_graph =
  let garbled v =
    let b = Buffer.create (String.length v) in
    let n = String.length v and i = ref 0 in
    while !i < n do
      if !i + 4 <= n && String.sub v !i 4 = "Ops." then begin
        Buffer.add_string b "Opz.";
        i := !i + 4
      end
      else begin
        Buffer.add_char b v.[!i];
        incr i
      end
    done;
    Buffer.contents b
  in
  Disk_cache.restated garbled

(* An entry whose lengths sum to its size only by wrapping past [max_int], with
   its digest stated anew: whole to the disk cache, no schedule. *)
let wrapping e =
  let line = String.index e '\n' + 1 in
  let klen = int_of_string (List.hd (String.split_on_char ' ' e)) in
  let key = String.sub e line klen in
  let rest = "no schedule" in
  let value =
    Printf.sprintf "0 %d %d %d\n%s" max_int max_int
      (String.length rest + 2)
      rest
  in
  Printf.sprintf "%d %d %s\n%s%s" (String.length key) (String.length value)
    (Digest.to_hex (Digest.string (key ^ value)))
    key value

(* Damage reads as a miss: the next process makes the schedule, computes what
   the sink computes, and replaces the entry, which a third reads back. *)
let recovers damage () =
  let db = Disk_cache.fresh () in
  ignore (output "arithmetic" db);
  Disk_cache.damage ~table:compiled db damage;
  let healed = output "arithmetic" db in
  greater ~msg:"programs made anew" ~than:0 int healed.programs;
  equal ~msg:"values" values (expected "arithmetic") healed.values;
  equal ~msg:"programs made after" int 0 (output "arithmetic" db).programs

(* Nothing is kept *)

let keeps_none env () =
  let db = Disk_cache.fresh () in
  ignore (output ~env "arithmetic" db);
  equal (list string) [] (kept db)

let keeps_none_collecting () =
  let db = Disk_cache.fresh () in
  ignore (output "collected" db);
  equal (list string) [] (kept db)

let () =
  Disk_cache.play parts;
  exit
    (run "Tolk_engine.compile on disk"
       [
         group "a later process compiling an equal sink"
           [
             test "reads its schedule back: an elementwise sink"
               (reads_back "arithmetic");
             test "reads its schedule back: a reduction"
               (reads_back "reduction");
             test "reads its schedule back: a loop of calls" (reads_back "loop");
             test "reads it back under other numbers of ranges and storage"
               renumbering;
             test "with its inputs in other places makes its own" positions;
             test "binds its own held buffers" held_buffers;
           ];
         group "an entry of other sources is made anew, and replaced"
           [
             test "the engine's digest is its own" (fun () ->
                 not_equal string Jit.source_digest Engine.source_digest);
             test "of another engine"
               (recovers (Disk_cache.of_other_sources Engine.source_digest));
             test "of another tolk"
               (recovers (Disk_cache.of_other_sources Jit.source_digest));
           ];
         group "nothing is kept"
           [
             test "under a beam search" (keeps_none [ ("JITBEAM", "1") ]);
             test "with DEBUG on" (keeps_none [ ("DEBUG", "1") ]);
             test "with CHECK_OOB on" (keeps_none [ ("CHECK_OOB", "1") ]);
             test "while collecting kernels" keeps_none_collecting;
             test "without the binary cache" (keeps_none [ ("CCACHE", "0") ]);
             test "with SCACHE at 1" (keeps_none [ ("SCACHE", "1") ]);
           ];
         group "a damaged entry is made anew, and replaced"
           [
             test "truncated" (recovers Disk_cache.truncated);
             test "holding no graph" (recovers not_a_graph);
             test "of another value"
               (recovers (Disk_cache.restated (fun v -> v ^ "garbage")));
             test "whose lengths wrap" (recovers wrapping);
           ];
       ])
