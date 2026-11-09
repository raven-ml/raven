(* A later process's first call of a compiled function: what the disk cache
   keeps of it is read back whole, and only when it is right. *)

open Windtrap

let f32 = Nx.float32
let vector l = Nx.create f32 [| List.length l |] (Array.of_list l)

(* Writes the values [x] holds on a line, then how many programs this process
   read or made. *)
let report_base = ref 0

let report x =
  print_endline
    (String.concat " " (List.map (Printf.sprintf "%.3f") (Array.to_list (Nx.to_array x))));
  print_endline (Printf.sprintf "programs %d" (Tolk.Codegen.programs () - !report_base))

let sum_of_scan () =
  let f =
    Rune.jit' (fun x ->
        let step acc y =
          let acc = Nx.add acc y in
          (acc, acc)
        in
        fst (Rune.scan' step ~init:(Nx.zeros f32 [| 1 |]) x))
  in
  report (f (Nx.reshape [| 4; 1 |] (vector [ 1.; 2.; 3.; 4. ])))

let arithmetic () =
  let f = Rune.jit' (fun x -> Nx.mul (Nx.add x (vector [ 1.; 2. ])) x) in
  report (f (vector [ 3.; 5. ]))

let reduction () =
  let f = Rune.jit' (fun x -> Nx.mean x) in
  report (Nx.reshape [| 1 |] (f (vector [ 1.; 3.; 5.; 7. ])))

(* The subtraction of [b] from [a], with its arguments taken in the order
   [a b] by one part and [b a] by the other: two lowerings that differ only in
   which argument is which. *)
let subtract_ab () =
  let f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) (fun a b -> Nx.sub a b) in
  report (f (vector [ 10. ]) (vector [ 3. ]))

let subtract_ba () =
  let f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) (fun b a -> Nx.sub a b) in
  report (f (vector [ 3. ]) (vector [ 10. ]))

(* The same function over captured constants of different values. *)
let times_two () =
  let c = vector [ 2.; 2. ] in
  let f = Rune.jit' (fun x -> Nx.mul x c) in
  report (f (vector [ 4.; 5. ]))

let times_three () =
  let c = vector [ 3.; 3. ] in
  let f = Rune.jit' (fun x -> Nx.mul x c) in
  report (f (vector [ 4.; 5. ]))

(* [arithmetic], after unrelated compiled calls moved the process's counters of
   ranges and storage: the numbers a schedule is kept under are its graph's. *)
let shifted () =
  let g = Rune.jit' (fun x -> Nx.add (Nx.mean x) (Nx.std x)) in
  ignore (g (vector [ 1.; 2.; 3. ]));
  ignore (g (vector [ 1.; 2.; 3.; 4. ]));
  report_base := Tolk.Codegen.programs ();
  arithmetic ()

(* [arithmetic], while the kernels a process compiles are collected. *)
let collected () =
  let (), found = Tolk.Realize.kernels arithmetic in
  print_endline (Printf.sprintf "kernels %d" (List.length found))

let parts =
  [
    ("arithmetic", arithmetic);
    ("shifted", shifted);
    ("collected", collected);
    ("reduction", reduction);
    ("scan", sum_of_scan);
    ("sub_ab", subtract_ab);
    ("sub_ba", subtract_ba);
    ("times_two", times_two);
    ("times_three", times_three);
  ]

let compiled = "compiled"

let output part ?env db =
  match Disk_cache.child ?env ~cachedb:db part with
  | Ok out -> out
  | Error err -> failf "%s failed: %s" part err

let lines out = String.split_on_char '\n' (String.trim out)

let values out = List.hd (lines out)

let programs out =
  match lines out with
  | [ _; p ] -> Scanf.sscanf p "programs %d" Fun.id
  | _ -> failf "unexpected output %S" out

let string = Testable.string

(* A second process compiles nothing and computes what the first computed. *)
let reads_back ?expected part () =
  let db = Disk_cache.fresh () in
  let first = output part db in
  Option.iter (equal string ~msg:"expected values" (values first)) expected;
  greater ~than:0 int (programs first);
  equal bool ~msg:"an entry was kept" true
    (Disk_cache.entries ~table:compiled db <> []);
  let second = output part db in
  equal string ~msg:"values" (values first) (values second);
  equal int ~msg:"no program read or made" 0 (programs second)

(* Two lowerings that take their arguments in another order are told apart. *)
let positions () =
  let db = Disk_cache.fresh () in
  let ab = output "sub_ab" db and ba = output "sub_ba" db in
  equal string ~msg:"a b" "7.000" (values ab);
  equal string ~msg:"b a" "7.000" (values ba)

(* A constant the function captures is this process's, whatever the entry was
   made from. *)
let captures () =
  let db = Disk_cache.fresh () in
  let two = output "times_two" db and three = output "times_three" db in
  equal string ~msg:"times two" "8.000 10.000" (values two);
  equal string ~msg:"times three" "12.000 15.000" (values three);
  equal int ~msg:"the entry the first made" 0 (programs three)

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

(* An entry whose lengths sum to its size only by wrapping past [max_int],
   with its digest stated anew: whole to the disk cache, no compiled
   schedule. *)
let wrapping e =
  let line = String.index e '\n' + 1 in
  let klen = int_of_string (List.hd (String.split_on_char ' ' e)) in
  let key = String.sub e line klen in
  let rest = "no schedule" in
  let value =
    Printf.sprintf "0 %d %d %d\n%s" max_int max_int
      (String.length rest + 2) rest
  in
  Printf.sprintf "%d %d %s\n%s%s" (String.length key) (String.length value)
    (Digest.to_hex (Digest.string (key ^ value)))
    key value

(* Damage reads as a miss: the next process computes the right values and
   rewrites the entry. *)
let recovers damage () =
  let db = Disk_cache.fresh () in
  let first = output "arithmetic" db in
  Disk_cache.damage ~table:compiled db damage;
  let healed = output "arithmetic" db in
  equal string ~msg:"values" (values first) (values healed);
  greater ~than:0 int ~msg:"made anew" (programs healed);
  let later = output "arithmetic" db in
  equal int ~msg:"rewritten, read back" 0 (programs later)

(* The engine's sources are digested apart from tolk's: an entry depends on
   both. *)
let digests_apart () =
  not_equal string ~msg:"engine against tolk" Tolk.Jit.source_digest
    Tolk_engine.source_digest

(* A process whose counters differ reads the entry another made, and gives each
   numbered node its own. *)
let renumbered () =
  let db = Disk_cache.fresh () in
  let first = output "arithmetic" db in
  equal string ~msg:"values" "12.000 35.000" (values first);
  let later = output "shifted" db in
  equal string ~msg:"after other calls" (values first) (values later);
  equal int ~msg:"read back" 0 (programs later)

(* While kernels are collected no entry is read or kept: each kernel is
   reported, over a disk that holds its entry too. *)
let collects () =
  let db = Disk_cache.fresh () in
  ignore (output "arithmetic" db);
  let reported out =
    match String.split_on_char '\n' (String.trim out) with
    | [ _; _; kernels ] -> Scanf.sscanf kernels "kernels %d" Fun.id
    | _ -> failf "unexpected output %S" out
  in
  greater ~than:0 int ~msg:"reported" (reported (output "collected" db));
  let fresh = Disk_cache.fresh () in
  ignore (output "collected" fresh);
  equal (list string) ~msg:"kept" [] (Disk_cache.entries ~table:compiled fresh)

(* A process whose settings show compilation, or that keeps no binaries, keeps
   no entry. *)
let keeps_none env () =
  let db = Disk_cache.fresh () in
  ignore (output "arithmetic" ~env db);
  equal (list string) ~msg:"entries" []
    (List.map Filename.basename (Disk_cache.entries ~table:compiled db))

let () =
  Disk_cache.play parts;
  exit
    (run "Rune.jit across processes"
    [
      group "a later process's first call"
        [
          test "compiles nothing, computes the same: arithmetic"
            (reads_back ~expected:"12.000 35.000" "arithmetic");
          test "compiles nothing, computes the same: a reduction"
            (reads_back "reduction");
          test "compiles nothing, computes the same: a scan"
            (reads_back "scan");
          test "reads an entry whatever its counters" renumbered;
          test "tells apart arguments in another order" positions;
          test "reads this process's captured constants" captures;
        ];
      group "a damaged entry is made anew, and replaced"
        [
          test "truncated" (recovers Disk_cache.truncated);
          test "holding no graph" (recovers not_a_graph);
          test "of another value"
            (recovers (Disk_cache.restated (fun v -> v ^ "garbage")));
          test "whose lengths wrap" (recovers wrapping);
        ];
      group "an entry of other sources is made anew, and replaced"
        [
          test "the engine's digest is its own" digests_apart;
          test "of another engine"
            (recovers (Disk_cache.of_other_sources Tolk_engine.source_digest));
          test "of another tolk"
            (recovers (Disk_cache.of_other_sources Tolk.Jit.source_digest));
        ];
      group "nothing is kept"
        [
          test "with DEBUG on" (keeps_none [ ("DEBUG", "1") ]);
          test "under a beam search" (keeps_none [ ("JITBEAM", "1") ]);
          test "with CHECK_OOB on" (keeps_none [ ("CHECK_OOB", "1") ]);
          test "while collecting kernels" collects;
          test "without the binary cache" (keeps_none [ ("CCACHE", "0") ]);
          test "with SCACHE at 1" (keeps_none [ ("SCACHE", "1") ]);
        ];
    ])
