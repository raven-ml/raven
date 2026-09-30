(* Tests of Tolk_next.Support_memory: the TLSF allocator on tinygrad's traces
   and cases, and its laws against a model of the addresses it hands out. *)

open Windtrap
open Tolk_next
module T = Support_memory.Tlsf_allocator

let alloc = option int
let rec bits n = if n <= 0 then 0 else 1 + bits (n lsr 1)

(* Traces

   Each trace of `traces.golden` replayed on one allocator: every allocation
   returns what tinygrad's returned. *)

let replay rows =
  let allocator = ref None in
  List.iter
    (fun (step, call, result) ->
      match String.split_on_char ' ' call with
      | [ "create"; size; base; block_size; lv2_cnt ] ->
          allocator :=
            Some
              (T.create ~base:(int_of_string base)
                 ~block_size:(int_of_string block_size)
                 ~lv2_cnt:(int_of_string lv2_cnt) (int_of_string size))
      | [ "alloc"; n; align ] ->
          equal
            ~msg:(Printf.sprintf "step %s: %s" step call)
            alloc (int_of_string_opt result)
            (T.alloc ~align:(int_of_string align) (Option.get !allocator)
               (int_of_string n))
      | [ "free"; x ] -> T.free (Option.get !allocator) (int_of_string x)
      | _ -> fail ("unknown call " ^ call))
    rows

let traces =
  let rows = Golden.rows "traces.golden" in
  let trials =
    List.sort_uniq compare
      (List.map (fun cell -> int_of_string (cell "trial")) rows)
  in
  cases ~name:(Printf.sprintf "trial %d") "traces.golden" trials (fun trial ->
      replay
        (List.filter_map
           (fun cell ->
             if int_of_string (cell "trial") = trial then
               Some (cell "step", cell "call", cell "result")
             else None)
           rows))

(* Cases *)

let fresh () = T.create ~block_size:16 1024

let examples =
  group "Tlsf_allocator"
    [
      test "blocks are handed out in address order, and a freed one again"
        (fun () ->
          let a = fresh () in
          equal alloc (Some 0) (T.alloc a 32);
          equal alloc (Some 32) (T.alloc a 64);
          T.free a 0;
          equal alloc (Some 0) (T.alloc a 32));
      test "a block is at least the block size" (fun () ->
          let a = fresh () in
          equal alloc (Some 0) (T.alloc a 1);
          equal alloc (Some 16) (T.alloc a 20);
          equal alloc (Some 36) (T.alloc a 35));
      test "freed neighbours merge" (fun () ->
          let a = fresh () in
          let x = T.alloc a 32 and y = T.alloc a 32 in
          ignore (T.alloc a 32);
          T.free a (Option.get x);
          T.free a (Option.get y);
          equal alloc x (T.alloc a 64));
      test "a freed block splits" (fun () ->
          let a = fresh () in
          let x = Option.get (T.alloc a 128) in
          T.free a x;
          equal alloc (Some x) (T.alloc a 32);
          equal alloc (Some (x + 32)) (T.alloc a 32));
      test "a block larger than the range is None" (fun () ->
          equal alloc None (T.alloc (fresh ()) 2048));
      test "an empty range hands out nothing" (fun () ->
          equal alloc None (T.alloc (T.create 0) 1));
      test "addresses start at the base" (fun () ->
          let a = T.create ~base:1000 1024 in
          equal alloc (Some 1000) (T.alloc a 32);
          equal alloc (Some 1032) (T.alloc a 64));
      test "an aligned block is aligned from the base" (fun () ->
          let a = T.create ~base:1000 ~block_size:16 4096 in
          ignore (T.alloc a 16);
          equal alloc (Some 1256) (T.alloc ~align:256 a 16));
      test "a block past the base is freed by its address" (fun () ->
          let a = T.create ~base:1000 1024 in
          let x = Option.get (T.alloc a 32) in
          T.free a x;
          equal alloc (Some 1000) (T.alloc a 1024));
    ]

let errors =
  let rejects ?substring f = raises_match (Exn.invalid_arg ?substring) f in
  group "Tlsf_allocator › errors"
    [
      test "a negative size is refused" (fun () ->
          rejects (fun () -> T.create (-1)));
      test "a block size or a level count that is not positive is refused"
        (fun () ->
          rejects (fun () -> T.create ~block_size:0 64);
          rejects (fun () -> T.create ~lv2_cnt:0 64));
      test "a block size of fewer bits than the level count is refused"
        (fun () ->
          rejects (fun () -> T.create ~block_size:8 ~lv2_cnt:16 64);
          ignore (T.create ~block_size:16 ~lv2_cnt:16 64));
      test "an alignment that is not positive is refused" (fun () ->
          rejects (fun () -> T.alloc ~align:0 (fresh ()) 16));
      test "freeing an address where no block starts is refused" (fun () ->
          let a = fresh () in
          ignore (T.alloc a 32);
          rejects (fun () -> T.free a 8));
      test "freeing a free block is refused" (fun () ->
          let a = fresh () in
          let x = Option.get (T.alloc a 32) in
          T.free a x;
          rejects (fun () -> T.free a x));
    ]

(* Laws

   A model of the addresses an allocator hands out: the blocks live, each
   [(start, length)]. A block is [max block_size n] addresses long, lies in the
   range, starts a multiple of [align] after the base, and overlaps no live
   block. An allocation is None exactly when no free stretch of the range holds
   a block of the smallest subdivision that fits [n] padded for its alignment:
   free neighbours merge, so the free stretches are the free blocks. *)

type model = {
  base : int;
  size : int;
  block_size : int;
  lv2_bits : int;
  mutable live : (int * int) list;
}

let pp_model ppf m =
  Format.fprintf ppf "[%s]"
    (String.concat "; "
       (List.map
          (fun (s, l) -> Printf.sprintf "%d+%d" s l)
          (List.sort compare m.live)))

(* The smallest length of the subdivision whose blocks all fit [n]. *)
let fitting m n =
  let n = max m.block_size n in
  let unit = 1 lsl max 0 (bits n - m.lv2_bits) in
  (n + unit - 1) / unit * unit

let largest_gap m =
  let rec gaps at = function
    | [] -> [ m.base + m.size - at ]
    | (s, l) :: rest -> (s - at) :: gaps (s + l) rest
  in
  List.fold_left max 0 (gaps m.base (List.sort compare m.live))

let judge_alloc m n align outcome =
  let length = max m.block_size n in
  match outcome with
  | Error e -> raise e
  | Ok None ->
      cover "no room" true;
      less ~msg:"a free stretch fits the request" int
        ~than:(fitting m (length + align - 1))
        (largest_gap m)
  | Ok (Some x) ->
      cover "an aligned block" (align > 1 && x - m.base > 0);
      at_least ~msg:"the block starts in the range" int ~than:m.base x;
      at_most ~msg:"the block ends in the range" int ~than:(m.base + m.size)
        (x + length);
      equal ~msg:"the block is aligned from the base" int 0
        ((x - m.base) mod align);
      List.iter
        (fun (s, l) ->
          is_true
            ~msg:(Printf.sprintf "the block overlaps the live block %d+%d" s l)
            (x + length <= s || s + l <= x))
        m.live;
      m.live <- (x, length) :: m.live

let allocator = abstract "a" ~pp:pp_model

let block =
  among int allocator (fun m -> List.map fst (List.sort compare m.live))

let shape =
  Gen.(
    let+ size =
      of_list ~pp:Format.pp_print_int [ 0; 256; 1024; 4096; 65536; 100_003 ]
    and+ base = of_list ~pp:Format.pp_print_int [ 0; 1000; 0x200000 ]
    and+ block_size, lv2_cnt =
      of_list
        ~pp:(fun ppf (b, l) -> Format.fprintf ppf "block %d, lv2 %d" b l)
        [ (16, 16); (256, 32); (64, 8); (16, 4); (32, 32); (16, 1) ]
    in
    (size, base, block_size, lv2_cnt))

let request =
  Gen.frequency
    [
      ( 3,
        Gen.of_list ~pp:Format.pp_print_int
          [ 1; 15; 16; 17; 255; 256; 257; 1000; 4096 ] );
      (1, Gen.int_range 1 70_000);
    ]

let alignment = Gen.of_list ~pp:Format.pp_print_int [ 1; 1; 8; 64; 256; 4096 ]

let commands =
  [
    command "create"
      (shape @-> makes allocator)
      (fun (size, base, block_size, lv2_cnt) ->
        { base; size; block_size; lv2_bits = bits lv2_cnt; live = [] })
      (fun (size, base, block_size, lv2_cnt) ->
        T.create ~base ~block_size ~lv2_cnt size);
    command "alloc"
      (request @-> alignment @-> allocator ^-> judges alloc)
      (fun n align m outcome -> judge_alloc m n align outcome)
      (fun n align a -> T.alloc ~align a n);
    command "free"
      (allocator ^-> block ^-> returns unit)
      (fun m x -> m.live <- List.filter (fun (s, _) -> s <> x) m.live)
      T.free;
  ]

(* The law that freeing every block restores the allocator: after any trace
   whose blocks are all freed, allocations answer as a fresh allocator's do. *)
let restores (shape, trace, probes) =
  let size, base, block_size, lv2_cnt = shape in
  let make () = T.create ~base ~block_size ~lv2_cnt size in
  let used = make () in
  let live = ref [] in
  List.iter
    (fun (n, align) ->
      match T.alloc ~align used n with
      | Some x -> live := x :: !live
      | None -> ())
    trace;
  List.iter (T.free used) !live;
  let fresh = make () in
  List.iter
    (fun (n, align) ->
      equal alloc (T.alloc ~align fresh n) (T.alloc ~align used n))
    probes

let laws =
  let requests =
    Gen.list ~size:(Gen.int_range 0 12) (Gen.pair request alignment)
  in
  group "Tlsf_allocator › laws"
    [
      stateful "blocks fit, align and never overlap, and None means no room"
        ~count:200 ~steps:40 commands;
      prop "freeing every block restores a fresh allocator"
        Gen.(triple shape requests requests)
        restores;
    ]

let () =
  exit (run "Tolk_next.Support_memory" [ traces; examples; errors; laws ])
