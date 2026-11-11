(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe: embeddings of TESSERA tiles by the released v2 models, and their
   check against the upstream reference's outputs.

   encode reads a tile, embeds its pixels and writes GeoTessera's files.
   validate embeds the fixture tiles with each released model and compares the
   result with the reference's, embedding by embedding and, for medium, block by
   block against its layer dumps. *)

open Tessera

let usage =
  {|usage: main.exe encode --model FILE --tile DIR --out NAME [OPTION...]
       main.exe validate --weights DIR --fixtures DIR [OPTION...]
       main.exe validate --training DIR --tile DIR [--masks] [--device D] [--bf16]
       main.exe corpus --tiles DIR[,DIR...] --out DIR [--seed N] [--bytes B]
       main.exe targets --teacher FILE --corpus DIR --out FILE [OPTION...]
       main.exe distil --corpus DIR --targets FILE --out DIR
                       (--init FILE [--heads FILE] | --arch A) [OPTION...]

encode options:
  --device D    cpu (eager, the default), cpu-jit, cuda, nv, amd or metal
  --dims K      write the Matryoshka prefix of K values (students only)
  --int8        write GeoTessera's int8 codes and scales
  --landmask F  write NAME.tiff, GeoTessera's float32 GeoTIFF, on the grid of
                the tile's landmask F, in place of the npy files
  --bf16        compute the linear layers and attention in bfloat16
  --pixels N    pixels per batch (default 4096)
  --slots S     pixels per row at most (default 32)

validate options:
  --device D, --pixels N, --slots S
                as for encode, for the students
  --models M    comma-separated, of nano, small, medium, large, teacher
                (default the four students)
  --bf16        validate the teacher in bfloat16 too

validate --training compares one distillation step's loss and gradients
with gen/grads.py's on a small student, dropout masks fed to it; --masks
writes raven's tokens and masks for it first.

corpus writes the pixels of the tiles with an observation, shuffled by
--seed, in runs of --bytes (default 1 GiB).

targets options:
  --device D, --bf16, --pixels N (default 512), --slots S
                as for encode
  --fp16        write float16 targets (default float32)
  --chunk N     pixels a block of the file (default 8192)

distil options:
  --init FILE   a student checkpoint, its heads from --heads FILE (npz of
                PyTorch's k.weight and k.bias) or drawn
  --arch A      a student drawn from --seed: nano, small, medium or large
  --device D    as for encode
  --steps N     (default 1000), --groups G (default 1) per step, --pixels N
                per group (default 2048), --slots S (default 32)
  --seed N      the run's key: corpus order after the first epoch, dropout,
                and a drawn student's and heads' parameters (default 0)
  --lr X (8e-4), --warmup F (0.05 of the steps), --decay X (1e-4, every
                leaf), --clip X (1.0), --dropout X (0.1), --reducer-dropout X
                (0.2): v2 App. J, Table 9
  --bf16        compute the products in bfloat16, parameters float32
  --memory B    refuse a step whose footprint, state and groups pass B bytes
  --every K     write OUT/state.safetensors every K steps (default 100)
  --until N     stop after step N of the --steps the schedule spans
  --resume      continue from OUT/state.safetensors, with the options of
                the run that wrote it (a run without --resume refuses an
                OUT that holds one)
  --heldout C:T report each prefix's mean cosine to the targets T of the
                corpus C after the last step

Memory grows linearly with --pixels. At 4096 pixels a batch takes about
7.7 GB for medium and 11.4 GB for large eager on the CPU, and 3.8 GB and
5.1 GB of a CUDA GPU's memory compiled.|}

let die fmt =
  Printf.ksprintf
    (fun m ->
      prerr_endline ("tessera: " ^ m);
      exit 2)
    fmt

(* Arguments *)

(* A command's [--name value] options and [--flag]s, in any order; any other
   argument is refused. *)
let parse ~options ~flags args =
  let rec go opts = function
    | [] -> opts
    | a :: rest when List.mem a flags -> go ((a, "") :: opts) rest
    | a :: v :: rest when List.mem a options -> go ((a, v) :: opts) rest
    | [ a ] when List.mem a options -> die "%s needs a value\n%s" a usage
    | a :: _ -> die "unknown argument %S\n%s" a usage
  in
  go [] args

let opt opts name = List.assoc_opt name opts

let req opts name =
  match opt opts name with
  | Some v -> v
  | None -> die "%s is required\n%s" name usage

let flag opts name = List.mem_assoc name opts

let int_opt opts name default =
  match opt opts name with
  | None -> default
  | Some v -> (
      match int_of_string_opt v with Some n -> n | None -> die "%s: %S" name v)

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type run = { run : 'd. 'd target -> unit }

let gpus =
  [
    ("cuda", fun () -> Rig.open_ (Rig_cuda.gpu 0));
    ("nv", fun () -> Rig.open_ (Rig_nv_nvidia.gpu 0));
    ("amd", fun () -> Rig.open_ (Rig_amd_amdgpu.gpu 0));
    ("metal", fun () -> Rig.open_ (Rig_metal.gpu 0));
  ]

let on device k =
  match device with
  | "cpu" -> k.run { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> k.run { at = Nx.Host.on; jit = true }
  | gpu -> (
      match List.assoc_opt gpu gpus with
      | None -> die "%S is not a device" gpu
      | Some open_ -> (
          match open_ () with
          | Error e -> die "%s: %s" gpu e
          | Ok g ->
              let module D = (val Nx.devices [ g ]) in
              k.run { at = D.on; jit = true }))

(* A model's forward pass at [t], compiled where [t] asks. *)
let forward (type s d) (t : d target) (m : ((float, s, d) Nx.t, d) Model.t) :
    d Batch.t -> d Nx.float32_t =
  if not t.jit then Model.apply m
  else
    let model = Nx.Ptree.instantiate (module Model) in
    let batch = Nx.Ptree.v Batch.walk in
    let f =
      Rune.jit Nx.Ptree.(model @-> batch @-> returns tensor) Model.apply
    in
    f m

let place (type d) (t : d target) m = Nx.Ptree.place (module Model) t.at m
let bf16 m = Nx.Ptree.cast (module Model) Nx.bfloat16 m

(* encode *)

(* The tile's grid, from its landmask's tags, as GeoTessera takes it. A tile of
   another shape is refused here, before it is embedded, as save_geotiff would
   refuse it after. *)
let landmask ~tile:(h, w) file =
  let tags = Nx_io.Tiff.tags (List.hd (Nx_io.load_tiff file)) in
  match Jord.Geotiff.grid tags with
  | Error e -> die "%s: %s" file e
  | Ok (g : Jord.Grid.t) when g.rows <> h || g.cols <> w ->
      die "%s: a grid of %d × %d, and the tile is %d × %d" file g.rows g.cols h
        w
  | Ok g -> g

let encode opts =
  let file = req opts "--model" and tile = req opts "--tile" in
  let out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let m = Model.load file in
  let dims =
    match opt opts "--dims" with
    | None -> Model.dims m
    | Some k -> (
        match int_of_string_opt k with
        | Some k when List.mem k m.prefixes -> k
        | _ ->
            die "--dims %s: the model's prefixes are [%s]" k
              (String.concat "; " (List.map string_of_int m.prefixes)))
  in
  let px, (h, w) = Pixels.of_tile tile in
  let geotiff = Option.map (landmask ~tile:(h, w)) (opt opts "--landmask") in
  if flag opts "--int8" && Option.is_some geotiff then
    die "--int8 and --landmask: GeoTessera's GeoTIFFs hold float32";
  let embed t =
    if flag opts "--bf16" then
      Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px
    else Embedding.encode ~shape t.at (forward t (place t m)) px
  in
  on device
    {
      run =
        (fun t ->
          let e = embed t in
          let e = Nx.reshape [| h; w; -1 |] e in
          let e = Nx.slice [ Nx.A; Nx.A; Nx.R (0, dims) ] e in
          match geotiff with
          | Some grid -> Embedding.save_geotiff out ~grid e
          | None when flag opts "--int8" -> Embedding.save_int8 out e
          | None -> Embedding.save out e);
    }

(* validate *)

let f64 x = Nx.to_array (Nx.cast Nx.float64 x)
let load_f32 file = Nx.unpack Nx.float32 (Nx_io.load_npy file)
let checks = ref 0
let failed = ref []

let report ~name ~pass text =
  Printf.printf "%-48s %s %s\n%!" name (if pass then "ok  " else "FAIL") text;
  incr checks;
  if not pass then failed := name :: !failed

(* A comparison no bound judges, reported for its numbers. *)
let note ~name text = Printf.printf "%-48s %s %s\n%!" name "info" text

(* Tolerances: twice the reference's spread against itself over devices, batch
   sizes and fused paths (max |Δ| 9.8e-6, 1 − cos 2e-12). *)
let max_abs_tol = 2e-5
let cos_tol = 1e-11
let int8_share_tol = 1e-4

(* Max |Δ| and least cosine over the rows [valid] marks, rows of [dims]. *)
let compare_rows ~dims ~valid a b =
  let worst = ref 0. and least = ref 1. in
  Array.iteri
    (fun p v ->
      if v then begin
        let dot = ref 0. and na = ref 0. and nb = ref 0. in
        for c = 0 to dims - 1 do
          let x = a.((p * dims) + c) and y = b.((p * dims) + c) in
          worst := Float.max !worst (Float.abs (x -. y));
          if Float.is_nan x || Float.is_nan y then worst := Float.nan;
          dot := !dot +. (x *. y);
          na := !na +. (x *. x);
          nb := !nb +. (y *. y)
        done;
        least := Float.min !least (!dot /. Float.sqrt (!na *. !nb))
      end)
    valid;
  (!worst, !least)

(* Compares [e] with [ref_file], judged at [tol] if given. *)
let validate_embedding ~label ~valid ?tol e ref_file =
  let r = f64 (load_f32 ref_file) in
  let dims = Nx.dim 1 e in
  let mine = f64 e in
  let worst, least = compare_rows ~dims ~valid mine r in
  let undefined_nan =
    let ok = ref true in
    Array.iteri
      (fun p v ->
        if not v then
          for c = 0 to dims - 1 do
            if not (Float.is_nan mine.((p * dims) + c)) then ok := false
          done)
      valid;
    !ok
  in
  let text =
    Printf.sprintf "max|d| %.3g  1-cos %.3g  undefined NaN %b" worst
      (1. -. least) undefined_nan
  in
  match tol with
  | None -> note ~name:label text
  | Some (abs_tol, cos_tol) ->
      let pass = worst <= abs_tol && 1. -. least <= cos_tol && undefined_nan in
      report ~name:label ~pass text

let validate_int8 ~label ~valid e ~codes ~scales =
  let c, s = Embedding.quantize e in
  let c = Nx.to_array c and s = f64 s in
  let rc = Nx.to_array (Nx.unpack Nx.int8 (Nx_io.load_npy codes)) in
  let rs = f64 (load_f32 scales) in
  let dims = Nx.dim 1 e in
  let differ = ref 0 and total = ref 0 and worst = ref 0 and scale = ref 0. in
  let undefined_ok = ref true in
  Array.iteri
    (fun p v ->
      let codes = Array.sub c (p * dims) dims in
      if not v then
        undefined_ok :=
          !undefined_ok && Float.is_nan s.(p) && Array.for_all (( = ) 0) codes
      else begin
        scale :=
          Float.max !scale (Float.abs (s.(p) -. rs.(p)) /. Float.abs rs.(p));
        Array.iteri
          (fun k code ->
            let d = abs (code - rc.((p * dims) + k)) in
            incr total;
            if d > 0 then incr differ;
            worst := max !worst d)
          codes
      end)
    valid;
  let share = Float.of_int !differ /. Float.of_int !total in
  let pass = !worst <= 1 && share <= int8_share_tol && !undefined_ok in
  report ~name:label ~pass
    (Printf.sprintf
       "codes differing %d / %d (max %d)  scale rel %.3g  undefined 0, NaN %b"
       !differ !total !worst !scale !undefined_ok)

let student_names = [ "nano"; "small"; "medium"; "large" ]

let weight_file dir = function
  | "teacher" -> Filename.concat dir "tessera_v2_2B_teacher.pt"
  | m -> Filename.concat dir ("student_" ^ m ^ ".pt")

let tiles = [ "small_64"; "nodesc_32" ]

let valid_of fixtures tile =
  let v =
    Nx.unpack Nx.uint8
      (Nx_io.load_npy (Filename.concat fixtures ("emb/" ^ tile ^ "/valid.npy")))
  in
  Array.map (fun x -> x <> 0) (Nx.to_array v)

(* The reference's name for a device's fixtures. *)
let ref_device = function "cpu" | "cpu-jit" -> "cpu" | _ -> "cuda"

(* Blocks *)

(* The position of each dumped pixel, from the dumps' index.json. *)
let dumped_pixels dir =
  let text =
    In_channel.with_open_bin
      (Filename.concat dir "index.json")
      In_channel.input_all
  in
  let find from sub =
    let n = String.length sub in
    let rec go i =
      if i + n > String.length text then die "%s: no %S" dir sub
      else if String.sub text i n = sub then i + n
      else go (i + 1)
    in
    go from
  in
  let number from key =
    let i = find from (Printf.sprintf "\"%s\": " key) in
    let j = ref i in
    while !j < String.length text && text.[!j] >= '0' && text.[!j] <= '9' do
      incr j
    done;
    int_of_string (String.sub text i (!j - i))
  in
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun n -> Sys.is_directory (Filename.concat dir n))
  |> List.sort compare
  |> List.map (fun n ->
      let i = find 0 (Printf.sprintf "\"%s\": {" n) in
      (n, number i "y", number i "x"))

(* Row [i]'s first [k] tokens of a [[rows; width; …]] block, as [k] rows of [c]
   values. *)
let tokens_of block i k =
  let c = Nx.numel block / (Nx.dim 0 block * Nx.dim 1 block) in
  f64 (Nx.reshape [| k; c |] (Nx.slice [ Nx.I i; Nx.R (0, k) ] block))

let max_abs a b =
  let w = ref 0. in
  Array.iteri (fun i x -> w := Float.max !w (Float.abs (x -. b.(i)))) a;
  !w

(* Token values as multisets: the reference's copies against our counts. *)
let same_multiset ~c dump mine counts =
  let tbl = Hashtbl.create 64 in
  let key a i = Array.to_list (Array.sub a (i * c) c) in
  for i = 0 to (Array.length dump / c) - 1 do
    let k = key dump i in
    Hashtbl.replace tbl k (1 + Option.value ~default:0 (Hashtbl.find_opt tbl k))
  done;
  Array.iteri
    (fun i m ->
      let k = key mine i in
      Hashtbl.replace tbl k
        (Option.value ~default:0 (Hashtbl.find_opt tbl k) - m))
    counts;
  Hashtbl.fold (fun _ v ok -> ok && v = 0) tbl true

let input name = name = "s2_in" || name = "s1_in"

(* Block tolerance: relative to the largest magnitude of the reference's block,
   the reference's own float32 spread (1e-5 of the output) with a margin for
   blocks that are not normalised. *)
let block_tol = 1e-4

let validate_blocks (type s d) (t : d target)
    (m : ((float, s, d) Nx.t, d) Model.t) ~fixtures ~device =
  let dir =
    Filename.concat fixtures
      (Printf.sprintf "layers/medium__%s__noflag" (ref_device device))
  in
  let pixels = dumped_pixels dir in
  let n = List.length pixels in
  let px, (_, w) = Pixels.of_tile (Filename.concat fixtures "tiles/small_64") in
  let indices =
    Nx.create Nx.int64 [| n |]
      (Array.of_list
         (List.map (fun (_, y, x) -> Int64.of_int ((y * w) + x)) pixels))
  in
  let sub =
    {
      Pixels.s2 = Nx_ragged.take ~indices px.s2;
      s1 = Nx_ragged.take ~indices px.s1;
    }
  in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 256 }
  in
  let b, _ = Batch.pack shape sub (Batch.v2_counts sub) ~from:0 in
  let blocks =
    let bt = Batch.place t.at b in
    if not t.jit then Model.blocks m (Model.inputs m bt)
    else
      let names = ref [] in
      let model = Nx.Ptree.instantiate (module Model) in
      let batch = Nx.Ptree.v Batch.walk in
      let f =
        Rune.jit
          Nx.Ptree.(model @-> batch @-> returns (list tensor))
          (fun m b ->
            let l = Model.blocks m (Model.inputs m b) in
            names := List.map fst l;
            List.map snd l)
      in
      let vs = f m bt in
      List.combine !names vs
  in
  let blocks = List.map (fun (k, v) -> (k, Nx.place Nx.Host.on v)) blocks in
  let tokens (r : Nx.host Batch.rows) i =
    Int32.to_int (Nx.item [ i; 0 ] r.stop)
  in
  let counts (r : Nx.host Batch.rows) i k =
    Array.init k (fun j ->
        Float.to_int (Float.round (Float.exp (Nx.item [ i; j ] r.log_count))))
  in
  List.iter
    (fun (name, v) ->
      let worst = ref 0. and scale = ref 0. and exact = ref true in
      List.iteri
        (fun i (pixel, _, _) ->
          let file =
            Filename.concat dir (Filename.concat pixel (name ^ ".npy"))
          in
          if Sys.file_exists file then begin
            let dump = f64 (load_f32 file) in
            let sensor = String.sub name 0 2 in
            let per_token =
              (sensor = "s2" || sensor = "s1")
              && not (String.ends_with ~suffix:"pooled" name)
            in
            let mine, dump =
              if per_token then begin
                let r = if sensor = "s2" then b.s2 else b.s1 in
                let k = tokens r i in
                let mine = tokens_of v i k in
                let c = Array.length mine / k in
                if input name then
                  exact :=
                    !exact
                    && same_multiset ~c dump mine (counts r i k)
                    && max_abs mine (Array.sub dump 0 (k * c)) = 0.;
                (mine, Array.sub dump 0 (k * c))
              end
              else (f64 (Nx.slice [ Nx.I i ] v), dump)
            in
            Array.iter (fun x -> scale := Float.max !scale (Float.abs x)) dump;
            worst := Float.max !worst (max_abs mine dump)
          end)
        pixels;
      let tol = block_tol *. Float.max 1. !scale in
      let pass = !worst <= tol && !exact in
      report
        ~name:(Printf.sprintf "medium/%s/blocks/%s" device name)
        ~pass
        (Printf.sprintf "max|d| %.3g  max|ref| %.3g%s" !worst !scale
           (if input name then
              Printf.sprintf "  tokens and counts exact %b" !exact
            else "")))
    blocks

(* Training *)

let float_opt opts name default =
  match opt opts name with
  | None -> default
  | Some v -> (
      match float_of_string_opt v with
      | Some x -> x
      | None -> die "%s: %S" name v)

let now = Unix.gettimeofday

(* Words the program allocated so far. *)
let words () =
  let s = Gc.quick_stat () in
  s.minor_words +. s.major_words -. s.promoted_words

let host_f32 a name =
  match Nx_io.Archive.find name a with
  | None -> die "%s: no entry" name
  | Some (Nx.P x) -> Nx_io.Archive.float ~shape:(Nx.shape x) Nx.float32 name a

(* A student's heads in PyTorch's layout, [k.weight] [[d; k]] and [k.bias]. *)
let load_heads file prefixes =
  let a = Nx_io.load_npz file in
  List.map
    (fun k ->
      let w = Nx.matrix_transpose (host_f32 a (Printf.sprintf "%d.weight" k)) in
      let b = host_f32 a (Printf.sprintf "%d.bias" k) in
      (k, [ { Kaun.Linear.w; b = Some b } ]))
    prefixes

(* Pixels at the corpus indices [indices]. *)
let take_pixels (px : Nx.host Pixels.t) indices =
  {
    Pixels.s2 = Nx_ragged.take ~indices px.s2;
    s1 = Nx_ragged.take ~indices px.s1;
  }

(* Each place's key: the run's dropout key indexed by the place's stream
   position, the first position at an empty place. *)
let place_keys key ~first (b : Nx.host Batch.t) =
  let p =
    Nx.add_s (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L)) (Int64.of_int first)
  in
  Nx.Rng.fold_in_tensor key p

(* corpus *)

let corpus opts =
  let tiles = String.split_on_char ',' (req opts "--tiles") in
  let out = req opts "--out" in
  let key = Nx.Rng.key (int_opt opts "--seed" 0) in
  let bytes = int_opt opts "--bytes" (1 lsl 30) in
  if not (Sys.file_exists out) then Sys.mkdir out 0o755;
  let parts =
    List.mapi
      (fun i tile ->
        let px, _ = Pixels.of_tile tile in
        let l2 = Nx.to_array (Nx_ragged.lengths px.s2) in
        let l1 = Nx.to_array (Nx_ragged.lengths px.s1) in
        let kept =
          List.filter
            (fun p -> l2.(p) > 0L || l1.(p) > 0L)
            (List.init (Pixels.length px) Fun.id)
        in
        let indices =
          Nx.create Nx.int64
            [| List.length kept |]
            (Array.of_list (List.map Int64.of_int kept))
        in
        let dir = Filename.concat out (Printf.sprintf "part%d" i) in
        Pixels.save dir (take_pixels px indices);
        Printf.printf "%s: %d pixels, %d with an observation\n%!" tile
          (Pixels.length px) (List.length kept);
        dir)
      tiles
  in
  let n = Pixels.shuffle ~key ~bytes out parts in
  List.iter
    (fun d ->
      Array.iter (fun f -> Sys.remove (Filename.concat d f)) (Sys.readdir d);
      Sys.rmdir d)
    parts;
  Printf.printf "%s: %d pixels, shuffled\n" out n

(* targets *)

let targets opts =
  let teacher = req opts "--teacher" and dir = req opts "--corpus" in
  let out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let chunk = int_opt opts "--chunk" 8192 in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 512)
      ~slots:(int_opt opts "--slots" 32)
  in
  let px = Pixels.load dir in
  let n = Pixels.length px in
  let m = Model.load teacher in
  on device
    {
      run =
        (fun t ->
          let f =
            if flag opts "--bf16" then forward t (place t (bf16 m))
            else forward t (place t m)
          in
          let start = now () in
          let block pos =
            let len = min chunk (n - pos) in
            let e = Embedding.encode ~shape t.at f (Pixels.sub px ~pos ~len) in
            Printf.printf "targets: %d / %d pixels, %.0f px/s\n%!" (pos + len) n
              (Float.of_int (pos + len) /. (now () -. start));
            e
          in
          let blocks =
            Seq.map block
              (Seq.init ((n + chunk - 1) / chunk) (fun i -> i * chunk))
          in
          let d = Model.dims m in
          if flag opts "--fp16" then
            Nx_io.save_npy_concat out Nx.float16 [| d |]
              (Seq.map (Nx.cast Nx.float16) blocks)
          else Nx_io.save_npy_concat out Nx.float32 [| d |] blocks);
    }

(* distil *)

(* A run's keys: the corpus order of each epoch after the first, and
   dropout's. *)
let epoch_site = 0
let dropout_site = 1
let heads_site = 2
let init_site = 3

(* The corpus index of each stream position of [p, p + len): epoch [e]'s
   ordering of [n], the corpus's own for epoch 0. *)
let indices key ~n p len =
  let pos = Nx.arange Nx.int64 p (p + len) 1 in
  let n' = Int64.of_int n in
  let e = Nx.div_s pos n' and i = Nx.mod_s pos n' in
  let shuffled =
    Nx.Rng.permute
      ~key:(Nx.Rng.fold_in_tensor (Nx.Rng.fold_in key epoch_site) e)
      ~n i
  in
  Nx.where (Nx.equal_s e 0L) i shuffled

(* The pixels a step's groups hold: from position [p], each group packed from
   where the previous one stopped, and the positions they took. *)
let step_groups ~key ~shape ~groups corpus targets p =
  let n = Pixels.length corpus in
  let window = groups * shape.Batch.pixels in
  let idx = indices key ~n p window in
  let px = take_pixels corpus idx in
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  let dkey = Nx.Rng.fold_in key dropout_site in
  let rec go g from acc =
    if g = groups then (List.rev acc, from)
    else
      let b, next = Batch.pack shape px counts ~from in
      let places = Nx.maximum b.pixel (Nx.scalar Nx.int64 0L) in
      let keys = place_keys dkey ~first:p b in
      let t = Nx.take ~axis:0 (Nx.take ~axis:0 places idx) targets in
      go (g + 1) next ({ Distil.batch = b; keys; targets = t } :: acc)
  in
  let gs, taken = go 0 0 [] in
  let stack f = Nx.stack (List.map f gs) in
  ( {
      Distil.batch =
        Batch.stack (List.map (fun (g : _ Distil.group) -> g.batch) gs);
      keys =
        Nx.Rng.of_tensor
          (stack (fun (g : _ Distil.group) -> Nx.Rng.to_tensor g.keys));
      targets = stack (fun (g : _ Distil.group) -> g.targets);
    },
    taken )

let tree_bytes s x = Nx.Ptree.fold s (fun _ a acc -> acc + Nx.nbytes a) x 0

(* Mean cosine of each prefix's head to the teacher over [px]. *)
let fidelity (type d) (t : d target) shape
    (st : (d Nx.float32_t, d) Distil.state) px targets =
  let e = Embedding.encode ~shape t.at (forward t st.model) px in
  let host =
    Nx.Ptree.place
      (module Distil.Params)
      Nx.Host.on
      { Distil.net = st.model.net; heads = st.heads }
  in
  let c = Distil.cosines host.heads e ~targets in
  let mean = Nx.to_array (Nx.mean ~axes:[ 0 ] c) in
  List.mapi (fun i (p, _) -> (p, mean.(i))) host.heads

let distil opts =
  let dir = req opts "--corpus" and out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let seed = int_opt opts "--seed" 0 in
  let key = Nx.Rng.key seed in
  let steps = int_opt opts "--steps" 1000 in
  let until = min steps (int_opt opts "--until" steps) in
  let groups = int_opt opts "--groups" 1 in
  let every = int_opt opts "--every" 100 in
  let lr = float_opt opts "--lr" 8e-4 in
  let warmup = float_opt opts "--warmup" 0.05 in
  let decay = float_opt opts "--decay" 1e-4 in
  let clip = float_opt opts "--clip" 1.0 in
  let dropout =
    {
      Model.layers = float_opt opts "--dropout" 0.1;
      reducer = float_opt opts "--reducer-dropout" 0.2;
    }
  in
  let corpus = Pixels.load dir in
  let targets =
    match Nx_io.load_npy (req opts "--targets") with
    | Nx.P x -> Nx.cast Nx.float32 x
  in
  let d_t = Nx.dim 1 targets in
  let m, heads =
    match (opt opts "--init", opt opts "--arch") with
    | Some file, _ ->
        let m = Model.load file in
        let heads =
          match opt opts "--heads" with
          | Some h -> load_heads h m.prefixes
          | None ->
              Distil.heads
                ~key:(Nx.Rng.fold_in key heads_site)
                ~prefixes:m.prefixes d_t
        in
        (m, heads)
    | None, Some a ->
        let arch =
          match a with
          | "nano" -> Model.nano
          | "small" -> Model.small
          | "medium" -> Model.medium
          | "large" -> Model.large
          | a -> die "--arch %S: nano, small, medium or large" a
        in
        let m = Model.init ~key:(Nx.Rng.fold_in key init_site) arch in
        ( m,
          Distil.heads
            ~key:(Nx.Rng.fold_in key heads_site)
            ~prefixes:m.prefixes d_t )
    | None, None -> die "--init or --arch is required\n%s" usage
  in
  (* Rows for the duplicate-filled tokens of the corpus's first pixels. *)
  let pixels = int_opt opts "--pixels" 2048
  and slots = int_opt opts "--slots" 32 in
  let width = 256 in
  let shape =
    let sample =
      Pixels.sub corpus ~pos:0 ~len:(min 4096 (Pixels.length corpus))
    in
    let _, c = Batch.expand sample (Batch.v2_counts sample) in
    let mean x =
      Float.of_int (Nx.numel x) /. Float.of_int (Pixels.length sample)
    in
    let rows tokens =
      max 1
        (Float.to_int
           (Float.ceil
              (1.05 *. tokens *. Float.of_int pixels /. Float.of_int width))
        + 1)
    in
    {
      Batch.pixels;
      s2_rows = rows (mean c.s2);
      s1_rows = rows (mean c.s1);
      slots;
      width;
    }
  in
  let warmup_steps =
    max 1 (Float.to_int (Float.round (warmup *. Float.of_int steps)))
  in
  let sched () =
    Vega.Schedule.warmup_cosine_decay ~init_value:0. ~peak_value:lr
      ~warmup_steps
      ~decay_steps:(max 1 (steps - warmup_steps))
      ()
  in
  let compute_bf16 = flag opts "--bf16" in
  let heldout =
    match opt opts "--heldout" with
    | None -> None
    | Some s -> (
        match String.split_on_char ':' s with
        | [ c; t ] ->
            let px = Pixels.load c in
            let t =
              match Nx_io.load_npy t with Nx.P x -> Nx.cast Nx.float32 x
            in
            Some (px, t)
        | _ -> die "--heldout CORPUS:TARGETS")
  in
  if not (Sys.file_exists out) then Sys.mkdir out 0o755;
  let ckpt = Filename.concat out "state.safetensors" in
  let log_file = Filename.concat out "log.tsv" in
  let resume = flag opts "--resume" in
  if resume && not (Sys.file_exists ckpt) then die "%s: no state to resume" ckpt;
  if (not resume) && (Sys.file_exists ckpt || Sys.file_exists log_file) then
    die "%s holds a run: --resume it, or write to another --out" out;
  (* The options a resume must repeat: every one that changes the steps. *)
  let run =
    String.concat " "
      ([
         Printf.sprintf "--corpus %s --targets %s" dir (req opts "--targets");
         (match (opt opts "--init", opt opts "--arch") with
         | Some f, _ ->
             Printf.sprintf "--init %s --heads %s" f
               (Option.value ~default:"drawn" (opt opts "--heads"))
         | None, a -> "--arch " ^ Option.value ~default:"" a);
         Printf.sprintf
           "--seed %d --steps %d --groups %d --pixels %d --slots %d --lr %h \
            --warmup %h --decay %h --clip %h --dropout %h --reducer-dropout %h"
           seed steps groups pixels slots lr warmup decay clip dropout.layers
           dropout.reducer;
       ]
      @ if compute_bf16 then [ "--bf16" ] else [])
  in
  (* A resumed run's log keeps the steps its state holds. *)
  let kept_log =
    if not resume then []
    else
      In_channel.with_open_bin log_file In_channel.input_all
      |> String.split_on_char '\n'
      |> List.filter (fun l -> l <> "")
  in
  let log = open_out log_file in
  let keep_log upto =
    List.iter
      (fun l ->
        match String.split_on_char '\t' l with
        | s :: _ when int_of_string s <= upto -> output_string log (l ^ "\n")
        | _ -> ())
      kept_log;
    flush log
  in
  Printf.printf
    "distil: %d pixels, shape {pixels %d; s2_rows %d; s1_rows %d; slots %d; \
     width %d}, %d groups, dropout %g/%g, %s\n\
     %!"
    (Pixels.length corpus) shape.pixels shape.s2_rows shape.s1_rows shape.slots
    shape.width groups dropout.layers dropout.reducer
    (if compute_bf16 then "bf16" else "f32");
  on device
    {
      run =
        (fun (type d) (t : d target) ->
          let st_ptree = Nx.Ptree.instantiate (module Distil.State) in
          let group = Nx.Ptree.instantiate (module Distil.Group) in
          let fresh =
            Nx.Ptree.place
              (module Distil.Params)
              t.at
              { Distil.net = m.net; heads }
          in
          let start =
            Distil.init
              { (Nx.Ptree.place (module Model) t.at m) with net = fresh.net }
              fresh.heads
          in
          let st, position =
            if resume then begin
              let st, k, position, saved =
                Checkpoint.load ckpt (module Distil.State) ~like:start
              in
              if saved <> run then
                die "%s is a run of\n  %s\nnot of\n  %s" ckpt saved run;
              if
                Nx.to_array (Nx.Rng.to_tensor k)
                <> Nx.to_array (Nx.Rng.to_tensor key)
              then die "%s: the run's key is not --seed %d's" ckpt seed;
              (st, position)
            end
            else (start, 0)
          in
          let sched = sched () in
          let step_fn
              (g :
                (d Nx.float32_t, d) Distil.group
                * (d Nx.float32_t, d) Distil.state) :
              d Nx.float32_t * (d Nx.float32_t, d) Distil.state =
            if compute_bf16 then
              Distil.step ~compute:Nx.bfloat16 ~dropout ~lr:sched
                ~weight_decay:(Vega.rate decay) ~clip g
            else
              Distil.step ~compute:Nx.float32 ~dropout ~lr:sched
                ~weight_decay:(Vega.rate decay) ~clip g
          in
          let signature :
              ( (d Nx.float32_t, d) Distil.group
                * (d Nx.float32_t, d) Distil.state ->
                d Nx.float32_t * (d Nx.float32_t, d) Distil.state,
                d )
              Nx.Ptree.fn =
            Nx.Ptree.(pair group st_ptree @-> returns (pair tensor st_ptree))
          in
          let step = if t.jit then Rune.jit signature step_fn else step_fn in
          let first =
            Int32.to_int (Nx.item [] (Nx.place Nx.Host.on st.opt.step))
          in
          keep_log first;
          let gs0, taken0 =
            step_groups ~key ~shape ~groups corpus targets position
          in
          let gs0 = Nx.Ptree.place (module Distil.Group) t.at gs0 in
          if t.jit then begin
            let fp = Rune.footprint signature step_fn (gs0, st) in
            let live = tree_bytes st_ptree st + tree_bytes group gs0 in
            List.iter
              (fun (r, b) ->
                Printf.printf
                  "footprint: %s %.2f GB, with the state and a step's groups \
                   %.2f GB\n\
                   %!"
                  (Rig.name r)
                  (Float.of_int b /. 1e9)
                  (Float.of_int (b + live) /. 1e9))
              fp;
            match opt opts "--memory" with
            | None -> ()
            | Some s ->
                let budget = float_of_string s in
                List.iter
                  (fun (r, b) ->
                    if Rig.name r <> "host" && Float.of_int (b + live) > budget
                    then
                      die "%s needs %.2f GB, above --memory %.2f GB"
                        (Rig.name r)
                        (Float.of_int (b + live) /. 1e9)
                        (budget /. 1e9))
                  fp
          end;
          let rec go st position i ~gs =
            if i >= until then (st, position)
            else begin
              let t0 = now () and w0 = words () in
              let gs, taken =
                match gs with
                | Some g -> g
                | None ->
                    let g, taken =
                      step_groups ~key ~shape ~groups corpus targets position
                    in
                    (Nx.Ptree.place (module Distil.Group) t.at g, taken)
              in
              (* A compiled step writes the new state over the old one. *)
              let st = if t.jit then Nx.Ptree.donate st_ptree st else st in
              let loss, st = step (gs, st) in
              let loss = Nx.item [] (Nx.place Nx.Host.on loss) in
              let dt = now () -. t0 and dw = words () -. w0 in
              let filled =
                Nx.to_array (Nx.place Nx.Host.on gs.batch.pixel)
                |> Array.fold_left (fun a p -> if p >= 0L then a + 1 else a) 0
              in
              Printf.fprintf log "%d\t%d\t%d\t%d\t%.9g\t%.4f\t%.0f\n%!" (i + 1)
                position taken filled loss dt dw;
              if (i + 1) mod 10 = 0 || i < 3 then
                Printf.printf
                  "step %d position %d pixels %d loss %.6f %.3f s %.0f words\n\
                   %!"
                  (i + 1) position filled loss dt dw;
              let position = position + taken in
              if (i + 1) mod every = 0 || i + 1 = until then
                Checkpoint.save ckpt
                  (module Distil.State)
                  st ~key ~position ~run;
              go st position (i + 1) ~gs:None
            end
          in
          let st, _ = go st position first ~gs:(Some (gs0, taken0)) in
          match heldout with
          | None -> ()
          | Some (px, tg) ->
              let shape = Batch.typical ~pixels:4096 ~slots:32 in
              List.iter
                (fun (k, c) ->
                  Printf.printf "fidelity: prefix %d mean cos %.6f\n" k c)
                (fidelity t shape st px tg));
    };
  close_out log

(* validate --training *)

(* The oracle's distillation: the student [student.pt], its heads [heads.npz],
   the targets and the tile's pixels [pixels.npy] of [gen/grads.py init], at
   v2's dropout, keyed by each pixel's index there. *)
let training_inputs dir tile =
  let file f = Filename.concat dir f in
  let m = Model.load (file "student.pt") in
  let heads = load_heads (file "heads.npz") m.prefixes in
  let targets = Nx.unpack Nx.float32 (Nx_io.load_npy (file "targets.npy")) in
  let px, _ = Pixels.of_tile tile in
  let px =
    take_pixels px (Nx.unpack Nx.int64 (Nx_io.load_npy (file "pixels.npy")))
  in
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  (m, heads, targets, px, counts)

let oracle_key = Nx.Rng.key 0

(* float32's spacing relative to its value, at most. *)
let f32_ulp = Float.ldexp 1. (-23)

(* Writes the tokens and masks of each pixel, one a row, for the oracle. *)
let write_masks dir tile =
  let m, _, _, px, counts = training_inputs dir tile in
  let n = Pixels.length px in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 256 }
  in
  let b, _ = Batch.pack shape px counts ~from:0 in
  let keys = place_keys oracle_key ~first:0 b in
  let masks = Model.masks ~dropout:Model.v2_dropout ~keys m b in
  List.iter
    (fun (name, x) ->
      Nx_io.save_npy
        (Filename.concat dir ("mask_" ^ name ^ ".npy"))
        (Nx.cast Nx.uint8 x))
    masks;
  let blocks = Model.blocks m (Model.inputs m b) in
  List.iter
    (fun s ->
      Nx_io.save_npy
        (Filename.concat dir ("raven_" ^ s ^ ".npy"))
        (List.assoc s blocks))
    [ "s2_in"; "s1_in" ];
  let len (r : Nx.host Batch.rows) =
    Nx.cast Nx.int64 (Nx.slice [ Nx.A; Nx.I 0 ] r.stop)
  in
  Nx_io.save_npy
    (Filename.concat dir "raven_lengths.npy")
    (Nx.stack ~axis:1 [ len b.s2; len b.s1 ]);
  Out_channel.with_open_bin (Filename.concat dir "raven_masks.json") (fun oc ->
      Printf.fprintf oc "{\"order\": [%s]}\n"
        (String.concat ", "
           (List.map (fun (k, _) -> Printf.sprintf "%S" k) masks)));
  Printf.printf "%s: %d masks of %d pixels\n" dir (List.length masks) n

let leaves s x =
  List.rev
    (Nx.Ptree.fold s
       (fun p a acc ->
         ( Nx.Ptree.Path.to_string p,
           Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on a)) )
         :: acc)
       x [])

let rel a b =
  let d = ref 0. and n = ref 0. in
  Array.iteri
    (fun i x ->
      d := !d +. ((x -. b.(i)) ** 2.);
      n := !n +. (b.(i) ** 2.))
    a;
  Float.sqrt (!d /. Float.max !n 1e-300)

let validate_training opts =
  let dir = req opts "--training" and tile = req opts "--tile" in
  if flag opts "--masks" then write_masks dir tile
  else begin
    let device = Option.value ~default:"cpu" (opt opts "--device") in
    let prec = if flag opts "--bf16" then "bf16" else "f32" in
    let m, heads, targets, px, counts = training_inputs dir tile in
    let n = Pixels.length px in
    let shape =
      { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 4; width = 256 }
    in
    let b, _ = Batch.pack shape px counts ~from:0 in
    let keys = place_keys oracle_key ~first:0 b in
    let host = Nx.Ptree.instantiate (module Distil.Params) in
    let file f = Filename.concat dir f in
    let oracle prec =
      let g = Model.load (file ("grads_" ^ prec ^ ".pt")) in
      let h = load_heads (file ("heads_grads_" ^ prec ^ ".npz")) m.prefixes in
      let loss =
        Nx.item []
          (Nx.unpack Nx.float64
             (Nx_io.load_npy (file ("loss_" ^ prec ^ ".npy"))))
      in
      (loss, leaves host { Distil.net = g.net; heads = h })
    in
    let l64, g64 = oracle "f64" in
    let lref, gref = oracle prec in
    on device
      {
        run =
          (fun (type d) (t : d target) ->
            let params =
              Nx.Ptree.place
                (module Distil.Params)
                t.at
                { Distil.net = m.net; heads }
            in
            let mt = Nx.Ptree.place (module Model) t.at m in
            let p = Nx.Ptree.instantiate (module Distil.Params) in
            let f (params, b, keys, targets) =
              let objective (type c) (dt : (float, c) Nx.dtype) q =
                let q = Nx.Ptree.cast (module Distil.Params) dt q in
                Nx.div_s
                  (Distil.loss ~dropout:Model.v2_dropout ~keys mt q b ~targets)
                  (Float.of_int n)
              in
              if prec = "bf16" then
                Rune.value_and_grad p (objective Nx.bfloat16) params
              else Rune.value_and_grad p (objective Nx.float32) params
            in
            let args =
              ( params,
                Batch.place t.at b,
                Nx.Rng.place t.at keys,
                Nx.place t.at targets )
            in
            let loss, grads =
              if not t.jit then f args
              else
                Rune.jit
                  Nx.Ptree.(
                    quad p (v Batch.walk) Nx.Rng.ptree tensor
                    @-> returns (pair tensor p))
                  f args
            in
            let loss = Nx.item [] (Nx.place Nx.Host.on loss) in
            let mine = leaves p grads in
            let lrel = Float.abs (loss -. l64) /. Float.abs l64 in
            (* The oracle sums its pixels' losses in float64; raven's is one
               float32 value, which its rounding alone can move by an ulp. *)
            let lbound =
              Float.max (2. *. Float.abs (lref -. l64) /. Float.abs l64) f32_ulp
            in
            report
              ~name:(Printf.sprintf "training/%s/%s/loss" device prec)
              ~pass:(lrel <= lbound)
              (Printf.sprintf
                 "raven %.10g  torch f64 %.10g  rel %.3g  bound %.3g" loss l64
                 lrel lbound);
            let worst = ref 0. in
            List.iter2
              (fun (path, g) ((path', r64), (_, rp)) ->
                if path <> path' then die "%s against %s" path path';
                let mine = rel g r64 and theirs = rel rp r64 in
                let ratio = mine /. (2. *. theirs) in
                worst := Float.max !worst ratio;
                report
                  ~name:(Printf.sprintf "training/%s/%s/%s" device prec path)
                  ~pass:(ratio <= 1.)
                  (Printf.sprintf "rel %.3g  torch %s rel %.3g  ratio %.2f" mine
                     prec theirs ratio))
              mine (List.combine g64 gref);
            Printf.printf "worst ratio to twice torch's %s spread: %.3f\n" prec
              !worst);
      }
  end

(* Main *)

(* The teacher's reference has one float32 fixture and two bfloat16 ones of
   equal bits, so it has no measured spread of its own at either precision.
   Float32 is judged at the students' float32 bound; bfloat16 against bfloat16
   autocast at twice the reference's bfloat16-against-float32 distance (max |Δ|
   2.9e-2, 1 − cos 8.8e-6), the most rounding to bfloat16 moved it. A pair
   across precisions is reported, not judged. *)
let teacher_bf16_tol = (2. *. 2.9e-2, 2. *. 8.8e-6)

let validate_teacher (type d) (t : d target) m ~fixtures ~bf16:with_bf16 =
  if not t.jit then die "the teacher validates on a GPU";
  let tile = "small_64" in
  let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
  let valid = valid_of fixtures tile in
  let shape = Batch.typical ~pixels:512 ~slots:32 in
  let reference kind =
    Printf.sprintf "%s/emb/%s/teacher__cuda__%s__bp512__sdpa.npy" fixtures tile
      kind
  in
  let check label e ~same ~tol ~other =
    let name kind = Printf.sprintf "teacher/%s/%s vs %s" label tile kind in
    validate_embedding ~label:(name same) ~valid ~tol e (reference same);
    validate_embedding ~label:(name other) ~valid e (reference other)
  in
  let e = Embedding.encode ~shape t.at (forward t (place t m)) px in
  check "fp32" e ~same:"fp32" ~tol:(max_abs_tol, cos_tol) ~other:"bf16-autocast";
  Gc.full_major ();
  if with_bf16 then begin
    let e = Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px in
    check "bf16" e ~same:"bf16-autocast" ~tol:teacher_bf16_tol ~other:"fp32"
  end

let validate_student (type d) (t : d target) m ~fixtures ~shape ~device name =
  let m = place t m in
  let f = forward t m in
  List.iter
    (fun tile ->
      let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
      let e = Embedding.encode ~shape t.at f px in
      let valid = valid_of fixtures tile in
      let base =
        Printf.sprintf "%s/emb/%s/%s__%s__fp32__bp4096__fast" fixtures tile name
          (ref_device device)
      in
      let label = Printf.sprintf "%s/%s/%s" name device tile in
      validate_embedding ~label ~valid ~tol:(max_abs_tol, cos_tol) e
        (base ^ ".npy");
      validate_int8 ~label:(label ^ "/int8") ~valid e
        ~codes:(base ^ "_int8.npy") ~scales:(base ^ "_scales.npy"))
    tiles;
  if name = "medium" then validate_blocks t m ~fixtures ~device

let summary () =
  Printf.printf "%d checks, %d failed\n" !checks (List.length !failed);
  List.iter (Printf.printf "failed: %s\n") (List.rev !failed);
  exit (if !failed = [] then 0 else 1)

let validate opts =
  if opt opts "--training" <> None then begin
    validate_training opts;
    summary ()
  end;
  let weights = req opts "--weights" and fixtures = req opts "--fixtures" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let models =
    match opt opts "--models" with
    | None -> student_names
    | Some l ->
        let ms = String.split_on_char ',' l in
        List.iter
          (fun m ->
            if not (List.mem m ("teacher" :: student_names)) then
              die "--models: %S is not a model\n%s" m usage)
          ms;
        ms
  in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let one t name =
    match Model.load (weight_file weights name) with
    | exception Failure why -> report ~name ~pass:false why
    | m when name = "teacher" ->
        validate_teacher t m ~fixtures ~bf16:(flag opts "--bf16")
    | m -> validate_student t m ~fixtures ~shape ~device name
  in
  on device
    {
      run =
        (fun t ->
          List.iter
            (fun name ->
              one t name;
              Gc.full_major ())
            models);
    };
  summary ()

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "encode" :: args ->
      encode
        (parse ~flags:[ "--int8"; "--bf16" ]
           ~options:
             [
               "--model";
               "--tile";
               "--out";
               "--landmask";
               "--device";
               "--dims";
               "--pixels";
               "--slots";
             ]
           args)
  | "validate" :: args ->
      validate
        (parse ~flags:[ "--bf16"; "--masks" ]
           ~options:
             [
               "--weights";
               "--fixtures";
               "--device";
               "--models";
               "--pixels";
               "--slots";
               "--training";
               "--tile";
             ]
           args)
  | "corpus" :: args ->
      corpus
        (parse ~flags:[]
           ~options:[ "--tiles"; "--out"; "--seed"; "--bytes" ]
           args)
  | "targets" :: args ->
      targets
        (parse ~flags:[ "--bf16"; "--fp16" ]
           ~options:
             [
               "--teacher";
               "--corpus";
               "--out";
               "--device";
               "--chunk";
               "--pixels";
               "--slots";
             ]
           args)
  | "distil" :: args ->
      distil
        (parse ~flags:[ "--bf16"; "--resume" ]
           ~options:
             [
               "--corpus";
               "--targets";
               "--out";
               "--init";
               "--heads";
               "--arch";
               "--device";
               "--seed";
               "--steps";
               "--groups";
               "--pixels";
               "--slots";
               "--every";
               "--lr";
               "--warmup";
               "--decay";
               "--clip";
               "--dropout";
               "--reducer-dropout";
               "--memory";
               "--heldout";
               "--until";
             ]
           args)
  | ("-h" | "--help" | "help") :: _ -> print_endline usage
  | _ -> die "%s" usage
