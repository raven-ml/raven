(* Rig's probe for run.py: rig.exe SCENARIO DRIVER [ARG...], DRIVER one of
   metal, cuda, nv and amd. Each scenario prints one line of KEY=VALUE fields;
   compare/README.md says what each one measures. *)

open Rig

external metal_fill : unit -> nativeint = "rig_compare_metal_fill"

let get = function Ok value -> value | Error why -> failwith why

(* The floats an addition adds. *)
let count = 64

(* The micro rows' samples: each after untimed warmups and an idle pause. *)
let warmups = 3
let samples = 5
let idle = 0.02

let open_driver = function
  | "metal" -> get (open_ (Rig_metal.gpu 0))
  | "cuda" -> get (open_ (Rig_cuda.gpu 0))
  | "nv" -> get (open_ (Rig_nv_nvidia.gpu 0))
  | "amd" -> get (open_ (Rig_amd_amdgpu.gpu 0))
  | driver -> failwith ("unknown driver " ^ driver)

let fill buffer value =
  let source = Buffer.create host (4 * count) in
  Buffer.write source Bigarray.float32 (fun a -> Bigarray.Array1.fill a value);
  Buffer.copy ~src:source ~dst:buffer

let floats device value =
  let buffer = Buffer.create device (4 * count) in
  fill buffer value;
  buffer

let check out expected =
  let actual = Buffer.create host (4 * count) in
  Buffer.copy ~src:out ~dst:actual;
  Buffer.read actual Bigarray.float32 (fun values ->
      for i = 0 to count - 1 do
        if values.{i} <> expected then
          failwith
            (Printf.sprintf "output %d: %g, expected %g" i values.{i} expected)
      done)

(* [adder device driver image a b out] submits [out = a + b] with the image's
   [add], 64 threads in one group. Metal runs a dispatch recorded in an
   indirect command buffer; the other drivers launch the kernel. *)
let adder device driver image a b out =
  let add = Option.get (Image.resolve image "add") in
  let run, buffers =
    if driver = "metal" then begin
      let cap = Option.get (capability device Rig_metal_abi.key) in
      let args = Buffer.create device 32 in
      let pipeline = Image.entry add in
      let dispatch =
        Rig_metal_abi.{ pipeline; offset = Buffer.offset args;
                        groups = (1, 1, 1); threads = (count, 1, 1) }
      in
      let icb = get (cap.icb (Buffer.handle args) [| dispatch |]) in
      let part = Submission.(fill (metal_fill ()) ~params:16) in
      let submission =
        Submission.make ~keep:(icb, add)
          ~fixed:[ (args, Read); (a, Read); (b, Read); (out, Read_write) ]
          device [| part |]
      in
      Buffer.write args Bigarray.int64 (fun params ->
          List.iteri (fun i buffer ->
              params.{i} <- Int64.of_int (Submission.address submission buffer))
            [ a; b; out ];
          params.{3} <- Int64.of_int count);
      let run = Submission.Run.make submission in
      let block = Submission.block submission 0 in
      Submission.Run.int64 run block 0 (Int64.of_nativeint icb.handle);
      Submission.Run.int64 run block 8 1L;
      (run, [||])
    end
    else begin
      (* add takes the three addresses, then the count. *)
      let refs = Submission.[| read 0 ~at:0; read 1 ~at:8; write 2 ~at:16 |] in
      let part = Submission.(launch ~refs add ~params:32) in
      let submission = Submission.make device [| part |] in
      let run = Submission.Run.make submission in
      let block = Submission.block submission 0 in
      Submission.Run.groups run block 1 1 1;
      Submission.Run.threads run block count 1 1;
      Submission.Run.int64 run block 24 (Int64.of_int count);
      (run, [| a; b; out |])
    end
  in
  fun () -> submit run ~buffers

let load_image device path =
  get (Image.load device (In_channel.with_open_bin path In_channel.input_all))

(* Scenarios *)

(* The first kernel: one addition, checked after it completed; the time it
   completed. *)
let first device driver path =
  let image = load_image device path in
  let a = floats device 41. and b = floats device 1. in
  let out = floats device nan in
  Point.wait (adder device driver image a b out ());
  let completed = Profile.now () in
  check out 42.;
  completed

let start device driver path =
  Printf.printf "completed_ns=%d\n" (first device driver path)

let fit device driver path =
  ignore (first device driver path);
  Printf.printf "pid=%d\n%!" (Unix.getpid ());
  ignore (read_line ())

(* The timer covers opening each file, allocating its device buffer and the
   completed copy; the check reads every byte back after it. *)
let load device paths =
  let started = Profile.now () in
  let loaded =
    List.map (fun path ->
        let source = get (Rig_disk.of_file path) in
        let destination = Buffer.create device (Buffer.length source) in
        Buffer.copy ~src:source ~dst:destination;
        (source, destination))
      paths
  in
  let elapsed = Profile.now () - started in
  let chunk = 4 * 1024 * 1024 in
  let expected = Buffer.create host chunk and actual = Buffer.create host chunk in
  List.iter (fun (source, destination) ->
      let length = Buffer.length source in
      for i = 0 to (length + chunk - 1) / chunk - 1 do
        let first = i * chunk and length = min chunk (length - (i * chunk)) in
        let expected = Buffer.view expected ~first:0 ~length in
        let actual = Buffer.view actual ~first:0 ~length in
        Buffer.copy ~src:(Buffer.view source ~first ~length) ~dst:expected;
        Buffer.copy ~src:(Buffer.view destination ~first ~length) ~dst:actual;
        let bytes b =
          Buffer.read b Bigarray.char (fun a ->
              String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a))
        in
        if bytes expected <> bytes actual then
          failwith (Printf.sprintf "checkpoint differs in chunk %d" i)
      done)
    loaded;
  let bytes = List.fold_left (fun n (s, _) -> n + Buffer.length s) 0 loaded in
  Printf.printf "bytes=%d load_ns=%d\n" bytes elapsed

(* [calls] dependent additions, alternating between two prepared submissions,
   and a wait on the last. *)
let add device driver calls path =
  let image = load_image device path in
  let a = Buffer.create device (4 * count) and b = floats device 1. in
  let out = Buffer.create device (4 * count) in
  let forward = adder device driver image a b out in
  let backward = adder device driver image out b a in
  let sample () =
    fill a 41.;
    fill out nan;
    Unix.sleepf idle;
    let started = Profile.now () in
    for i = 0 to calls - 1 do
      let point = if i mod 2 = 0 then forward () else backward () in
      if i = calls - 1 then Point.wait point
    done;
    let elapsed = Profile.now () - started in
    check (if calls mod 2 = 0 then a else out) (float_of_int (41 + calls));
    elapsed
  in
  for _ = 1 to warmups do ignore (sample ()) done;
  let times = List.init samples (fun _ -> sample ()) in
  Printf.printf "samples_ns=%s\n" (String.concat "," (List.map string_of_int times))

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "start" :: driver :: [ image ] ->
      let device = open_driver driver in
      start device driver image;
      close device
  | "fit" :: driver :: [ image ] ->
      let device = open_driver driver in
      fit device driver image;
      close device
  | "load" :: driver :: (_ :: _ as paths) ->
      let device = open_driver driver in
      load device paths;
      close device
  | "add" :: driver :: calls :: [ image ] ->
      let device = open_driver driver in
      add device driver (int_of_string calls) image;
      close device
  | _ ->
      prerr_endline
        "usage: rig.exe start DRIVER IMAGE | fit DRIVER IMAGE | load DRIVER FILE... \
         | add DRIVER CALLS IMAGE";
      exit 2
