(* Copyright (c) 2026 The Raven authors. ISC License. *)
open Windtrap
open Tolk
open Tolk_uop

let retained_allocations () =
  let first = Tolk_cuda.create "CUDA:81" in
  let destination = Tolk_cuda.create "CUDA:82" in
  let make ~host device =
    let spec = { Device.Buffer_spec.default with host; nolru = true } in
    let buffer = Device.create_buffer ~spec ~size:8 ~dtype:Dtype.uint8 device in
    Device.Buffer.ensure_allocated buffer;
    buffer in
  let old_pinned = make ~host:true first and old_private = make ~host:false first in
  let replacement = Tolk_cuda.create "CUDA:81" in
  is_false (Device.allocator first == Device.allocator replacement);
  let new_pinned = make ~host:true replacement and new_private = make ~host:false replacement in
  let buffers = [old_pinned; old_private; new_pinned; new_private] in
  Fun.protect ~finally:(fun () -> List.iter Device.Buffer.deallocate buffers) (fun () ->
      List.iteri (fun i buffer ->
          let expected = Bytes.make 8 (Char.chr (17 + i)) in
          Device.Buffer.copyin buffer expected;
          let view = Device.Buffer.view buffer ~offset:2 ~size:4 ~dtype:Dtype.uint8 in
          Fun.protect ~finally:(fun () -> Device.Buffer.deallocate view) (fun () ->
              let address = Device.Buffer.addr ~target:(Device.allocator destination) view in
              equal nativeint (Nativeint.add (Option.get (Device.Buffer.host_addr buffer)) 2n) address;
              equal bytes (Bytes.sub expected 2 4) (Device.Buffer.as_bytes view)))
        [old_pinned; new_pinned];
      List.iter (fun buffer ->
          match Device.Buffer.addr ~target:(Device.allocator destination) buffer with
          | _ -> fail "device-only CUDA storage must use the shared staging path"
          | exception Storage.Mapping_unavailable _ -> ())
        [old_private; new_private])

let isolated_driver () =
  if Sys.getenv_opt "TOLK_TEST_CUDA_ALLOCATOR" = Some "1" then retained_allocations ()
  else begin
    let binary = Sys.executable_name in
    let directory = Filename.concat (Filename.dirname binary) "cuda_allocator_fixture" in
    let directory = if Filename.is_relative directory then
        Filename.concat (Sys.getcwd ()) directory else directory in
    let names = ["LD_LIBRARY_PATH="; "DYLD_LIBRARY_PATH="; "TOLK_TEST_CUDA_ALLOCATOR="] in
    let environment = Unix.environment () |> Array.to_list |> List.filter (fun entry ->
        not (List.exists (fun prefix -> String.starts_with ~prefix entry) names)) in
    let environment = Array.of_list
        (("LD_LIBRARY_PATH=" ^ directory) :: ("DYLD_LIBRARY_PATH=" ^ directory)
         :: "TOLK_TEST_CUDA_ALLOCATOR=1" :: environment) in
    let pid = Unix.create_process_env binary [|binary|] environment
        Unix.stdin Unix.stdout Unix.stderr in
    match snd (Unix.waitpid [] pid) with
    | Unix.WEXITED 0 -> ()
    | _ -> fail "isolated CUDA allocator fixture failed"
  end

let () = exit (run "CUDA allocation ownership" [
  test "retained allocations survive device replacement and stage device peers" isolated_driver;
])
