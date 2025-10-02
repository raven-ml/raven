module C = Configurator.V1

let get_llvm_config c =
  (* Try different llvm-config names in order of preference *)
  let possible_names = [ "llvm-config-21"; "llvm-config-20"; "llvm-config-19"; "llvm-config" ] in

  let rec find_llvm_config = function
    | [] -> None
    | cmd :: rest ->
        if C.Process.run_ok c cmd [ "--version" ] then Some cmd
        else find_llvm_config rest
  in

  match find_llvm_config possible_names with
  | None ->
      (* Try homebrew paths *)
      let brew_paths =
        [
          "/opt/homebrew/opt/llvm/bin/llvm-config";
          "/opt/homebrew/opt/llvm@21/bin/llvm-config";
          "/opt/homebrew/opt/llvm@20/bin/llvm-config";
          "/opt/homebrew/opt/llvm@19/bin/llvm-config";
          "/opt/homebrew/opt/llvm@18/bin/llvm-config";
          "/usr/bin/llvm-config";
          "/usr/local/opt/llvm/bin/llvm-config";
          "/usr/local/opt/llvm@21/bin/llvm-config";
          "/usr/local/opt/llvm@20/bin/llvm-config";
          "/usr/local/opt/llvm@19/bin/llvm-config";
        ]
      in
      find_llvm_config brew_paths
  | Some cmd -> Some cmd

let run_capture_opt c cmd args =
  (* Use run_ok to check if command succeeds without logging errors *)
  if C.Process.run_ok c cmd args then
    Some (C.Process.run_capture_exn c cmd args)
  else None

let () =
  C.main ~name:"llvm-config" (fun c ->
      match get_llvm_config c with
      | None ->
          C.die
            "Could not find llvm-config. Please install LLVM and ensure \
             llvm-config is in your PATH."
      | Some llvm_config ->
          let version =
            C.Process.run_capture_exn c llvm_config [ "--version" ]
            |> String.trim
          in
          Printf.printf "Found LLVM version %s using %s\n%!" version llvm_config;

          (* Get compiler flags *)
          let cflags =
            C.Process.run_capture_exn c llvm_config [ "--cflags" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          (* Add -fPIC on Linux and BSD systems for position-independent code.
             This is required when building shared libraries on x86-64 Linux to
             avoid relocation errors like "relocation R_X86_64_32 against
             `.data' can not be used when making a shared object" *)
          let cflags =
            match C.ocaml_config_var c "system" with
            | Some "linux"
            | Some "freebsd"
            | Some "netbsd"
            | Some "openbsd"
            | Some "dragonfly"
            | Some "gnu" ->
                "-fPIC" :: cflags
            | _ -> cflags
          in

          (* Get linker flags *)
          let ldflags =
            C.Process.run_capture_exn c llvm_config [ "--ldflags" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          (* Get libraries for different components *)
          let libs_core =
            C.Process.run_capture_exn c llvm_config
              [ "--libs"; "core"; "support" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          let libs_executionengine =
            C.Process.run_capture_exn c llvm_config
              [ "--libs"; "executionengine"; "mcjit"; "native" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          let libs_analysis =
            C.Process.run_capture_exn c llvm_config [ "--libs"; "analysis" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          let libs_target =
            C.Process.run_capture_exn c llvm_config
              [ "--libs"; "target"; "asmparser"; "asmprinter" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          let libs_bitreader =
            match run_capture_opt c llvm_config [ "--libs"; "bitreader" ] with
            | Some output ->
                output |> String.trim |> String.split_on_char ' '
                |> List.filter (fun s -> String.length s > 0)
            | None -> []
          in

          let libs_bitwriter =
            match run_capture_opt c llvm_config [ "--libs"; "bitwriter" ] with
            | Some output ->
                output |> String.trim |> String.split_on_char ' '
                |> List.filter (fun s -> String.length s > 0)
            | None -> []
          in

          let libs_transforms =
            (* Try different transform library names as they vary by LLVM
               version *)
            match
              run_capture_opt c llvm_config
                [ "--libs"; "transformutils"; "scalaropt"; "ipo"; "vectorize" ]
            with
            | Some output ->
                output |> String.trim |> String.split_on_char ' '
                |> List.filter (fun s -> String.length s > 0)
            | None -> (
                (* Try newer LLVM names *)
                match
                  run_capture_opt c llvm_config
                    [ "--libs"; "passes"; "transforms" ]
                with
                | Some output ->
                    output |> String.trim |> String.split_on_char ' '
                    |> List.filter (fun s -> String.length s > 0)
                | None -> [])
          in

          let libs_all =
            C.Process.run_capture_exn c llvm_config [ "--libs"; "all" ]
            |> String.trim |> String.split_on_char ' '
            |> List.filter (fun s -> String.length s > 0)
          in

          (* Deduplicate flags - important when LLVM uses monolithic shared library *)
          let deduplicate lst =
            let tbl = Hashtbl.create 16 in
            List.filter (fun x ->
              if Hashtbl.mem tbl x then false
              else (Hashtbl.add tbl x (); true)
            ) lst
          in

          (* Separate ldflags into library paths (-L) and libraries (-l) *)
          let is_lib_path flag = String.length flag >= 2 && String.sub flag 0 2 = "-L" in
          let lib_paths = List.filter is_lib_path ldflags in
          let lib_flags = List.filter (fun f -> not (is_lib_path f)) ldflags in

          (* Check if using monolithic shared library (all components return same libs) *)
          let using_monolithic =
            libs_core = libs_executionengine &&
            libs_core = libs_analysis &&
            libs_core = libs_target
          in

          (* For monolithic builds, only include -lLLVM-XX in core library *)
          let make_libs is_core libs =
            if using_monolithic && not is_core then
              deduplicate (lib_paths @ lib_flags)
            else
              deduplicate (lib_paths @ libs @ lib_flags)
          in

          (* Write sexp files for each library *)
          C.Flags.write_sexp "llvm_cflags.sexp" cflags;
          C.Flags.write_sexp "llvm_libs_core.sexp" (make_libs true libs_core);
          C.Flags.write_sexp "llvm_libs_executionengine.sexp"
            (make_libs false libs_executionengine);
          C.Flags.write_sexp "llvm_libs_analysis.sexp" (make_libs false libs_analysis);
          C.Flags.write_sexp "llvm_libs_target.sexp" (make_libs false libs_target);
          C.Flags.write_sexp "llvm_libs_bitreader.sexp"
            (make_libs false libs_bitreader);
          C.Flags.write_sexp "llvm_libs_bitwriter.sexp"
            (make_libs false libs_bitwriter);
          C.Flags.write_sexp "llvm_libs_transforms.sexp"
            (make_libs false libs_transforms);
          C.Flags.write_sexp "llvm_libs_all.sexp" (make_libs false libs_all);

          (* Generate version file for conditional compilation *)
          let version_parts = String.split_on_char '.' version in
          let version_major =
            match version_parts with
            | major :: _ -> ( try int_of_string major with _ -> 0)
            | [] -> 0
          in

          let version_file = open_out "llvm_version.ml" in
          Printf.fprintf version_file "(* Auto-generated by discover.ml *)\n";
          Printf.fprintf version_file "let major = %d\n" version_major;
          Printf.fprintf version_file "let version = \"%s\"\n" version;
          Printf.fprintf version_file "let llvm_config = \"%s\"\n" llvm_config;
          close_out version_file)
