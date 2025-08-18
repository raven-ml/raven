module C = Configurator.V1

let get_llvm_config c =
  (* Try different llvm-config names in order of preference *)
  let possible_names = [
    "llvm-config";
    "llvm-config-20";
    "llvm-config-19";
    "llvm-config-18";
    "llvm-config-17";
    "llvm-config-16";
    "llvm-config-15";
    "llvm-config-14";
  ] in
  
  let rec find_llvm_config = function
    | [] -> None
    | cmd :: rest ->
        if C.Process.run_ok c cmd ["--version"] then Some cmd
        else find_llvm_config rest
  in
  
  match find_llvm_config possible_names with
  | None -> 
      (* Try homebrew paths *)
      let brew_paths = [
        "/opt/homebrew/opt/llvm/bin/llvm-config";
        "/opt/homebrew/opt/llvm@20/bin/llvm-config";
        "/opt/homebrew/opt/llvm@19/bin/llvm-config";
        "/opt/homebrew/opt/llvm@18/bin/llvm-config";
        "/usr/local/opt/llvm/bin/llvm-config";
        "/usr/local/opt/llvm@20/bin/llvm-config";
        "/usr/local/opt/llvm@19/bin/llvm-config";
      ] in
      find_llvm_config brew_paths
  | Some cmd -> Some cmd

let run_capture_opt c cmd args =
  (* Use run_ok to check if command succeeds without logging errors *)
  if C.Process.run_ok c cmd args then
    Some (C.Process.run_capture_exn c cmd args)
  else
    None

let () =
  C.main ~name:"llvm-config" (fun c ->
    match get_llvm_config c with
    | None ->
        C.die "Could not find llvm-config. Please install LLVM and ensure llvm-config is in your PATH."
    | Some llvm_config ->
        let version = 
          C.Process.run_capture_exn c llvm_config ["--version"]
          |> String.trim
        in
        Printf.printf "Found LLVM version %s using %s\n%!" version llvm_config;
        
        (* Get compiler flags *)
        let cflags = 
          C.Process.run_capture_exn c llvm_config ["--cflags"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        (* Get linker flags *)
        let ldflags = 
          C.Process.run_capture_exn c llvm_config ["--ldflags"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        (* Get libraries for different components *)
        let libs_core = 
          C.Process.run_capture_exn c llvm_config ["--libs"; "core"; "support"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        let libs_executionengine = 
          C.Process.run_capture_exn c llvm_config 
            ["--libs"; "executionengine"; "mcjit"; "native"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        let libs_analysis = 
          C.Process.run_capture_exn c llvm_config ["--libs"; "analysis"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        let libs_target = 
          C.Process.run_capture_exn c llvm_config ["--libs"; "target"; "asmparser"; "asmprinter"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        let libs_bitreader = 
          match run_capture_opt c llvm_config ["--libs"; "bitreader"] with
          | Some output ->
              output |> String.trim |> String.split_on_char ' '
              |> List.filter (fun s -> String.length s > 0)
          | None -> []
        in
        
        let libs_bitwriter = 
          match run_capture_opt c llvm_config ["--libs"; "bitwriter"] with
          | Some output ->
              output |> String.trim |> String.split_on_char ' '
              |> List.filter (fun s -> String.length s > 0)
          | None -> []
        in
        
        let libs_transforms = 
          (* Try different transform library names as they vary by LLVM version *)
          match run_capture_opt c llvm_config ["--libs"; "transformutils"; "scalaropt"; "ipo"; "vectorize"] with
          | Some output -> 
              output |> String.trim |> String.split_on_char ' ' 
              |> List.filter (fun s -> String.length s > 0)
          | None ->
              (* Try newer LLVM names *)
              match run_capture_opt c llvm_config ["--libs"; "passes"; "transforms"] with
              | Some output ->
                  output |> String.trim |> String.split_on_char ' '
                  |> List.filter (fun s -> String.length s > 0)
              | None -> []
        in
        
        let libs_all = 
          C.Process.run_capture_exn c llvm_config ["--libs"; "all"]
          |> String.trim
          |> String.split_on_char ' '
          |> List.filter (fun s -> String.length s > 0)
        in
        
        (* Write sexp files for each library *)
        C.Flags.write_sexp "llvm_cflags.sexp" cflags;
        C.Flags.write_sexp "llvm_libs_core.sexp" (ldflags @ libs_core);
        C.Flags.write_sexp "llvm_libs_executionengine.sexp" (ldflags @ libs_executionengine);
        C.Flags.write_sexp "llvm_libs_analysis.sexp" (ldflags @ libs_analysis);
        C.Flags.write_sexp "llvm_libs_target.sexp" (ldflags @ libs_target);
        C.Flags.write_sexp "llvm_libs_bitreader.sexp" (ldflags @ libs_bitreader);
        C.Flags.write_sexp "llvm_libs_bitwriter.sexp" (ldflags @ libs_bitwriter);
        C.Flags.write_sexp "llvm_libs_transforms.sexp" (ldflags @ libs_transforms);
        C.Flags.write_sexp "llvm_libs_all.sexp" (ldflags @ libs_all);
        
        (* Generate version file for conditional compilation *)
        let version_parts = String.split_on_char '.' version in
        let version_major = 
          match version_parts with
          | major :: _ -> (try int_of_string major with _ -> 0)
          | [] -> 0
        in
        
        let version_file = open_out "llvm_version.ml" in
        Printf.fprintf version_file "(* Auto-generated by discover.ml *)\n";
        Printf.fprintf version_file "let major = %d\n" version_major;
        Printf.fprintf version_file "let version = \"%s\"\n" version;
        Printf.fprintf version_file "let llvm_config = \"%s\"\n" llvm_config;
        close_out version_file
  )