(* Episode visualization module
   Generates SVG visualizations and HTML replay pages for collected episodes *)

open Slide3  (* For episode_data type *)
open Slide4  (* For training_history type with collected_episodes *)

(* Helper to ensure directory exists *)
let ensure_dir dir =
  if not (Sys.file_exists dir) then
    Sys.command (Printf.sprintf "mkdir -p %s" dir) |> ignore

(* Convert gridworld state to SVG *)
let gridworld_state_to_svg state ~grid_size =
  let cell_size = 40 in
  let width = grid_size * cell_size in
  let height = grid_size * cell_size in

  (* Get state as array *)
  let state_array = Rune.to_array state in

  let buffer = Buffer.create 1024 in

  (* SVG header *)
  Buffer.add_string buffer
    (Printf.sprintf {|<svg width="%d" height="%d" xmlns="http://www.w3.org/2000/svg">
<rect width="%d" height="%d" fill="white" stroke="black" stroke-width="2"/>
|} width height width height);

  (* Draw grid cells *)
  for y = 0 to grid_size - 1 do
    for x = 0 to grid_size - 1 do
      let idx = y * grid_size + x in
      let value = state_array.(idx) in
      let px = x * cell_size in
      let py = y * cell_size in

      (* Cell colors based on value *)
      let color =
        if value = 0.0 then "#f0f0f0"  (* Empty *)
        else if value = 1.0 then "#333333"  (* Wall *)
        else if value = 2.0 then "#ff0000"  (* Player *)
        else if value = 3.0 then "#00ff00"  (* Goal *)
        else if value = 4.0 then "#0000ff"  (* Obstacle *)
        else "#888888"  (* Unknown *)
      in

      Buffer.add_string buffer
        (Printf.sprintf
          {|<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="gray" stroke-width="1"/>
|}
          px py cell_size cell_size color);

      (* Add text label for non-empty cells *)
      if value > 0.0 then
        let label =
          if value = 1.0 then "#"
          else if value = 2.0 then "P"
          else if value = 3.0 then "G"
          else if value = 4.0 then "O"
          else "?"
        in
        Buffer.add_string buffer
          (Printf.sprintf
            {|<text x="%d" y="%d" text-anchor="middle" font-size="20" font-weight="bold">%s</text>
|}
            (px + cell_size / 2) (py + cell_size / 2 + 7) label)
    done
  done;

  Buffer.add_string buffer "</svg>\n";
  Buffer.contents buffer

(* Convert Sokoban state to SVG *)
let sokoban_state_to_svg state ~grid_size =
  let cell_size = 40 in
  let width = grid_size * cell_size in
  let height = grid_size * cell_size in

  (* Get state as array *)
  let state_array = Rune.to_array state in

  let buffer = Buffer.create 1024 in

  (* SVG header *)
  Buffer.add_string buffer
    (Printf.sprintf {|<svg width="%d" height="%d" xmlns="http://www.w3.org/2000/svg">
<rect width="%d" height="%d" fill="#e8d4b0" stroke="black" stroke-width="2"/>
|} width height width height);

  (* Draw grid cells *)
  for y = 0 to grid_size - 1 do
    for x = 0 to grid_size - 1 do
      let idx = y * grid_size + x in
      let value = int_of_float state_array.(idx) in
      let px = x * cell_size in
      let py = y * cell_size in

      (* Sokoban cell types:
         0: Empty
         1: Wall
         2: Box
         3: Target
         4: BoxOnTarget
         5: Player
         6: PlayerOnTarget *)

      (* Background based on target/empty *)
      let bg_color =
        if value = 3 || value = 4 || value = 6 then "#ffcc99"  (* Target background *)
        else if value = 1 then "#8b4513"  (* Wall *)
        else "#e8d4b0"  (* Floor *)
      in

      Buffer.add_string buffer
        (Printf.sprintf
          {|<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#666" stroke-width="1"/>
|}
          px py cell_size cell_size bg_color);

      (* Draw entities *)
      let cx = px + cell_size / 2 in
      let cy = py + cell_size / 2 in

      (* Draw target marker if it's a target cell *)
      if value = 3 || value = 4 || value = 6 then begin
        Buffer.add_string buffer
          (Printf.sprintf
            {|<circle cx="%d" cy="%d" r="%d" fill="none" stroke="#cc6600" stroke-width="2" stroke-dasharray="3,3"/>
|}
            cx cy (cell_size / 3))
      end;

      (* Draw box *)
      if value = 2 || value = 4 then begin
        let box_size = cell_size * 3 / 4 in
        let box_color = if value = 4 then "#00aa00" else "#8b6914" in
        Buffer.add_string buffer
          (Printf.sprintf
            {|<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#654321" stroke-width="2" rx="3"/>
|}
            (cx - box_size / 2) (cy - box_size / 2) box_size box_size box_color)
      end;

      (* Draw player *)
      if value = 5 || value = 6 then begin
        Buffer.add_string buffer
          (Printf.sprintf
            {|<circle cx="%d" cy="%d" r="%d" fill="#ff6b6b" stroke="#cc0000" stroke-width="2"/>
<text x="%d" y="%d" text-anchor="middle" font-size="20" font-weight="bold" fill="white">@</text>
|}
            cx cy (cell_size / 3) cx (cy + 7))
      end;

      (* Draw wall pattern *)
      if value = 1 then begin
        (* Simple brick pattern *)
        Buffer.add_string buffer
          (Printf.sprintf
            {|<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#654321" stroke-width="1"/>
<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#654321" stroke-width="1"/>
|}
            px (py + cell_size / 2) (px + cell_size) (py + cell_size / 2)
            (px + cell_size / 2) py (px + cell_size / 2) (py + cell_size / 2))
      end
    done
  done;

  Buffer.add_string buffer "</svg>\n";
  Buffer.contents buffer

(* Generate HTML page for episode replay *)
let generate_episode_html _episode_dir episode_length =
  let buffer = Buffer.create 4096 in

  Buffer.add_string buffer {|<!DOCTYPE html>
<html>
<head>
    <title>Episode Replay</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-color: #f5f5f5;
        }
        #controls {
            margin: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #frame-info {
            font-size: 18px;
            margin: 0 20px;
        }
        #state-display {
            border: 2px solid #333;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        #speed-control {
            margin-left: 20px;
        }
        input[type="range"] {
            width: 150px;
        }
    </style>
</head>
<body>
    <h1>Episode Replay</h1>

    <div id="controls">
        <button id="play-pause">Play</button>
        <button id="reset">Reset</button>
        <button id="prev-frame">Previous</button>
        <button id="next-frame">Next</button>
        <span id="frame-info">Frame: <span id="current-frame">0</span> / <span id="total-frames">|};

  Buffer.add_string buffer (string_of_int (episode_length - 1));
  Buffer.add_string buffer {|</span></span>
        <div id="speed-control">
            <label for="speed">Speed: </label>
            <input type="range" id="speed" min="1" max="10" value="5">
            <span id="speed-value">5</span>
        </div>
    </div>

    <div id="state-display">
        <img id="state-image" src="0.svg" alt="State visualization">
    </div>

    <script>
        let currentFrame = 0;
        const totalFrames = |};
  Buffer.add_string buffer (string_of_int (episode_length - 1));
  Buffer.add_string buffer {|;
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 5;

        const stateImage = document.getElementById('state-image');
        const currentFrameSpan = document.getElementById('current-frame');
        const playPauseBtn = document.getElementById('play-pause');
        const prevBtn = document.getElementById('prev-frame');
        const nextBtn = document.getElementById('next-frame');
        const resetBtn = document.getElementById('reset');
        const speedSlider = document.getElementById('speed');
        const speedValue = document.getElementById('speed-value');

        function updateFrame() {
            stateImage.src = currentFrame + '.svg';
            currentFrameSpan.textContent = currentFrame;

            prevBtn.disabled = currentFrame === 0;
            nextBtn.disabled = currentFrame === totalFrames;
        }

        function play() {
            if (currentFrame >= totalFrames) {
                currentFrame = 0;
            }
            isPlaying = true;
            playPauseBtn.textContent = 'Pause';

            playInterval = setInterval(() => {
                if (currentFrame < totalFrames) {
                    currentFrame++;
                    updateFrame();
                } else {
                    pause();
                }
            }, 1000 / playSpeed);
        }

        function pause() {
            isPlaying = false;
            playPauseBtn.textContent = 'Play';
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }

        playPauseBtn.addEventListener('click', () => {
            if (isPlaying) {
                pause();
            } else {
                play();
            }
        });

        prevBtn.addEventListener('click', () => {
            if (currentFrame > 0) {
                currentFrame--;
                updateFrame();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentFrame < totalFrames) {
                currentFrame++;
                updateFrame();
            }
        });

        resetBtn.addEventListener('click', () => {
            pause();
            currentFrame = 0;
            updateFrame();
        });

        speedSlider.addEventListener('input', (e) => {
            playSpeed = parseInt(e.target.value);
            speedValue.textContent = playSpeed;
            if (isPlaying) {
                pause();
                play();
            }
        });

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    playPauseBtn.click();
                    break;
                case 'ArrowLeft':
                    prevBtn.click();
                    break;
                case 'ArrowRight':
                    nextBtn.click();
                    break;
                case 'r':
                    resetBtn.click();
                    break;
            }
        });

        // Initialize
        updateFrame();
    </script>
</body>
</html>
|};

  Buffer.contents buffer

(* Visualize all episodes from training history *)
let visualize_episodes history environment algorithm env_type ~grid_size =
  Printf.printf "Visualizing %d episodes for %s/%s\n"
    (List.length history.collected_episodes) environment algorithm;

  List.iteri (fun episode_idx episode_data ->
    let episode_num = (episode_idx + 1) * 10 in  (* Episodes are collected every 10 *)
    let episode_dir = Printf.sprintf "episodes/%s/%s/%d"
      environment algorithm episode_num in

    ensure_dir episode_dir;

    (* Generate SVG for each state *)
    let n_states = Array.length episode_data.states in
    for step = 0 to n_states - 1 do
      let state = episode_data.states.(step) in
      let svg_content =
        match env_type with
        | `Gridworld -> gridworld_state_to_svg state ~grid_size
        | `Sokoban -> sokoban_state_to_svg state ~grid_size
      in

      let svg_file = Printf.sprintf "%s/%d.svg" episode_dir step in
      let oc = open_out svg_file in
      output_string oc svg_content;
      close_out oc
    done;

    (* Generate HTML replay page *)
    let html_content = generate_episode_html episode_dir n_states in
    let html_file = Printf.sprintf "%s/replay.html" episode_dir in
    let oc = open_out html_file in
    output_string oc html_content;
    close_out oc;

    Printf.printf "  Episode %d: %d states -> %s/replay.html\n"
      episode_num n_states episode_dir
  ) history.collected_episodes

(* Main visualization function *)
let visualize_all_episodes () =
  Printf.printf "Starting episode visualization...\n";

  (* This would be called from plots.ml after training with the collected histories *)
  (* For now, just print a message *)
  Printf.printf "Visualization module ready. Call visualize_episodes with training history.\n"