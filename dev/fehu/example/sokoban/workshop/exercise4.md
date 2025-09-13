# Exercise 4: Upgrading to CNN-based Policy and Value Networks

## Background

So far, our policy networks have used simple fully-connected layers after flattening the grid:

```ocaml
let create_policy_network grid_size num_actions =
  Kaun.Layer.sequential [
    Kaun.Layer.flatten ();  (* 5×5 grid → 25 values *)
    Kaun.Layer.linear ~in_features:(grid_size * grid_size)
                      ~out_features:32 ();
    Kaun.Layer.relu ();
    (* ... more FC layers ... *)
  ]
```

This approach loses spatial structure! A box at (1,2) and at (2,1) are treated as completely unrelated features.

## The Power of Convolutional Neural Networks

CNNs preserve spatial relationships through:
- **Local connectivity**: Each filter looks at a small region
- **Weight sharing**: Same pattern detector across the grid
- **Translation invariance**: "Box next to wall" detected anywhere

## Your Task

Replace the fully-connected networks with CNN architectures for both policy and value networks.

## Implementation Guide

### Step 1: CNN Policy Network

```ocaml
let create_cnn_policy_network num_actions =
  Kaun.Layer.sequential [
    (* Input shape: [batch, 1, 5, 5] - single channel for grid values *)

    (* First conv layer: detect basic patterns *)
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                      ~kernel_size:(3, 3) ~padding:(1, 1) ();
    Kaun.Layer.relu ();

    (* Second conv layer: combine patterns *)
    Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32
                      ~kernel_size:(3, 3) ~padding:(1, 1) ();
    Kaun.Layer.relu ();

    (* Global pooling or flatten for final layers *)
    Kaun.Layer.flatten ();
    (* Output size after 5×5 with padding: 32 * 5 * 5 = 800 *)
    Kaun.Layer.linear ~in_features:800 ~out_features:64 ();
    Kaun.Layer.relu ();

    (* Output layer for actions *)
    Kaun.Layer.linear ~in_features:64 ~out_features:num_actions ();
  ]
```

### Step 2: CNN Value Network

```ocaml
let create_cnn_value_network () =
  Kaun.Layer.sequential [
    (* Similar architecture but single value output *)
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                      ~kernel_size:(3, 3) ~padding:(1, 1) ();
    Kaun.Layer.relu ();

    Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32
                      ~kernel_size:(3, 3) ~padding:(1, 1) ();
    Kaun.Layer.relu ();

    (* Spatial average pooling to reduce dimensions *)
    Kaun.Layer.adaptive_avg_pool2d ~output_size:(1, 1) ();
    Kaun.Layer.flatten ();

    (* Small FC layer for value estimation *)
    Kaun.Layer.linear ~in_features:32 ~out_features:16 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:16 ~out_features:1 ();
  ]
```

### Step 3: Adjust Input Shape

The grid needs a channel dimension for conv2d:

```ocaml
(* In collect_episode or when processing states *)
let prepare_state_for_cnn state =
  (* state shape: [5, 5] *)
  (* Add channel dimension: [1, 5, 5] *)
  Rune.reshape [|1; 5; 5|] state

(* When batching for training *)
let prepare_batch_for_cnn states =
  (* states: array of [5, 5] tensors *)
  (* Stack and add channel: [batch_size, 1, 5, 5] *)
  let states_with_channel = Array.map (fun s ->
    Rune.reshape [|1; 1; 5; 5|] s
  ) states in
  Rune.concat ~axis:0 (Array.to_list states_with_channel)
```

## Key Design Decisions

### 1. Kernel Size
- **3×3**: Standard choice, captures local patterns
- **5×5**: For our small grid, could use entire grid as one kernel!

### 2. Number of Filters
- Start small (16-32) since our grid is tiny
- More filters = more pattern types detected

### 3. Padding
- `padding:(1,1)` for 3×3 kernels maintains spatial dimensions
- Important for small grids to avoid losing edge information

### 4. Pooling Strategy
- **No pooling**: Keep all spatial information (5×5 is already small)
- **Global average pooling**: Reduce to single value per channel
- **Max pooling**: Could lose important details on small grids

## Advanced Techniques

### Residual Connections
```ocaml
let conv_residual_block in_channels out_channels =
  fun input ->
    let conv1 = Kaun.Layer.conv2d ~in_channels ~out_channels
                                  ~kernel_size:(3, 3) ~padding:(1, 1) () in
    let conv2 = Kaun.Layer.conv2d ~in_channels:out_channels ~out_channels
                                  ~kernel_size:(3, 3) ~padding:(1, 1) () in
    let x = input |> conv1 |> Kaun.Layer.relu () |> conv2 in
    (* Add skip connection *)
    Rune.add input x |> Kaun.Layer.relu ()
```

### Spatial Attention
Focus on important regions (player, boxes, targets):

```ocaml
let spatial_attention_layer channels =
  Kaun.Layer.sequential [
    (* Reduce channels to 1 for attention map *)
    Kaun.Layer.conv2d ~in_channels:channels ~out_channels:1
                      ~kernel_size:(1, 1) ();
    Kaun.Layer.sigmoid ();  (* Attention weights 0-1 *)
  ]

(* Apply: multiply features by attention weights *)
```

## Testing Your Implementation

### 1. Verify Shapes
```ocaml
let test_cnn_shapes () =
  let net = create_cnn_policy_network 4 in
  let params = Kaun.init net ~rngs:(Rune.Rng.split_n ~n:1 rng).(0)
                          ~device ~dtype:Rune.float32 in

  (* Test single state *)
  let state = Rune.randn device Rune.float32 [|1; 1; 5; 5|] in
  let output = Kaun.apply net params ~training:false state in
  assert (Rune.shape output = [|1; 4|]);  (* batch=1, actions=4 *)

  (* Test batch *)
  let batch = Rune.randn device Rune.float32 [|32; 1; 5; 5|] in
  let output = Kaun.apply net params ~training:false batch in
  assert (Rune.shape output = [|32; 4|])  (* batch=32, actions=4 *)
```

### 2. Compare Performance

Track metrics with both architectures:
- **Learning speed**: Episodes to reach 80% success
- **Final performance**: Success rate after 1000 episodes
- **Stability**: Variance in returns
- **Generalization**: Test on unseen grid configurations

## Expected Improvements

CNNs should provide:
1. **Faster learning**: Spatial patterns recognized immediately
2. **Better generalization**: "Box next to wall" works anywhere
3. **Lower parameter count**: Weight sharing reduces total parameters
4. **Interpretable filters**: Can visualize what patterns are detected

## Common Pitfalls

1. **Forgetting channel dimension**: Conv2d needs [batch, channels, height, width]
2. **Wrong padding**: Without padding, 5×5 → 3×3 → 1×1 (too much reduction!)
3. **Too many filters**: 5×5 grid doesn't need 256 filters like ImageNet
4. **Batch norm on small batches**: Can be unstable with batch_size < 32

## Bonus Challenges

1. **Visualize learned filters**: What patterns do first-layer filters detect?
2. **Multi-channel input**: Separate channels for walls, boxes, targets, player
3. **Dilated convolutions**: Capture longer-range dependencies
4. **Comparison study**: Plot learning curves for FC vs CNN on increasingly complex grids

## Solution Outline

A complete solution is provided in `solutions/exercise4_cnn_networks.ml`. Key points:
- Separate CNN architectures for policy and value
- Careful shape management throughout
- Performance comparison with original FC networks
- Visualization of learned spatial features

Remember: For a 5×5 Sokoban grid, the performance difference might be subtle. The real benefits emerge with larger grids or when transferring to new puzzles!