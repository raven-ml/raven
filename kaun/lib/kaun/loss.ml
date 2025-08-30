let softmax_cross_entropy logits labels =
  Rune.debug_with_context "softmax_cross_entropy" (fun () ->
      (* Assumes labels are one-hot encoded *)
      let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
      let exp_logits = Rune.exp (Rune.sub logits max_logits) in
      let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
      let log_softmax =
        Rune.sub logits (Rune.add max_logits (Rune.log sum_exp))
      in
      let loss =
        Rune.neg (Rune.sum (Rune.mul labels log_softmax) ~axes:[| -1 |])
      in
      Rune.mean loss)

let softmax_cross_entropy_with_indices logits indices =
  (* Convert indices to one-hot encoding *)
  let indices_int = Rune.cast Rune.int32 indices in
  let num_classes = (Rune.shape logits).(1) in
  let one_hot = Rune.one_hot ~num_classes indices_int in
  let one_hot_float = Rune.cast (Rune.dtype logits) one_hot in
  softmax_cross_entropy logits one_hot_float

let binary_cross_entropy predictions labels =
  Rune.debug_with_context "binary_cross_entropy" (fun () ->
      let dtype = Rune.dtype predictions in
      let dev = Rune.device predictions in
      let one = Rune.scalar dev dtype 1.0 in
      let eps = Rune.scalar dev dtype 1e-7 in
      (* Clip predictions to avoid log(0) *)
      let predictions_clipped =
        Rune.maximum eps (Rune.minimum (Rune.sub one eps) predictions)
      in
      let term1 = Rune.mul labels (Rune.log predictions_clipped) in
      let term2 =
        Rune.mul (Rune.sub one labels)
          (Rune.log (Rune.sub one predictions_clipped))
      in
      let loss_per_example = Rune.neg (Rune.add term1 term2) in
      Rune.mean loss_per_example)

let sigmoid_binary_cross_entropy logits labels =
  Rune.debug_with_context "sigmoid_binary_cross_entropy" (fun () ->
      let dtype = Rune.dtype logits in
      let dev = Rune.device logits in
      let one = Rune.scalar dev dtype 1.0 in
      let log_sig = Rune.log_sigmoid logits in
      let log_sig_neg = Rune.log_sigmoid (Rune.neg logits) in
      let term1 = Rune.mul labels log_sig in
      let term2 = Rune.mul (Rune.sub one labels) log_sig_neg in
      Rune.neg (Rune.add term1 term2))

let mse predictions targets =
  Rune.debug_with_context "mse" (fun () ->
      let diff = Rune.sub predictions targets in
      let squared = Rune.mul diff diff in
      Rune.mean squared)

let mae predictions targets =
  Rune.debug_with_context "mae" (fun () ->
      let diff = Rune.sub predictions targets in
      let abs_diff = Rune.abs diff in
      Rune.mean abs_diff)
