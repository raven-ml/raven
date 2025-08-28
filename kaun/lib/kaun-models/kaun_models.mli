(** Kaun_models: Model zoo with pre-defined neural network architectures.

    Provides implementations of popular models that can be easily instantiated
    and used with pretrained weights from HuggingFace or trained from scratch.
*)

(** {1 Classic Vision Models} *)

module LeNet = Lenet
(** LeNet-5: Classic CNN for handwritten digit recognition.
    @inline *)

(** {1 Language Models} *)

module Bert = Bert
(** BERT: Bidirectional Encoder Representations from Transformers.
    @inline *)

module GPT2 = Gpt2
(** GPT-2: Generative Pre-trained Transformer 2 for causal language modeling.
    @inline *)
