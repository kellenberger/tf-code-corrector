# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""standard hparams utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def standard_hparams():
    return {
        # network
        "num_units": 32,
        "num_layers": 2,
        "num_encoder_layers": None,
        "num_decoder_layers": None,
        "encoder_type": "uni",
        "residual": False,
        "time_major": True,
        "num_embeddings_partitions": 0,

        # attention mechanisms
        "attention": "",
        "attention_architecture": "standard",
        "output_attention": True,
        "pass_hidden_state": True,

        # optimizer
        "optimizer": "sgd",
        "learning_rate": 1.0,
        "warmup_steps": 0,
        "warmup_scheme": "t2t",
        "decay_scheme": "",
        "num_train_steps": 12000,
        "colocate_gradients_with_ops": True,

        # initializer
        "init_op": "uniform",
        "init_weight": 0.1,

        # data
        "src": None,
        "tgt": None,
        "train_prefix": None,
        "dev_prefix": None,
        "test_prefix": None,
        "out_dir": None,

        # Vocab
        "vocab_prefix": None,
        "embed_prefix": None,
        "sos": "<s>",
        "eos": "</s>",
        "share_vocab": False,
        "check_special_token": True,

        # Sequence lengths
        "src_max_len": 50,
        "tgt_max_len": 50,
        "src_max_len_infer": None,
        "tgt_max_len_infer": None,

        # Default settings works well (rarely need to change)
        "unit_type": "lstm",
        "forget_bias": 1.0,
        "dropout": 0.2,
        "max_gradient_norm": 5.0,
        "batch_size": 128,
        "steps_per_stats": 100,
        "max_train": 0,
        "num_buckets": 5,

        # SPM
        "subword_option": "",

        # Misc
        "num_gpus": 1,
        "log_device_placement": False,
        "metrics": "bleu",
        "steps_per_external_eval": None,
        "scope": None,
        "hparams_path": None,
        "random_seed": None,
        "override_loaded_hparams": False,
        "num_keep_ckpts": 5,
        "avg_ckpts": False,

        # Inference
        "ckpt": "",
        "inference_input_file": None,
        "inference_list": None,
        "infer_batch_size": 32,
        "inference_output_file": None,
        "inference_ref_file": None,
        "beam_width": 0,
        "length_penalty_weight": 0.0,
        "sampling_temperature": 0.0,
        "num_translations_per_input": 1,

        # Job info
        "jobid": 0,
        "num_workers": 1,
        "num_inter_threads": 0,
        "num_intra_threads": 0,
    }
