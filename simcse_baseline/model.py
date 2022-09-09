# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SimCSE(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, margin=0.0, scale=20, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768, output_emb_size, weight_attr=weight_attr)
        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.sacle = scale

    def get_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, with_pooler=False):
        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        if not with_pooler:
            cls_embedding = sequence_output[:, 0, :]
        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding

    def cosine_sim(
        self,
        query_input_ids,
        key_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        key_token_type_ids=None,
        key_position_ids=None,
        key_attention_mask=None,
        with_pooler=False
    ):
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler
        )
        key_cls_embedding = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler
        )
        cosine_sim = paddle.sum(query_cls_embedding * key_cls_embedding, axis=-1)
        return cosine_sim

    def forward(
        self,
        query_input_ids,
        key_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        key_token_type_ids=None,
        key_position_ids=None,
        key_attention_mask=None,
        with_pooler=False
    ):
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler
        )
        key_cls_embedding = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler
        )
        cosine_sim = paddle.matmul(query_cls_embedding, key_cls_embedding, transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype()
        )

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)
        return loss

