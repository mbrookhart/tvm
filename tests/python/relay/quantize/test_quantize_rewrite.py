# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from functools import partial
import pytest
import tvm
from tvm import relay
from tvm.relay.transform.quantize import requantize
from tvm.relay.frontend.common import infer_type

import numpy as np


def check_requantize(pre_graph, expected_graph, skip_list=[]):
    post_graph = requantize(pre_graph, skip_list)

    post_graph = infer_type(post_graph)
    expected_graph = infer_type(expected_graph)
    assert tvm.ir.structural_equal(post_graph, expected_graph)


def create_pre_graph(pre_op, ndims):
    data_shape = list(range(ndims))
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype="int8"))

    scale1 = relay.const(np.array(1).astype("float32"))
    zp1 = relay.const(np.array(2).astype("int32"))
    scale2 = relay.const(np.array(3).astype("float32"))
    zp2 = relay.const(np.array(4).astype("int32"))

    dequantize = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    fp32_op = pre_op(dequantize)
    quantize = relay.qnn.op.quantize(fp32_op, scale2, zp2)

    return dequantize, relay.Function([int8_data], quantize)


def test_quantize_relu():
    dq, pre_graph = create_pre_graph(relay.op.nn.relu, 4)
    q = pre_graph.body
    quant = relay.qnn.op.quantize(dq, q.args[1], q.args[2])
    zp = relay.op.cast_like(q.args[2], quant)
    int8_op = relay.op.maximum(quant, zp)

    expected_graph = relay.Function(pre_graph.params, int8_op)

    check_requantize(pre_graph, expected_graph)
    # test skippping
    check_requantize(pre_graph, pre_graph, ["nn.relu"])


@pytest.mark.parametrize(
    "pre_op",
    [
        (4, relay.op.argmax),
        (4, relay.op.argmin),
        (4, relay.op.argsort),
        (4, partial(relay.op.broadcast_to, shape=[2, 2, 3, 4])),
        (4, relay.op.copy),
        (4, partial(relay.op.expand_dims, axis=0)),
        (4, partial(relay.op.layout_transform, src_layout="NCHW", dst_layout="NCHW2c")),
        (4, relay.op.max),
        (4, relay.op.min),
        (4, relay.op.nn.adaptive_max_pool2d),
        (5, relay.op.nn.adaptive_max_pool3d),
        (4, relay.op.nn.batch_flatten),
        (4, partial(relay.op.nn.depth_to_space, block_size=1)),
        (3, relay.op.nn.max_pool1d),
        (4, relay.op.nn.max_pool2d),
        (5, relay.op.nn.max_pool3d),
        (4, partial(relay.op.nn.space_to_depth, block_size=1)),
        (4, partial(relay.op.repeat, axis=0, repeats=2)),
        (4, partial(relay.op.reshape, newshape=[1, -1])),
        (4, partial(relay.op.reverse, axis=3)),
        (4, relay.op.shape_of),
        (4, relay.op.sort),
        (4, relay.op.squeeze),
        (4, partial(relay.op.strided_slice, begin=[0, 0, 0, 0], end=[1, 1, 1, 1])),
        (4, partial(relay.op.tile, reps=[2, 2, 1, 1])),
        (4, relay.op.transpose),
    ],
)
def test_identity_op(pre_op):
    ndims, pre_op = pre_op
    dq, pre_graph = create_pre_graph(pre_op, ndims)
    q = pre_graph.body
    quant = relay.qnn.op.quantize(dq, q.args[1], q.args[2])
    int8_op = pre_op(quant)
    expected_graph = relay.Function(pre_graph.params, int8_op)
    check_requantize(pre_graph, expected_graph)
