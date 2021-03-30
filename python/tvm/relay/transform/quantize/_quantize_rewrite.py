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
# pylint: disable=no-else-return, invalid-name, unused-argument, too-many-arguments, consider-using-in
"""op quantization registration"""
from ... import op


def identity(post, scale, zp):
    return post


op.register_quantize_rewrite("argmax", identity)
op.register_quantize_rewrite("argmin", identity)
op.register_quantize_rewrite("argsort", identity)
op.register_quantize_rewrite("broadcast_to", identity)
op.register_quantize_rewrite("copy", identity)
op.register_quantize_rewrite("expand_dims", identity)
op.register_quantize_rewrite("layout_transform", identity)
op.register_quantize_rewrite("max", identity)
op.register_quantize_rewrite("min", identity)
op.register_quantize_rewrite("nn.adaptive_max_pool2d", identity)
op.register_quantize_rewrite("nn.adaptive_max_pool3d", identity)
op.register_quantize_rewrite("nn.batch_flatten", identity)
op.register_quantize_rewrite("nn.batch_to_space_nd", identity)
op.register_quantize_rewrite("nn.depth_to_space", identity)
op.register_quantize_rewrite("nn.max_pool1d", identity)
op.register_quantize_rewrite("nn.max_pool2d", identity)
op.register_quantize_rewrite("nn.max_pool3d", identity)
op.register_quantize_rewrite("nn.space_to_depth", identity)
op.register_quantize_rewrite("repeat", identity)
op.register_quantize_rewrite("reshape", identity)
op.register_quantize_rewrite("reverse", identity)
op.register_quantize_rewrite("shape_of", identity)
op.register_quantize_rewrite("sort", identity)
op.register_quantize_rewrite("squeeze", identity)
op.register_quantize_rewrite("strided_slice", identity)
op.register_quantize_rewrite("tile", identity)
op.register_quantize_rewrite("transpose", identity)


@op.register_quantize_rewrite("nn.relu")
def relu_quantize_rewrite(post, scale, zp):
    i = post.args[0]
    return op.maximum(i, op.cast_like(zp, i))
