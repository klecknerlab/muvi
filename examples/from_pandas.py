#!/usr/bin/python3
#
# Copyright 2022 Dustin Kleckner
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

import pickle
from muvi import geometry
import os

odir = 'geometry_examples'
if not os.path.exists(odir):
    os.makedirs(odir)
    
with open('test_link_df.pkl', 'rb') as f:
    dat = pickle.load(f)

geometry.from_pandas(dat).save(os.path.join(odir, 'pandas_test.vtp'))
