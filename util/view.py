#!/usr/bin/python3
#
# Copyright 2019 Dustin Kleckner
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

from muvi.view.qtview import view_volume
import sys
from muvi import open_4D_movie

vol = open_4D_movie(sys.argv[1])

ret = view_volume(vol)

vol.close()

sys.exit(ret)
