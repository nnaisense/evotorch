# Copyright 2022 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from evotorch.tools.versionchecking import check_version


class FakeModule1:
    __version__ = "1.4.5"


class FakeModule2:
    __version__ = "2.5.notinteger"


class FakeModule3:
    pass


class FakeModule4:
    __version__ = "incompatible.version.string"


def test_check_version():
    assert check_version(FakeModule1, 1) == (1,)
    assert check_version(FakeModule1, 2) == (1, 4)
    assert check_version(FakeModule1, 3) == (1, 4, 5)
    assert check_version(FakeModule1, 4) is None
    assert check_version(FakeModule1, 5) is None

    assert check_version(FakeModule2, 1) == (2,)
    assert check_version(FakeModule2, 2) == (2, 5)
    assert check_version(FakeModule2, 3) is None
    assert check_version(FakeModule2, 4) is None

    for i in range(1, 6):
        assert check_version(FakeModule3, i) is None
        assert check_version(FakeModule4, i) is None
