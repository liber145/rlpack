"""
如何写一个类的测试



pytest测试module, 所以test目录下应当有__init__.py

依照tensorflow的方式,所有的module内的文件用test_xxxx.py来测试,不单独设立tests文件夹


在终端运行

$ pytest

即可看到结果


============================= test session starts ==============================
platform linux -- Python 3.6.6, pytest-3.10.0, py-1.7.0, pluggy-0.8.0
rootdir: /home/chenyu/dev/rl-algo, inifile:
collected 1 item

test_pytest_format.py .                                                  [100%]

=========================== 1 passed in 0.01 seconds ===========================

"""

import pytest

from .pytest_format import MyCustomClass, MyCustomClass2


@pytest.fixture()
def instance1():
    """
        使用pytest.fixture装饰器创建实例
    """
    ins = MyCustomClass(1)
    return ins


@pytest.fixture()
def instance2():
    # create a breakout gym environment
    ins = MyCustomClass2(3, 4)
    return ins


@pytest.mark.unit_test
def test_sample(instance1):
    # observation space
    assert type(instance1.sample(6)) is list
    assert len(instance1.sample(6)) == 6


@pytest.mark.unit_test
def test_sample(instance2):
    # observation space
    assert hasattr(instance2, 'attr')
