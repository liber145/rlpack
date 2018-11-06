"""
测试类的模板
"""


class MyCustomClass:
    def __init__(self, attr):
        self.attr = self._attr = attr

    def sample(self, n):
        return list(range(n))

    def foo(self, a, b):
        return a + b


class MyCustomClass2(MyCustomClass):
    def __init__(self, attr, new_attr):
        super().__init__(attr)
        self.new_attr = attr
