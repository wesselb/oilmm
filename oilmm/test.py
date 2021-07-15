def create_test_class():
    class Test:
        def __getattr__(self, item):
            if item in self.__dict__:
                return self.__dict__[item]
            else:
                raise AttributeError(
                    f'Test attribute "{item}" requested but not available.'
                )

    return Test


def test_sample_prior():
    class TestSamplePrior(create_test_class()):
        def test_go(self):
            assert self.result

    return TestSamplePrior
