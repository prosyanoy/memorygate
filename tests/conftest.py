import pytest

from memorygate.generator import generate
from memorygate.models import GeneratorConfig


@pytest.fixture(scope="session")
def dataset():
    return generate(GeneratorConfig(seed=11, n_customers=30))
