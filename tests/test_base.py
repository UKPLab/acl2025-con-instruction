# Tests are defined here
from acl2025_con_instruction import BaseClass
from acl2025_con_instruction.subpackage import SubPackageClass

def test_template():
    assert True

def test_base_class():
    bc1 = BaseClass(name="test1")
    bc2 = BaseClass(name="test2")

    assert str(bc1) == "test1"
    assert repr(bc1) == "test1"
    assert bc1 != bc2
    assert bc1.something() == "something"

def test_subpackage():
    spc = SubPackageClass(name="test")

    assert str(spc) == "SubPackage - test"
    assert repr(spc) == "SubPackage - test"
    assert spc.something() == "SubPackage - something"