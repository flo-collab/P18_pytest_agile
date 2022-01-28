# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(currentdir)
# sys.path.append(parentdir)
import pytest
from torchimport import *
from app import *

dataDir,classNames,numClass,imageFiles,nnumEach,imageFilesList,imageClass,numTotal,imageWidth, imageHeight 

@pytest.mark.parametrize("classname", [(cls) for cls in classNames])
def test_classnames_is_str(classname):
    assert type(classname)==str

def test_numclass_is_int():
    assert type(numClass)==int

def test_classnames_len_equal_numclass():
    assert len(classNames)==numClass

@pytest.mark.parametrize("img_size", [imageWidth, imageHeight ])
def test_img_size_is_int(img_size):
    assert type(img_size)==int


