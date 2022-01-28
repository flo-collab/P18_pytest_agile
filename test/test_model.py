import pytest
from app import *
import numpy as np

# test constructeur
def test_numClass_high():
    with pytest.raises(RuntimeError) as high_nb_class:
        numClass = 12
        model = MedNet(imageWidth,imageHeight,numClass).to(dev)
        m_state_dict = torch.load('model_mednist_statedict_9983.pt', map_location=torch.device('cpu'))
        model.load_state_dict(m_state_dict)
    assert "size mismatch" in str(high_nb_class.value)

def test_numClass_low():
    with pytest.raises(RuntimeError) as low_nb_class:
        numClass = 4
        model = MedNet(imageWidth,imageHeight,numClass).to(dev)
        m_state_dict = torch.load('model_mednist_statedict_9983.pt', map_location=torch.device('cpu'))
        model.load_state_dict(m_state_dict)
    assert "size mismatch" in str(low_nb_class.value)

def test_numClass_good():
    numClass = 6
    model = MedNet(imageWidth,imageHeight,numClass).to(dev)
    m_state_dict = torch.load('model_mednist_statedict_9983.pt', map_location=torch.device('cpu'))
    model.load_state_dict(m_state_dict)

@pytest.mark.parametrize("method",['num_flat_features','forward'])
def test_method_exists(method):
    assert method in dir(model)

# test methode forward
def test_forward_no_arg():
    with pytest.raises(TypeError) as forward_no_arg:
        model = MedNet(imageWidth,imageHeight,numClass).to(dev)
        m_state_dict = torch.load('model_mednist_statedict_9983.pt', map_location=torch.device('cpu'))
        model.load_state_dict(m_state_dict)
        model.forward()
    assert "required positional argument" in str(forward_no_arg.value)

def test_forward():
    x=np.random.randint(255, size=(64,64))
    tensor_img = scaleImage(x)
    tensor_img = tensor_img[None,:]
    model.forward(tensor_img)

def test_forward_3_dim():
    with pytest.raises(RuntimeError) as forward_3_dim: 
        x=np.random.randint(255, size=(64,64))
        tensor_img = toTensor(x)
        model.forward(tensor_img)
    assert "Expected 4-dimensional input" in str(forward_3_dim.value)

def test_forward_case():
    with pytest.raises(AttributeError) as forward_case:
        x=np.random.randint(255, size=(64,64))
        tensor_img = scaleImage(x)
        tensor_img = tensor_img[None,:]
        model.Forward(tensor_img)
        model.FORWARD(tensor_img)

# test methode  num_flat_features
def test_num_flat_features_no_arg():
    with pytest.raises(TypeError) as flat_no_arg:
        model.num_flat_features()
    assert "required positional argument" in str(flat_no_arg.value)

@pytest.mark.parametrize("arg",['toto',666,3.14,None])
def test_num_flat_features_wrong_arg(arg):
    with pytest.raises(AttributeError) as flat_wrong_arg:
        model.num_flat_features(arg)

def test_num_flat_features():
    x=np.random.randint(255, size=(64,64))
    tensor_img = scaleImage(x)
    tensor_img = tensor_img[None,:]
    model.num_flat_features(tensor_img)

def test_num_flat_features_return_int():
    x=np.random.randint(255, size=(64,64))
    tensor_img = scaleImage(x)
    tensor_img = tensor_img[None,:]
    y = model.num_flat_features(tensor_img)
    assert type(y)==int

def test_num_flat_features_not_null():
    x=np.random.randint(255, size=(64,64))
    tensor_img = scaleImage(x)
    tensor_img = tensor_img[None,:]
    y = model.num_flat_features(tensor_img)
    assert y !=0

