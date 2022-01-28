# from app import *
# import io
from torchimport import get_accuracy_confmat_ontest
from conftest import *


def test_homepage_route(test_client):
        response = test_client.get('/')
        assert response.status_code == 200

def test_wrong_route(test_client):
    response = test_client.get('/toto')
    assert response.status_code == 404


def test_predict_route(test_client,img_data):
    response = test_client.response = test_client.post('/predict', buffered=True,
                    content_type='multipart/form-data',
                    data = img_data)
    assert response.status_code == 200


def test_predict_output(test_client,img_data):
    response = test_client.response = test_client.post('/predict', buffered=True,
                    content_type='multipart/form-data',
                    data = img_data)
    assert response.data.decode("utf-8") in classNames

        # print('\n ------- just response \n',response)
        # print('\n ------- just response.data \n',response.data)
        # print(type(response.data))

def test_confuse_matrix():
    _ , confuseMtx = get_accuracy_confmat_ontest()
    assert confuseMtx.shape == (6,6)

def test_accuracy():
    accuracy, _ = get_accuracy_confmat_ontest()
    assert accuracy > 0.99


