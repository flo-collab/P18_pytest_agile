from app import *
# import io
from torchimport import get_accuracy_confmat_ontest

flask_app = app

def test_homepage_route():
    with flask_app.test_client() as test_client:
        response = test_client.get('/')
        assert response.status_code == 200

def test_wrong_route():
    with flask_app.test_client() as test_client:
        response = test_client.get('/toto')
        assert response.status_code == 404

def test_predict_route():
    with flask_app.test_client() as test_client:
        with open('test/000006.jpeg', 'rb') as f:
            response = test_client.response = test_client.post('/predict', buffered=True,
                            content_type='multipart/form-data',
                            data={'image_title' : 'New york from the top',
                                    'description' : 'lalalala',
                                    'file' :( f, '000006.jpeg')})
        assert response.status_code == 200

def test_predict_output():
    with flask_app.test_client() as test_client:
        with open('test/000006.jpeg', 'rb') as f:
            response = test_client.response = test_client.post('/predict', buffered=True,
                            content_type='multipart/form-data',
                            data={
                                #'image_title' : 'Turlututu',
                                   #'description' : 'lalalala',
                                    'file' :( f, '000006.jpeg')})
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
