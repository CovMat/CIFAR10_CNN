def my_read( fdir ):
    import pickle
    import numpy as np

    training_image = np.empty( [50000, 32, 32, 3] )
    training_label = np.empty( [50000] )
    for i in range(5):
        fin = open( fdir+"/data_batch_"+str(i+1) , 'rb' )
        tmp_dict = pickle.load( fin, encoding='bytes' )
        tmp_data = tmp_dict[b'data']
        tmp_label= tmp_dict[b'labels']
        tmp_data = tmp_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        training_image[ i*10000:(i+1)*10000, :, :, : ] = tmp_data
        training_label[ i*10000:(i+1)*10000 ] = np.asarray( tmp_label )
        fin.close()
    training_image -= 128
    training_image /= 128

    testing_image = np.empty( [10000, 32, 32, 3] )
    testing_label = np.empty( [10000] )
    fin = open( fdir+"/test_batch", "rb" )
    tmp_dict = pickle.load( fin, encoding='bytes' )
    tmp_data = tmp_dict[b'data']
    tmp_label= tmp_dict[b'labels']
    tmp_data = tmp_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    testing_image[ 0:10000, :, :, : ] = tmp_data
    testing_label[ 0:10000 ] = np.asarray( tmp_label )
    fin.close()
    testing_image -= 128
    testing_image /= 128

    return training_image, training_label, testing_image, testing_label
