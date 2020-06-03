## eGeMAPS
* 1 layer Fully Connected dev acc: `0.7028`
* 2 layer Fully Connected dev acc: `0.7080`
* 3 layer Fully Connected dev acc: `0.6355`

* With Dropout (best with 2 layers, dropout = 0.2): `0.71`
* With Batch Normalization: `0.7125`

* Manually tried different optimizers, learning rates, added ReduceLROnPlateau, tried out different hidden layer sizes
and eventually ended up with the following best hyperparameters:
    * nodes1 = 64
    * nodes2 = 32
    * nodes3 = 16
    * ReduceLROnPlateau: patience 2
    * epochs = 50
    * learning_rate = 0.001
    * l2_decay = 0.05
    * batch_size = 128
    * dropout = 0.5
Final Train acc: 0.7936
Final Train prf:  0.7858975932517323, 0.7792982733931817, 0.7820799551278901
Final dev acc: 0.7244
Final dev prf:  0.7079146217963699, 0.7185134171948142, 0.7108585393086813



## IS11
* 1 layer Fully Connected dev acc: `0.7423`
* 2 layer Fully Connected dev acc: `0.7409`
* 3 layer Fully Connected dev acc: `0.7341`

* With Dropout (best with 3 layers, dropout = 0.3): `0.7438`
* With Batch Normalization: `0.7423`

* best hyperparameters:
    * nodes1 = 128
    * nodes2 = 32
    * nodes3 = 16
    * ReduceLROnPlateau: patience 2
    * epochs = 50
    * learning_rate = 0.001
    * l2_decay = 0.05
    * batch_size = 128
    * dropout = 0.5
Final Train acc: 0.8880
Final Train prf:  0.8862263861499764, 0.8786650401836968, 0.8820074611816899
Final dev acc: 0.7446
Final dev prf:  0.7289749984859246, 0.7002875113174472, 0.7081695547772356
