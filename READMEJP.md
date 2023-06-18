# NumPyCNNを使った画像分類

[English](https://github.com/KanaMeisa/ImageClassifier-CNN/blob/master/README.md)

これは私の評価作品で、NumPyを使ったCNNと、PyQtで実装されたGUIが付いているんだ。プロジェクトの面白さと難しさは、CNN全体をNumPyで実現していて、DeepLearningフレームワーク(PyTorch、TensorFlowなど)は一切使っていないところなんだ。

このプログラムでは、NumPyで構築されたCNNを使って、小さなニューラルネットワークフレームワークを実現していて、CNNのいろんなレイヤー（線形、畳み込み、プーリングレイヤーなど）や活性化関数（ReLu、Sigmoid、Softmaxなど）やクロスエントロピー損失関数が含まれているんだ。このプロジェクトは、CNNの基本的な仕組みを理解したい人にとってすごく役に立つよ。



## CNNフレームワーク

![Framework](./.idea/framework.jpg)



## データセット1: 車両

![Database1](./.idea/training_history1.png)



## データセット2: シーン

![Database2](./.idea/training_history2.png)



## GUI

<img src="./.idea/GUI.jpg" style="width:50%;height:50%;" />
