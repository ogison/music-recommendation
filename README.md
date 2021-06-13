# music-recommendation

対象ユーザの視聴履歴を使って学習して次に再生される楽曲を予測します。論文["Session-based Recommendations With Recurrent Neural Networks"](https://arxiv.org/abs/1511.06939)を元に作成しました。コードの書き方は[master_thesis](https://github.com/olesls/master_thesis)を参考にしてます。
 
# Requirement
 
Python 3.7.10  
tensorflow 2.5.0  
Numpy   
Pickle  
 
# Usage
## Datasets
[lastfmの視聴履歴のデータセット](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)を使います。  
ダウンロードして解凍したら以下の場所にデータを配置してください。　  
～/datasets/lastfm/

## Preprocessing
`preprocess.py`を実行して計4段階に分けてデータの前処理を行います。

## Running the RNN models
`main.py`を実行してデータの学習と評価を行います。Recall@5の結果がよくなるたび、`save`フォルダにモデルのパラメータが保存されます。  
また、`testlog`フォルダに評価結果が保存されます。`save`、`testlog`フォルダは作成されていない場合、自動的に作成します。
