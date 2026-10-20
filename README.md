# duckscout

MicroDuckがカメラで対象を探し、歩いて近づいて停止することを目指す、Physical AI応用1の最終課題プロジェクトです。

## セットアップ

Python 3.12とuvを使用します。以下はリポジトリのルートで実行してください。Python環境は `.venv/` に作成されます。

```sh
uv sync
uv run python scripts/fetch_assets.py
```

素材はPollen Roboticsの固定revisionから取得します（モデル・メッシュ・ライセンス42ファイルと、立位方策1ファイル、計約23 MB）。取得済みファイルは検証して再利用します。接続切断・タイムアウト・一時的なHTTPエラーは、1・2・4・8秒の間隔で最大4回再試行します。途中で停止しても同じコマンドで再開できます。破損や変更を検出した場合は上書きせず停止するため、表示された対象ファイルを確認し、必要なら退避して取得をやり直してください。

## モデル表示

通常のパッケージ起動を使用します。macOSでは内部で `mjpython` へ切り替え、実行中のPythonの共有ライブラリ探索パスを補います。設定は起動プロセス内だけに適用します。

```sh
uv run python -m duckscout
```

ロボットと床を一時停止状態で表示します。ウィンドウにフォーカスを合わせて操作してください。

| 操作 | 動作 |
|---|---|
| Space | 物理更新の再開・一時停止 |
| R | 初期姿勢へ戻して一時停止 |
| Esc／ウィンドウを閉じる | 終了 |

ビューアは公式XMLの位置制御を使用し、学習済み方策による姿勢維持は行いません。物理更新を再開すると倒れる場合があります。

初期姿勢は `--keyframe SIT` などで選べます。別のシーンには `--scene /absolute/path/to/scene.xml` を指定します。macOS以外でも同じコマンドを使用できます。

## 立位維持

```sh
uv run python -m duckscout --policy
```

取得済みの公式 `velstand.onnx` を使い、移動・頭部・胴体の指令をすべてゼロにして立位を維持します。起動時は一時停止しています。Spaceで開始・一時停止、Rで姿勢と制御履歴を初期化して一時停止します。倒れた場合や開始位置から10 cm以上ずれた場合は、エラーを表示して終了します。

`--policy /path/to/velstand.onnx` で別の保存先も指定できます。同じ観測・関節・動作仕様の方策が必要で、任意のONNXモデルには対応しません。立位モードの初期姿勢はSTANDに限定します。

ウィンドウを開かず、30秒分の立位を確認するには次を実行します。

```sh
uv run python -m duckscout --policy --headless --steps 6000
```

胴体の最低高さ・最大傾き・最大水平ずれを出力します。立位モードは物理200 Hz・推論50 Hzで動作し、モデル表示のみのモードとは1ステップの時間が異なります。判定条件と学習環境との差は[設計](docs/design.md#立位制御)を参照してください。

## 検証

```sh
uv run python scripts/fetch_assets.py --check
uv run python -m duckscout --headless --steps 300
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

`--check` はネットワークを使わず素材を検証します。`--headless` は物理更新と数値状態を確認します。`--policy` 併用時は立位条件も検証します。表示やキー操作の確認は別途必要です。実方策のテストは素材取得後に実行され、ファイルがない場合はskipと表示されます。

コードのlint・フォーマットには開発用依存のRuffを使用します。整形を適用する場合は `uv run ruff format .` を実行してください。

## ドキュメント

- [設計](docs/design.md)：構成、責務、制御方針、評価方法。
- [外部素材](THIRD_PARTY_NOTICES.md)：参照元とライセンス情報。
- [開発ルール](AGENTS.md)：変更・レビュー・検証のルール。
- [進捗](docs/progress.md)：動作確認状況と確認待ち事項。
