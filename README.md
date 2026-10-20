# duckscout

MicroDuckがカメラで対象を探し、歩いて近づいて停止することを目指す、Physical AI応用1の最終課題プロジェクトです。

## セットアップ

Python 3.12とuvを使用します。以下はリポジトリのルートで実行してください。Python環境は `.venv/` に作成されます。

```sh
uv sync
uv run python scripts/fetch_assets.py
```

素材はPollen Roboticsの固定revisionから取得します（モデル・メッシュ・ライセンス42ファイルと、移動方策1ファイル、計約23 MB）。取得済みファイルは検証して再利用します。接続切断・タイムアウト・一時的なHTTPエラーは、1・2・4・8秒の間隔で最大4回再試行します。途中で停止しても同じコマンドで再開できます。破損や変更を検出した場合は上書きせず停止するため、表示された対象ファイルを確認し、必要なら退避して取得をやり直してください。

## モデル表示

通常のパッケージ起動を使用します。macOSでは内部で `mjpython` へ切り替え、実行中のPythonの共有ライブラリ探索パスを補います。設定は起動プロセス内だけに適用します。

```sh
uv run python -m duckscout
```

ロボット・床・正面の赤い対象物を一時停止状態で表示します。ウィンドウにフォーカスを合わせて操作してください。

| 操作 | 動作 |
|---|---|
| Space | 物理更新の再開・一時停止 |
| R | 初期姿勢へ戻して一時停止 |
| Esc／ウィンドウを閉じる | 終了 |

ビューアは公式XMLの位置制御を使用し、学習済み方策による姿勢維持は行いません。物理更新を再開すると倒れる場合があります。

初期姿勢は `--keyframe SIT` などで選べます。別のシーンには `--scene /absolute/path/to/scene.xml` を指定します。macOS以外でも同じコマンドを使用できます。

## 立位・前進・旋回・停止

```sh
uv run python -m duckscout --policy
```

取得済みの公式 `velstand.onnx` を使います。起動時は移動指令ゼロ・一時停止状態です。ウィンドウにフォーカスを合わせ、Spaceで物理更新を開始してください。

| 操作 | 動作 |
|---|---|
| ↑ | 前進（目標0.30 m/s） |
| ← | その場で左旋回（目標+0.80 rad/s） |
| → | その場で右旋回（目標−0.80 rad/s） |
| ↓ | 移動指令をゼロにして停止。姿勢制御は継続 |
| Space | シミュレーション時間の一時停止・再開 |
| R | 姿勢・移動指令・制御履歴を初期化し、一時停止 |
| Esc／ウィンドウを閉じる | 終了 |

矢印キーは一度押すと指令を保持し、次の矢印キーで置き換えます。前進中に←を押すと前進指令をゼロにして左旋回へ切り替わります。キーを離しても停止しないため、停止には↓を使ってください。一時停止中も指令は変更でき、Spaceで再開するとその指令で動きます。カメラはロボットの位置を追従します。

指令値と実速度は一致するとは限りません。停止指令後も減速・姿勢調整の動きがあり、微小な揺れが残る場合があります。転倒相当の高さ・傾き、非有限値を検出した場合はエラーを表示して終了します。

`--policy /path/to/velstand.onnx` で別の保存先も指定できます。同じ観測・関節・動作仕様の方策が必要で、任意のONNXモデルには対応しません。方策使用時の初期姿勢はSTANDに限定します。

ウィンドウを開かず、30秒分の立位を確認するには次を実行します。

```sh
uv run python -m duckscout --policy --headless --steps 6000
```

胴体の最低高さ・最大傾き・開始位置からの最大水平変位を出力します。方策使用時は物理200 Hz・推論50 Hzで動作し、モデル表示のみのモードとは1ステップの時間が異なります。headless起動では移動指令はゼロです。移動と停止のシナリオは下記の自動テストで実行します。判定条件と学習環境との差は[設計](docs/design.md#移動制御)を参照してください。

## 対象物と頭部カメラ

既定のシーンは `assets/scenes/search.xml` です。初期位置の正面80 cmに、一辺8 cmの赤い立方体を床へ固定しています。配置を変えるにはXMLの `target` の `pos` を編集します。長さの単位はm、`size` は各辺の半分です。物体との衝突は有効です。

頭部視点で起動するには、次を実行してください。

```sh
uv run python -m duckscout --policy --camera head
```

起動直後は一時停止しています。Space・矢印キーによる操作は外部視点と同じです。左側の **Rendering → Camera** で `head_camera` と `Tracking`（外部追従視点）を切り替えられます。パネルが隠れている場合はTabで表示します。Trackingは選択中のbodyを追跡し、方策使用時は起動時に胴体を選択します。

**Enter** を押すと、その時点の頭部カメラ画像を `runs/head_camera.ppm`、検出結果付き画像を `runs/head_camera_detection.ppm` に保存します。表示中の視点に関係なく頭部画像を保存し、同じパスのファイルは上書きします。`--snapshot runs/another.ppm` で保存先を変えられます。PPMはRGBをそのまま保存する画像形式で、追加ライブラリは不要です。macOSでは次のコマンドでPNGへ変換できます。

```sh
sips -s format png runs/head_camera.ppm --out runs/head_camera.png
open runs/head_camera.png
```

RGB画像は320×240で、シミュレーション時間基準で10 Hzで取得します。一時停止中は定期取得も止まり、Enterで現在の画像を取得・保存します。ビューアの更新周期・表示領域の縦横比は保存画像とは異なります。元画像はビューアのUI・選択マーク・検出枠を含みません。

ウィンドウを開かず、2秒立位後の元画像と検出結果付き画像を保存する場合：

```sh
uv run python -m duckscout --policy --headless --steps 400 --snapshot runs/head_camera.ppm
```

画像保存を指定したheadless実行にはOpenGLの描画環境が必要です。`--snapshot` を付けないheadless実行は描画コンテキストを作りません。

## 対象検出

```sh
uv run python -m duckscout --policy
```

右下に頭部カメラ画像を表示し、赤い対象候補を**緑の枠**、その中心を**黄色の十字**で示します。上部の `x`・`y` は画像左上を原点とする中心座標、`area` は検出した赤い画素の数です。検出できない場合は枠が消えて `Target: not detected` と表示します。外部視点・頭部視点のどちらを選んでも、この小窓は頭部の画像を表示します。小窓への画像転送は画像更新時とウィンドウサイズ変更時に行います。小窓はウィンドウに合わせて拡縮しますが、認識には常に320×240の元画像を使います。

画像取得と同じ10 Hzで毎回検出します。一時停止中は最後の結果を表示し、リセットとEnterで画像・検出結果を更新します。Spaceで開始し、←／→で対象を視野外へ出してから戻すと、見失い・再検出を確認できます。動きを止めるには↓を使ってください。

検出結果は表示・保存に使用し、移動は矢印キーで操作します。色に基づく検出なので、赤い別物体を区別することや、暗さ・照明変化への完全な対応はできません。

Enterで保存した検出結果付き画像は、macOSでは次で開けます。

```sh
sips -s format png runs/head_camera_detection.ppm --out runs/head_camera_detection.png
open runs/head_camera_detection.png
```

`--snapshot runs/example.ppm` を指定すると元画像と `runs/example_detection.ppm` を保存します。両ファイルは同じフレームから生成され、同名ファイルを上書きします。

## 検証

```sh
uv run python scripts/fetch_assets.py --check
uv run python -m duckscout --headless --steps 300
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

`--check` はネットワークを使わず素材を検証します。`--headless` は物理更新と数値状態を確認します。`--policy` 併用時は高さ・傾きも検証します。表示やキー操作の確認は別途必要です。実方策のテストは素材取得後に実行され、ファイルがない場合はskipと表示されます。立位維持に加え、前進・左右旋回からの停止、旋回後の前進をシミュレーション時間で検証します。

テストファイルはソース単位で対応させています。`test_main.py` は `__main__.py`、`test_locomotion.py` は `locomotion.py`、`test_perception.py` は `perception.py`、`test_simulation.py` は `simulation.py`、`test_fetch_assets.py` は `scripts/fetch_assets.py` を対象にします。シミュレーションを通じた移動やカメラの検証も `test_simulation.py` に置き、共通の素材パスとモック方策の準備だけを `tests/helpers.py` にまとめています。

コードのlint・フォーマットには開発用依存のRuffを使用します。整形を適用する場合は `uv run ruff format .` を実行してください。

## ドキュメント

- [設計](docs/design.md)：構成、責務、制御方針、評価方法。
- [外部素材](THIRD_PARTY_NOTICES.md)：参照元とライセンス情報。
- [開発ルール](AGENTS.md)：変更・レビュー・検証のルール。
- [進捗](docs/progress.md)：動作確認状況と確認待ち事項。
