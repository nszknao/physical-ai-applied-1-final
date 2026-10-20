# duckscout

MicroDuckがカメラで対象を探し、歩いて近づいて停止することを目指す、Physical AI応用1の最終課題プロジェクトです。

## セットアップ

Python 3.12とuvを使用します。以下はリポジトリのルートで実行してください。Python環境は `.venv/` に作成されます。

```sh
uv sync
uv run python scripts/fetch_assets.py
```

素材はPollen Roboticsの固定revisionから取得します（42ファイル、約22 MB）。取得済みファイルは検証して再利用します。接続切断・タイムアウト・一時的なHTTPエラーは、1・2・4・8秒の間隔で最大4回再試行します。途中で停止しても同じコマンドで再開できます。破損や変更を検出した場合は上書きせず停止するため、表示された対象ファイルを確認し、必要なら退避して取得をやり直してください。

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

## 検証

```sh
uv run python scripts/fetch_assets.py --check
uv run python -m duckscout --headless --steps 300
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

`--check` はネットワークを使わず素材を検証します。`--headless` は物理更新と数値状態を確認し、表示・歩行の安定性は評価しません。

コードのlint・フォーマットには開発用依存のRuffを使用します。整形を適用する場合は `uv run ruff format .` を実行してください。

## ドキュメント

- [設計](docs/design.md)：構成、責務、制御方針、評価方法。
- [外部素材](THIRD_PARTY_NOTICES.md)：参照元とライセンス情報。
- [開発ルール](AGENTS.md)：変更・レビュー・検証のルール。
- [進捗](docs/progress.md)：動作確認状況と確認待ち事項。
