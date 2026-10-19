# duckscout

MicroDuckがカメラで対象を探し、歩いて近づいて停止することを目指す、Physical AI応用1の最終課題プロジェクトです。

## 技術構成

- Python 3.12、uvによる環境・依存管理。
- MuJoCoによる物理シミュレーションと表示。
- 公式の学習済みONNX方策を使った歩行制御。
- カメラによる対象認識と、探索・接近・停止の行動判断。

対象環境はmacOS / Apple Siliconです。Python環境はリポジトリ内の `.venv/` に隔離します。

## ドキュメント

- [設計](docs/design.md)：構成、責務、制御方針、評価方法。
- [外部素材](THIRD_PARTY_NOTICES.md)：参照元とライセンス情報。
- [開発ルール](AGENTS.md)：変更・レビュー・検証のルール。
- [進捗](docs/progress.md)：実装状況と確認待ち事項。
