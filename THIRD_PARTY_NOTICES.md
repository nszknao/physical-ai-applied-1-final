# 外部コード・モデル

## MicroDuckモデル

モデルの著作権者は **Pollen Robotics** です。

- 出典：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
- 固定revision：`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`
- 利用対象：`scene.xml`、`robot_groundcontact.xml`、これらが参照する38個のSTLメッシュ。
- 取得時の改変：なし。上流のファイル内容と相対配置を保持。
- 実行時の変更：頭部カメラの向きをMuJoCoの画像座標系に合わせてメモリ上で補正。独自シーンから固定対象物を追加。BAM使用時にはアクチュエータ・摩擦設定をメモリ上で変更。
- MJCF（XML）：[Apache-2.0](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/LICENSE)。
- STLメッシュ：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。範囲・帰属表示の根拠は[上流のライセンス説明](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/LICENSE-MODELS.md)。

素材はGitリポジトリに同梱せず、`scripts/fetch_assets.py` で取得します。上流の `LICENSE` と `LICENSE-MODELS.md` も取得・保持します。取得対象とGit blobハッシュは `assets/sources.toml`、取得結果のSHA-256は `assets/vendor/microduck_rl/checksums.json` に記録します。

メッシュを共有・改変する際はPollen Roboticsへの帰属表示、出典とライセンスへのリンク、改変の表示、適用ライセンスの条件を保持してください。

## 立位方策と制御仕様

- 方策：[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)、`velstand.onnx`。
- 固定revision：`665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`。
- ライセンス：配布元の[モデルカード](https://huggingface.co/pollen-robotics/microduck-policies/blob/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448/README.md)でApache-2.0と表示。方策の改変なし。
- 方策はGit管理せず、`assets/vendor/microduck-policies/` に取得。サイズとSHA-256を `assets/sources.toml` に固定し、取得記録を同ディレクトリの `checksums.json` に保存。
- 観測順序、基準姿勢、BAM設定はPollen Roboticsの[公式CPU推論コード](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/scripts/infer_policy.py)（Apache-2.0）を参照・適用。
- 本プロジェクトでの変更：前進・旋回・停止の指令、CLI、互換性検証、一時停止・リセット、姿勢判定、テストを独自実装。

## 実行ライブラリ

[MuJoCo](https://github.com/google-deepmind/mujoco)を物理シミュレーションと表示に使用します。方策の推論には[ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）、関節制御には[BAM](https://github.com/Rhoban/bam)（Apache-2.0、Marc Duclusaud & Grégoire Passault）を使用します。BAMは上流の依存ロックに合わせてrevision `62bd8ce12154340be97e06f7f41a0ca8f116d967` に固定します。Python依存は `pyproject.toml` で宣言し、解決結果を `uv.lock` に保存します。
